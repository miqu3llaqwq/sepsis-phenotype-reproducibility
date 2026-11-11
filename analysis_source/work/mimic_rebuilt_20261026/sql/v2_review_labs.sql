WITH rebuilt_soi AS (-- note this duplicates prescriptions
-- each ICU stay in the same hospitalization will get a copy of
-- all prescriptions for that hospitalization
WITH ab_tbl AS (
    SELECT
        abx.subject_id, abx.hadm_id, abx.stay_id
        , abx.antibiotic
        , abx.starttime AS antibiotic_time
        -- date is used to match microbiology cultures with only date available
        , DATETIME_TRUNC(abx.starttime, DAY) AS antibiotic_date
        , abx.stoptime AS antibiotic_stoptime
        -- create a unique identifier for each patient antibiotic
        , ROW_NUMBER() OVER
        (
            PARTITION BY subject_id
            ORDER BY starttime, stoptime, antibiotic
        ) AS ab_id
    FROM `physionet-data.mimiciv_3_1_derived.antibiotic` abx
)

, me AS (
    SELECT micro_specimen_id
        -- the following columns are identical for all rows
        -- of the same micro_specimen_id
        -- these aggregates simply collapse duplicates down to 1 row
        , MAX(subject_id) AS subject_id
        , MAX(hadm_id) AS hadm_id
        , CAST(MAX(chartdate) AS DATE) AS chartdate
        , MAX(charttime) AS charttime
        , MAX(spec_type_desc) AS spec_type_desc
        -- identify negative cultures as NULL organism
        -- or a specific itemid saying "NEGATIVE"
        , MAX(
            CASE WHEN org_name IS NOT NULL
                      AND org_itemid != 90856
                      AND org_name != ''
                THEN 1 ELSE 0
            END) AS positiveculture
    FROM `physionet-data.mimiciv_3_1_hosp.microbiologyevents`
    GROUP BY micro_specimen_id
)

-- culture followed by an antibiotic
, me_then_ab AS (
    SELECT
        ab_tbl.subject_id
        , ab_tbl.hadm_id
        , ab_tbl.stay_id
        , ab_tbl.ab_id

        , me72.micro_specimen_id
        , COALESCE(me72.charttime, DATETIME(me72.chartdate)) AS last72_charttime
        , me72.positiveculture AS last72_positiveculture
        , me72.spec_type_desc AS last72_specimen

        -- we will use this partition to select the earliest culture
        -- before this abx
        -- this ensures each antibiotic is only matched to a single culture
        -- and consequently we have 1 row per antibiotic
        , ROW_NUMBER() OVER
        (
            PARTITION BY ab_tbl.subject_id, ab_tbl.ab_id
            ORDER BY me72.chartdate, me72.charttime NULLS LAST
        ) AS micro_seq
    FROM ab_tbl
    -- abx taken after culture, but no more than 72 hours after
    LEFT JOIN me me72
        ON ab_tbl.subject_id = me72.subject_id
            AND
            (
                (
                    -- if charttime is available, use it
                    me72.charttime IS NOT NULL
                    AND ab_tbl.antibiotic_time > me72.charttime
                    AND ab_tbl.antibiotic_time <= DATETIME_ADD(
                        me72.charttime, INTERVAL 72 HOUR
                    )
                )
                OR
                (
                    -- if charttime is not available, use chartdate
                    me72.charttime IS NULL
                    AND antibiotic_date >= me72.chartdate
                    AND antibiotic_date <= DATE_ADD(
                        me72.chartdate, INTERVAL 3 DAY
                    )
                )
            )
)

, ab_then_me AS (
    SELECT
        ab_tbl.subject_id
        , ab_tbl.hadm_id
        , ab_tbl.stay_id
        , ab_tbl.ab_id

        , me24.micro_specimen_id
        , COALESCE(me24.charttime, DATETIME(me24.chartdate)) AS next24_charttime
        , me24.positiveculture AS next24_positiveculture
        , me24.spec_type_desc AS next24_specimen

        -- we will use this partition to select the earliest culture
        -- before this abx
        -- this ensures each antibiotic is only matched to a single culture
        -- and consequently we have 1 row per antibiotic
        , ROW_NUMBER() OVER
        (
            PARTITION BY ab_tbl.subject_id, ab_tbl.ab_id
            ORDER BY me24.chartdate, me24.charttime NULLS LAST
        ) AS micro_seq
    FROM ab_tbl
    -- culture in subsequent 24 hours
    LEFT JOIN me me24
        ON ab_tbl.subject_id = me24.subject_id
            AND
            (
                (
                    -- if charttime is available, use it
                    me24.charttime IS NOT NULL
                    AND ab_tbl.antibiotic_time >= DATETIME_SUB(me24.charttime, INTERVAL 24 HOUR)  -- noqa: L016
                    AND ab_tbl.antibiotic_time < me24.charttime
                )
                OR
                (
                    -- if charttime is not available, use chartdate
                    me24.charttime IS NULL
                    AND ab_tbl.antibiotic_date >= DATE_SUB(me24.chartdate, INTERVAL 1 DAY)  -- noqa: L016
                    AND ab_tbl.antibiotic_date <= me24.chartdate
                )
            )
)

SELECT
    ab_tbl.subject_id
    , ab_tbl.stay_id
    , ab_tbl.hadm_id
    , ab_tbl.ab_id
    , ab_tbl.antibiotic
    , ab_tbl.antibiotic_time

    , CASE
        WHEN last72_specimen IS NULL AND next24_specimen IS NULL
            THEN 0
        ELSE 1
    END AS suspected_infection
    -- time of suspected infection:
    --    (1) the culture time (if before antibiotic)
    --    (2) or the antibiotic time (if before culture)
    , CASE
        WHEN last72_specimen IS NULL AND next24_specimen IS NULL
            THEN NULL
        ELSE COALESCE(last72_charttime, antibiotic_time)
    END AS suspected_infection_time

    , COALESCE(last72_charttime, next24_charttime) AS culture_time

    -- the specimen that was cultured
    , COALESCE(last72_specimen, next24_specimen) AS specimen

    -- whether the cultured specimen ended up being positive or not
    , COALESCE(
        last72_positiveculture, next24_positiveculture
    ) AS positive_culture

FROM ab_tbl
LEFT JOIN ab_then_me ab2me
    ON ab_tbl.subject_id = ab2me.subject_id
        AND ab_tbl.ab_id = ab2me.ab_id
        AND ab2me.micro_seq = 1
LEFT JOIN me_then_ab me2ab
    ON ab_tbl.subject_id = me2ab.subject_id
        AND ab_tbl.ab_id = me2ab.ab_id
        AND me2ab.micro_seq = 1
), rebuilt_sepsis3 AS (-- Creates a table with "onset" time of Sepsis-3 in the ICU.
-- That is, the earliest time at which a patient had SOFA >= 2
-- and suspicion of infection.
-- As many variables used in SOFA are only collected in the ICU,
-- this query can only define sepsis-3 onset within the ICU.

-- extract rows with SOFA >= 2
-- implicitly this assumes baseline SOFA was 0 before ICU admission.
WITH sofa AS (
    SELECT stay_id
        , starttime, endtime
        , respiration_24hours AS respiration
        , coagulation_24hours AS coagulation
        , liver_24hours AS liver
        , cardiovascular_24hours AS cardiovascular
        , cns_24hours AS cns
        , renal_24hours AS renal
        , sofa_24hours AS sofa_score
    FROM `physionet-data.mimiciv_3_1_derived.sofa`
    WHERE sofa_24hours >= 2
)

, s1 AS (
    SELECT
        soi.subject_id
        , soi.stay_id
        -- suspicion columns
        , soi.ab_id
        , soi.antibiotic
        , soi.antibiotic_time
        , soi.culture_time
        , soi.suspected_infection
        , soi.suspected_infection_time
        , soi.specimen
        , soi.positive_culture
        -- sofa columns
        , starttime, endtime
        , respiration, coagulation, liver, cardiovascular, cns, renal
        , sofa_score
        -- All rows have an associated suspicion of infection event
        -- Therefore, Sepsis-3 is defined as SOFA >= 2.
        -- Implicitly, the baseline SOFA score is assumed to be zero,
        -- as we do not know if the patient has preexisting
        -- (acute or chronic) organ dysfunction before the onset
        -- of infection.
        , sofa_score >= 2 AND suspected_infection = 1 AS sepsis3
        -- subselect to the earliest suspicion/antibiotic/SOFA row
        , ROW_NUMBER() OVER
        (
            PARTITION BY soi.stay_id
            ORDER BY
                suspected_infection_time, antibiotic_time, culture_time, endtime
        ) AS rn_sus
    FROM rebuilt_soi AS soi
    INNER JOIN sofa
        ON soi.stay_id = sofa.stay_id
            AND sofa.endtime >= DATETIME_SUB(
                soi.suspected_infection_time, INTERVAL '48' HOUR
            )
            AND sofa.endtime <= DATETIME_ADD(
                soi.suspected_infection_time, INTERVAL '24' HOUR
            )
    -- only include in-ICU rows
    WHERE soi.stay_id IS NOT NULL
)

SELECT
    subject_id, stay_id
    -- note: there may be more than one antibiotic given at this time
    , antibiotic_time
    -- culture times may be dates, rather than times
    , culture_time
    , suspected_infection_time
    -- endtime is latest time at which the SOFA score is valid
    , endtime AS sofa_time
    , sofa_score
    , respiration, coagulation, liver, cardiovascular, cns, renal
    , sepsis3
FROM s1
WHERE rn_sus = 1), candidate AS (
 SELECT s.subject_id, s.stay_id, i.hadm_id, s.sepsis3,
 s.sofa_time, s.suspected_infection_time, s.antibiotic_time, s.culture_time,
 p.gender, p.anchor_age, p.anchor_year, p.anchor_year_group,
 p.anchor_age + EXTRACT(YEAR FROM a.admittime) - p.anchor_year AS age,
 i.intime AS icu_intime, i.outtime AS icu_outtime,
 a.admittime, a.dischtime, a.deathtime,
 a.hospital_expire_flag AS hosp_mortality,
 GREATEST(s.sofa_time, s.antibiotic_time, s.culture_time, i.intime) AS t0
 FROM rebuilt_sepsis3 s
 LEFT JOIN `physionet-data.mimiciv_3_1_icu.icustays` i ON s.stay_id = i.stay_id AND s.subject_id = i.subject_id
 LEFT JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a ON i.hadm_id = a.hadm_id AND s.subject_id = a.subject_id
 LEFT JOIN `physionet-data.mimiciv_3_1_hosp.patients` p ON s.subject_id = p.subject_id
), eligibility AS (
 SELECT *, CASE
 WHEN sepsis3 IS NULL OR NOT sepsis3 THEN 'not_sepsis3'
 WHEN hadm_id IS NULL OR gender IS NULL THEN 'unlinked_record'
 WHEN age IS NULL OR age < 18 THEN 'age_ineligible'
 WHEN t0 IS NULL OR admittime IS NULL OR dischtime IS NULL
   OR icu_intime IS NULL OR icu_outtime IS NULL THEN 'missing_time'
 WHEN hosp_mortality IS NULL OR hosp_mortality NOT IN (0,1) THEN 'missing_outcome'
 WHEN t0 < admittime OR t0 >= dischtime OR t0 < icu_intime
   OR t0 >= icu_outtime THEN 'landmark_outside_stay'
 WHEN hosp_mortality = 1 AND deathtime IS NULL THEN 'missing_death_time'
 WHEN hosp_mortality = 1 AND (deathtime <= t0 OR deathtime > dischtime)
   THEN 'invalid_death_time'
 WHEN hosp_mortality = 0 AND deathtime IS NOT NULL THEN 'inconsistent_death_flag'
 ELSE NULL END AS exclusion_reason
 FROM candidate
), eligible_ranked AS (
 SELECT *, ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY t0, hadm_id, stay_id) AS eligible_rank
 FROM eligibility WHERE exclusion_reason IS NULL
), cohort AS (
 SELECT * EXCEPT(exclusion_reason, eligible_rank),
 IF(hosp_mortality = 1, 1, 2) AS event_type,
 IF(hosp_mortality = 1, deathtime, dischtime) AS endpoint,
 DATETIME_DIFF(IF(hosp_mortality = 1, deathtime, dischtime), t0, SECOND)/86400.0 AS follow_up_days
 FROM eligible_ranked WHERE eligible_rank = 1
) , old_cohort AS (WITH candidate AS (
 SELECT s.subject_id, s.stay_id, i.hadm_id, s.sepsis3,
 s.sofa_time, s.suspected_infection_time, s.antibiotic_time, s.culture_time,
 p.gender, p.anchor_age, p.anchor_year, p.anchor_year_group,
 p.anchor_age + EXTRACT(YEAR FROM a.admittime) - p.anchor_year AS age,
 i.intime AS icu_intime, i.outtime AS icu_outtime,
 a.admittime, a.dischtime, a.deathtime,
 a.hospital_expire_flag AS hosp_mortality,
 GREATEST(s.sofa_time, s.antibiotic_time, s.culture_time, i.intime) AS t0
 FROM `physionet-data.mimiciv_3_1_derived.sepsis3` s
 LEFT JOIN `physionet-data.mimiciv_3_1_icu.icustays` i ON s.stay_id = i.stay_id AND s.subject_id = i.subject_id
 LEFT JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a ON i.hadm_id = a.hadm_id AND s.subject_id = a.subject_id
 LEFT JOIN `physionet-data.mimiciv_3_1_hosp.patients` p ON s.subject_id = p.subject_id
), eligibility AS (
 SELECT *, CASE
 WHEN sepsis3 IS NULL OR NOT sepsis3 THEN 'not_sepsis3'
 WHEN hadm_id IS NULL OR gender IS NULL THEN 'unlinked_record'
 WHEN age IS NULL OR age < 18 THEN 'age_ineligible'
 WHEN t0 IS NULL OR admittime IS NULL OR dischtime IS NULL
   OR icu_intime IS NULL OR icu_outtime IS NULL THEN 'missing_time'
 WHEN hosp_mortality IS NULL OR hosp_mortality NOT IN (0,1) THEN 'missing_outcome'
 WHEN t0 < admittime OR t0 >= dischtime OR t0 < icu_intime
   OR t0 >= icu_outtime THEN 'landmark_outside_stay'
 WHEN hosp_mortality = 1 AND deathtime IS NULL THEN 'missing_death_time'
 WHEN hosp_mortality = 1 AND (deathtime <= t0 OR deathtime > dischtime)
   THEN 'invalid_death_time'
 WHEN hosp_mortality = 0 AND deathtime IS NOT NULL THEN 'inconsistent_death_flag'
 ELSE NULL END AS exclusion_reason
 FROM candidate
), eligible_ranked AS (
 SELECT *, ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY t0, hadm_id, stay_id) AS eligible_rank
 FROM eligibility WHERE exclusion_reason IS NULL
), cohort AS (
 SELECT * EXCEPT(exclusion_reason, eligible_rank),
 IF(hosp_mortality = 1, 1, 2) AS event_type,
 IF(hosp_mortality = 1, deathtime, dischtime) AS endpoint,
 DATETIME_DIFF(IF(hosp_mortality = 1, deathtime, dischtime), t0, SECOND)/86400.0 AS follow_up_days
 FROM eligible_ranked WHERE eligible_rank = 1
)  SELECT * FROM cohort), delta_cohort AS (
SELECT n.* FROM cohort n LEFT JOIN old_cohort o USING(subject_id)
WHERE o.subject_id IS NULL OR n.stay_id != o.stay_id OR n.t0 != o.t0) SELECT c.subject_id,c.stay_id,c.hadm_id,c.t0,e.charttime,e.storetime,
e.itemid,e.valuenum,e.valueuom,e.specimen_id
FROM `physionet-data.mimiciv_3_1_hosp.labevents` e JOIN delta_cohort c ON e.hadm_id=c.hadm_id AND e.subject_id=c.subject_id
WHERE e.itemid IN (50813,50882,50885,50912,50931,50971,50983,51006,51222,51265,51301) AND e.valuenum IS NOT NULL
AND e.charttime BETWEEN DATETIME_SUB(c.t0, INTERVAL 24 HOUR) AND c.t0
AND e.storetime IS NOT NULL AND e.storetime <= c.t0