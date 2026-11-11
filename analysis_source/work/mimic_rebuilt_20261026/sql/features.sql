-- Generated from config/analysis_v2.json by analysis/08_rebuild.py.
-- Retrospective evidence-complete landmark; NOT a real-time sepsis alert.
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
),
vitals_raw AS (
 SELECT c.stay_id, e.charttime, e.storetime, e.itemid,
 CASE WHEN e.itemid IN (220045) THEN 'heart_rate' WHEN e.itemid IN (220050,220179) THEN 'sbp' WHEN e.itemid IN (220052,220181) THEN 'map' WHEN e.itemid IN (220210) THEN 'resp_rate' WHEN e.itemid IN (223761,223762) THEN 'temp_c' WHEN e.itemid IN (220277) THEN 'spo2' END AS feature, CASE WHEN e.itemid = 223761 THEN (e.valuenum-32)*5.0/9.0 ELSE e.valuenum END AS value,
 LOWER(REPLACE(REPLACE(REPLACE(e.valueuom,' ',''),'碌','u'),'渭','u')) AS unit,
 e.storetime IS NOT NULL AND e.storetime <= c.t0 AS available
 FROM `physionet-data.mimiciv_3_1_icu.chartevents` e JOIN cohort c ON e.stay_id = c.stay_id
 WHERE e.itemid IN (220045,220050,220179,220052,220181,220210,223761,223762,220277) AND e.valuenum IS NOT NULL
 AND e.charttime BETWEEN DATETIME_SUB(c.t0, INTERVAL 6 HOUR) AND c.t0
), vitals_screened AS (
 SELECT *, CASE WHEN feature = 'heart_rate' THEN value BETWEEN 1 AND 400 WHEN feature = 'sbp' THEN value BETWEEN 1 AND 400 WHEN feature = 'map' THEN value BETWEEN 1 AND 300 WHEN feature = 'resp_rate' THEN value BETWEEN 1 AND 100 WHEN feature = 'temp_c' THEN value BETWEEN 25 AND 45 WHEN feature = 'spo2' THEN value BETWEEN 1 AND 100 ELSE FALSE END AS valid,
 CASE WHEN feature='heart_rate' THEN unit IN ('beats/min','bpm') WHEN feature='sbp' THEN unit IN ('mmhg') WHEN feature='map' THEN unit IN ('mmhg') WHEN feature='resp_rate' THEN unit IN ('breaths/min','insp/min') WHEN feature='temp_c' AND itemid=223761 THEN unit IN ('f','deg.f','deg.fahrenheit','掳f','?f') WHEN feature='temp_c' THEN unit IN ('c','deg.c','deg.celsius','掳c','?c') WHEN feature='spo2' THEN unit IN ('%') ELSE FALSE END AS unit_valid
 FROM vitals_raw
), vitals_wide AS (
 SELECT stay_id,
 APPROX_QUANTILES(IF(feature = 'heart_rate' AND available AND valid AND unit_valid, value, NULL), 2)[SAFE_OFFSET(1)] AS heart_rate, APPROX_QUANTILES(IF(feature = 'sbp' AND available AND valid AND unit_valid, value, NULL), 2)[SAFE_OFFSET(1)] AS sbp, APPROX_QUANTILES(IF(feature = 'map' AND available AND valid AND unit_valid, value, NULL), 2)[SAFE_OFFSET(1)] AS map, APPROX_QUANTILES(IF(feature = 'resp_rate' AND available AND valid AND unit_valid, value, NULL), 2)[SAFE_OFFSET(1)] AS resp_rate, APPROX_QUANTILES(IF(feature = 'temp_c' AND available AND valid AND unit_valid, value, NULL), 2)[SAFE_OFFSET(1)] AS temp_c, APPROX_QUANTILES(IF(feature = 'spo2' AND available AND valid AND unit_valid, value, NULL), 2)[SAFE_OFFSET(1)] AS spo2,
 COUNTIF(feature = 'heart_rate' AND available AND NOT valid) AS heart_rate_invalid_n, COUNTIF(feature = 'heart_rate' AND NOT available) AS heart_rate_unavailable_n, COUNTIF(feature = 'heart_rate' AND available AND (unit_valid IS NULL OR NOT unit_valid)) AS heart_rate_unit_unknown_n, COUNTIF(feature = 'sbp' AND available AND NOT valid) AS sbp_invalid_n, COUNTIF(feature = 'sbp' AND NOT available) AS sbp_unavailable_n, COUNTIF(feature = 'sbp' AND available AND (unit_valid IS NULL OR NOT unit_valid)) AS sbp_unit_unknown_n, COUNTIF(feature = 'map' AND available AND NOT valid) AS map_invalid_n, COUNTIF(feature = 'map' AND NOT available) AS map_unavailable_n, COUNTIF(feature = 'map' AND available AND (unit_valid IS NULL OR NOT unit_valid)) AS map_unit_unknown_n, COUNTIF(feature = 'resp_rate' AND available AND NOT valid) AS resp_rate_invalid_n, COUNTIF(feature = 'resp_rate' AND NOT available) AS resp_rate_unavailable_n, COUNTIF(feature = 'resp_rate' AND available AND (unit_valid IS NULL OR NOT unit_valid)) AS resp_rate_unit_unknown_n, COUNTIF(feature = 'temp_c' AND available AND NOT valid) AS temp_c_invalid_n, COUNTIF(feature = 'temp_c' AND NOT available) AS temp_c_unavailable_n, COUNTIF(feature = 'temp_c' AND available AND (unit_valid IS NULL OR NOT unit_valid)) AS temp_c_unit_unknown_n, COUNTIF(feature = 'spo2' AND available AND NOT valid) AS spo2_invalid_n, COUNTIF(feature = 'spo2' AND NOT available) AS spo2_unavailable_n, COUNTIF(feature = 'spo2' AND available AND (unit_valid IS NULL OR NOT unit_valid)) AS spo2_unit_unknown_n
 FROM vitals_screened GROUP BY stay_id
),
labs_raw AS (
 SELECT c.stay_id, e.charttime, e.storetime, e.itemid,
 CASE WHEN e.itemid IN (50813) THEN 'lactate' WHEN e.itemid IN (51301) THEN 'wbc' WHEN e.itemid IN (51265) THEN 'platelet' WHEN e.itemid IN (51222) THEN 'hemoglobin' WHEN e.itemid IN (50912) THEN 'creatinine' WHEN e.itemid IN (51006) THEN 'bun' WHEN e.itemid IN (50885) THEN 'bilirubin' WHEN e.itemid IN (50983) THEN 'sodium' WHEN e.itemid IN (50971) THEN 'potassium' WHEN e.itemid IN (50882) THEN 'bicarbonate' WHEN e.itemid IN (50931) THEN 'glucose' END AS feature, e.valuenum AS value,
 LOWER(REPLACE(REPLACE(REPLACE(e.valueuom,' ',''),'碌','u'),'渭','u')) AS unit,
 e.storetime IS NOT NULL AND e.storetime <= c.t0 AS available
 FROM `physionet-data.mimiciv_3_1_hosp.labevents` e JOIN cohort c ON e.hadm_id = c.hadm_id AND e.subject_id = c.subject_id
 WHERE e.itemid IN (50813,51301,51265,51222,50912,51006,50885,50983,50971,50882,50931) AND e.valuenum IS NOT NULL
 AND e.charttime BETWEEN DATETIME_SUB(c.t0, INTERVAL 24 HOUR) AND c.t0
), labs_screened AS (
 SELECT *, CASE WHEN feature = 'lactate' THEN value BETWEEN 0.01 AND 100 WHEN feature = 'wbc' THEN value BETWEEN 0.01 AND 1000 WHEN feature = 'platelet' THEN value BETWEEN 0 AND 2000 WHEN feature = 'hemoglobin' THEN value BETWEEN 0.1 AND 30 WHEN feature = 'creatinine' THEN value BETWEEN 0.01 AND 50 WHEN feature = 'bun' THEN value BETWEEN 0.1 AND 500 WHEN feature = 'bilirubin' THEN value BETWEEN 0 AND 100 WHEN feature = 'sodium' THEN value BETWEEN 50 AND 200 WHEN feature = 'potassium' THEN value BETWEEN 0.5 AND 15 WHEN feature = 'bicarbonate' THEN value BETWEEN 0 AND 60 WHEN feature = 'glucose' THEN value BETWEEN 1 AND 3000 ELSE FALSE END AS valid,
 CASE WHEN feature='lactate' THEN unit IN ('mmol/l') WHEN feature='wbc' THEN unit IN ('k/ul') WHEN feature='platelet' THEN unit IN ('k/ul') WHEN feature='hemoglobin' THEN unit IN ('g/dl') WHEN feature='creatinine' THEN unit IN ('mg/dl') WHEN feature='bun' THEN unit IN ('mg/dl') WHEN feature='bilirubin' THEN unit IN ('mg/dl') WHEN feature='sodium' THEN unit IN ('meq/l','mmol/l') WHEN feature='potassium' THEN unit IN ('meq/l','mmol/l') WHEN feature='bicarbonate' THEN unit IN ('meq/l','mmol/l') WHEN feature='glucose' THEN unit IN ('mg/dl') ELSE FALSE END AS unit_valid
 FROM labs_raw
), labs_wide AS (
 SELECT stay_id,
 APPROX_QUANTILES(IF(feature = 'lactate' AND available AND valid AND unit_valid, value, NULL), 2)[SAFE_OFFSET(1)] AS lactate, APPROX_QUANTILES(IF(feature = 'wbc' AND available AND valid AND unit_valid, value, NULL), 2)[SAFE_OFFSET(1)] AS wbc, APPROX_QUANTILES(IF(feature = 'platelet' AND available AND valid AND unit_valid, value, NULL), 2)[SAFE_OFFSET(1)] AS platelet, APPROX_QUANTILES(IF(feature = 'hemoglobin' AND available AND valid AND unit_valid, value, NULL), 2)[SAFE_OFFSET(1)] AS hemoglobin, APPROX_QUANTILES(IF(feature = 'creatinine' AND available AND valid AND unit_valid, value, NULL), 2)[SAFE_OFFSET(1)] AS creatinine, APPROX_QUANTILES(IF(feature = 'bun' AND available AND valid AND unit_valid, value, NULL), 2)[SAFE_OFFSET(1)] AS bun, APPROX_QUANTILES(IF(feature = 'bilirubin' AND available AND valid AND unit_valid, value, NULL), 2)[SAFE_OFFSET(1)] AS bilirubin, APPROX_QUANTILES(IF(feature = 'sodium' AND available AND valid AND unit_valid, value, NULL), 2)[SAFE_OFFSET(1)] AS sodium, APPROX_QUANTILES(IF(feature = 'potassium' AND available AND valid AND unit_valid, value, NULL), 2)[SAFE_OFFSET(1)] AS potassium, APPROX_QUANTILES(IF(feature = 'bicarbonate' AND available AND valid AND unit_valid, value, NULL), 2)[SAFE_OFFSET(1)] AS bicarbonate, APPROX_QUANTILES(IF(feature = 'glucose' AND available AND valid AND unit_valid, value, NULL), 2)[SAFE_OFFSET(1)] AS glucose,
 COUNTIF(feature = 'lactate' AND available AND NOT valid) AS lactate_invalid_n, COUNTIF(feature = 'lactate' AND NOT available) AS lactate_unavailable_n, COUNTIF(feature = 'lactate' AND available AND (unit_valid IS NULL OR NOT unit_valid)) AS lactate_unit_unknown_n, COUNTIF(feature = 'wbc' AND available AND NOT valid) AS wbc_invalid_n, COUNTIF(feature = 'wbc' AND NOT available) AS wbc_unavailable_n, COUNTIF(feature = 'wbc' AND available AND (unit_valid IS NULL OR NOT unit_valid)) AS wbc_unit_unknown_n, COUNTIF(feature = 'platelet' AND available AND NOT valid) AS platelet_invalid_n, COUNTIF(feature = 'platelet' AND NOT available) AS platelet_unavailable_n, COUNTIF(feature = 'platelet' AND available AND (unit_valid IS NULL OR NOT unit_valid)) AS platelet_unit_unknown_n, COUNTIF(feature = 'hemoglobin' AND available AND NOT valid) AS hemoglobin_invalid_n, COUNTIF(feature = 'hemoglobin' AND NOT available) AS hemoglobin_unavailable_n, COUNTIF(feature = 'hemoglobin' AND available AND (unit_valid IS NULL OR NOT unit_valid)) AS hemoglobin_unit_unknown_n, COUNTIF(feature = 'creatinine' AND available AND NOT valid) AS creatinine_invalid_n, COUNTIF(feature = 'creatinine' AND NOT available) AS creatinine_unavailable_n, COUNTIF(feature = 'creatinine' AND available AND (unit_valid IS NULL OR NOT unit_valid)) AS creatinine_unit_unknown_n, COUNTIF(feature = 'bun' AND available AND NOT valid) AS bun_invalid_n, COUNTIF(feature = 'bun' AND NOT available) AS bun_unavailable_n, COUNTIF(feature = 'bun' AND available AND (unit_valid IS NULL OR NOT unit_valid)) AS bun_unit_unknown_n, COUNTIF(feature = 'bilirubin' AND available AND NOT valid) AS bilirubin_invalid_n, COUNTIF(feature = 'bilirubin' AND NOT available) AS bilirubin_unavailable_n, COUNTIF(feature = 'bilirubin' AND available AND (unit_valid IS NULL OR NOT unit_valid)) AS bilirubin_unit_unknown_n, COUNTIF(feature = 'sodium' AND available AND NOT valid) AS sodium_invalid_n, COUNTIF(feature = 'sodium' AND NOT available) AS sodium_unavailable_n, COUNTIF(feature = 'sodium' AND available AND (unit_valid IS NULL OR NOT unit_valid)) AS sodium_unit_unknown_n, COUNTIF(feature = 'potassium' AND available AND NOT valid) AS potassium_invalid_n, COUNTIF(feature = 'potassium' AND NOT available) AS potassium_unavailable_n, COUNTIF(feature = 'potassium' AND available AND (unit_valid IS NULL OR NOT unit_valid)) AS potassium_unit_unknown_n, COUNTIF(feature = 'bicarbonate' AND available AND NOT valid) AS bicarbonate_invalid_n, COUNTIF(feature = 'bicarbonate' AND NOT available) AS bicarbonate_unavailable_n, COUNTIF(feature = 'bicarbonate' AND available AND (unit_valid IS NULL OR NOT unit_valid)) AS bicarbonate_unit_unknown_n, COUNTIF(feature = 'glucose' AND available AND NOT valid) AS glucose_invalid_n, COUNTIF(feature = 'glucose' AND NOT available) AS glucose_unavailable_n, COUNTIF(feature = 'glucose' AND available AND (unit_valid IS NULL OR NOT unit_valid)) AS glucose_unit_unknown_n
 FROM labs_screened GROUP BY stay_id
),
gcs_components AS (
 SELECT c.stay_id, e.charttime,
 MAX(IF(e.itemid=220739 AND e.valuenum BETWEEN 1 AND 4, e.valuenum, NULL)) AS eyes,
 MAX(IF(e.itemid=223900 AND e.valuenum BETWEEN 1 AND 5, e.valuenum, NULL)) AS verbal,
 MAX(IF(e.itemid=223901 AND e.valuenum BETWEEN 1 AND 6, e.valuenum, NULL)) AS motor,
 COUNTIF(e.itemid=223900 AND (e.value='No Response-ETT' OR e.valuenum=0))>0 AS unable
 FROM `physionet-data.mimiciv_3_1_icu.chartevents` e JOIN cohort c ON e.stay_id=c.stay_id
 WHERE e.itemid IN (220739,223900,223901)
 AND e.charttime BETWEEN DATETIME_SUB(c.t0, INTERVAL 6 HOUR) AND c.t0
 AND e.storetime IS NOT NULL AND e.storetime <= c.t0
 GROUP BY c.stay_id, e.charttime
), gcs_wide AS (
 SELECT stay_id, MIN(IF(NOT unable, eyes+verbal+motor, NULL)) AS gcs_min,
 COUNTIF(unable) AS gcs_unable_n
 FROM gcs_components GROUP BY stay_id
), fio2_observations AS (
 SELECT c.stay_id, e.charttime,
 CASE WHEN e.valuenum BETWEEN 0.21 AND 1 THEN e.valuenum*100
 WHEN e.valuenum BETWEEN 21 AND 100 THEN e.valuenum ELSE NULL END AS fio2
 FROM `physionet-data.mimiciv_3_1_icu.chartevents` e JOIN cohort c ON e.stay_id=c.stay_id
 WHERE e.itemid=223835
 AND e.charttime BETWEEN DATETIME_SUB(c.t0, INTERVAL 24 HOUR) AND c.t0
 AND e.storetime IS NOT NULL AND e.storetime<=c.t0
), bloodgas_raw AS (
 SELECT c.stay_id, e.specimen_id, e.charttime,
 MAX(IF(e.itemid=52033, e.value, NULL)) AS specimen,
 MAX(IF(e.itemid=50821 AND e.valuenum>0 AND e.valuenum<=1000
 AND LOWER(REPLACE(e.valueuom,' ',''))='mmhg',e.valuenum,NULL)) AS po2,
 MAX(IF(e.itemid=50816,
 CASE WHEN e.valuenum BETWEEN 0.21 AND 1 THEN e.valuenum*100
 WHEN e.valuenum BETWEEN 21 AND 100 THEN e.valuenum ELSE NULL END,NULL)) AS fio2
 FROM `physionet-data.mimiciv_3_1_hosp.labevents` e JOIN cohort c
 ON e.hadm_id=c.hadm_id AND e.subject_id=c.subject_id
 WHERE e.itemid IN (52033,50821,50816) AND e.specimen_id IS NOT NULL
 AND e.charttime BETWEEN DATETIME_SUB(c.t0, INTERVAL 24 HOUR) AND c.t0
 AND e.storetime IS NOT NULL AND e.storetime<=c.t0
 GROUP BY c.stay_id,e.specimen_id,e.charttime
), arterial_samples AS (
 SELECT stay_id,specimen_id,charttime,po2 AS pao2,fio2 AS sample_fio2
 FROM bloodgas_raw bg WHERE bg.specimen LIKE 'ART%' AND bg.po2 IS NOT NULL
), pairs AS (
 SELECT b.*, COALESCE(b.sample_fio2, f.fio2) AS paired_fio2,
 IF(b.sample_fio2 IS NOT NULL, 0, DATETIME_DIFF(b.charttime,f.charttime,SECOND)) AS pair_lag_seconds
 FROM arterial_samples b LEFT JOIN fio2_observations f
 ON b.stay_id=f.stay_id AND f.fio2 IS NOT NULL
 AND f.charttime BETWEEN DATETIME_SUB(b.charttime, INTERVAL 60 MINUTE) AND b.charttime
 QUALIFY ROW_NUMBER() OVER (PARTITION BY b.stay_id,b.specimen_id ORDER BY f.charttime DESC, f.fio2 DESC)=1
), oxygen_wide AS (
 SELECT stay_id, APPROX_QUANTILES(pao2, 2)[SAFE_OFFSET(1)] AS pao2,
 APPROX_QUANTILES(paired_fio2, 2)[SAFE_OFFSET(1)] AS fio2,
 APPROX_QUANTILES(SAFE_DIVIDE(pao2,paired_fio2/100.0), 2)[SAFE_OFFSET(1)] AS pao2_fio2_ratio,
 COUNT(*) AS arterial_samples_n, COUNTIF(paired_fio2 IS NOT NULL) AS oxygen_pairs_n,
 MAX(IF(paired_fio2 IS NOT NULL, pair_lag_seconds, NULL)) AS pair_max_lag_seconds
 FROM pairs GROUP BY stay_id
)
SELECT c.*,
 v.* EXCEPT(stay_id), l.* EXCEPT(stay_id), g.* EXCEPT(stay_id), o.* EXCEPT(stay_id)
FROM cohort c
LEFT JOIN vitals_wide v USING(stay_id)
LEFT JOIN labs_wide l USING(stay_id)
LEFT JOIN gcs_wide g USING(stay_id)
LEFT JOIN oxygen_wide o USING(stay_id)
ORDER BY c.subject_id
