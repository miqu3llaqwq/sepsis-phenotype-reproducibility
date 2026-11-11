WITH candidate AS (
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
) SELECT * FROM eligibility