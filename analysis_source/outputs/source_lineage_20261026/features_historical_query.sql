-- Generated from config/analysis_v2.json by analysis/08_rebuild.py.
-- Retrospective evidence-complete landmark; NOT a real-time sepsis alert.
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
),
vitals_raw AS (
 SELECT c.stay_id, e.charttime, e.storetime, e.itemid,
 CASE WHEN e.itemid IN (220045) THEN 'heart_rate' WHEN e.itemid IN (220050,220179) THEN 'sbp' WHEN e.itemid IN (220052,220181) THEN 'map' WHEN e.itemid IN (220210) THEN 'resp_rate' WHEN e.itemid IN (223761,223762) THEN 'temp_c' WHEN e.itemid IN (220277) THEN 'spo2' END AS feature, CASE WHEN e.itemid = 223761 THEN (e.valuenum-32)*5.0/9.0 ELSE e.valuenum END AS value,
 LOWER(REPLACE(REPLACE(REPLACE(e.valueuom,' ',''),'µ','u'),'μ','u')) AS unit,
 e.storetime IS NOT NULL AND e.storetime <= c.t0 AS available
 FROM `physionet-data.mimiciv_3_1_icu.chartevents` e JOIN cohort c ON e.stay_id = c.stay_id
 WHERE e.itemid IN (220045,220050,220179,220052,220181,220210,223761,223762,220277) AND e.valuenum IS NOT NULL
 AND e.charttime BETWEEN DATETIME_SUB(c.t0, INTERVAL 6 HOUR) AND c.t0
), vitals_screened AS (
 SELECT *, CASE WHEN feature = 'heart_rate' THEN value BETWEEN 1 AND 400 WHEN feature = 'sbp' THEN value BETWEEN 1 AND 400 WHEN feature = 'map' THEN value BETWEEN 1 AND 300 WHEN feature = 'resp_rate' THEN value BETWEEN 1 AND 100 WHEN feature = 'temp_c' THEN value BETWEEN 25 AND 45 WHEN feature = 'spo2' THEN value BETWEEN 1 AND 100 ELSE FALSE END AS valid,
 CASE WHEN feature='heart_rate' THEN unit IN ('beats/min','bpm') WHEN feature='sbp' THEN unit IN ('mmhg') WHEN feature='map' THEN unit IN ('mmhg') WHEN feature='resp_rate' THEN unit IN ('breaths/min','insp/min') WHEN feature='temp_c' AND itemid=223761 THEN unit IN ('f','deg.f','deg.fahrenheit','°f','?f') WHEN feature='temp_c' THEN unit IN ('c','deg.c','deg.celsius','°c','?c') WHEN feature='spo2' THEN unit IN ('%') ELSE FALSE END AS unit_valid
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
 LOWER(REPLACE(REPLACE(REPLACE(e.valueuom,' ',''),'µ','u'),'μ','u')) AS unit,
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
