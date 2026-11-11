"""Aggregate completed source-query results before REST download, preserving hourly maxima."""
import json
from google.cloud import bigquery
from extract_common_core_20261030 import run_query
from build_common_identified_20261030 import OUT,WORK

def main():
    m=json.loads((OUT/'extraction_manifest.json').read_text());j=bigquery.Client().get_job(m['common_sofa_chart']['job_id'])
    assert j.state=='DONE' and j.error_result is None
    dest=j.destination
    sql=f'''WITH raw AS (SELECT * FROM `{dest.project}.{dest.dataset_id}.{dest.table_id}`),
gcs AS (SELECT stay_id,offset_min,
MAX(IF(itemid=220739,value,NULL))+MAX(IF(itemid=223900,value,NULL))+MAX(IF(itemid=223901,value,NULL)) AS total,
COUNT(DISTINCT itemid)=3 AND COUNTIF(null_rows>0 OR distinct_values!=1 OR unscorable_rows>0 OR value IS NULL OR value!=FLOOR(value) OR value<1 OR value>CASE itemid WHEN 220739 THEN 4 WHEN 223900 THEN 5 ELSE 6 END)=0
AND COUNT(DISTINCT IF(itemid=220739,value,NULL))=1 AND COUNT(DISTINCT IF(itemid=223900,value,NULL))=1 AND COUNT(DISTINCT IF(itemid=223901,value,NULL))=1 AS valid
FROM raw WHERE itemid IN(220739,223900,223901) GROUP BY stay_id,offset_min),
points AS (
SELECT stay_id,offset_min,'cardiovascular' AS domain,1 AS lower_score FROM raw
WHERE itemid IN(220052,220181) AND null_rows=0 AND distinct_values=1 AND value>=1 AND value<70 AND LOWER(valueuom)='mmhg'
UNION ALL
SELECT stay_id,offset_min,'cns',CASE WHEN total<6 THEN 4 WHEN total<10 THEN 3 WHEN total<13 THEN 2 WHEN total<15 THEN 1 ELSE 0 END
FROM gcs WHERE valid AND total>=3 AND total<15)
SELECT stay_id,CEIL(offset_min/60)*60 AS offset_min,domain,MAX(lower_score) AS lower,
COUNT(*) AS contributing_points,MIN(offset_min) AS first_point_min,MAX(offset_min) AS last_point_min
FROM points GROUP BY stay_id,CEIL(offset_min/60)*60,domain'''
    run_query('common_sofa_chart_compact',sql)
    # Save a reproducible query independent of the temporary result table.
    source=(WORK/'sql/common_sofa_chart.sql').read_text()
    (WORK/'sql/common_sofa_chart_compact_reproducible.sql').write_text(sql.replace(f'SELECT * FROM `{dest.project}.{dest.dataset_id}.{dest.table_id}`',source),encoding='utf-8')
    base=(WORK/'sql/mimic_orders.sql').read_text(encoding='utf-8')
    ct=f'WITH ab AS ({base}), s AS (SELECT DISTINCT subject_id,hadm_id,stay_id,intime,outtime,admittime FROM ab)'
    run_query('common_sofa_gas',ct+'''
SELECT s.stay_id,DATETIME_DIFF(e.charttime,s.intime,SECOND)/60.0 AS offset_min,
 e.itemid,e.valuenum,e.value,e.valueuom,e.specimen_id,e.labevent_id
FROM s JOIN `physionet-data.mimiciv_3_1_hosp.labevents` e ON s.hadm_id=e.hadm_id AND s.subject_id=e.subject_id
WHERE e.itemid IN (50821,50816,52033)
AND e.charttime BETWEEN GREATEST(s.admittime,DATETIME_SUB(s.intime,INTERVAL 48 HOUR)) AND s.outtime''')

if __name__=='__main__':main()
