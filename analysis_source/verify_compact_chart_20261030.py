"""Check remote aggregation against original chart rows for a hash-selected sample."""
import json
import numpy as np
import pandas as pd
from google.cloud import bigquery
from extract_common_core_20261030 import run_query
from build_common_identified_20261030 import OUT,WORK,DATA,frame,dump

def main():
    m=json.loads((OUT/'extraction_manifest.json').read_text());j=bigquery.Client().get_job(m['common_sofa_chart']['job_id']);d=j.destination
    sql=f'SELECT * FROM `{d.project}.{d.dataset_id}.{d.table_id}` WHERE MOD(ABS(FARM_FINGERPRINT(CAST(stay_id AS STRING))),1000)<3'
    p=run_query('common_chart_qa_raw',sql);raw=frame(p);rows=[]
    for (stay,t),g in raw.groupby(['stay_id','offset_min']):
        map_rows=g[g.itemid.isin([220052,220181])]
        for r in map_rows.itertuples():
            if r.null_rows==0 and r.distinct_values==1 and pd.notna(r.value) and 1<=r.value<70 and str(r.valueuom).lower()=='mmhg':rows.append((stay,float(np.ceil(t/60)*60),'cardiovascular',1))
        a=g[g.itemid.isin([220739,223900,223901])];bounds={220739:4,223900:5,223901:6};values=[];good=True
        for item,upper in bounds.items():
            z=a[a.itemid.eq(item)]
            if z.empty or z.value.nunique()!=1 or not (z.null_rows.eq(0)&z.distinct_values.eq(1)&z.unscorable_rows.eq(0)&z.value.between(1,upper)&z.value.mod(1).eq(0)).all():good=False;break
            values.append(float(z.value.iloc[0]))
        if good:
            total=sum(values);score=4 if total<6 else 3 if total<10 else 2 if total<13 else 1 if total<15 else 0
            if score:rows.append((stay,float(np.ceil(t/60)*60),'cns',score))
    keys=['stay_id','offset_min','domain'];expected=pd.DataFrame(rows,columns=keys+['lower']).groupby(keys).lower.max().sort_index()
    actual=frame(DATA/'common_sofa_chart_compact_local.parquet');actual=actual[actual.stay_id.isin(raw.stay_id)].set_index(keys).lower.sort_index()
    assert expected.index.equals(actual.index)
    assert np.array_equal(expected.to_numpy(),actual.to_numpy())
    dump('remote_chart_aggregation_qa.json',{'status':'pass','raw_chart_groups_checked':len(raw),'sample_stays':raw.stay_id.nunique(),'positive_hour_domains_checked':len(expected),'sample_rule':'deterministic hashed stay sample <3/1000','comparison':'independent Python numeric/units/completeness checks versus remote hourly aggregation'})
    original=(WORK/'sql/common_sofa_chart.sql').read_text();(WORK/'sql/common_chart_qa_reproducible.sql').write_text(sql.replace(f'`{d.project}.{d.dataset_id}.{d.table_id}`',f'({original})'),encoding='utf-8')
    print('REMOTE CHART QA PASSED',len(raw),'raw groups',len(expected),'hour domains',flush=True)

if __name__=='__main__':main()
