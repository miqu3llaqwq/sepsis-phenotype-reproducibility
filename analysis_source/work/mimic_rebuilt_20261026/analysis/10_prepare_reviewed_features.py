"""Review training source observations; apply registered transforms to frozen partitions.

No clustering/model fitting; the original v2 cohort and splits are not overwritten.
"""
from pathlib import Path
import hashlib
import json
import sys
import importlib
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
build=importlib.import_module('08_rebuild')


def normalize_units(series):
    return series.fillna('').astype(str).str.lower().str.replace(' ','',regex=False).str.replace('µ','u',regex=False).str.replace('μ','u',regex=False)


def screen_and_collapse(raw, c, domain):
    raw=raw.copy()
    raw['unit']=normalize_units(raw.valueuom)
    raw['value']=pd.to_numeric(raw.valuenum,errors='coerce')
    raw.loc[raw.itemid.eq(223761),'value']=(raw.loc[raw.itemid.eq(223761),'value']-32)*5/9
    mapping={i:n for n,s in c[domain].items() for i in s['itemids']}
    mapping.update({220051:'dbp',220180:'dbp',225309:'sbp',225310:'dbp',225312:'map',224690:'resp_rate'})
    raw['feature']=raw.itemid.map(mapping)
    unit_aliases={'beats/min':['beats/min','bpm'],'breaths/min':['breaths/min','insp/min'],'mmHg':['mmhg'],'C':['c','deg.c','deg.celsius','°c','?c'],'%':['%'],'K/uL':['k/ul'],'mmol/L':['mmol/l'],'mg/dL':['mg/dl'],'g/dL':['g/dl'],'mEq/L':['meq/l','mmol/l']}
    raw['valid']=False
    for name in raw.feature.dropna().unique():
        spec=c[domain].get(name,{'range':[1,300],'unit':'mmHg'})
        lo,hi=spec['range']; units=unit_aliases[spec['unit']]
        mask=raw.feature.eq(name)
        valid=raw.value.between(lo,hi)&raw.unit.isin(units)
        if name=='temp_c':
            valid=raw.value.between(lo,hi)&np.where(raw.itemid.eq(223761),raw.unit.isin(['f','deg.f','deg.fahrenheit','°f','?f']),raw.unit.isin(units))
        raw.loc[mask,'valid']=valid.loc[mask]
    key=['stay_id','itemid','charttime','specimen_id']
    valid=raw[raw.valid].copy()
    unique=valid.drop_duplicates(key+['value','unit'])
    collapsed=unique.groupby(key,dropna=False,as_index=False).agg(value=('value','median'),feature=('feature','first'),n_distinct=('value','size'))
    return raw,unique,collapsed


def aggregate(raw,c,domain):
    screened,unique,x=screen_and_collapse(raw,c,domain)
    bp_meta=None
    if domain=='vitals':
        bp=x[x.feature.isin(['sbp','map','dbp'])].copy()
        bp['source']=np.where(bp.itemid.isin([220179,220180,220181]),'noninvasive','arterial')
        # Equal weight per source/time; multiple ART item codes at the same time are collapsed.
        pairs=bp.pivot_table(index=['stay_id','charttime','source'],columns='feature',values='value',aggfunc='median').reset_index()
        for col in ['sbp','map','dbp']:
            if col not in pairs:pairs[col]=np.nan
        pairs['valid_pair']=pairs.sbp.notna()&pairs['map'].notna()&pairs['map'].le(pairs.sbp)&(pairs.dbp.isna()|pairs.dbp.le(pairs['map']))
        valid=pairs[pairs.valid_pair].copy()
        arterial_stays=set(valid.loc[valid.source.eq('arterial'),'stay_id'])
        chosen=valid[valid.source.eq('arterial')|~valid.stay_id.isin(arterial_stays)]
        bp_out=chosen.groupby('stay_id')[['sbp','map']].median()
        bp_meta=pairs
        # Avoid letting ventilator RR replace independently charted RR when it is available.
        preferred_stays=set(x.loc[x.itemid.eq(220210),'stay_id'])
        x=x[~(x.itemid.eq(224690)&x.stay_id.isin(preferred_stays))]
        nonbp=x[~x.feature.isin(['sbp','map','dbp'])]
        out=nonbp.groupby(['stay_id','feature']).value.median().unstack().join(bp_out,how='outer')
    else:
        out=x.groupby(['stay_id','feature']).value.median().unstack()
    return out,screened,unique,x,bp_meta


def encode_age(df,cap):
    age=pd.to_numeric(df.age,errors='coerce').clip(upper=cap)
    return age.mask(df.anchor_age.eq(91),cap)


def fit_imputer(df,features):
    fit=df[df.split.eq('train')&df.phenotype_eligible]
    if fit.empty:raise ValueError('No eligible training records')
    result={}
    for col in features:
        value=fit[col].mode().iloc[0] if col=='gender' else fit[col].median()
        if pd.isna(value):raise ValueError('Entire training feature missing: '+col)
        result[col]=str(value) if col=='gender' else float(value)
    return result


def main():
    loc=json.loads((ROOT/'logs/observation_review_latest.json').read_text(encoding='utf-8'))
    frozen=json.loads((ROOT/'config/frozen_split_v2.json').read_text(encoding='utf-8'))
    base=Path(loc['input_path'])
    if hashlib.sha256(base.read_bytes()).hexdigest()!=frozen['dataset_sha256']:raise ValueError('Frozen input changed')
    c=build.load_config(); rules=json.loads((ROOT/'config/preprocessing_review_v3.json').read_text(encoding='utf-8'))
    df=pd.read_parquet(base); original=df[['subject_id','stay_id','split']].copy()
    train_stays=set(df.loc[df.split.eq('train'),'stay_id'])
    dest=Path(loc['data_path'])/'derived_v3';dest.mkdir(exist_ok=True)
    reports=ROOT/'outputs/reports/preprocessing_review_v3';reports.mkdir(exist_ok=True)
    tables=ROOT/'outputs/tables/preprocessing_review_v3';tables.mkdir(exist_ok=True)
    observations={}; duplicates=[];extremes=[];source_checks=[]
    for domain in ['vitals','labs']:
        raw=pd.read_parquet(Path(loc['data_path'])/f'{domain}.parquet')
        if not raw.stay_id.isin(df.stay_id).all():raise ValueError('Source has unexpected stay')
        check=raw[['stay_id','t0']].drop_duplicates().merge(df[['stay_id','t0']],on='stay_id',validate='one_to_one',suffixes=('_raw','_base'))
        if not pd.to_datetime(check.t0_raw).eq(pd.to_datetime(check.t0_base)).all():raise ValueError('Source landmark mismatch')
        times=pd.to_datetime(raw.charttime); stores=pd.to_datetime(raw.storetime); landmark=pd.to_datetime(raw.t0)
        hours=c['vitals_hours'] if domain=='vitals' else c['labs_hours']
        if stores.isna().any() or times.isna().any() or (stores>landmark).any() or (times>landmark).any() or (times<landmark-pd.Timedelta(hours=hours)).any():raise ValueError('Observation unavailable or outside window')
        out,screened,unique,collapsed,pairs=aggregate(raw,c,domain)
        for col in c[domain]:df[col]=df.stay_id.map(out[col] if col in out else pd.Series(dtype=float))
        tr=screened[screened.stay_id.isin(train_stays)]
        u=unique[unique.stay_id.isin(train_stays)];cx=collapsed[collapsed.stay_id.isin(train_stays)]
        duplicates.append({'partition':'train','domain':domain,'source_rows':len(tr),'valid_rows':int(tr.valid.sum()),'exact_repeat_rows_removed':int(tr.valid.sum()-len(u)),'conflicting_item_time_groups':int(cx.n_distinct.gt(1).sum())})
        # Numeric review flags are NOT diagnoses; count support using adjacent same-source item records.
        for name,(lo,hi) in rules['strict_sensitivity_ranges'].items():
            group=cx[cx.feature.eq(name)].copy()
            if group.empty:continue
            group=group.sort_values(['stay_id','itemid','charttime'])
            flag=~group.value.between(lo,hi)
            g=group.groupby(['stay_id','itemid'],dropna=False)
            prev_val=g.value.shift();next_val=g.value.shift(-1)
            prev_time=g.charttime.shift();next_time=g.charttime.shift(-1)
            # Evidence within30min indicates persistence; absence cannot prove artifact.
            previous=(group.charttime-prev_time).dt.total_seconds().le(1800)&~prev_val.between(lo,hi)&prev_val.notna()
            following=(next_time-group.charttime).dt.total_seconds().le(1800)&~next_val.between(lo,hi)&next_val.notna()
            extremes.append({'partition':'train','feature':name,'flagged_source_time_groups':int(flag.sum()),'flagged_with_adjacent_extreme_within30min':int((flag&(previous|following)).sum()),'flagged_without_adjacent_extreme_support':int((flag&~(previous|following)).sum())})
        if pairs is not None:
            tp=pairs[pairs.stay_id.isin(train_stays)]
            source_checks.append({'partition':'train','paired_source_times':len(tp),'invalid_order_or_incomplete_pairs':int((~tp.valid_pair).sum()),'arterial_valid_pairs':int((tp.valid_pair&tp.source.eq('arterial')).sum()),'noninvasive_valid_pairs':int((tp.valid_pair&tp.source.eq('noninvasive')).sum())})
        observations[domain]=hashlib.sha256((Path(loc['data_path'])/f'{domain}.parquet').read_bytes()).hexdigest()
    df['age_raw']=df.age;df['age_deidentified']=df.anchor_age.eq(91)
    df['age']=encode_age(df,rules['age_topcode'])
    features=c['primary_features']; physiology=[x for x in features if x not in ['age','gender']]
    df['physiology_missing_n']=df[physiology].isna().sum(axis=1)
    df['phenotype_eligible']=df.physiology_missing_n.lt(len(physiology))
    if not df[['subject_id','stay_id','split']].equals(original):raise ValueError('Frozen partitions changed')
    imputer=fit_imputer(df,features)
    pre=df.copy();pre.to_parquet(dest/'reviewed_unimputed.parquet',index=False)
    strict=pre.copy()
    for name,(lo,hi) in rules['strict_sensitivity_ranges'].items():strict[name]=strict[name].where(strict[name].between(lo,hi))
    strict['physiology_missing_n']=strict[physiology].isna().sum(axis=1)
    strict['phenotype_eligible']=strict.physiology_missing_n.lt(len(physiology))
    strict.to_parquet(dest/'strict_screen_sensitivity_unimputed.parquet',index=False)
    strict_imputer=fit_imputer(strict,features)
    for target,values in [(df,imputer),(strict,strict_imputer)]:
        for col in features:target[col+'_missing']=target[col].isna();target[col]=target[col].fillna(values[col])
        if target[features].isna().any().any():raise ValueError('Imputation left missing values')
    df.to_parquet(dest/'reviewed_imputed.parquet',index=False)
    strict.to_parquet(dest/'strict_screen_sensitivity_imputed.parquet',index=False)
    (dest/'imputer.json').write_text(json.dumps({'fit_partition':'eligible_train','primary':imputer,'strict_sensitivity':strict_imputer},indent=2),encoding='utf-8')
    pd.DataFrame(duplicates).to_csv(tables/'training_duplicates.csv',index=False)
    pd.DataFrame(extremes).to_csv(tables/'training_extreme_evidence.csv',index=False)
    pd.DataFrame(source_checks).to_csv(tables/'training_bp_source_checks.csv',index=False)
    training=pre[pre.split.eq('train')]
    rows=[]
    for col in features:
        row={'partition':'train','feature':col,'missing_pct':round(training[col].isna().mean()*100,2),'imputation_value':imputer[col]}
        if col!='gender':row.update(min=training[col].min(),median=training[col].median(),max=training[col].max())
        rows.append(row)
    pd.DataFrame(rows).to_csv(tables/'training_reviewed_features.csv',index=False)
    training.physiology_missing_n.value_counts().sort_index().rename_axis('missing_physiology_features').to_csv(tables/'training_joint_missingness.csv',header=['patients'])
    pre.groupby('split').agg(n=('subject_id','size'),phenotype_eligible=('phenotype_eligible','sum')).to_csv(tables/'population_eligibility.csv')
    age_summary={'partition':'train','n_anchor_age91':int(training.anchor_age.eq(91).sum()),'n_age_proxy_changed':int(training.age.ne(training.age_raw).sum()),'encoded_age_max':float(training.age.max())}
    (reports/'age_encoding.json').write_text(json.dumps(age_summary,indent=2),encoding='utf-8')
    snapshots=reports/'source_snapshot';snapshots.mkdir(exist_ok=True)
    source_hashes={}
    for rel in ['analysis/08_rebuild.py','analysis/09_observation_review.py','analysis/10_prepare_reviewed_features.py','analysis/test_preprocessing_review.py','config/analysis_v2.json','config/preprocessing_review_v3.json','sql/v2_review_vitals.sql','sql/v2_review_labs.sql']:
        src=ROOT/rel; content=src.read_bytes(); (snapshots/src.name).write_bytes(content);source_hashes[rel]=hashlib.sha256(content).hexdigest()
    partition_hashes={name:hashlib.sha256((base.parent/(name+'_ids.parquet')).read_bytes()).hexdigest() for name in ['train','validation','test']}
    manifest={'date':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'status':'training_source_review_and_imputation_complete_not_clinically_certified','input_sha256':frozen['dataset_sha256'],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_hashes':source_hashes,'partition_file_sha256':partition_hashes,'rules':rules,'observation_sha256':observations,'output_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir() if p.is_file()},'source_partitions_unchanged':True,'fit_partition':'eligible_train','clinical_extreme_flags_resolved_as_errors':False,'clustering_started':False}
    (reports/'run_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    (ROOT/'logs/preprocessing_review_latest.json').write_text(json.dumps({'data_path':str(dest),'report_path':str(reports),'tables_path':str(tables)},indent=2,ensure_ascii=False),encoding='utf-8')
    print('Training-only review and imputation complete. Frozen cohort and partitions unchanged.')
    print('Training no-physiology rows:',int((~training.phenotype_eligible).sum()))
    print('Clinical authenticity of extreme measurements remains unproven; sensitivity data saved.')


if __name__=='__main__':main()
