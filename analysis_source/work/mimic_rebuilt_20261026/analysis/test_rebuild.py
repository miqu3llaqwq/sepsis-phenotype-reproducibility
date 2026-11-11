"""Synthetic boundary tests; no MIMIC records are used."""
import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from importlib import import_module
m=import_module('08_rebuild')


def fixture():
    c=m.load_config()
    df=pd.DataFrame({name:[1.0]*20 for name in c['primary_features']})
    df['gender']='F'; df['age']=60
    for name in ['subject_id','hadm_id','stay_id']:
        df[name]=range(20)
    for name in ['admittime','icu_intime','antibiotic_time','culture_time','sofa_time']:
        df[name]='2020-01-01 00:00:00'
    df['t0']='2020-01-01 01:00:00'
    for name in ['dischtime','icu_outtime','endpoint']:
        df[name]='2020-01-02 01:00:00'
    df['follow_up_days']=1.0
    df['hosp_mortality']=0; df['event_type']=2; df['sepsis3']=True
    df['pair_max_lag_seconds']=0
    return c,df


class RebuildTests(unittest.TestCase):
    def test_valid_competing_discharge(self):
        c,df=fixture()
        self.assertTrue(m.validate_dataset(df,c))

    def test_rejects_future_infection_evidence(self):
        c,df=fixture(); df.loc[0,'antibiotic_time']='2020-01-03'
        with self.assertRaises(ValueError): m.validate_dataset(df,c)

    def test_rejects_zero_followup_and_bad_cause(self):
        for name,val in [('endpoint','2020-01-01 01:00:00'),('event_type',1),('follow_up_days',0)]:
            c,df=fixture(); df.loc[0,name]=val
            with self.assertRaises(ValueError): m.validate_dataset(df,c)

    def test_rejects_duplicate_and_null_key(self):
        for val in [1,None]:
            c,df=fixture(); df.loc[0,'subject_id']=val
            with self.assertRaises(ValueError): m.validate_dataset(df,c)

    def test_partition_is_disjoint_order_invariant_and_outcome_blind(self):
        c,df=fixture()
        a=m.split_ids(df,c)[['subject_id','split']].sort_values('subject_id').reset_index(drop=True)
        df['hosp_mortality']=1
        b=m.split_ids(df.iloc[::-1],c)[['subject_id','split']].sort_values('subject_id').reset_index(drop=True)
        assert_frame_equal(a,b)
        self.assertEqual(a.split.value_counts().to_dict(),{'train':14,'validation':3,'test':3})

    def test_pairing_and_eligibility_sql_contract(self):
        c=m.load_config(); sql=m.feature_sql(c)
        self.assertIn("bg.specimen LIKE 'ART%'",sql)
        self.assertIn('AND b.charttime',sql)
        self.assertIn('INTERVAL 60 MINUTE',sql)
        self.assertIn('e.storetime<=c.t0',sql)
        self.assertIn('e.specimen_id',sql)
        self.assertLess(sql.index('eligible_ranked AS'),sql.index('WHERE eligible_rank = 1'))
        self.assertNotIn('charlson',sql)
        self.assertNotIn('sofa_score',sql)


if __name__=='__main__': unittest.main()
