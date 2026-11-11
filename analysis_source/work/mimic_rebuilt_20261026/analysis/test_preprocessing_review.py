"""Synthetic checks for source aggregation and train-only preprocessing."""
import importlib
import unittest
import numpy as np
import pandas as pd

p=importlib.import_module('10_prepare_reviewed_features')
c=p.build.load_config()

def observations(rows):
    return pd.DataFrame([{'stay_id':stay,'itemid':item,'charttime':pd.Timestamp('2020-01-01')+pd.Timedelta(minutes=minute),'specimen_id':np.nan,'valuenum':value,'valueuom':unit} for stay,item,minute,value,unit in rows])

class PreprocessingChecks(unittest.TestCase):
    def test_repeats_do_not_overweight_median(self):
        raw=observations([(1,220045,0,60,'bpm')]*8+[(1,220045,1,100,'bpm')])
        out,_,unique,_,_=p.aggregate(raw,c,'vitals')
        self.assertEqual(len(unique),2)
        self.assertEqual(out.loc[1,'heart_rate'],80)

    def test_same_time_conflict_collapses_before_window(self):
        raw=observations([(1,220045,0,60,'bpm'),(1,220045,0,100,'bpm'),(1,220045,1,120,'bpm')])
        out,_,_,_,_=p.aggregate(raw,c,'vitals')
        self.assertEqual(out.loc[1,'heart_rate'],100)

    def test_bp_pair_priority_and_no_cross_source_pair(self):
        raw=observations([(1,220050,0,110,'mmHg'),(1,220052,0,70,'mmHg'),(1,220179,0,150,'mmHg'),(1,220181,0,90,'mmHg'),(2,220050,0,100,'mmHg'),(2,220181,0,70,'mmHg'),(3,220050,0,80,'mmHg'),(3,220052,0,90,'mmHg'),(3,220179,1,120,'mmHg'),(3,220181,1,75,'mmHg')])
        out,*_=p.aggregate(raw,c,'vitals')
        self.assertEqual(out.loc[1,'sbp'],110)
        self.assertEqual(out.loc[1,'map'],70)
        self.assertNotIn(2,out.dropna(subset=['sbp','map']).index)
        self.assertEqual(out.loc[3,'sbp'],120)

    def test_age_topcode_is_proxy(self):
        df=pd.DataFrame({'age':[88,96,85],'anchor_age':[86,91,91]})
        self.assertEqual(p.encode_age(df,90).tolist(),[88,90,90])

    def test_imputation_ignores_holdouts_and_ineligible(self):
        df=pd.DataFrame({'split':['train','train','train','validation','test'],'phenotype_eligible':[True,True,False,True,True],'age':[20,40,999,1000,-1000],'gender':['F','F','M','M','M']})
        self.assertEqual(p.fit_imputer(df,['age','gender']),{'age':30.0,'gender':'F'})

    def test_resp_rate_fallback_and_unit_screen(self):
        raw=observations([(1,220210,0,18,'breaths/min'),(1,224690,1,40,'breaths/min'),(2,224690,0,22,'breaths/min'),(3,220045,0,100,'mmHg')])
        out,*_=p.aggregate(raw,c,'vitals')
        self.assertEqual(out.loc[1,'resp_rate'],18)
        self.assertEqual(out.loc[2,'resp_rate'],22)
        self.assertNotIn(3,out.index)

if __name__=='__main__':unittest.main()
