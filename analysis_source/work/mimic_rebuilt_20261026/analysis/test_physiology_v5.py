import importlib
import unittest
import numpy as np
import pandas as pd

m=importlib.import_module('18_physiology_v5')

class PhysiologyV5Tests(unittest.TestCase):
    def test_coverage_requires_both_domains_before_imputation(self):
        df=pd.DataFrame(np.nan,index=range(3),columns=m.FEATURES)
        df.loc[0,m.LABS[:8]]=1
        df.loc[1,m.VITALS]=1;df.loc[1,m.LABS[:2]]=1
        df.loc[2,m.VITALS]=1;df.loc[2,m.LABS[:1]]=1
        self.assertEqual(m.eligible(df,8).tolist(),[False,True,False])

    def test_donor_pool_contains_only_original_observed_values(self):
        rng=np.random.default_rng(1)
        result=m.donor_draw(np.linspace(-100,100,100),np.array([0.,1.,2.]),np.array([3.,8.,17.]),rng,5)
        self.assertTrue(set(result).issubset({3.,8.,17.}))

    def test_nearest_donors_without_duplicating_edge(self):
        rng=np.random.default_rng(5)
        result=m.donor_draw(np.full(1000,-100.),np.arange(8.),np.arange(8.),rng,5)
        self.assertEqual(set(result),{0.,1.,2.,3.,4.})

    def test_all_tied_predictions_can_draw_every_donor(self):
        result=m.donor_draw(np.zeros(2000),np.zeros(30),np.arange(30.),np.random.default_rng(8),5)
        self.assertEqual(set(result),set(np.arange(30.)))

    def test_opposite_equal_distance_donors_both_supported(self):
        predictions=np.r_[np.full(20,-1.),np.full(20,1.)]
        result=m.donor_draw(np.zeros(5000),predictions,np.arange(40.),np.random.default_rng(8),5)
        self.assertEqual(set(result),set(np.arange(40.)))
        self.assertLess(abs(float(np.mean(result<20))-.5),.04)

    def test_changing_hidden_truth_has_no_effect_after_fixed_mask(self):
        rng=np.random.default_rng(11)
        df=pd.DataFrame(rng.normal(size=(40,len(m.FEATURES))),columns=m.FEATURES)
        hidden=rng.random(df.shape)<.2
        changed=df.copy();changed[hidden]=99999.
        cfg={'cycles':1,'trees':3,'max_depth':3,'min_samples_leaf':2,'n_jobs':1,'donors':3}
        a,_=m.rf_pmm(df.mask(hidden),cfg,12)
        b,_=m.rf_pmm(changed.mask(hidden),cfg,12)
        np.testing.assert_array_equal(a,b)

    def test_rf_pmm_preserves_observed_seed_and_empirical_support(self):
        rng=np.random.default_rng(7)
        x=rng.normal(size=(80,len(m.FEATURES)));x[rng.random(x.shape)<.15]=np.nan
        df=pd.DataFrame(x,columns=m.FEATURES)
        cfg={'cycles':2,'trees':4,'max_depth':3,'min_samples_leaf':2,'n_jobs':1,'donors':3}
        a,_=m.rf_pmm(df,cfg,5);b,_=m.rf_pmm(df,cfg,5)
        np.testing.assert_array_equal(a,b)
        observed=np.isfinite(x)
        np.testing.assert_array_equal(a.to_numpy()[observed],x[observed])
        for j in range(len(m.FEATURES)):
            self.assertTrue(set(a.to_numpy()[~observed[:,j],j]).issubset(set(x[observed[:,j],j])))

    def test_hidden_truth_cannot_be_a_donor(self):
        rng=np.random.default_rng(12)
        df=pd.DataFrame(rng.normal(size=(40,len(m.FEATURES))),columns=m.FEATURES)
        df.iloc[0,0]=np.nan
        cfg={'cycles':1,'trees':3,'max_depth':3,'min_samples_leaf':2,'n_jobs':1,'donors':3}
        a,_=m.rf_pmm(df,cfg,42)
        self.assertIn(a.iloc[0,0],set(df.iloc[1:,0]))

    def test_numerical_distance_excludes_demographics(self):
        a=pd.DataFrame(np.zeros((2,16)),columns=m.FEATURES);a.iloc[1,0]=1
        a['age']=[18,90];a['gender']=['F','M']
        D=m.g.numeric_gower(a,a,m.FEATURES,{f:[0,1] for f in m.FEATURES})
        self.assertAlmostEqual(float(D[0,1]),1/16)

    def test_parallel_restarts_equal_serial_seeded_solution(self):
        import tempfile
        from pathlib import Path
        from scipy.spatial.distance import cdist
        from concurrent.futures import ProcessPoolExecutor
        x=np.random.default_rng(7).random((60,16))
        D=(cdist(x,x,'cityblock')/16).astype(np.float32)
        expected,rows=m.c.fit_pam(D,3,5,12,200)
        test_root=m.ROOT/'data/synthetic_v5_tests';test_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_root) as td:
            assert Path(td).resolve().is_relative_to(test_root.resolve())
            path=Path(td)/'synthetic.npy';np.save(path,D)
            with ProcessPoolExecutor(2,initializer=m.initialize_pam_worker,initargs=(str(path),)) as pool:
                actual,actual_rows=m.parallel_pam(pool,len(D),3,5,12,200)
        np.testing.assert_array_equal(expected['labels'],actual['labels'])
        np.testing.assert_array_equal(expected['medoids'],actual['medoids'])
        self.assertEqual(expected['loss'],actual['loss'])
        self.assertEqual(rows,actual_rows)

if __name__=='__main__':unittest.main()
