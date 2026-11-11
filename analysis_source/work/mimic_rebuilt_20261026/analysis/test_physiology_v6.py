import importlib
import unittest
import numpy as np
import pandas as pd

m=importlib.import_module('22_physiology_v6')

class V6DiagnosticTests(unittest.TestCase):
    def test_quantiles_use_observed_cells_and_preserve_input(self):
        a=pd.DataFrame({f:[0.,1.,2.,3.,100.,np.nan] for f in m.m.FEATURES})
        before=a.copy(deep=True);limits=m.observed_limits(a,[.01,.99])
        pd.testing.assert_frame_equal(a,before)
        for f in m.m.FEATURES:
            np.testing.assert_allclose(limits[f],np.quantile([0.,1.,2.,3.,100.],[.01,.99]))

    def test_clipping_is_representation_only(self):
        a=pd.DataFrame({f:[-100.,0.,1.,100.] for f in m.m.FEATURES})
        before=a.copy(deep=True)
        D=m.m.g.numeric_gower(a,a,m.m.FEATURES,{f:[0.,1.] for f in m.m.FEATURES})
        pd.testing.assert_frame_equal(a,before)
        self.assertEqual(float(D[0,1]),0.)
        self.assertEqual(float(D[2,3]),0.)
        self.assertEqual(float(D[0,3]),1.)

    def test_constant_features_zero_contribution(self):
        a=pd.DataFrame({f:[2.,2.,2.] for f in m.m.FEATURES})
        D=m.m.g.numeric_gower(a,a,m.m.FEATURES,{f:[2.,2.] for f in m.m.FEATURES})
        np.testing.assert_array_equal(D,np.zeros((3,3)))

    def test_common_patient_comparison_joins_all_keys_and_aligns(self):
        a=pd.DataFrame({'subject_id':[1,2,3,4],'stay_id':[10,20,30,40],'hadm_id':[100,200,300,400],'cluster':[1,1,2,2]})
        b=a.iloc[[2,1,3,0]].copy();b.cluster=3-b.cluster
        result=m.compare(a,b,2,'synthetic')
        self.assertEqual(result['ari'],1.)
        self.assertEqual(result['aligned_agreement'],1.)
        self.assertEqual(result['n_common'],4)

if __name__=='__main__':unittest.main()
