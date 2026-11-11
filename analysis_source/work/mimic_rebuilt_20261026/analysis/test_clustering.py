import importlib
import unittest
import numpy as np
import pandas as pd

c=importlib.import_module('14_cluster_training')
diagnostic=importlib.import_module('15_gender_diagnostic')

class ClusteringTests(unittest.TestCase):
    def setUp(self):
        self.features=['x','y','gender']
        self.frame=pd.DataFrame({'x':[0.,10.,5.],'y':[0.,20.,0.],'gender':['F','M','F']})
        self.limits={'x':[0.,10.],'y':[0.,20.]}

    def test_equal_weight_mixed_gower(self):
        d=c.gower(self.frame,self.frame,self.features,self.limits)
        self.assertAlmostEqual(float(d[0,1]),1.)
        self.assertAlmostEqual(float(d[0,2]),1/6,places=6)
        np.testing.assert_allclose(d,d.T)
        np.testing.assert_array_equal(np.diag(d),0)

    def test_constant_feature_and_clipped_assignment(self):
        limits={'x':[0.,0.],'y':[0.,20.]}
        extreme=pd.DataFrame({'x':[1000.],'y':[200.],'gender':['F']})
        d=c.gower(self.frame.iloc[:1],extreme,self.features,limits)
        self.assertAlmostEqual(float(d[0,0]),1/3,places=6)

    def test_fit_parameters_from_fitting_rows_only(self):
        train=pd.DataFrame({'x':[2.,4.,np.nan],'gender':['F','F',None]})
        _,params,limits=c.fit_transform(train,['x','gender'])
        held=pd.DataFrame({'x':[1000.,np.nan],'gender':['M',None]})
        result=c.apply_transform(held,params)
        self.assertEqual(params,{'x':3.,'gender':'F'})
        self.assertEqual(limits,{'x':[2.,4.]})
        self.assertEqual(result.x.tolist(),[1000.,3.])

    def test_label_permutation_does_not_imply_change(self):
        left=np.array([0,0,1,1,2,2]);right=np.array([2,2,0,0,1,1])
        agreement,mapped,_=c.aligned(left,right,3)
        self.assertEqual(agreement,1.)
        np.testing.assert_array_equal(mapped,left)

    def test_numeric_diagnostic_actually_removes_gender_term(self):
        d=diagnostic.numeric_gower(self.frame,self.frame,['x','y'],self.limits)
        self.assertAlmostEqual(float(d[0,1]),1.)
        self.assertAlmostEqual(float(d[0,2]),.25)
        other=self.frame.copy();other['gender']='M'
        np.testing.assert_allclose(d,diagnostic.numeric_gower(other,other,['x','y'],self.limits))

    def test_fastpam_matches_reference_given_initial_medoids(self):
        d=np.array([[0,1,6,7],[1,0,5,6],[6,5,0,1],[7,6,1,0]],dtype=np.float32)
        init=np.array([0,1],dtype=np.uintp)
        fast=c.kmedoids.fastpam1(d,init.copy(),max_iter=200)
        reference=c.kmedoids.pam(d,init.copy(),max_iter=200)
        self.assertAlmostEqual(fast.loss,reference.loss)

    def test_pam_result_is_one_swap_local_optimum(self):
        d=c.gower(self.frame,self.frame,self.features,self.limits)
        fit,_=c.fit_pam(d,2,3,42,200)
        for slot in range(2):
            for candidate in range(len(d)):
                medoids=fit['medoids'].copy();medoids[slot]=candidate
                if len(set(medoids))<2:continue
                loss=d[:,medoids].min(axis=1).sum(dtype=np.float64)
                self.assertGreaterEqual(loss+1e-6,fit['loss'])

if __name__=='__main__':unittest.main()
