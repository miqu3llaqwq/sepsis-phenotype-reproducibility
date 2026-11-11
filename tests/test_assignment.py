import sys,unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sepsis_repro import assign,distance_to_medoids

class AssignmentTests(unittest.TestCase):
    def setUp(self):self.cfg={'features':['x','y'],'limits':{'x':[0.,10.],'y':[0.,20.]}}
    def test_clipping_and_equal_feature_weights(self):
        x=pd.DataFrame({'x':[-5.],'y':[30.]});m=pd.DataFrame({'x':[10.],'y':[0.],'cluster':[1]})
        self.assertEqual(float(distance_to_medoids(x,m,self.cfg)[0,0]),1.)
    def test_one_unknown_medoid_blocks_assignment(self):
        x=pd.DataFrame({'x':[2.],'y':[np.nan]});m=pd.DataFrame({'x':[1.,np.nan],'y':[np.nan,2.],'cluster':[1,2]})
        labels,ok,d=assign(x,m,self.cfg);self.assertFalse(ok[0]);self.assertTrue(np.isnan(labels[0]));self.assertTrue(np.isnan(d[0,1]))
    def test_tie_uses_first_label(self):
        x=pd.DataFrame({'x':[2.],'y':[2.]});m=pd.DataFrame({'x':[2.,2.],'y':[2.,2.],'cluster':[1,2]})
        labels,ok,_=assign(x,m,self.cfg);self.assertTrue(ok[0]);self.assertEqual(labels[0],1)
    def test_label_order_is_validated(self):
        x=pd.DataFrame({'x':[1.],'y':[2.]});m=pd.DataFrame({'x':[1.,2.],'y':[2.,3.],'cluster':[2,1]})
        with self.assertRaises(ValueError):assign(x,m,self.cfg)
    def test_matches_independent_scalar_distance(self):
        rng=np.random.default_rng(42);a=rng.uniform(-5,30,(53,2));b=rng.uniform(-5,30,(6,2));a[rng.random(a.shape)<.25]=np.nan;b[rng.random(b.shape)<.25]=np.nan
        actual=distance_to_medoids(pd.DataFrame(a,columns=['x','y']),pd.DataFrame(b,columns=['x','y']),self.cfg)
        expected=np.full((53,6),np.nan)
        for i in range(53):
            for j in range(6):
                vals=[]
                for c,hi in enumerate([10.,20.]):
                    if np.isfinite(a[i,c]) and np.isfinite(b[j,c]):vals.append(abs(min(max(a[i,c],0),hi)-min(max(b[j,c],0),hi))/hi)
                if vals:expected[i,j]=sum(vals)/len(vals)
        np.testing.assert_allclose(actual,expected,atol=2e-7,equal_nan=True)

if __name__=='__main__':unittest.main()
