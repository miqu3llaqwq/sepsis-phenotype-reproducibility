"""Observed-value assignment kernel extracted unchanged from the study source.
No clinical eligibility or event-time inference is performed here.
"""
import numpy as np
import pandas as pd

def distance_to_medoids(rows,medoids,config):
    features=config['features'];lim=config['limits']
    a=rows[features].to_numpy(float);b=medoids[features].to_numpy(float)
    va=np.isfinite(a);vb=np.isfinite(b)
    numerator=np.zeros((len(a),len(b)),np.float32);denominator=np.zeros_like(numerator)
    for j,f in enumerate(features):
        lo,hi=lim[f]
        if not np.isfinite([lo,hi]).all() or hi<=lo:raise ValueError('Invalid frozen bounds: '+f)
        av=(np.clip(np.nan_to_num(a[:,j],nan=0.),lo,hi)-lo)/(hi-lo)
        bv=(np.clip(np.nan_to_num(b[:,j],nan=0.),lo,hi)-lo)/(hi-lo)
        common=va[:,j,None]&vb[None,:,j]
        numerator+=np.where(common,np.abs(av[:,None]-bv[None,:]),0.).astype(np.float32)
        denominator+=common
    return np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>0)

def assign(rows,medoids,config):
    if not medoids.cluster.tolist()==list(range(1,len(medoids)+1)):raise ValueError('Invalid medoid/label ordering')
    distance=distance_to_medoids(rows,medoids,config)
    # Unknown distances could conceal the true nearest medoid; do not silently rank a subset.
    complete=np.isfinite(distance).all(axis=1)
    labels=np.full(len(rows),np.nan)
    labels[complete]=np.argmin(distance[complete],axis=1)+1
    return labels,complete,distance
