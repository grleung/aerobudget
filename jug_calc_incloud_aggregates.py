#This script reads in RAMS output data (files starting with a-A-) and
#calculates in-cloud aggregates across the domain. Output files are 2-D 
#arrays where one axis is time and the other is height/altitude. Fields
#are conditionally sampled to get the mean values among grid cells that
#satisfy the vertical velocity and cloud condensate thresholds for a
#cloudy updraft.

import pandas as pd
import numpy as np
import h5py
from jug import TaskGenerator
import os

#RAMS power law constants
cfmas = 524 #for rain, cloud, drizzle
pwmas = 3

runs = ['sulf.100','sulf.500','sulf.1000','sulf.1500',
        'absc.100','absc.500','absc.1000','absc.1500',
        'salt.100','salt.500','salt.1000','salt.1500',
        'dust.100','dust.500','dust.1000','dust.1500'
       ]

variables = ['WC',
             'RCP','CCP',
             'RRP','CRP',
            'CDP','RDP']

@TaskGenerator
def calc_hydro(run, v, wthresh=1, tconthresh=1E-5):
    #This calculates a mean vertical profile per timestep of a given variable only taken over in-cloud and in-updraft points
    
    if os.path.isdir(f"/moonbow/gleung/detrain-2/{run}/"):
        dataPath = f"/moonbow/gleung/detrain-2/{run}/"
    else:
        dataPath = f"/squall/gleung/detrain-2/{run}/"
    anaPath = f"/squall/gleung/detrain-2-analysis/{run}/"

    paths = [f"{dataPath}{p}" for p in sorted(os.listdir(dataPath)) if p.startswith('a-A-') and p.endswith('.h5')]
    
    out = np.zeros((len(paths),120))
    
    if v == 'WC':
        count = np.zeros((len(paths),120))

    for i, p in enumerate(paths):
        with h5py.File(p) as f:
            rtp = f['RTP'][:,:,:]
            rv = f['RV'][:,:,:]
            tcon = rtp - rv

            w = f['WC'][:,:,:]
            
            ovar = f[v][:,:,:]
                
            mask = (w>=wthresh) & (tcon>=tconthresh)

            out[i,:] = np.nanmean(np.where(mask,ovar,np.nan),axis=(1,2))
            
            if v == 'WC':
                count[i,:] = np.count_nonzero(mask,axis=(1,2))

    pd.DataFrame(out).to_pickle(f"{anaPath}domain_byalt/{v}_incloud_mean.pkl")
        
    if v == 'WC':
        pd.DataFrame(count).to_pickle(f"{anaPath}domain_byalt/cloudpts_count.pkl")
    
for run in runs:
    for v in variables:
        calc_hydro(run, v)
