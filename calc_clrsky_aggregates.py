#This script reads in RAMS output data (files starting with a-A-) and
#calculates clear-sky aggregates across the domain. Output files are 2-D 
#arrays where one axis is time and the other is height/altitude. Fields
#are conditionally sampled to get the mean values among grid cells that
#satisfy the total condensate thresholds for clear sky.

import pandas as pd
import numpy as np
import h5py
from jug import TaskGenerator
import os

runs = ['sulf.100','sulf.500','sulf.1000','sulf.1500',
        'absc.100','absc.500','absc.1000','absc.1500',
        'salt.100','salt.500','salt.1000','salt.1500',
        'dust.100','dust.500','dust.1000','dust.1500'
       ]

variables = ['THETA','PI','RV','FTHRD',
             'SWUP','SWDN','LWUP','LWDN',
             'REGEN_AERO1_NP','REGEN_AERO1_MP',
            'REGEN_AERO2_NP','REGEN_AERO2_MP']

@TaskGenerator
def calc_hydro(run, v, tconthresh=1E-5):
    #This calculates a mean vertical profile per timestep of a given variable only taken over in-cloud and in-updraft points
    
    if os.path.isdir(f"/moonbow/gleung/aerobudget/{run}/"):
        dataPath = f"/moonbow/gleung/aerobudget/{run}/"
    else:
        dataPath = f"/squall/gleung/aerobudget/{run}/"
        
    anaPath = f"/squall/gleung/aerobudget-analysis/{run}/"
    if not os.path.isdir(anaPath):
        os.mkdir(anaPath)
    if not os.path.isdir(f"{anaPath}domain_byalt"):
        os.mkdir(f"{anaPath}domain_byalt")
        
   
    paths = [f"{dataPath}{p}" for p in sorted(os.listdir(dataPath)) if p.startswith('a-A-') and p.endswith('.h5')]
    
    out = np.zeros((len(paths),120))
    
    for i, p in enumerate(paths):
        with h5py.File(p) as f:
            rtp = f['RTP'][:,:,:]
            rv = f['RV'][:,:,:]
            tcon = rtp - rv

            ovar = f[v][:,:,:]
                
            mask = (tcon>=tconthresh)
            mask = ~np.any(mask, axis=0)
            mask = np.tile(mask,(120,1,1))
            out[i,:] = np.nanmean(np.where(mask,
                                           ovar,np.nan),
                                  axis=(1,2))


    pd.DataFrame(out).to_pickle(f"{anaPath}domain_byalt/{v}_clrsky_mean.pkl")
    
    
for run in runs:
    if run.split('.')[0] == 'sulf':
        mas = 'CCCMP'
        num = 'CCCNP'
    elif run.split('.')[0] == 'absc':
        mas = 'ABC2MP'
        num = 'ABC2NP'
    elif run.split('.')[0] == 'salt':
        mas = 'SALT_FILM_MP'
        num = 'SALT_FILM_NP'
    elif run.split('.')[0] == 'dust':
        mas = 'MD1MP'
        num = 'MD1NP'
 
    aero = [mas,num]
    for v in variables+aero:
        calc_hydro(run, v)
