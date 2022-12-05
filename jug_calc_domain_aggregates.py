#This script reads in RAMS output data (files starting with a-A-) and
#calculates aggregates across the domain. Output files are 2-D arrays
#where one axis is time and the other is height/altitude. Depending on the
#variable, either the horizontal sum, mean, or maximum is given. Fields
#such as rain rate and vertical velocity are also conditionally sampled
#to get the mean maining rain rate or mean updraft velocity.

import pandas as pd
import numpy as np
import h5py
from jug import TaskGenerator
import os
from run_params import nx,ny,nz

runs = ['sulf.100','sulf.500','sulf.1000','sulf.1500',
        'absc.100','absc.500','absc.1000','absc.1500',
        'salt.100','salt.500','salt.1000','salt.1500',
        'dust.100','dust.500','dust.1000','dust.1500']

@TaskGenerator
def domain_agg(run):
    if os.path.exists(f"/moonbow/gleung/aerobudget/{run}"):
        dataPath = f"/moonbow/gleung/aerobudget/{run}/"
    else:
        dataPath = f"/squall/gleung/aerobudget/{run}/"
        
    anaPath = f"/squall/gleung/aerobudget-analysis/{run}/"
                
    if not os.path.isdir(anaPath):
        os.mkdir(anaPath)
    if not os.path.isdir(f"{anaPath}domain_byalt"):
        os.mkdir(f"{anaPath}domain_byalt")
        
    paths = [f"{dataPath}{p}" for p in sorted(os.listdir(dataPath)) if p.startswith('a-A-') and p.endswith('.h5')]
    
    tconnorain = np.zeros((len(paths),nz-2))
    tcon = np.zeros((len(paths),nz-2))
    pcprr = np.zeros((len(paths)))
    accpr = np.zeros((len(paths)))
    wc = np.zeros((len(paths),nz-2))
    up = np.zeros(len(paths))
    rain = np.zeros(len(paths))
    hydronum = np.zeros((len(paths),nz-2))
    
    for i, p in enumerate(paths):
        with h5py.File(p) as f:
            o = f['CSP'][1:nz-1,1:ny-1,1:nx-1] + f['CCP'][1:nz-1,1:ny-1,1:nx-1] + f['CPP'][1:nz-1,1:ny-1,1:nx-1]+f['CAP'][1:nz-1,1:ny-1,1:nx-1] + f['CRP'][1:nz-1,1:ny-1,1:nx-1] + f['CDP'][1:nz-1,1:ny-1,1:nx-1]+f['CHP'][1:nz-1,1:ny-1,1:nx-1] + f['CGP'][1:nz-1,1:ny-1,1:nx-1]
            hydronum[i,:] = np.nansum(o,axis=(1,2)) 
            
            o = f['RSP'][1:nz-1,1:ny-1,1:nx-1] + f['RCP'][1:nz-1,1:ny-1,1:nx-1] + f['RPP'][1:nz-1,1:ny-1,1:nx-1]
            tconnorain[i,:] = np.nansum(o,axis=(1,2))

            o = f['RTP'][1:nz-1,1:ny-1,1:nx-1] - f['RV'][1:nz-1,1:ny-1,1:nx-1]
            tcon[i,:] = np.nansum(o,axis=(1,2))
            
            o = f['PCPRR'][1:ny-1,1:nx-1]          
            pcprr[i] = np.nanmean(o,axis=(0,1))
            
            o = f['ACCPR'][1:ny-1,1:nx-1]          
            accpr[i] = np.nansum(o,axis=(0,1))
                      
            o = f['WC'][1:nz-1,1:ny-1,1:nx-1]
            wc[i,:] = np.nanmax(o,axis=(1,2))
            
            o = f['PCPRR'][1:ny-1,1:nx-1]    
            o = o[o>=2E-6]      
            rain[i] = np.nanmean(o)

            o = f['WC'][1:nz-1,1:ny-1,1:nx-1]
            o = o[o>=0.01]
            up[i] = np.nanmean(o)
                      

   
    pd.DataFrame(tcon).to_pickle(f"{anaPath}domain_byalt/tcon_sum.pkl")
    pd.DataFrame(tconnorain).to_pickle(f"{anaPath}domain_byalt/tconnorain_sum.pkl")
    pd.DataFrame(pcprr).to_pickle(f"{anaPath}domain_byalt/pcprr_mean.pkl")
    pd.DataFrame(accpr).to_pickle(f"{anaPath}domain_byalt/accpr_sum.pkl")
    pd.DataFrame(wc).to_pickle(f"{anaPath}domain_byalt/wc_max.pkl")
    pd.DataFrame(rain).to_pickle(f"{anaPath}domain_byalt/pcprr_raining_mean.pkl")
    pd.DataFrame(up).to_pickle(f"{anaPath}domain_byalt/wc_updraft_mean.pkl")
    pd.DataFrame(hydronum).to_pickle(f"{anaPath}domain_byalt/hydro_aero_num.pkl")
    
for run in runs:
    domain_agg(run)

