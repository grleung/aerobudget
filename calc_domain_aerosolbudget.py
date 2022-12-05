#This script reads in RAMS output data (files starting with a-A-) and
#calculates the horizontally integrated aerosol mass in each aerosol budget 
#term. Output files are 2-D arrays where one axis is time and the other is height/altitude. 

import pandas as pd
import numpy as np
import h5py
from jug import TaskGenerator
import os
import time
from run_params import nx,ny,nz

runs = ['sulf.100','sulf.500','sulf.1000','sulf.1500',
        'absc.100','absc.500','absc.1000','absc.1500',
        'salt.100','salt.500','salt.1000','salt.1500',
        'dust.100','dust.500','dust.1000','dust.1500']

@TaskGenerator
def aero_budget(run):
    #This calculates the total aerosol mass in each category per vertical level at each timeseries
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

    regen = np.zeros((len(paths),nz-2))
    hydro = np.zeros((len(paths),nz-2))
    unpro = np.zeros((len(paths),nz-2))
    wtdep = np.zeros((len(paths)))
    unpronum = np.zeros((len(paths),nz-2))
    
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
    
    for i, p in enumerate(paths):
        with h5py.File(p) as f:
            #unactivated aerosol mass
            o = f[mas][1:nz-1,1:ny-1,1:nx-1]
            unpro[i,:] = np.nansum(o,axis=(1,2))
            
            #unactivated aerosol number
            o = f[num][1:nz-1,1:ny-1,1:nx-1]
            unpronum[i,:] = np.nansum(o,axis=(1,2))
            
            #accumulated wet-deposited/rained-out aerosol mass
            o = f['ACCPAERO'][1:ny-1,1:nx-1]
            wtdep[i] = np.nansum(o,axis=(0,1))
            
            #regenerated aerosol mass
            o = f['REGEN_AERO1_MP'][1:nz-1,1:ny-1,1:nx-1]+f['REGEN_AERO2_MP'][1:nz-1,1:ny-1,1:nx-1]
            regen[i,:] = np.nansum(o,axis=(1,2))
            
            #in-hydrometeor aerosol mass
            o = f['CNMCP'][1:nz-1,1:ny-1,1:nx-1]+f['CNMDP'][1:nz-1,1:ny-1,1:nx-1]+f['CNMRP'][1:nz-1,1:ny-1,1:nx-1]+f['CNMPP'][1:nz-1,1:ny-1,1:nx-1]+f['CNMSP'][1:nz-1,1:ny-1,1:nx-1]+f['CNMAP'][1:nz-1,1:ny-1,1:nx-1]+f['CNMGP'][1:nz-1,1:ny-1,1:nx-1]+f['CNMHP'][1:nz-1,1:ny-1,1:nx-1]
            hydro[i,:] = np.nansum(o,axis=(1,2))
            

    pd.DataFrame(wtdep).to_pickle(f"{anaPath}domain_byalt/wtdep_aero.pkl")
    pd.DataFrame(unpronum).to_pickle(f"{anaPath}domain_byalt/unpro_aero_num.pkl")
    pd.DataFrame(unpro).to_pickle(f"{anaPath}domain_byalt/unpro_aero.pkl") 
    pd.DataFrame(regen).to_pickle(f"{anaPath}domain_byalt/regen_aero.pkl") 
    pd.DataFrame(hydro).to_pickle(f"{anaPath}domain_byalt/hydro_aero.pkl")   
    
for run in runs:
    aero_budget(run)

    
