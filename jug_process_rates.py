#This reads from REVU output of vertically integrated aerosol and makes a
#domain sum for each run as well as one comparison file for all runs.
#Note that the REVU must be run for the namelists REVU.proc.{run} before
#this script can be run. That generates a file starting with "procliq".
#Also requires jug_calc_domain_aggregates.py to be run for rain rates.

import pandas as pd
import numpy as np
import h5py
from jug import TaskGenerator, barrier
import os
from run_params import nx,ny,nz
import xarray as xr

runs = ['sulf.100','sulf.500','sulf.1000','sulf.1500',
        'absc.100','absc.500','absc.1000','absc.1500',
        'salt.100','salt.500','salt.1000','salt.1500',
        'dust.100','dust.500','dust.1000','dust.1500'
       ]

anaPath = f"/squall/gleung/aerobudget-analysis/"

if os.path.isdir(f"{anaPath}/comparison/"):
    os.mkdir(f"{anaPath}/comparison/")

@TaskGenerator
def revu_procliq(run):
    #domain sum of liquid processes in REVU
    ds = xr.open_dataset(f"{anaPath}{run}/procliq-a-AC-2019-09-16-000000-g1.h5").isel(x_coords=slice(1,nx-2), y_coords=slice(1,ny-2))
    
    ds = ds.groupby('t_coords').sum(...).to_dataframe()
    
    ds.to_pickle(f"{anaPath}{run}/domain_sum_procliq.pkl")
    

for run in runs:
    if not os.path.exists(f"{anaPath}{run}/domain_sum_procliq.pkl"):
        revu_procliq(run)
    
barrier()

#Second half of script takes the domain_sum_aero and calculates aerosol budget as percentages of total process rate

liqvariables = ['vt_cld2raint', 'vt_nuccldrt',
            'vt_vapcldt','vt_vapraint','vt_vapliqt', 
            'vt_evapcldt','vt_evapraint','vt_evapliqt']

runs = [r for r in os.listdir(anaPath) if os.path.exists(f"{anaPath}{r}/domain_sum_procliq.pkl")]

for var in liqvariables:
    dfOut = pd.DataFrame()
    
    for run in runs:
        df = pd.read_pickle(f"{anaPath}{run}/domain_sum_procliq.pkl")[var]
        
        dfOut[run] = df

    dfOut.to_pickle(f"{anaPath}/comparison/proc-{var}.pkl")

var = 'accpr_sum'

df = pd.read_pickle(f"{anaPath}/comparison/proc-vt_vapliqt.pkl")
outdf = pd.DataFrame(columns = [], index = df.index)

for run, col in zip(runs, cs):
    df = pd.read_pickle(f"{anaPath}{ver}.{conc}/domain_byalt/{var}.pkl")
    outdf[run] =df.values

outdf.to_pickle(f"{anaPath}/comparison/proc-accpr.pkl")