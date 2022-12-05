#This script runs tobac segmentation of cloud condensate (3D)
#This requires that tobac_updrafttracking.py has already been 
#run so updraft features are already identified. Note that this
#splits up the segmentation into 12 time sections to avoid loading 
#very large arrays into memory.

import os 
import iris
import numpy as np
import pandas as pd
from run_params import *
import tobac
import time
from jug import TaskGenerator
import datetime as ddt
from scipy.ndimage import labeled_comprehension

anaPath = f"/squall/gleung/aerobudget-analysis/"

runs = ['sulf.100','sulf.500','sulf.1000','sulf.1500',
        'absc.100','absc.500','absc.1000','absc.1500',
        'salt.100','salt.500','salt.1000','salt.1500',
        'dust.100','dust.500','dust.1000','dust.1500']

parameters_segmentation_tcon={}
parameters_segmentation_tcon['method']='watershed'
parameters_segmentation_tcon['threshold']=1E-5  # kg/kg mixing ratio
parameters_segmentation_tcon['seed_3D_flag']='box'
parameters_segmentation_tcon['PBC_flag']='both'

@TaskGenerator
def run_segment(run,sectionid):
    if os.path.exists(f"/moonbow/gleung/aerobudget/{run}"):
        dataPath = "/moonbow/gleung/aerobudget/"
    else:
        dataPath = "/squall/gleung/aerobudget/"
        
    if not os.path.exists(f"{anaPath}{run}/tobac/"):
        os.mkdir(f"{anaPath}{run}/tobac/")
    
    Features_3d = pd.read_pickle(f"{anaPath}{run}/tobac/updraft_features.pkl")

    startdate = ddt.datetime(year=2019,month=9,day=16)
    
    Features_3d = Features_3d[(Features_3d.time>=startdate+ddt.timedelta(hours=(sectionid)*4)) &(Features_3d.time<startdate+ddt.timedelta(hours=(sectionid+1)*4))]

    paths = [f"{dataPath}{run}/{p}" for p in 
             sorted(os.listdir(f"{dataPath}{run}")) 
             if p.startswith('a-A-') and p.endswith('.h5') 
             and (pd.to_datetime(p.split('/')[-1][4:-6]) in Features_3d.time.unique())]
    
    if len(paths)==0:
        return()
    
    lat = iris.load(paths[0],'GLAT')[0]
    lon = iris.load(paths[0],'GLON')[0]

    xs = iris.coords.DimCoord(np.arange(0,nx*dx,dx), 
                              standard_name='projection_x_coordinate',
                             units = 'metre')
    ys = iris.coords.DimCoord(np.arange(0,ny*dy,dy), 
                              standard_name='projection_y_coordinate',
                             units = 'metre')
    zs = iris.coords.DimCoord(np.arange(0,nz,1), 
                              standard_name='model_level_number')

    lat = iris.coords.AuxCoord(lat.data,
                        standard_name='latitude',
                        units='degrees')

    lon = iris.coords.AuxCoord(lon.data,
                        standard_name='longitude',
                        units='degrees')

    altitude = iris.coords.AuxCoord(alt.values,
                        standard_name='altitude',
                        units='metre')

    times = [(pd.to_datetime(p.split('/')[-1][4:-6])-ddt.datetime(year=2019,month=9,day=16))/ddt.timedelta(minutes=1) for p in sorted(paths)]
    times = iris.coords.DimCoord(times, 
                              standard_name='time', 
                                               units = 'minutes since 2019-09-16 00:00:00')
    
    
    #Segmentation
    temp = iris.load(paths,['RCP','RPP','RSP'])
    tcons = []
    for i,p in enumerate(paths):
        tcon = temp[i] + temp[len(temp)//3 + i] + temp[2*len(temp)//3 + i]
        tcon.add_aux_coord(iris.coords.AuxCoord(
                                pd.to_datetime(p.split('/')[-1][4:-6]),
                                standard_name='time'))

        tcons.append(tcon)

    tcon = iris.cube.CubeList(tcons).merge()[0]
    tcon.remove_coord('time')
    tcon.var_name = 'cloud_condensate'
    tcon.add_dim_coord(times,0)
    tcon.add_dim_coord(zs,1)
    tcon.add_dim_coord(ys,2)
    tcon.add_dim_coord(xs,3)
    tcon.add_aux_coord(lat, data_dims=[2,3])
    tcon.add_aux_coord(lon, data_dims=[2,3])
    tcon.add_aux_coord(altitude, data_dims=1)

    ws = iris.load(paths,'WC')

    for w,p in zip(ws, paths):
        w.add_aux_coord(iris.coords.AuxCoord(
                                pd.to_datetime(p.split('/')[-1][4:-6]),
                                standard_name='time'))

    ws = iris.cube.CubeList(ws).merge()[0]

    ws.remove_coord('time')
    ws.add_dim_coord(times,0)
    ws.add_dim_coord(zs,1)
    ws.add_dim_coord(ys,2)
    ws.add_dim_coord(xs,3)
    ws.add_aux_coord(altitude, data_dims=1)
    ws.add_aux_coord(lat, data_dims=[2,3])
    ws.add_aux_coord(lon, data_dims=[2,3])

    dxy = 100

    mask,Features_tcon_3d=tobac.segmentation_3D(Features_3d,tcon,dxy,**parameters_segmentation_tcon)

    #Calculating Parameters
    volume  = np.array([100*100*dz[i] for i in range(len(dz))])
    volume = np.tile(volume, (len(paths),nx,ny,1))
    volume = np.transpose(volume, (0, 3, 1, 2))

    Features_tcon_3d['condensate_volume'] = labeled_comprehension(
         volume, mask.data, Features_tcon_3d["feature"], np.sum, volume.dtype, np.nan)
    zs  = np.array([i/1000 for i in alt])
    zs = np.tile(zs, (len(paths),nx,ny,1))
    zs = np.transpose(zs, (0, 3, 1, 2))

    Features_tcon_3d['CTH'] = labeled_comprehension(
        zs, mask.data, Features_tcon_3d["feature"], np.max, zs.dtype, np.nan)
    Features_tcon_3d['CBH'] = labeled_comprehension(
        zs, mask.data, Features_tcon_3d["feature"], np.min, zs.dtype, np.nan)
    Features_tcon_3d['maxW'] = labeled_comprehension(
        ws.data, mask.data, Features_tcon_3d["feature"], np.max, ws.data.dtype, np.nan)
                                  
    Features_tcon_3d.to_pickle(f"{anaPath}{run}/tobac/condensate_features-{sectionid}.pkl")
 
for run in runs:
    if (os.path.exists(f"{anaPath}{run}/tobac/updraft_features.pkl")):
        for sectionid in [0,1,2,3,4,5,6,7,8,9,10,11]:
            run_segment(run, sectionid)
