#This script runs tobac segmentation of rain rates and rainout rates 
#(2D surface fields). This requires that tobac_updrafttracking.py has
#already been run so updraft features are already identified.

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

parameters_segmentation_pcprr={}
parameters_segmentation_pcprr['method']='watershed'
parameters_segmentation_pcprr['threshold']= 0.01/3600  # mm/hr converted rain rate
parameters_segmentation_pcprr['PBC_flag']='both' 

parameters_segmentation_pcpraero={}
parameters_segmentation_pcpraero['method']='watershed'
parameters_segmentation_pcpraero['threshold']=1E-8/3600  # mm/hr converted rain rate
parameters_segmentation_pcpraero['PBC_flag']='both' 


@TaskGenerator
def run_segment_2d(run, var, parameters_segmentation):
    if os.path.isdir(f"/moonbow/gleung/aerobudget/{run}"):
        dataPath = "/moonbow/gleung/aerobudget/"
    else:
        dataPath = "/squall/gleung/aerobudget/"
        
    if not os.path.exists(f"{anaPath}{run}/tobac/"):
        os.mkdir(f"{anaPath}{run}/tobac/")
    
    Features_3d = pd.read_pickle(f"{anaPath}{run}/tobac/updraft_features.pkl")

    paths = [f"{dataPath}{run}/{p}" for p in 
             sorted(os.listdir(f"{dataPath}{run}")) 
             if p.startswith('a-A-') and p.endswith('.h5') 
             and (pd.to_datetime(p.split('/')[-1][4:-6]) in Features_3d.time.unique())]
    
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
    data = iris.load(paths,var)

    for w,p in zip(data, paths):
        w.add_aux_coord(iris.coords.AuxCoord(
                                pd.to_datetime(p.split('/')[-1][4:-6]),
                                standard_name='time'))

    data = iris.cube.CubeList(data).merge()[0] 

    data.remove_coord('time')
    data.add_dim_coord(times,0)
    data.add_dim_coord(ys,1)
    data.add_dim_coord(xs,2)
    data.add_aux_coord(lat, data_dims=[1,2])
    data.add_aux_coord(lon, data_dims=[1,2])

    dxy, dt = tobac.get_spacings(data)
    mask,Features=tobac.segmentation_2D(Features_3d,data,
                                              dxy,**parameters_segmentation)

    #Calculating Parameters
    if not (mask.coord("projection_x_coordinate").has_bounds()):
        mask.coord("projection_x_coordinate").guess_bounds()

    if not (mask.coord("projection_y_coordinate").has_bounds()):
        mask.coord("projection_y_coordinate").guess_bounds()

    area = np.outer(
        np.diff(mask.coord("projection_x_coordinate").bounds, axis=1),
        np.diff(mask.coord("projection_y_coordinate").bounds, axis=1),
            )

    Features['area'] = labeled_comprehension(
        area, mask.data, Features["feature"], np.sum, area.dtype, np.nan
        )

    Features['mean'] = labeled_comprehension(
        data.data, mask.data, Features["feature"], np.mean, data.data.dtype, np.nan
    )

    Features['sum'] = labeled_comprehension(
        data.data, mask.data, Features["feature"], np.sum, data.data.dtype, np.nan
    )

    Features['max'] = labeled_comprehension(
        data.data, mask.data, Features["feature"], np.max, data.data.dtype, np.nan
    )

    Features.to_pickle(f"{anaPath}{run}/tobac/{var}_features.pkl")    
    
for run in runs:
    if os.path.exists(f"{anaPath}{run}/tobac/updraft_tracks.pkl"):
        for var, parameters_segmentation in zip(['PCPRR','PCPRAERO'],
                                               [parameters_segmentation_pcprr, parameters_segmentation_pcpraero]):
            run_segment_2d(run, var, parameters_segmentation)
