#This script runs main tobac tracking in 3D with periodic boundaries 
#accounted for. This identifies updraft features and tracks them in time,
#and places output as pkl files in {anaPath}/tobac/ folder

import os 
import iris
import numpy as np
import pandas as pd
from run_params import *
import tobac
from jug import TaskGenerator
import datetime as ddt

anaPath = f"/squall/gleung/aerobudget-analysis/"

runs = ['sulf.100','sulf.500','sulf.1000','sulf.1500',
        'absc.100','absc.500','absc.1000','absc.1500',
        'salt.100','salt.500','salt.1000','salt.1500',
        'dust.100','dust.500','dust.1000','dust.1500']

#Feature Detection
parameters_features={}
parameters_features['position_threshold']='weighted_diff'
parameters_features['min_distance']=0
parameters_features['sigma_threshold']=1
parameters_features['threshold']=[1,3,5] #m/s
parameters_features['n_erosion_threshold']=0
parameters_features['n_min_threshold']=10
parameters_features['PBC_flag'] = 'both'
parameters_features['vertical_coord'] = 'altitude'
#Tracking
parameters_linking={}
parameters_linking['method_linking']='predict'
parameters_linking['adaptive_stop']=0.2
parameters_linking['adaptive_step']=0.95
parameters_linking['extrapolate']=0
parameters_linking['order']=1
parameters_linking['subnetwork_size']=100
parameters_linking['memory']=0
parameters_linking['time_cell_min']=5*60
parameters_linking['method_linking']='predict'
parameters_linking['v_max']=10
parameters_linking['d_min']=2000  
parameters_linking['min_h1']=0  
parameters_linking['max_h1']=1000  
parameters_linking['min_h2']=0  
parameters_linking['max_h2']=1000  
parameters_linking['PBC_flag']='both' 
parameters_linking['vertical_coord'] = 'altitude'  

@TaskGenerator
def run_tobac(run):
    if os.path.exists(f"/moonbow/gleung/aerobudget/{run}"):
        dataPath = f"/moonbow/gleung/aerobudget/{run}/"
    else:
        dataPath = f"/squall/gleung/aerobudget/{run}/"   
    
    anaPath = f"/squall/gleung/aerobudget-analysis/{run}/"
    
    if not os.path.isdir(anaPath):
        os.mkdir(anaPath)
    if not os.path.isdir(f"{anaPath}tobac"):
        os.mkdir(f"{anaPath}tobac")

    paths = [f"{dataPath}{run}/{p}" for p in 
             sorted(os.listdir(f"{dataPath}{run}")) 
             if p.startswith('a-A-') and p.endswith('.h5')]
    
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

    dxy,dt=tobac.get_spacings(ws)
    Features_3d=tobac.feature_detection_multithreshold(ws,dxy,**parameters_features)

   #Tracking
    Track_3d=tobac.linking_trackpy(Features_3d,ws,dt=dt,dxy=dxy,**parameters_linking)
    Track_3d['lifetime'] = Track_3d.cell.map(Track_3d.groupby('cell').time_cell.max())
    
    Features_3d = Features_3d[Features_3d.feature.isin(Track_3d[(Track_3d.lifetime>=ddt.timedelta(minutes=5))].feature)]
    
    Features_3d.to_pickle(f"{anaPath}{run}/tobac/updraft_features.pkl")
    Track_3d.to_pickle(f"{anaPath}{run}/tobac/updraft_tracks.pkl")
     
    
for run in runs:
    run_tobac(run)
