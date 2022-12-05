#This script takes in the input from tobac feature detection (tobac_updrafttracking.py)
#as well as segmentation (2D: tobac_surfacesegmentation.py; 
#3d: tobac_condensatesegmentation.py) and does quality control checks
#to eliminate spurious features with insufficiently long tracks,
#non-continuous condensate segmentations, etc. It then puts the final
#tobac output into a file named {anaPath}/tobac/full_tobac_output.pkl

import pandas as pd
import numpy as np
import os
import datetime as ddt
from jug import TaskGenerator

@TaskGenerator
def process_tobac(run): 
    #condenstate features
    cond = pd.DataFrame(columns=[])
    for p in [p for p in sorted(os.listdir(f"{anaPath}{run}/tobac")) if p.startswith('condensate_features-')]:
        cond = cond.append(pd.read_pickle(f"{anaPath}{run}/tobac/{p}"))
    cond = cond.set_index('feature')[['condensate_volume','CTH','CBH','maxW']]
    cond['condensate_volume'] = cond.condensate_volume / (1000*1000*1000)

    #updraft features
    up = pd.read_pickle(f"{anaPath}{run}/tobac/updraft_features.pkl").set_index('feature')[['vdim','hdim_1','hdim_2','frame','num','time']]
    track = pd.read_pickle(f"{anaPath}{run}/tobac/updraft_tracks.pkl").set_index('feature')[['cell','time_cell', 'lifetime']].dropna()    

    pcp = pd.read_pickle(f"{anaPath}{run}/tobac/PCPRR_features.pkl").set_index('feature')
    pcp = pcp[['area','mean','sum','max']]
    pcp['sum_pcprr'] = pcp['sum'] * 3600
    pcp['mean_pcprr'] = pcp['mean'] * 3600
    pcp['max_pcprr'] = pcp['max'] * 3600
    pcp['raining_area'] = pcp['area'] / (1000*1000)

    pcp = pcp[['sum_pcprr','mean_pcprr','max_pcprr','raining_area']]

    aero = pd.read_pickle(f"{anaPath}{run}/tobac/PCPRAERO_features.pkl").set_index('feature')
    aero = aero[['area','mean','sum','max']]
    aero['sum_wtdep'] = aero['sum'] * 3600
    aero['mean_wtdep'] = aero['mean'] * 3600
    aero['max_wtdep'] = aero['max'] * 3600
    aero['wtdep_area'] = aero['area'] / (1000*1000)
    aero = aero[['sum_wtdep','mean_wtdep','max_wtdep','wtdep_area']]

    out = up[up.index.isin(cond.index)]
    out = pd.concat([up,pcp,aero,cond,track],axis=1)

    #remove updraft cells with no associated cloud volume
    out = out[~out.condensate_volume.isnull()]

    cells = out.cell.unique()

    temp = cells[out.groupby('cell').lifetime.count()!=(out.groupby('cell').lifetime.first()/ddt.timedelta(minutes=5)+1)]

    #ensure cells have a consecutive track
    dropd = 0
    for c in temp:
        x = (consecutive(out[out.cell==c].time_cell/ddt.timedelta(minutes=5)))

        if len(x)!=1:
            #there is a gap in the track
            i = 0
            for y in x:
                if len(y)<=1:
                    out = out.drop(y.index.values)
                    dropd+=1
                else:
                    #removed some features which are not consecutive, but there is still a remaining cell
                    if i!=0:
                        out.loc[y.index,'cell'] = out.cell.max() + c + i
                    i+=1

                    out.loc[y.index,'time_cell'] = out.loc[y.index,'time'] - out.loc[y.index,'time'].min() 
                    out.loc[y.index,'lifetime'] = out.loc[y.index,'time_cell'].max()
        else:
            x = x[0]
            #stub
            if len(x)<=1:
                dropd+=1
                out = out.drop(x.index.values)

    out['featmean_cloud_area'] = out.condensate_volume / out.CTH
    out['cellmax_CTH'] = out.cell.map(out.groupby('cell').CTH.max())
    out['cellmean_maxW'] = out.cell.map(out.groupby('cell').maxW.mean())
    out['cellmax_maxW'] = out.cell.map(out.groupby('cell').maxW.max())

    out['cellmax_raining_area'] = out.cell.map(out.groupby('cell').raining_area.max())
    out['cellmean_raining_area'] = out.cell.map(out.groupby('cell').raining_area.mean())
    out['cellmax_wtdep_area'] = out.cell.map(out.groupby('cell').wtdep_area.max())
    out['cellmean_wtdep_area'] = out.cell.map(out.groupby('cell').wtdep_area.mean())

    out['cellmax_sum_pcprr'] = out.cell.map(out.groupby('cell').sum_pcprr.max())
    out['cellsum_sum_pcprr'] = out.cell.map(out.groupby('cell').sum_pcprr.sum())
    out['cellmean_mean_pcprr'] = out.cell.map(out.groupby('cell').mean_pcprr.mean())
    out['cellmax_mean_pcprr'] = out.cell.map(out.groupby('cell').mean_pcprr.max())
    out['cellmean_max_pcprr'] = out.cell.map(out.groupby('cell').max_pcprr.mean())
    out['cellmax_max_pcprr'] = out.cell.map(out.groupby('cell').max_pcprr.max())

    out['cellmax_sum_wtdep'] = out.cell.map(out.groupby('cell').sum_wtdep.max())
    out['cellsum_sum_wtdep'] = out.cell.map(out.groupby('cell').sum_wtdep.sum())
    out['cellmean_mean_wtdep'] = out.cell.map(out.groupby('cell').mean_wtdep.mean())
    out['cellmax_mean_wtdep'] = out.cell.map(out.groupby('cell').mean_wtdep.max())
    out['cellmean_max_wtdep'] = out.cell.map(out.groupby('cell').max_wtdep.mean())
    out['cellmax_max_wtdep'] = out.cell.map(out.groupby('cell').max_wtdep.max())

    out['time_raining'] = out.cell.map((out.groupby('cell').mean_pcprr.count()-1)*ddt.timedelta(minutes=5))
    out['time_raining'] = out.time_raining.replace(ddt.timedelta(minutes=-5), ddt.timedelta(minutes=0))
    out['perc_lifetime_raining'] = out.time_raining/out.lifetime

    out.to_pickle(f"{anaPath}{run}/tobac/full_tobac_output.pkl")

