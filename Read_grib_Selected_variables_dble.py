#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 15:14:29 2025

@author: Selvaraj

Algorithm that
- detects clear skies in obs and model
- calculates the bias and standard deviation for each case from ground measurements
- works using the hour-by-hour method
- detection with SAT data for observations, and cloud cover in AROME
- on zones and not point by point comparison

Modified on Thu Sep 24 11:05:00 2025 by Selvaraj

"""

### ============================================================================
### LIBRARY AND TOOLS IMPORTATION

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from scipy import stats
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfea
from datetime import datetime as dt, timedelta
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as md
from matplotlib import cm
import time
import array as arr
from array import *

import sys; sys.path.insert(0,'../tools')
from num_jour_between import num_jour_between
import sys; sys.path.insert(0,'../cloud_detection')
import pandas as pd
import sys; sys.path.insert(0,'../../coding/')
from read_dble_cy48t1 import read_dble, read_dble_2_param, datehr_v, intV, Array_lon_lat, extract_time_V
import glob as glb
import copy as cpy
import cfgrib

 

work_dir = "/home/gmap/mrmn/selvarajd/SAVE/python/SWD/scores/"
cmd_should = 'rm -rf '+work_dir+'shouldfly-*'

grid_dir = "/scratch/work/selvarajd/GRIB/" 
grib_dir_vtx = "/scratch/mtool/selvarajd/cache/vortex/arome/pefrance/DBLE/"
grib_dble_pfx = "grid.arome-forecast.eurw1s40+00"

start = time.time()

### ============================================================================
### DONNEES ENTREES

date_temp1 = '20240406' # sys.argv[1]
date_temp2 = '20240408' # sys.argv[2]
y1 = int(date_temp1[:4])
m1 = int(date_temp1[4:6])
d1 = int(date_temp1[6:8])
y2 = int(date_temp2[:4])
m2 = int(date_temp2[4:6])
d2 = int(date_temp2[6:8])
date1 = dt(y1,m1,d1,0,0,0)
date2 = dt(y2,m2,d2,0,0,0)

PNT = "EPS"
K = 2 # int(sys.argv[3]) #Defaults value : 2
fn_lim = 0.02 # float( sys.argv[4]) #Defaults value : 0.02
###--- READING PEAROME at 1.3km of cy48t1----
run_UT = '2100P'
suite = "dble"
T_mem = 2 ###----ensemble members
forecast_time_Mx = 3 ###----Forecast terms

if suite=='dble':
        st_res = '_1k300m'
        print('suite is '+suite)
elif suite=='H3Z3':
        st_res = '_750m'
        print('suite is '+suite)
else:
        print('suite is not defined')


grib_dir = "/scratch/work/selvarajd/GRIB/"+suite+"/"
pre_fle = ''

fle_names = glb.glob(grib_dir+"*P")

for datei in fle_names:
        for mbN in range(T_mem):
                for f_ti in range(2, forecast_time_Mx): 
                        dir_nme_i = datei+"/mb0"+str(int(mbN))+"/forecast/"
                        fle_nme_i = grib_dble_pfx+str(int(f_ti+1))+':00.grib'
                        print( dir_nme_i + fle_nme_i ) 
                        dat_g = cfgrib.open_datasets(dir_nme_i + fle_nme_i)
                        ssrd_750m = dat_g[0]['ssrd'].data
                        lat_750m = dat_g[0]['ssrd']['latitude'].data
                        lng_750m = dat_g[0]['ssrd']['longitude'].data
                        del dat_g;
                        
                        if len(str(mbN))==1:
                                mb_str = "0"+str(mbN)
                        else:
                                mb_str = str(mbN)
                        
                        if len(str(f_ti))==1:
                                fxi_str = "0"+str(f_ti)
                                fxi_str_t1 = "0"+str(f_ti-1)
                        else:
                                fxi_str = str(f_ti)
                                fxi_str_t1 = str(f_ti-1)
                        
                        datei_dt = datei[-16:]
                        dir_sfx = datei_dt.replace('-','')+"/mb0"+mb_str+"/forecast/"
                        dat_vtx_t = cfgrib.open_datasets(grib_dir_vtx+dir_sfx+grib_dble_pfx+fxi_str+":00.grib")
                        jjkqjfk
                        ssrd_vtx_t = dat_vtx_t[20]['ssrd'].data
                        lat_1k3 = dat_vtx_t[20]['ssrd']['latitude'].data
                        lng_1k3 = dat_vtx_t[20]['ssrd']['longitude'].data
                        #kqjjkqg
                        del dat_vtx_t;
                        dat_vtx_t1 = cfgrib.open_datasets(grib_dir_vtx+dir_sfx+grib_dble_pfx+fxi_str_t1+":00.grib")
                        ssrd_vtx_t1 = dat_vtx_t1[20]['ssrd'].data
                        ssrd_1k3 =  (ssrd_vtx_t - ssrd_vtx_t1)/3600 ###--Wm-2
                        del ssrd_vtx_t1, ssrd_vtx_t;

                        fle_nme_png = grib_dir+dir_sfx.replace('/','_')+mb_str+'_'+fxi_str+'_ssrd.png'
                        fig = plt.figure(1, figsize =(9, 8))
                        plt.subplot(1,2,1)
                        plt.pcolor(lng_1k3,lat_1k3,np.log10(ssrd_1k3), vmin=-14,vmax=-13.8); plt.colorbar()
                        plt.xlim([0, 4]); plt.ylim([47.75, 50]);
                        plt.title(suite+st_res+' from vortex')
                        plt.subplot(1,2,2)
                        plt.pcolor(lng_750m,lat_750m,np.log10(ssrd_750m), vmin=-14,vmax=-13.8); plt.colorbar()
                        plt.xlim([0, 4]); plt.ylim([47.75, 50])
                        plt.title(suite+st_res+' from regriding')
                        plt.show()
                        fig.savefig(fle_nme_png, dpi = 300, bbox_inches='tight')
                        plt.close('all')

