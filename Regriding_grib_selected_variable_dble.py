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
from datetime import datetime, timedelta
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as md
from matplotlib import cm
import time
import array as arr
from array import *
from pathlib import Path

import sys; sys.path.insert(0,'../tools')
from num_jour_between import num_jour_between
import sys; sys.path.insert(0,'../cloud_detection')
import pandas as pd
import sys; sys.path.insert(0,'../../coding/')
from read_dble_cy48t1 import read_dble, read_dble_2_param, read_dble_45_param, datehr_v, intV, Array_lon_lat, extract_time_V
import glob as glb
import copy as cpy
import epygram

work_dir = "/home/gmap/mrmn/selvarajd/SAVE/python/SWD/scores/"
cmd_should = 'rm -rf '+work_dir+'shouldfly-*'

grid_dir = "/scratch/work/selvarajd/GRIB/" 
###---grib_pfx = 'grid.arome-forecast.paris1s100+00'
grib_pfx = 'grid.arome-forecast.eurw1s40+00'

start = time.time()

### ============================================================================
### DONNEES ENTREES

date_temp1 = '20240406' # sys.argv[1]
date_temp2 = '20240406' # sys.argv[2]
y1 = int(date_temp1[:4])
m1 = int(date_temp1[4:6])
d1 = int(date_temp1[6:8])
y2 = int(date_temp2[:4])
m2 = int(date_temp2[4:6])
d2 = int(date_temp2[6:8])
date1 = datetime(y1,m1,d1,0,0,0)
date2 = datetime(y2,m2,d2,0,0,0)

PNT = "EPS"
###--- READING PEAROME at 1.3km of cy48t1----
run_UT = '2100P'
suite = "dble" ###-----"dble"---"H3Z3"---
T_mem = 2 ###--24--ensemble members
forecast_time_Mx = 5 ###--51--Forecast terms

tic_h = time.time();

def detect_hourly(swd_l, h_term_l, dates_swd, date_v_l):
	swd_h = swd_l
	h_term = h_term_l
	dates_l = dates_swd
	date_v = date_v_l
	run_date = date_v+timedelta(hours = h_term) 
	run_date_ID = np.where(dates_l==run_date)[0][0]
	print(run_date_ID)
	return run_date_ID

num_jour = num_jour_between(date1,date2)
if __name__ == "__main__":
	for d in range(0,num_jour):
		##--- LECTURE JOUR ET FICHIER AROME/AROME FILES LECTURE-------
		date_v = date1 + timedelta(days = d)
		annee = date_v.year
		mois = date_v.month
		jour = date_v.day
		
                ###---f_srf = read_dble(date_v,run_UT,suite,T_mem,forecast_time_Mx,param)
		param_1 = 'H_COULIM'	
		param_2 = 'SRD'  ###--param=--ssrd----SURFRAYT SOLA DE	SURFRAYT SOLA DE	W.m-2	Cum. Downward solarflux at surface
		param_3 = 'NEBUL'
		param_4 = 'NEBBAS'
		param_5 = 'NEBMOY'
		param_6 = 'NEBHAUT'
		param_7 = 'B_NUAGE'
		param_8 = 'CC10'
		param_9 = 'CC20'
		param_10 = 'VISI'
		param_11 = 'T850'
		param_12 = 'T700'
		param_13 = 'T500'
		param_14 = 'T300'
		param_15 = 'T200'
		param_16 = 'HU850'
		param_17 = 'HU700'
		param_18 = 'HU500'
		param_19 = 'HU300'
		param_20 = 'HU200'
		param_21 = 'U850'
		param_22 = 'U700'
		param_23 = 'U500'
		param_24 = 'U300'
		param_25 = 'U200'
		param_26 = 'V850'
		param_27 = 'V700'
		param_28 = 'V500'
		param_29 = 'V300'
		param_30 = 'V200'
		param_31 = 'W850'
		param_32 = 'W700'
		param_33 = 'W500'
		param_34 = 'W300'
		param_35 = 'W200'
		param_36 = 'P2K'
		param_37 = 'P2K25'
		param_38 = 'P2K5'
		param_39 = 'P2K75'
		param_40 = 'P3K'
		param_41 = 'TKE1K'
		param_42 = 'TKE1K5'
		param_43 = 'TKE2K'
		param_44 = 'TKE2K5'
		param_45 = 'TKE3K'
		#param_46 = 'NEBCON'

		param_1 = 'H_COULIM'	
		param_2 = 'SRD' 
		param_3 = 'NEBUL'
		param_4 = 'NEBBAS'
		param_5 = 'NEBMOY'
		param_6 = 'NEBHAUT'
		param_7 = 'B_NUAGE'
		param_8 = 'CC10'
		param_9 = 'CC20'
		param_10 = 'VISI'
		param_11 = 'T850'
		param_12 = 'T700'
		param_13 = 'T500'
		param_14 = 'T300'
		param_15 = 'T200'
		param_16 = 'HU850'
		param_17 = 'HU700'
		param_18 = 'HU500'
		param_19 = 'HU300'
		param_20 = 'HU200'
		param_21 = 'U850'
		param_22 = 'U700'
		param_23 = 'U500'
		param_24 = 'U300'
		param_25 = 'U200'
		param_26 = 'V850'
		param_27 = 'V700'
		param_28 = 'V500'
		param_29 = 'V300'
		param_30 = 'V200'
		param_31 = 'W850'
		param_32 = 'W700'
		param_33 = 'W500'
		param_34 = 'W300'
		param_35 = 'W200'
		param_36 = 'P2K'
		param_37 = 'P2K25'
		param_38 = 'P2K5'
		param_39 = 'P2K75'
		param_40 = 'P3K'
		param_41 = 'TKE1K'
		param_42 = 'TKE1K5'
		param_43 = 'TKE2K'
		param_44 = 'TKE2K5'
		param_45 = 'TKE3K'
		#param_46 = 'NEBCON'
		param_var = [param_1, param_2, param_3, param_4, param_5, param_6, param_7,  param_8,  param_9,  param_10,  param_11,  param_12,  param_13,  param_14,  param_15,  param_16,  param_17,  param_18,  param_19,  param_20,  param_21,  param_22,  param_23,  param_24,  param_25,  param_26,  param_27,  param_28,  param_29,  param_30,  param_31,  param_32,  param_33,  param_34,  param_35, param_36,  param_37,  param_38,  param_39,  param_40,  param_41,  param_42,  param_43,  param_44,  param_45]

		fields_param = read_dble_45_param(date_v,run_UT,suite,T_mem,forecast_time_Mx,param_var)
		#f_srf, f_ssrd  = read_dble_2_param(date_v,run_UT,suite,T_mem,forecast_time_Mx,param_1, param_2)
		dates_l = extract_time_V(fields_param[0])
		lon_a, lat_a = Array_lon_lat(fields_param[0][0], fields_param[0][0].data.shape[0], fields_param[0][0].data.shape[1])
		
		mem_id = np.where(dates_l[0]  ==dates_l)[0]
		mem_N = len(mem_id)
		list_N = len(fields_param[0])
		tme_N = (int(list_N/mem_N))
		ssr_id = 1
		ssrd_arr = np.reshape(fields_param[ssr_id], (tme_N, mem_N))
		date_arr_A = np.reshape(dates_l, (tme_N, mem_N))
		srf_cld = np.reshape(fields_param[ssr_id+1], (tme_N, mem_N))
		
		print([ssrd_arr[0,0].data.min(), ssrd_arr[0,0].data.max()])
		##------------------------------------------------------------
		swd_h = [];
		srf_cld_atm = []
		if suite=='dble':
			for mi in range(mem_N):
				swd_mem = []; atm_mem = [];
				ssrd_zoom_0 = ssrd_arr[0,mi].extract_zoom(dict(lonmin=0.67,lonmax=4.02,latmin=47.75,latmax=49.95))
				ssrd_zoom_0.geometry
				for ti in range(0,tme_N):
					###--------Selection of First Variable----------
                                        ssrd_zoom = ssrd_arr[ti,mi].extract_zoom(dict(lonmin=0.67,lonmax=4.02,latmin=47.75,latmax=49.95))
                                        ssrd_zoom.geometry
                                        swd_zoom_wm2 =(ssrd_zoom.data-ssrd_zoom_0.data)/3600 ###---Wm-2
                                        ssrd_zoom_wm2F = cpy.deepcopy( ssrd_zoom )
                                        ssrd_zoom_wm2F.setdata( np.flip(swd_zoom_wm2, axis=0) ) # on écrase dans l'objet g, avec setdata, en ayant inversé le champ selon l'axe des latitudes avec numpy.flip
                                        ssrd_zoom_0 = ssrd_zoom
                                        ###------------------
                                        ###-------Selection of Second Variable-----------
                                        srf_cld_zm = srf_cld[ti,mi].extract_zoom(dict(lonmin=0.67,lonmax=4.02,latmin=47.75,latmax=49.95))
                                        srf_cld_zm.geometry
                                        srf_cld_zm.setdata( np.flip(srf_cld_zm.data, axis=0) ) # on écrase dans l'objet g, avec setdata, en ayant inversé le champ selon l'axe des latitudes avec numpy.flip
                                        
                                        ###-------Write a grib file for the selected variables-----------
                                        TmpFle = grib_pfx+str(int(ti+1))+':00.grib'
                                        grib_dir_out = grid_dir+suite+"/"+dates_l[0].isoformat()[:-8]+run_UT+"/mb0"+str(int(mi))+"/forecast/"      
                                        path = Path(grib_dir_out)
                                        path.mkdir(parents=True, exist_ok=True)
                                        
                                        wr=epygram.formats.resource(filename = grib_dir_out+TmpFle, openmode='w', fmt='GRIB')
                                        wr.open()
                                        wr.writefield(ssrd_zoom_wm2F)	
                                        wr.writefield(srf_cld_zm)
                                        wr.close()
                                        toc_h = time.time();
                                        print('Writing re-grided Grib for 1 forcast term is Done in {:.4f} seconds'.format(toc_h-tic_h));
                                        del wr, grib_dir_out, TmpFle, ssrd_zoom_wm2F, srf_cld_zm, swd_zoom_wm2, ssrd_zoom;
					###-----------------
                                        
		elif suite=='oper':
			for mi in range(mem_N):
				swd_mem = []; atm_mem = [];
				ssrd_zoom_0 = ssrd_arr[0,mi].extract_zoom(dict(lonmin=0.67,lonmax=4.02,latmin=47.75,latmax=49.95))
				ssrd_zoom_0.geometry
				for ti in range(1,tme_N):
					ssrd_zoom = ssrd_arr[ti,mi].extract_zoom(dict(lonmin=0.67,lonmax=4.02,latmin=47.75,latmax=49.95))
					ssrd_zoom.geometry
					swd_mem.append( [(ssrd_zoom.data-ssrd_zoom_0.data)/3600] ) ###---Wm-2
					srf_cld_zm = srf_cld[ti,mi].extract_zoom(dict(lonmin=0.67,lonmax=4.02,latmin=47.75,latmax=49.95))
					atm_mem.append([srf_cld_zm.data])
					ssrd_zoom_0 = ssrd_zoom	
					del srf_cld_zm, ssrd_zoom
				if mi==0:
					swd_h = swd_mem
					srf_cld_atm = atm_mem
				else:
					swd_h = np.hstack((swd_h, swd_mem))
					srf_cld_atm = np.hstack((srf_cld_atm, atm_mem))
		else:	
			for mi in range(mem_N):
				swd_mem = []; atm_mem = [];
				ssrd_zoom_0 = ssrd_arr[0,mi]
				ssrd_zoom_0.geometry
				for ti in range(1,tme_N):
					ssrd_zoom = ssrd_arr[ti,mi]
					ssrd_zoom.geometry
					swd_mem.append( [(ssrd_zoom.data-ssrd_zoom_0.data)/3600] ) ###---Wm-2
					srf_cld_zm = srf_cld[ti,mi]
					atm_mem.append([srf_cld_zm.data])
					ssrd_zoom_0 = ssrd_zoom	
				if mi==0:
					swd_h = swd_mem
					srf_cld_atm = atm_mem
				else:
					swd_h = np.hstack((swd_h, swd_mem))
					srf_cld_atm = np.hstack((srf_cld_atm, atm_mem))
			
		toc_h = time.time();
		print('Writing regrided gribs are done for all forcast terms in {:.4f} seconds'.format(toc_h-start));
			

	
