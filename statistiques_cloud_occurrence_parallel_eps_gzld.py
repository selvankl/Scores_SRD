#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 15:14:29 2021

@author: magnaldom

Algorithm that
- detects clear skies in obs and model
- calculates the bias and standard deviation for each case from ground measurements
- works using the hour-by-hour method
- detection with SAT data for observations, and cloud cover in AROME
- on zones and not point by point comparison

Modified on Thu March 13 11:05:00 2025 by Selvaraj

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
import netCDF4 as nc
from netCDF4 import Dataset
import time
import array as arr
from array import *
from sklearn.metrics import mean_squared_error
from itertools import repeat
import itertools

import sys; sys.path.insert(0,'../tools')
from TOA import TOA_moyen_heure
from plot_AROME import zone_M_N_lon_lat_opt
from zenith_angle import zenith_angle_ephem
from get_solar_constant import get_solar_constant
from plot_BDClim_v2 import plot_BDClim_station_h
from num_jour_between import num_jour_between
import sys; sys.path.insert(0,'../cloud_detection')
from detection_cc_AROME_SAT_opt import dcc_AROME_SAT_cumul_zone
from detection_cc_CEMS import dcc_CEMS_cumul_moypoints, fournearestpoints
import pandas as pd
import h5py
import sys; sys.path.insert(0,'../../coding/')
from read_dble_cy48t1 import read_dble, read_dble_2_param, AROME_SAT_cumul_zone, datehr_v, intV, AROME_SAT_instant_zone_nest, Array_lon_lat, extract_time_V
import glob as glb
import copy as cpy
from multiprocessing import Pool, set_start_method
from concurrent.futures import ProcessPoolExecutor as pool
#set_start_method("fork")
 

work_dir = "/home/gmap/mrmn/selvarajd/SAVE/python/SWD/scores/"
cmd_should = 'rm -rf '+work_dir+'shouldfly-*'

start = time.time()

### ============================================================================
### DONNEES ENTREES

date_temp1 = '20240913' # sys.argv[1]
date_temp2 = '20241009' # sys.argv[2]
y1 = int(date_temp1[:4])
m1 = int(date_temp1[4:6])
d1 = int(date_temp1[6:8])
y2 = int(date_temp2[:4])
m2 = int(date_temp2[4:6])
d2 = int(date_temp2[6:8])
date1 = datetime(y1,m1,d1,0,0,0)
date2 = datetime(y2,m2,d2,0,0,0)

PNT = "EPS"
K = 4 # int(sys.argv[3]) #Defaults value : 2
fn_lim = 0.02 # float( sys.argv[4]) #Defaults value : 0.02
###--- READING PEAROME at 1.3km of cy48t1----
run_UT = '0300P'
suite = "GZLD"
T_mem = 24 ###----ensemble members
forecast_time_Mx = 23 ###----Forecast terms

### ===========BDClim====================================================
BD_data_dir = "/scratch/work/selvarajd/BDClim/obs_BDClim/"
BD_dir = "/scratch/work/selvarajd/BDClim/BDClim_site/"
#BD_home_dir = "/home/gmap/mrmn/selvarajd/SAVE/BDClim_site/"
if suite=='dble':
	fle_name = 'BDClim_recup_station_zoom_'+suite+'_v2.h5'
elif suite=='GZLD':
	fle_name = 'BDClim_recup_station_use_750m_v2.h5'
elif suite=='GZQB':
	fle_name = 'BDClim_recup_station_use_750m_v2.h5'
else:
	print('SUITE is not defined to read the corresponding BDClim file')

BD_dat = pd.read_hdf(BD_dir+fle_name)
num_station = BD_dat['ID'].values
lon = BD_dat['lon'].values
lat = BD_dat['lat'].values
alti = BD_dat['Alt_m'].values
M = BD_dat['M_mod'].values
N = BD_dat['N_mod'].values
M_sat = BD_dat['M_sat'].values
N_sat = BD_dat['N_sat'].values
classe = BD_dat['Quality'].values
nb_station = len(num_station)

### ===========Satellite============================
Sat_dir = "/scratch/work/selvarajd/SPACEBORNE/cloud/"
Sat_dat_list = glb.glob(Sat_dir+'S_NWC_CMA*')

def csd_nebul_BDclim_arome_instant_zone(lon, lat, classe, num_station, alti, M, N, Z_cems, nebul_cems_h, run_date, min_tme, srf_m, K, fn_lim):
	cos_zen =  zenith_angle_ephem(lat, lon, run_date+timedelta(minutes = -30)) #Only when zenith angle is lower than a threshold
	if classe<4 and num_station!='6088001' and alti<1000 and cos_zen > 0.1 : #Selection of the stations
		nebul_cems_h = nebul_cems_h + Z_cems
		if min_tme=='00': #on identifie pour AROME une seule fois par heure
					###---csd_a, nebul = AROME_SAT_instant_zone(f_srf[run_date_ID], run_date, lon[i], lat[i], int(M[i]), int(N[i]), int(K), fn_lim)
					srf_cloud_mem = srf_m[0]
					#print(len(srf_m), len(srf_cloud_mem))
					csd_neb_arr = []
					#print(srf_cloud_mem.shape, K)
					with pool(max_workers=10) as P:
						csd_neb_nb = P.map(AROME_SAT_instant_zone_nest, srf_cloud_mem, repeat(M),repeat(N), repeat(K))
						
					for k_i in range(len(srf_cloud_mem) ):
						v_csd = next(csd_neb_nb) 	#Calculate GHI in AROME with the spatial tolerance method
						#print(v_csd)
						if k_i==0:
							csd_neb_arr = np.array( v_csd[0], v_csd[1] )
						else:
							csd_neb_arr = np.vstack((csd_neb_arr, np.array( v_csd[0], v_csd[1]  )))
						del v_csd
					#	csd_neb_nb = AROME_SAT_instant_zone_nest_vec(srf_cloud_mem, M,N, K)
					#csd_neb_arr[:,0] = np.array(csd_neb_nb[0])
					#csd_neb_arr[:,1] = np.array(csd_neb_nb[1])
					#print(np.array(csd_neb_nb[0]), np.array(csd_neb_nb[1]), csd_neb_arr)
					del csd_neb_nb
					nebul_csd = [nebul_cems_h, csd_neb_arr]
		else:
			nebul_csd = [nebul_cems_h]
		
					
	return nebul_csd

def scores_cases_BDclim_zone_M_N_lon_lat_opt(csd_tot, mem_N, nebul_cems_h, lon, lat, BD_data_dir, num_station, csd_arome_st, swd_m_l, lon_a, lat_a, M, N, K, h):
	toc_h = time.time();
	run_date = date_v+timedelta(hours = h) 
	if csd_tot == mem_N:
		if nebul_cems_h>0: #To take only valid station/cosSZA
			csd_cems_h=1
		else :
			csd_cems_h=0

		swd_m = swd_m_l[0]
		#print(len(swd_m), swd_m[0].shape)
        #--------------------------------------------------
		#---------Statistiques calculation----------------------------
		#------------------------------------------
		#print(lat, lon, run_date)
		cos_zen =  zenith_angle_ephem(lat, lon, run_date+timedelta(minutes = -30))
		S0 = get_solar_constant(date_v)
		S = 1365*S0
		TOA_estimation = S*cos_zen
		
		#Recover information each hour and each station
		ghi_b = plot_BDClim_station_h(BD_data_dir, num_station,date_v, h) #inf si pas present
		csd_a = csd_arome_st
		csd_b = csd_cems_h
		
		#Calculate GHI in AROME with the spatial tolerance method
		###---Z, lon_zone, lat_zone = zone_M_N_lon_lat_opt(swd_m[run_date_ID], lon_a, lat_a, M[i],N[i], date_v, h,K)
		tic = time.time()
		with pool(max_workers=10) as P:
			Z_zone = P.map(zone_M_N_lon_lat_opt, swd_m, repeat(lon_a), repeat(lat_a), repeat(M), repeat(N), repeat(date_v), repeat(h), repeat(K))
		toc = time.time()
		#print('Done in {:.4f} seconds'.format(toc-tic))
		
		v_csd = next(Z_zone) #Calculate GHI in AROME with the spatial tolerance method
		Z = v_csd[0]
		del v_csd
		for k_i in range(len(swd_m)-1):
			v_csd = next(Z_zone)
			#Calculate GHI in AROME with the spatial tolerance method
			Z = np.dstack((Z, v_csd[0]))
		#print(Z, v_csd)
	return [ghi_b, csd_b, csd_cems_h, Z, TOA_estimation]


def bias_rmse_score(ghi_b):
	distribution_1 = []
	####--------HIT-------####
	if csd_a == 1 and csd_b== 1 and ghi_b<20000 and ghi_b !=0: #If clouds present in the mod and obs
		score = 1
		biais_norm = flux_mod_norm - flux_obs_norm
		biais = ghi_zone - ghi_b
		MAPE = (ghi_zone - ghi_b)/ghi_b
		MAE = np.abs(ghi_zone - ghi_b)
		if ghi_zone - ghi_b > 0:
			biais_pos = ghi_zone - ghi_b
			score_pos = 1
		elif ghi_zone - ghi_b <0:
			biais_neg = ghi_zone - ghi_b
			score_neg = 1
			flu_moy_norm = flux_obs_norm
			flu_moy =  ghi_b
			ecart_type = (ghi_zone - ghi_b)**2
			array_cas1 = np.vstack((array_cas1, [ghi_zone, flux_mod_norm, ghi_b,flux_obs_norm ]))
			distribution_1.append(ghi_zone - ghi_b)
	return array_cas1

def detect_hourly(swd_l, h_term_l, dates_swd, date_v_l):
	swd_h = swd_l
	h_term = h_term_l
	dates_l = dates_swd
	date_v = date_v_l
	run_date = date_v+timedelta(hours = h_term) 
	run_date_ID = np.where(dates_l==run_date)[0][0]
	print(run_date_ID)
	return run_date_ID


## -------------------------------
### DEBUT DE LA BOUCLE TEMPORELLE/ BEGINNING OF TEMPORAL LOOP
num_jour = num_jour_between(date1,date2)
for d in range(0,num_jour):
        ##--- LECTURE JOUR ET FICHIER AROME/AROME FILES LECTURE-------
	date_v = date1 + timedelta(days = d)
	annee = date_v.year
	mois = date_v.month
	jour = date_v.day
	###--- READING PEAROME at 1.3km of cy48t1----

if __name__ == "__main__":
	for d in range(0,num_jour):
		##--- LECTURE JOUR ET FICHIER AROME/AROME FILES LECTURE-------
		date_v = date1 + timedelta(days = d)
		annee = date_v.year
		mois = date_v.month
		jour = date_v.day
		param_1 = 'srfNtot'	
		param_2 = 'ssrd' ###--param=--ssrd----SURFRAYT SOLA DE	SURFRAYT SOLA DE	W.m-2	Cum. Downward solarflux at surface
		###---f_srf = read_dble(date_v,run_UT,suite,T_mem,forecast_time_Mx,param)
		f_srf, f_ssrd  = read_dble_2_param(date_v,run_UT,suite,T_mem,forecast_time_Mx,param_1, param_2)
		dates_l = extract_time_V(f_srf)
		lon_a, lat_a = Array_lon_lat(f_srf[0], f_srf[0].data.shape[0], f_srf[0].data.shape[1])
		
		mem_id = np.where(dates_l[0]  ==dates_l)[0]
		mem_N = len(mem_id)
		list_N = len(f_ssrd)
		tme_N = (int(list_N/mem_N))
		ssrd_arr = np.reshape(f_ssrd, (tme_N, mem_N))
		date_arr_A = np.reshape(dates_l, (tme_N, mem_N))
		srf_cld = np.reshape(f_srf, (tme_N, mem_N))
		
		print([ssrd_arr[0,0].data.min(), ssrd_arr[0,0].data.max()])
		##------------------------------------------------------------
		swd_h = [];
		srf_cld_atm = []
		if suite=='dble':
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
				for ti in range(1,tme_N):
					ssrd_zoom = ssrd_arr[ti,mi]
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
			
		print([swd_h.min(), swd_h.max()], [srf_cld_atm.min(), srf_cld_atm.max()])
		date_arr = date_arr_A[1:]
		#####----------------- Loop on hours
		h_min = 6
		h_term = np.arange(6,forecast_time_Mx,1)
		h_max_t = h_term.max()
		
		for h in range(h_min, h_max_t):

			tic_h = time.time();
			run_date = date_v+timedelta(hours = h) 
			run_date_ID = np.where(date_arr==run_date)
			
			### ===========DEFINITIONS MATRICES DE SCORES=================
			score = np.zeros((nb_station,4))
			score_pos = np.zeros((nb_station,4))
			score_neg = np.zeros((nb_station,4))
			ecart_type = np.zeros((nb_station,4))
			biais = np.zeros((nb_station,4))
			biais_pos = np.zeros((nb_station,4))
			biais_neg = np.zeros((nb_station,4))
			biais_norm = np.zeros((nb_station,4))
			flu_moy = np.zeros((nb_station,4))
			flu_moy_norm = np.zeros((nb_station,4))
			MAE = np.zeros((nb_station,4))
			MAPE = np.zeros((nb_station,4))
			flu_moy_mod = np.zeros((nb_station,4))
			flu_moy_mod_norm = np.zeros((nb_station,4))
			std_deviation = np.zeros((nb_station,4))
			std_deviation_norm = np.zeros((nb_station,4))
			nb_heure = np.zeros((nb_station,4))
			array_cas1 = np.empty((0,4), float) #arr.array('f')
			array_cas2 = np.empty((0,4), float)
			array_cas3 = np.empty((0,4), float)
			array_cas4 = np.empty((0,4), float)
			expe = "_"
			
			distribution_1 = []
			distribution_2 = []
			distribution_3 = []
			distribution_4 = []
			
			date_hr_tme = date_arr[run_date_ID] ####-----List of dates-times from list array for cross check---
			swd_m = swd_h[run_date_ID[0], run_date_ID[1],:,:]
			srf_m = srf_cld_atm[run_date_ID[0], run_date_ID[1],:,:]
			
			nebul_cems_h = np.zeros((nb_station)) #Save data for each station each hour (initialisation chaque heure)
			csd_arome_h = np.zeros((nb_station, mem_N))*np.nan
			csd_cems_h = [np.inf for st in range(nb_station )] #Nan by default
			for m in ['00', '15', '30', '45']: #Satellite product frequency
				#------------------------------
				#------ LECTURE CEMS-----------
				#------------------------------
				if m != '00':
					h_cems = h-1
				else :
					h_cems = h
				Sat_slt_dt = date_v.strftime("%Y%m%dT")+'%s%s'%(str(h_cems).zfill(2),m) 
				if (any(Sat_slt_dt in wrd for wrd in Sat_dat_list))==True:
					for wrd in Sat_dat_list:
						if wrd[-19:-6]==Sat_slt_dt:
							fichier=wrd; #print(wrd)
				else:
					print("Fichier non trouvé : S_NWC_%s_MSG4_globeM-VISIR_%s%s%sT%s%s00Z.nc" %("CMA", str(date_v.year), str(date_v.month).zfill(2), str(date_v.day).zfill(2),str(h).zfill(2), m))
					#file = Sat_str_prefix3+date_v.strftime("%Y%m%dT")+'%s%s'%(str(h_cems).zfill(2),m)+'00Z.nc'
					#fichier = Sat_dir+file
			
				CEMS = Dataset(fichier, "r", format="NETCDF4")
				Z_cems =  [CEMS.variables['cma'][M_sat[ij], N_sat[ij]] for ij in range(nb_station)]
				with pool(max_workers=10) as P:
					csd_neb_l = P.map(csd_nebul_BDclim_arome_instant_zone, lon, lat, classe, num_station, alti, M, N, Z_cems, nebul_cems_h, repeat(run_date), repeat(m), repeat([srf_m]), repeat(int(K)), repeat(fn_lim))
					
				for k_i in range(nb_station):
					try:
						cems_csd_neb_l = next(csd_neb_l)
						nebul_cems_h[k_i] = cems_csd_neb_l[0]
					except:
						nebul_cems_h[k_i] = nebul_cems_h[k_i]

					if m == '00':
						try:
							csd_arome_h[k_i,:] = cems_csd_neb_l[1][:,0];	#nebul = cems_csd_neb_l[1][:,1]
						except:
							a=1
			#--------------------------------------------
			#- DEBUT POSTPROCESSING (chaque heure)-------			
			#--------------------------------------------			
			csd_arome_st = np.nanmean(csd_arome_h, axis=1)
			cld_id_mem = mem_N ###----- ensemble : mem_N / Deterministic : 0
			csd_tot = mem_N
			with pool(max_workers=10) as P:
				scores_l = P.map(scores_cases_BDclim_zone_M_N_lon_lat_opt, repeat(csd_tot), repeat(mem_N), nebul_cems_h, lon, lat, repeat(BD_data_dir), num_station, csd_arome_st, repeat([swd_m]), repeat(lon_a), repeat(lat_a), M, N, repeat(K), repeat(h))
			
			good_stat_ID = np.where(~np.isnan(csd_arome_st))[0]
			if good_stat_ID.shape[0] != 0:

				for i in good_stat_ID:
					toc_h = time.time();
					cems_csd_scores_l = next(scores_l) ###--[ghi_b, csd_b, csd_cems_h, Z, TOA_estimation]---SWD for selected stations from Pool 
					ghi_b = cems_csd_scores_l[0] 
					csd_b = cems_csd_scores_l[1] 
					csd_cems_h = cems_csd_scores_l[2]
					TOA_estimation = cems_csd_scores_l[4]
					csd_a = int(csd_arome_st[i])
					flux_obs_norm = ghi_b/TOA_estimation
					
					Z = cems_csd_scores_l[3][:,:,:cld_id_mem]  ###-----Deterministic
					Z_tot = (np.median(Z, axis=2)).flatten()
					
					ecart = [np.abs(Z_tot[bou]-(ghi_b) )for bou in range(len(Z_tot))]
					val_min = stats.scoreatpercentile(ecart, 10, interpolation_method="lower")
					position = ecart.index(val_min)
					ghi_zone = Z_tot[position]
					flux_mod_norm = ghi_zone/TOA_estimation
					
					####--------HIT-------####
					if csd_a == 1 and csd_b== 1 and ghi_b<20000 and ghi_b !=0: #If clouds present in the mod and obs
						score[i][0] = score[i][0] + 1
						biais_norm[i][0] = biais_norm[i][0] + flux_mod_norm - flux_obs_norm
						biais[i][0] = biais[i][0] + ghi_zone - ghi_b
						MAPE[i][0] = MAPE[i][0] + (ghi_zone - ghi_b)/ghi_b
						MAE[i][0] = MAE[i][0] + np.abs(ghi_zone - ghi_b)
						if ghi_zone - ghi_b > 0:
							biais_pos[i][0] = biais_pos[i][0] + ghi_zone - ghi_b
							score_pos[i][0] = score_pos[i][0] + 1
						elif ghi_zone - ghi_b <0:
							biais_neg[i][0] = biais_neg[i][0] + ghi_zone - ghi_b
							score_neg[i][0] = score_neg[i][0] + 1
						flu_moy_norm[i][0] = flu_moy_norm[i][0] + flux_obs_norm
						flu_moy[i][0] = flu_moy[i][0] + ghi_b
						ecart_type[i][0] = ecart_type[i][0] + (ghi_zone - ghi_b)**2
						array_cas1 = np.vstack((array_cas1, [ghi_zone, flux_mod_norm, ghi_b,flux_obs_norm ]))
						distribution_1.append(ghi_zone - ghi_b)

        			####--------FALSE ALARM-------####
					if csd_a == 1 and csd_b == 0 and ghi_b<20000 and ghi_b !=0: #If clouds present in the mod but not in the obs
						score[i][1] = score[i][1] + 1
						biais_norm[i][1] = biais_norm[i][1] + flux_mod_norm - flux_obs_norm
						MAE[i][1] = MAE[i][1] + np.abs(ghi_zone - ghi_b)
						MAPE[i][1] = MAPE[i][1] + (ghi_zone - ghi_b)/ghi_b
						biais[i][1] = biais[i][1] + ghi_zone - ghi_b
						if ghi_zone - ghi_b > 0:
							biais_pos[i][1] = biais_pos[i][1] + ghi_zone - ghi_b
							score_pos[i][1] = score_pos[i][1] + 1
						elif ghi_zone - ghi_b <0:
							biais_neg[i][1] = biais_neg[i][1] + ghi_zone - ghi_b
							score_neg[i][1] = score_neg[i][1] + 1
						flu_moy_norm[i][1] = flu_moy_norm[i][1] + flux_obs_norm
						flu_moy[i][1] = flu_moy[i][1] + ghi_b
						ecart_type[i][1] = ecart_type[i][1] + (ghi_zone - ghi_b)**2
						array_cas2 = np.vstack((array_cas2, [ghi_zone, flux_mod_norm, ghi_b,flux_obs_norm ]))
						distribution_2.append(ghi_zone - ghi_b)
					
					####--------MISSED-------####
					if csd_a == 0 and csd_b == 1 and ghi_b<20000 and ghi_b !=0: #If clouds present in the obs but not in the mod
						score[i][2] = score[i][2] + 1
						biais_norm[i][2] = biais_norm[i][2] + flux_mod_norm - flux_obs_norm
						MAE[i][2] = MAE[i][2] + np.abs(ghi_zone - ghi_b)
						MAPE[i][2] = MAPE[i][2] + (ghi_zone - ghi_b)/ghi_b
						biais[i][2] = biais[i][2] + ghi_zone - ghi_b
						if ghi_zone - ghi_b > 0:
							biais_pos[i][2] = biais_pos[i][2] + ghi_zone - ghi_b
							score_pos[i][2] = score_pos[i][2] + 1
						elif ghi_zone - ghi_b <0:
							biais_neg[i][2] = biais_neg[i][2] + ghi_zone - ghi_b
							score_neg[i][2] = score_neg[i][2] + 1
						flu_moy_norm[i][2] = flu_moy_norm[i][2] + flux_obs_norm
						flu_moy[i][2] = flu_moy[i][2] + ghi_b
						ecart_type[i][2] = ecart_type[i][2] + (ghi_zone - ghi_b)**2
						array_cas3 = np.vstack((array_cas3, [ghi_zone, flux_mod_norm, ghi_b,flux_obs_norm ]))
						distribution_3.append(ghi_zone - ghi_b)
					
					####--------CORRECT NEGATIVE-------####
					if csd_a == 0 and csd_b == 0 and ghi_b<20000 and ghi_b !=0: #If clouds not present in the mod and obs
						score[i][3] = score[i][3] + 1
						biais_norm[i][3] = biais_norm[i][3] + flux_mod_norm - flux_obs_norm
						MAE[i][3] = MAE[i][3] + np.abs(ghi_zone - ghi_b)
						MAPE[i][3] = MAPE[i][3] + (ghi_zone - ghi_b)/ghi_b
						biais[i][3] = biais[i][3] + ghi_zone - ghi_b
						if ghi_zone - ghi_b > 0:
							biais_pos[i][3] = biais_pos[i][3] + ghi_zone - ghi_b
							score_pos[i][3] = score_pos[i][3] + 1
						elif ghi_zone - ghi_b <0:
							biais_neg[i][3] = biais_neg[i][3] + ghi_zone - ghi_b
							score_neg[i][3] = score_neg[i][3] + 1
						flu_moy_norm[i][3] = flu_moy_norm[i][3] + flux_obs_norm
						flu_moy[i][3] = flu_moy[i][3] + ghi_b
						ecart_type[i][3] = ecart_type[i][3] + (ghi_zone - ghi_b)**2
						array_cas4 = np.vstack((array_cas4, [ghi_zone, flux_mod_norm, ghi_b,flux_obs_norm ]))
						distribution_4.append(ghi_zone - ghi_b)

			### ============================================================================
			### SCORES CALCULATION


			tc1 = 0
			tc2 = 0
			tc3 = 0
			tc4 = 0

			b1 = 0
			b2 = 0
			b3 = 0
			b4 = 0
			b1_pos = 0
			b2_pos = 0
			b3_pos = 0
			b4_pos = 0
			b1_neg = 0
			b2_neg = 0
			b3_neg = 0
			b4_neg = 0
			b1_norm = 0
			b2_norm = 0
			b3_norm = 0
			b4_norm = 0

			bt = 0
			fm1 = 0
			fm2 = 0
			fm3 = 0
			fm4 = 0

			ft = 0
			MAEt = 0
			MAPEt = 0
			ft_norm = 0
			ett = 0

			fm1_norm = 0
			fm2_norm = 0
			fm3_norm = 0
			fm4_norm= 0


			et1 = 0
			et2 = 0
			et3 = 0
			et4 = 0



			for i in range(nb_station):
				tc1 = tc1 + score[i][0]
				tc2 = tc2 + score[i][1]
				tc3 = tc3 + score[i][2]
				tc4 = tc4 + score[i][3]
				
				b1 = b1 + biais[i][0]
				b2 = b2 + biais[i][1]
				b3 = b3 + biais[i][2]
				b4 = b4 + biais[i][3]
				b1_pos = b1_pos + biais_pos[i][0]
				b2_pos = b2_pos + biais_pos[i][1]
				b3_pos = b3_pos + biais_pos[i][2]
				b4_pos = b4_pos + biais_pos[i][3]
				b1_neg = b1_neg + biais_neg[i][0]
				b2_neg = b2_neg + biais_neg[i][1]
				b3_neg = b3_neg + biais_neg[i][2]
				b4_neg = b4_neg + biais_neg[i][3]
				b1_norm = b1_norm + biais_norm[i][0]
				b2_norm = b2_norm + biais_norm[i][1]
				b3_norm = b3_norm + biais_norm[i][2]
				b4_norm = b4_norm + biais_norm[i][3]
				
				bt = bt + biais[i][0]+ biais[i][1] + biais[i][2] + biais[i][3]
				MAEt = MAEt + MAE[i][0]+ MAE[i][1] + MAE[i][2] + MAE[i][3]
				MAPEt = MAPEt + MAPE[i][0]+ MAPE[i][1] + MAPE[i][2] + MAPE[i][3]
				ft = ft + flu_moy[i][0]+ flu_moy[i][1] + flu_moy[i][2] + flu_moy[i][3]
				ett = ett + ecart_type[i][0]+ ecart_type[i][1] + ecart_type[i][2] + ecart_type[i][3]
				ft_norm = ft_norm + flu_moy_norm[i][0]+ flu_moy_norm[i][1] + flu_moy_norm[i][2] + flu_moy_norm[i][3]
				fm1 = fm1 + flu_moy[i][0]
				fm2 = fm2 + flu_moy[i][1]
				fm3 = fm3 + flu_moy[i][2]
				fm4 = fm4 + flu_moy[i][3]
				
				fm1_norm = fm1_norm + flu_moy_norm[i][0]
				fm2_norm = fm2_norm + flu_moy_norm[i][1]
				fm3_norm = fm3_norm + flu_moy_norm[i][2]
				fm4_norm = fm4_norm + flu_moy_norm[i][3]
				
				
				et1 = et1 + ecart_type[i][0]
				et2 = et2 + ecart_type[i][1]
				et3 = et3 + ecart_type[i][2]
				et4 = et4 + ecart_type[i][3]

			Nh = tc1 + tc2 + tc3 + tc4
			freq = [tc1/Nh, tc2/Nh, tc3/Nh, tc4/Nh]
			
			biais = [b1/tc1, b2/tc2, b3/tc3, b4/tc4]
			biais_pos = [b1_pos/tc1, b2_pos/tc2, b3_pos/tc3, b4_pos/tc4]
			biais_neg = [b1_neg/tc1, b2_neg/tc2, b3_neg/tc3, b4_neg/tc4]
			biais_norm = [b1_norm/tc1, b2_norm/tc2, b3_norm/tc3, b4_norm/tc4]
			
			cont1 = b1/Nh
			cont2 = b2/Nh
			cont3 = b3/Nh
			cont4 = b4/Nh
			
			cont = [cont1, cont2, cont3, cont4]
			ecart_type = [np.sqrt(et1/tc1), np.sqrt(et2/tc2), np.sqrt(et3/tc3), np.sqrt(et4/tc4)]
			biais_rel = [(b1/tc1)/(fm1/tc1),(b2/tc2)/(fm2/tc2),(b3/tc3)/(fm3/tc3),(b4/tc4)/(fm4/tc4)]
			biais_rel_norm = [(b1_norm/tc1)/(fm1_norm/tc1), (b2_norm/tc2)/(fm2_norm/tc2), (b3_norm/tc3)/(fm3_norm/tc3), (b4_norm/tc4)/(fm4_norm/tc4)]
			ecart_type_rel = [np.sqrt(et1/tc1)/(fm1/tc1), np.sqrt(et2/tc2)/(fm2/tc2),np.sqrt(et3/tc3)/(fm3/tc3), np.sqrt(et4/tc4)/(fm4/tc4)]
			
			flux_moyen = [fm1/tc1, fm2/tc2, fm3/tc3, fm4/tc4]
			flux_moyen_norm = [fm1_norm/tc1, fm2_norm/tc2, fm3_norm/tc3, fm4_norm/tc4]
			
			standard_deviation_1 = np.std(array_cas1, 0)
			standard_deviation_2 = np.std(array_cas2, 0)
			standard_deviation_3 = np.std(array_cas3, 0)
			standard_deviation_4 = np.std(array_cas4, 0)
			
			mat1 = np.corrcoef(array_cas1[:,0],array_cas1[:,2], rowvar = False)
			mat2 = np.corrcoef(array_cas2[:,0],array_cas2[:,2], rowvar = False)
			mat3 = np.corrcoef(array_cas3[:,0],array_cas3[:,2], rowvar = False)
			mat4 = np.corrcoef(array_cas4[:,0],array_cas4[:,2], rowvar = False)
			
			corrcoeff = [mat1[0][1],mat2[0][1],mat3[0][1],mat4[0][1]]
			
			#RMSE_norm = [np.sqrt(mean_squared_error(array_cas1[:,1],array_cas1[:,3])), np.sqrt(mean_squared_error(array_cas2[:,1],array_cas2[:,3])),np.sqrt(mean_squared_error(array_cas3[:,1],array_cas3[:,3])),np.sqrt(mean_squared_error(array_cas4[:,1],array_cas4[:,3]))]
			
			all_obs =  np.empty((0,1), float)
			all_obs = np.append(all_obs, array_cas1[:,2])
			all_obs = np.append(all_obs, array_cas2[:,2])
			all_obs = np.append(all_obs, array_cas3[:,2])
			all_obs = np.append(all_obs, array_cas4[:,2])
			
			STD_all_obs = np.std(all_obs, 0)
			
			all_obs_norm =  np.empty((0,1), float)
			all_obs_norm = np.append(all_obs_norm, array_cas1[:,3])

			all_obs_norm = np.append(all_obs_norm, array_cas2[:,3])
			all_obs_norm = np.append(all_obs_norm, array_cas3[:,3])
			all_obs_norm = np.append(all_obs_norm, array_cas4[:,3])
			STD_all_obs_norm = np.std(all_obs_norm, 0)
			
			### ============================================================================
			## ### WRITING IN A .NC
			fle_name_nc = suite+"/"+run_UT+"/RESULT/"+PNT+"/"+run_date.strftime('%Y%m%d%H')+'_mem'+str(T_mem)+'_memU'+str(cld_id_mem)+"_K"+str(K)+"_"+str(fn_lim)+".nc"
			file_nc = nc.Dataset(BD_dir[:-12]+fle_name_nc,"w",format="NETCDF4")
			file_nc.createDimension('const', 1)
			file_nc.createDimension('cas', 4)
			file_nc.createDimension('aux', 1)
			K_nc   = file_nc.createVariable('K', 'f4',dimensions=('aux'))
			fn_lim_nc   = file_nc.createVariable('fn_lim', 'f4', dimensions=('aux'))
			Nh_nc = file_nc.createVariable('Nh', 'f4', dimensions=('aux'))
			biais_total = file_nc.createVariable('biais_total', 'f4', dimensions=('const'))
			flux_total = file_nc.createVariable('flux_total', 'f4', dimensions=('const'))
			MAE_total = file_nc.createVariable('MAE_total', 'f4', dimensions=('const'))
			MAPE_total = file_nc.createVariable('MAPE_total', 'f4', dimensions=('const'))
			RMSE_total = file_nc.createVariable('RMSE_total', 'f4', dimensions=('const'))
			flux_norm_total = file_nc.createVariable('flux_total_norm', 'f4', dimensions=('const'))
			STD = file_nc.createVariable('STD_all_obs', 'f4', dimensions=('const'))
			STD[:] = STD_all_obs
			K_nc[:] = K
			fn_lim_nc[:] = fn_lim
			Nh_nc[:] = Nh
			biais_total[:] = bt/Nh
			MAE_total[:] = MAEt/Nh
			MAPE_total[:] = MAPEt/Nh
			flux_total[:] = ft/Nh
			flux_norm_total[:] = ft_norm/Nh
			RMSE_total[:] = np.sqrt(ett/Nh)

			frequence_nc = file_nc.createVariable('frequence', 'f4', dimensions=('cas'), zlib=True, complevel=1)
			frequence_nc[:] = freq
			biais_nc = file_nc.createVariable('biais', 'f4', dimensions=('cas'), zlib=True, complevel=1)
			biais_nc[:] = biais
			biais_pos_nc = file_nc.createVariable('biais_positif', 'f4', dimensions=('cas'), zlib=True, complevel=1)
			biais_pos_nc[:] = biais_pos
			biais_neg_nc = file_nc.createVariable('biais_negatif', 'f4', dimensions=('cas'), zlib=True, complevel=1)
			biais_neg_nc[:] = biais_neg
			biais_norm_nc = file_nc.createVariable('biais_norm', 'f4', dimensions=('cas'), zlib=True, complevel=1)
			biais_norm_nc[:] = biais_norm
			cont_nc = file_nc.createVariable('contribution_au_biais', 'f4', dimensions=('cas'), zlib=True, complevel=1)
			cont_nc[:] = cont
			ecart_type_nc = file_nc.createVariable('ecart_type', 'f4', dimensions=('cas'), zlib=True, complevel=1)
			ecart_type_nc[:] = ecart_type
			biais_rel_nc = file_nc.createVariable('biais_relatif', 'f4', dimensions=('cas'), zlib=True, complevel=1)
			biais_rel_nc[:] = biais_rel
			biais_rel_norm_nc = file_nc.createVariable('biais_relatif_norm', 'f4', dimensions=('cas'), zlib=True, complevel=1)
			biais_rel_norm_nc[:] = biais_rel_norm
			ecart_type_rel_nc = file_nc.createVariable('ecart_type_relatif', 'f4', dimensions=('cas'), zlib=True, complevel=1)
			
			ecart_type_rel_nc[:] = ecart_type_rel
			
			flux_moyen_nc = file_nc.createVariable('flux_moyen', 'f4', dimensions=('cas'), zlib=True, complevel=1)
			flux_moyen_nc[:] = flux_moyen
			
			standard_deviation_mod_nc = file_nc.createVariable('standard_deviation_mod', 'f4', dimensions=('cas'), zlib=True, complevel=1)
			standard_deviation_mod_nc[:] = [standard_deviation_1[0],standard_deviation_2[0],standard_deviation_3[0],standard_deviation_4[0]]
			
			standard_deviation_obs_nc = file_nc.createVariable('standard_deviation_obs', 'f4', dimensions=('cas'), zlib=True, complevel=1)
			standard_deviation_obs_nc[:] = [standard_deviation_1[2],standard_deviation_2[2],standard_deviation_3[2],standard_deviation_4[0]]
			
			corrcoeff_nc = file_nc.createVariable('corrcoeff', 'f4', dimensions=('cas'), zlib=True, complevel=1)
			corrcoeff_nc[:] = corrcoeff

			file_nc.close()
			del swd_m, srf_m;
			print('Writing the file of '+fle_name_nc)
			toc_h = time.time();
			print('Scores computation for 1 forcast term is Done in {:.4f} seconds'.format(toc_h-tic_h));
			
			

	
