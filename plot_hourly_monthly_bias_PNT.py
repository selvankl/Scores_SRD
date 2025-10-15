#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 15:14:29 2021

@author: Selvaraj

Algorithm that
- Reads the scores from *.nc file for each hour of the day from PEAROME-1.3km and 750m
- Recalculate the scores for each hour from the entire datasets of that hour
- Plot them

"""

### ============================================================================
### LIBRARY AND TOOLS IMPORTATION

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import numpy as np
from datetime import datetime as dt, timedelta
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as md
from matplotlib import cm
import netCDF4 as nc
from netCDF4 import Dataset
import time
from sklearn.metrics import mean_squared_error

from itertools import repeat

import pandas as pd
import glob as glb
import copy as cpy
from multiprocessing import set_start_method
from concurrent.futures import ProcessPoolExecutor as pool
#set_start_method("fork")
from dask.distributed import Client
#client = Client(processes=False) 


def f_1(x1,x2,x3,x4):
     return dt(int(x1),int(x2),int(x3), int(x4))
datehr_v = np.vectorize(f_1)

start = time.time()

PNT = 'EPS'
run_UT = '0300P'
fle_sfx = '.nc'
work_dir = "/home/gmap/mrmn/selvarajd/SAVE/python/SWD/scores/"
fn_lim = 0.02 # float( sys.argv[4]) #Defaults value : 0.02
if PNT=='EPS':
	T_mem = 24
	bias_lim=[-10, 50]
elif PNT=='DET':
	T_mem = 1
	bias_lim=[-10, 50]
else:
	print('PNT or type of the model is not defined and it is a fatal error')

h_min = 6 ###----Starting time of the day to get the scores by considering the solar zenith angle
tme_end = 21 ###---Limit the time to day by considering the solar zenith angle

### ========--- READING PEAROME at 1.3 km of cy48t1----===============
suite = "dble"
K = 2 # int(sys.argv[3]) #Defaults value : 2
nc_dir = "/scratch/work/selvarajd/BDClim/"+suite+"/"+run_UT+"/RESULT/"+PNT+"/"
str_l_path = len(nc_dir)
fle_str_sfx = str(T_mem)+'_memU*'+"_K"+str(K)+"_"+str(fn_lim)+fle_sfx

### ========--- READING PEAROME -750m at 0.75 km of cy48t1----===============
suite = "H1SN"
K = 4 # int(sys.argv[3]) #Defaults value : 2
nc_dir_750m = "/scratch/work/selvarajd/BDClim/"+suite+"/"+run_UT+"/RESULT/"+PNT+'/'
str_750m_path = len(nc_dir_750m)
fle_str_sfx_750m = str(T_mem)+'_memU*'+"_K"+str(K)+"_"+str(fn_lim)+fle_sfx

no_fore_terms = (tme_end+1 - h_min) ##---End time - Start time
no_days_term = int( np.ceil(len(glb.glob(nc_dir+'*'+fle_str_sfx))/no_fore_terms) )
### ========--- SAVE THE PLOTS of PEAROME of cy48t1 at----===============
png_dir = "/scratch/work/selvarajd/BDClim/plots_swd/"
### ============================================================================

def nc_read_score(fle_n, str_l_path):
	#print(fle_n)
	nc_dat = nc.Dataset(fle_n)
	Yrmon = fle_n[str_l_path:str_l_path+8]
	bias = nc_dat.variables['biais'][:]
	freq = nc_dat.variables['frequence'][:]
	rmse = nc_dat.variables['ecart_type'][:]
	return [bias, freq, rmse, Yrmon]

def stat_collect(h, nc_dir, fle_str_sfx, str_l_path):
	if len(str(h))==1:
		hr_st = "0"+str(h)
	elif len(str(h))==2:
		hr_st = str(h)
	
	str_hr = "*"+hr_st+"_mem"
	fle_name_nc = glb.glob(nc_dir+"*"+str_hr+fle_str_sfx)
	###---qx, bx, cx, dx = nc_read_score(fle_name_nc[0], str_l_path)
	with pool(max_workers=10) as P:
		nc_val = P.map(nc_read_score, fle_name_nc, repeat(str_l_path))
	
	for ki in range(len(fle_name_nc)):
		nc_data = next(nc_val) #Get the values from pool's map of multiprocessing	
		if ki==0:
			bias_h = nc_data[0]
			freq_h = nc_data[1]
			rmse_h = nc_data[2]
			date_int = np.array([int(nc_data[3][0:4]), int(nc_data[3][4:6]), int(nc_data[3][6:8]), int(hr_st)])
			date_h = date_int
		else:
			bias_h = np.vstack((bias_h, nc_data[0]))
			freq_h = np.vstack((freq_h, nc_data[1]))
			rmse_h = np.vstack((rmse_h, nc_data[2]))
			date_int = np.array([int(nc_data[3][0:4]), int(nc_data[3][4:6]), int(nc_data[3][6:8]), int(hr_st)])
			date_h = np.vstack((date_h, date_int))
			#print(date_int)
		del nc_data, date_int;

	dates_l = datehr_v(date_h[:,0], date_h[:,1], date_h[:,2], date_h[:,3])
	return [bias_h, freq_h, rmse_h, dates_l]

if __name__ == "__main__":
	h_term = np.arange(h_min,tme_end,1) ###---- Time vector
	tme_len = tme_end-h_min

	rr_days = no_days_term###----Number of days
	cc_cases = 4 ###---HIT --- FA --- MISSED --- Correct Negative--
	barWidth = 0.20

	case_cld = {}
	case_cld["HIT"] = {'name':'HIT at 1.3 km', 'color':'brown'}
	case_cld["FALSE ALARM"] = {'name':'FALSE ALARM at 1.3 km', 'color':'g'}
	case_cld["MISSED"] = {'name':'MISSED at 1.3 km', 'color':'b'}
	case_cld["CLEAR SKY"] = {'name':'CLEAR SKY at 1.3 km', 'color':'r'}
	
	case_cld["HIT m"] = {'name':'HIT at 750 m', 'color':'chocolate'}
	case_cld["FALSE ALARM m"] = {'name':'FALSE ALARM at 750 m', 'color':'darkviolet'}
	case_cld["MISSED m"] = {'name':'MISSED at 750 m', 'color':'darkslategrey'}
	case_cld["CLEAR SKY m"] = {'name':'CLEAR SKY at 750 m', 'color':'deeppink'}

	###==========================PEAROME - DOUBLE at 1.3 km=====================####
	bias_lh = np.nan * np.zeros((tme_len, rr_days, cc_cases))
	freq_lh = np.nan * np.zeros((tme_len, rr_days, cc_cases))
	rmse_lh = np.nan * np.zeros((tme_len, rr_days, cc_cases))
	
	bias_UT = np.nan * np.zeros((tme_len, cc_cases))
	freq_UT = np.nan * np.zeros((tme_len, cc_cases))
	rmse_UT = np.nan * np.zeros((tme_len, cc_cases))
	
	bias_dble_mon = np.nan * np.zeros((31*(tme_len+1), cc_cases, 12))
	freq_dble_mon = np.nan * np.zeros((31*(tme_len+1), cc_cases, 12))
	rmse_dble_mon = np.nan * np.zeros((31*(tme_len+1), cc_cases, 12))
	month_n = [1, 7, 8, 9, 10, 12]
	
	with pool(max_workers=10) as P:
		stat_val = P.map(stat_collect, h_term, repeat(nc_dir), repeat(fle_str_sfx), repeat(str_l_path))
		#dsl = stat_collect(h, nc_dir, fle_str_sfx, str_l_path)
	
	date_lh = {}
	UT_tme = []
	for k_i in range(tme_len):
		nc_data = next(stat_val) #Get the values from pool's map of multiprocessing
		rr = nc_data[0].shape[0]
		cc = nc_data[0].shape[1]
		
		bias_lh[k_i,:rr,:cc] = nc_data[0]
		freq_lh[k_i,:rr,:cc]  = nc_data[1]
		rmse_lh[k_i,:rr,:cc]  = nc_data[2]
		date_lh_ki = nc_data[3]
		date_lh[str(h_term[k_i])]  = date_lh_ki

		if k_i == 0:
			dble_MN_xax = np.unique( pd.PeriodIndex(pd.DataFrame(index=date_lh_ki).index, freq="M") )

		UT_tme.append(nc_data[3].min())
		del nc_data, rr, cc;
		
		bias_UT[k_i,:]	 = np.nanmean( bias_lh[k_i, :,:], axis=0 )
		freq_UT[k_i,:]	 = np.nansum( freq_lh[k_i, :,:], axis=0 )
		rmse_UT[k_i,:]	 = np.nanmean( rmse_lh[k_i, :,:], axis=0 )
		
        ###======Monthly========
		dmn = 0+(31*k_i)
		#k_i=0; print(0+(31*k_i), 31+(31*k_i))
		for mni in month_n:
			mon_ids = np.where(pd.DataFrame(index=date_lh_ki).index.month == mni)[0]
			dmx = len(mon_ids)+(31*k_i)
			bias_dble_mon[dmn:dmx,:,mni-1] = bias_lh[k_i, mon_ids,:]
			freq_dble_mon[dmn:dmx,:,mni-1] = freq_lh[k_i, mon_ids,:]
			rmse_dble_mon[dmn:dmx,:,mni-1] = rmse_lh[k_i, mon_ids,:]
			del mon_ids, dmx
		del date_lh_ki

	####=====Hourly - Standard Deviation Error=====
	freq_UT = freq_UT / len(date_lh[str(h_term[1])])
	SDE2 = pow(rmse_UT,2) - pow(bias_UT,2)
	SDE = np.sqrt(SDE2)
	del stat_val

	xax = bias_UT[:,0]
	xax_lbl = [UT_tme[r].strftime('%H') for r in range(len(UT_tme))]
	
	br1 = np.arange(len(xax)) 
	br2 = [x + barWidth for x in br1] 
	br3 = [x + barWidth for x in br2] 
	br4 = [x + barWidth for x in br3] 
	tlte = (date_lh['6'].min()).strftime('%d-%b-%Y') + ' - ' + (date_lh['6'].max()).strftime('%d-%b-%Y')+' ( '+str(len(date_lh['6']))+' days)'
	
    ###=====Monthly=====
	bias_dble_MN = np.nanmean( bias_dble_mon, axis=0 ).T
	freq_dble_MN = np.nanmean( freq_dble_mon, axis=0 ).T
	rmse_dble_MN = np.nanmean( rmse_dble_mon, axis=0 ).T
	SDE2_dble_MN = pow(rmse_dble_MN,2) - pow(bias_dble_MN,2)
	SDE_dble_MN = np.sqrt(SDE2_dble_MN)

	xax_dble_MN = list(np.arange(1,13,1))

	br1_dble_MN = np.arange(len(xax_dble_MN)) 
	br2_dble_MN = [x + barWidth for x in br1_dble_MN] 
	br3_dble_MN = [x + barWidth for x in br2_dble_MN] 
	br4_dble_MN = [x + barWidth for x in br3_dble_MN] 
	
    ###==========================PEAROME - 750m at 0.75 km=====================####
	bias750_lh = np.nan * np.zeros((tme_len, rr_days, cc_cases))
	freq750_lh = np.nan * np.zeros((tme_len, rr_days, cc_cases))
	rmse750_lh = np.nan * np.zeros((tme_len, rr_days, cc_cases))
	bias750_UT = np.nan * np.zeros((tme_len, cc_cases))
	freq750_UT = np.nan * np.zeros((tme_len, cc_cases))
	rmse750_UT = np.nan * np.zeros((tme_len, cc_cases))

	bias_750_mon = np.nan * np.zeros((31*(tme_len+1), cc_cases, 12))
	freq_750_mon = np.nan * np.zeros((31*(tme_len+1), cc_cases, 12))
	rmse_750_mon = np.nan * np.zeros((31*(tme_len+1), cc_cases, 12))
	
	with pool(max_workers=10) as P:
		stat_val = P.map(stat_collect, h_term, repeat(nc_dir_750m), repeat(fle_str_sfx_750m), repeat(str_750m_path))
		#dsl = stat_collect(h, nc_dir, fle_str_sfx, str_l_path)
	
	date750_lh = {}
	UT750_tme = []
	for k_i in range(tme_len):
		nc_data = next(stat_val) #Get the values from pool's map of multiprocessing
		rr = nc_data[0].shape[0]
		cc = nc_data[0].shape[1]

		bias750_lh[k_i,:rr,:cc] = nc_data[0]
		freq750_lh[k_i,:rr,:cc]  = nc_data[1]
		rmse750_lh[k_i,:rr,:cc]  = nc_data[2]
		date_lh_ki = nc_data[3]
		
		if k_i == 0:
			m750_MN_xax = np.unique( pd.PeriodIndex(pd.DataFrame(index=date_lh_ki).index, freq="M") )
			for ij in range(len(m750_MN_xax)):
				str_mon = str(m750_MN_xax[ij])
				ij_mn = int(str_mon[-2:])
				xax_dble_MN[ij_mn-1] = str_mon
			
		date750_lh[str(h_term[k_i])]  = date_lh_ki
		UT750_tme.append(nc_data[3].min())
		del nc_data, rr, cc;  
		
		bias750_UT[k_i,:]	 = np.nanmean( bias750_lh[k_i, :,:], axis=0 )
		freq750_UT[k_i,:]	 = np.nansum( freq750_lh[k_i, :,:], axis=0 )
		rmse750_UT[k_i,:]	 = np.nanmean( rmse750_lh[k_i, :,:], axis=0 )
			
        ###======Monthly========
		dmn = 0+(31*k_i)
		#k_i=0; print(0+(31*k_i), 31+(31*k_i))
		for mni in month_n:
			mon_ids = np.where(pd.DataFrame(index=date_lh_ki).index.month == mni)[0]
			dmx = len(mon_ids)+(31*k_i)
			bias_750_mon[dmn:dmx,:,mni-1] = bias750_lh[k_i, mon_ids,:]
			freq_750_mon[dmn:dmx,:,mni-1] = freq750_lh[k_i, mon_ids,:]
			rmse_750_mon[dmn:dmx,:,mni-1] = rmse750_lh[k_i, mon_ids,:]
			del mon_ids, dmx; 
		del date_lh_ki
		
	####=====Hourly - Standard Deviation Error=====
	freq750_UT = freq750_UT / len(date750_lh[str(h_term[1])])
	SDE2_750m = pow(rmse750_UT,2) - pow(bias750_UT,2)
	SDE_750m = np.sqrt(SDE2_750m)
	del stat_val

	xax_750m = bias750_UT[:,0]
	xax_750m_lbl = [UT750_tme[r].strftime('%H') for r in range(len(UT750_tme))]
	
	br1_750m = np.arange(len(xax_750m)) 
	br2_750m = [x + barWidth for x in br1_750m] 
	br3_750m = [x + barWidth for x in br2_750m] 
	br4_750m = [x + barWidth for x in br3_750m] 
	tlte_750m = (date750_lh['6'].min()).strftime('%d-%b-%Y') + ' - ' + (date750_lh['6'].max()).strftime('%d-%b-%Y')+' ( '+str(len(date750_lh['6']))+' days)'
	
	###=====Monthly=====
	bias_750_MN = np.nanmean( bias_750_mon, axis=0 ).T
	freq_750_MN = np.nanmean( freq_750_mon, axis=0 ).T
	rmse_750_MN = np.nanmean( rmse_750_mon, axis=0 ).T
	SDE2_750_MN = pow(rmse_750_MN,2) - pow(bias_750_MN,2)
	SDE_750_MN = np.sqrt(SDE2_750_MN)

	xax_750_MN = np.arange(1,13,1)

	br1_750_MN = np.arange(len(xax_750_MN)) 
	br2_750_MN = [x + barWidth for x in br1_750_MN] 
	br3_750_MN = [x + barWidth for x in br2_750_MN] 
	br4_750_MN = [x + barWidth for x in br3_750_MN] 
	
	###==========================Plotting=====================####
	####========================== Hourly Comparison=====================####
	add_wid = 0.1
	fig = plt.figure(1, figsize =(9, 8))
	fig.subplots_adjust(hspace=0.3)
	plt.subplot(2,1,1); #plt.grid(which='major', zorder=-1.0, alpha=0.3)
	plt.hlines(0, np.min(br1), np.max(br4), color='k')
	plt.bar(br1, bias_UT[:,0], fill=False, hatch='/////', color = case_cld['HIT']['color'], width = (barWidth+add_wid)*freq_UT[:,0], edgecolor =case_cld['HIT']['color'], label =case_cld['HIT']['name']) 
	plt.bar(br2, bias_UT[:,1], fill=False, hatch='/////', color = case_cld['FALSE ALARM']['color'], width = (barWidth+add_wid)*freq_UT[:,1], edgecolor = case_cld['FALSE ALARM']['color'], label =case_cld['FALSE ALARM']['name']) 
	plt.bar(br3, bias_UT[:,2], fill=False, hatch='/////', color = case_cld['MISSED']['color'], width = (barWidth+add_wid)*freq_UT[:,2], edgecolor = case_cld['MISSED']['color'], label =case_cld['MISSED']['name']) 
	plt.bar(br4, bias_UT[:,3], fill=False, hatch='/////', color = case_cld['CLEAR SKY']['color'], width = (barWidth+add_wid)*freq_UT[:,3], edgecolor = case_cld['CLEAR SKY']['color'], label =case_cld['CLEAR SKY']['name']) 
	
	plt.bar(br1_750m, bias750_UT[:,0], fill=False, hatch='\\\\\\\\', color = case_cld['HIT m']['color'], width = (barWidth+add_wid)*freq750_UT[:,0], edgecolor =case_cld['HIT m']['color'], label =case_cld['HIT m']['name']) 
	plt.bar(br2_750m, bias750_UT[:,1], fill=False, hatch='\\\\\\\\', color = case_cld['FALSE ALARM m']['color'], width = (barWidth+add_wid)*freq750_UT[:,1], edgecolor =case_cld['FALSE ALARM m']['color'], label =case_cld['FALSE ALARM m']['name']) 
	plt.bar(br3_750m, bias750_UT[:,2], fill=False, hatch='\\\\\\\\', color = case_cld['MISSED m']['color'], width = (barWidth+add_wid)*freq750_UT[:,2], edgecolor =case_cld['MISSED m']['color'], label =case_cld['MISSED m']['name']) 
	plt.bar(br4_750m, bias750_UT[:,3], fill=False, hatch='\\\\\\\\', color = case_cld['CLEAR SKY m']['color'], width = (barWidth+add_wid)*freq750_UT[:,3], edgecolor =case_cld['CLEAR SKY m']['color'], label =case_cld['CLEAR SKY m']['name']) 
	
	plt.title('(a) Bias & Relative Frequency of Incoming Short Wave Radiation ' +tlte, fontweight ='bold', fontsize = 15) 
	plt.xlabel('Time (UT)', fontweight ='bold', fontsize = 15) 
	plt.ylabel('Bias ($Wm^{-2}$)', fontweight ='bold', fontsize = 15); plt.ylim([bias_lim[0], bias_lim[1]])
	plt.xticks([r + barWidth for r in range(len(xax_lbl))], xax_lbl); 
	plt.legend(loc=2, fontsize = 'x-small', ncol=2)
	
	plt.subplot(2,1,2); 
	plt.hlines(0, np.min(br1), np.max(br4), color='k')
	plt.bar(br1, SDE[:,0], fill=False, hatch='/////', color = case_cld['HIT']['color'], width = barWidth, edgecolor =case_cld['HIT']['color'], label =case_cld['HIT']['name']) 
	plt.bar(br2, SDE[:,1], fill=False, hatch='/////', color = case_cld['FALSE ALARM']['color'], width = barWidth, edgecolor = case_cld['FALSE ALARM']['color'], label =case_cld['FALSE ALARM']['name']) 
	plt.bar(br3, SDE[:,2], fill=False, hatch='/////', color = case_cld['MISSED']['color'], width = barWidth, edgecolor = case_cld['MISSED']['color'], label =case_cld['MISSED']['name']) 
	plt.bar(br4, SDE[:,3], fill=False, hatch='/////', color = case_cld['CLEAR SKY']['color'], width = barWidth, edgecolor = case_cld['CLEAR SKY']['color'], label =case_cld['CLEAR SKY']['name']) 
	
	plt.bar(br1_750m, SDE_750m[:,0], fill=False, hatch='\\\\\\\\', color = case_cld['HIT m']['color'], width = barWidth, edgecolor =case_cld['HIT m']['color'], label =case_cld['HIT m']['name']) 
	plt.bar(br2_750m, SDE_750m[:,1], fill=False, hatch='\\\\\\\\', color = case_cld['FALSE ALARM m']['color'], width = barWidth, edgecolor =case_cld['FALSE ALARM m']['color'], label =case_cld['FALSE ALARM m']['name']) 
	plt.bar(br3_750m, SDE_750m[:,2], fill=False, hatch='\\\\\\\\', color = case_cld['MISSED m']['color'], width = barWidth, edgecolor =case_cld['MISSED m']['color'], label =case_cld['MISSED m']['name']) 
	plt.bar(br4_750m, SDE_750m[:,3], fill=False, hatch='\\\\\\\\', color = case_cld['CLEAR SKY m']['color'], width = barWidth, edgecolor =case_cld['CLEAR SKY m']['color'], label =case_cld['CLEAR SKY m']['name']) 
	
	plt.title('(b) Standard Deviation of errors', fontweight ='bold', fontsize = 15) 
	plt.xlabel('Time (UT)', fontweight ='bold', fontsize = 15) 
	plt.ylabel('SDE ($Wm^{-2}$)', fontweight ='bold', fontsize = 15);
	plt.xticks([r + barWidth for r in range(len(xax_lbl))], xax_lbl)
	#plt.legend()
	fle_out_png = suite+'_Comparison_Hourly_'+PNT+"_" + run_UT + "_SWD_Bias_" + (date_lh['6'].min()).strftime('%Y%m%dT%H')+ "_" + (date_lh['6'].max()).strftime('%Y%m%dT%H') + "_K"+str(K)+"_"+str(fn_lim)+ "h_hist.png";
	fig.savefig(png_dir+ fle_out_png, dpi = 300, bbox_inches='tight')
	plt.close(fig)  

	fig = plt.figure(2, figsize =(9, 8))
	fig.subplots_adjust(hspace=0.3)
	plt.subplot(2,1,1); 
	plt.hlines(0, np.min(br1), np.max(br4), color='k')
	plt.bar(br1, freq_UT[:,0], fill=False, hatch='/////', color = case_cld['HIT']['color'], width = barWidth, edgecolor =case_cld['HIT']['color'], label =case_cld['HIT']['name']) 
	plt.bar(br2, freq_UT[:,1], fill=False, hatch='/////', color = case_cld['FALSE ALARM']['color'], width = barWidth, edgecolor = case_cld['FALSE ALARM']['color'], label =case_cld['FALSE ALARM']['name']) 
	plt.bar(br3, freq_UT[:,2], fill=False, hatch='/////', color = case_cld['MISSED']['color'], width = barWidth, edgecolor = case_cld['MISSED']['color'], label =case_cld['MISSED']['name']) 
	plt.bar(br4, freq_UT[:,3], fill=False, hatch='/////', color = case_cld['CLEAR SKY']['color'], width = barWidth, edgecolor = case_cld['CLEAR SKY']['color'], label =case_cld['CLEAR SKY']['name']) 
	
	plt.bar(br1_750m, freq750_UT[:,0], fill=False, hatch='\\\\\\\\', color = case_cld['HIT m']['color'], width = barWidth, edgecolor =case_cld['HIT m']['color'], label =case_cld['HIT m']['name']) 
	plt.bar(br2_750m, freq750_UT[:,1], fill=False, hatch='\\\\\\\\', color = case_cld['FALSE ALARM m']['color'], width = barWidth, edgecolor =case_cld['FALSE ALARM m']['color'], label =case_cld['FALSE ALARM m']['name']) 
	plt.bar(br3_750m, freq750_UT[:,2], fill=False, hatch='\\\\\\\\', color = case_cld['MISSED m']['color'], width = barWidth, edgecolor =case_cld['MISSED m']['color'], label =case_cld['MISSED m']['name']) 
	plt.bar(br4_750m, freq750_UT[:,3], fill=False, hatch='\\\\\\\\', color = case_cld['CLEAR SKY m']['color'], width = barWidth, edgecolor =case_cld['CLEAR SKY m']['color'], label =case_cld['CLEAR SKY m']['name']) 
	
	plt.title('Normalized Frequency', fontweight ='bold', fontsize = 15) 
	plt.xlabel('Time (UT)', fontweight ='bold', fontsize = 15) 
	plt.ylabel('Frequency', fontweight ='bold', fontsize = 15);
	plt.xticks([r + barWidth for r in range(len(xax_lbl))], xax_lbl)
	plt.legend(loc=1, fontsize = 5, ncol=2)
	fle_out_png = suite+'_Comparison_Hourly'+PNT+"_" + run_UT + "_Frequency_" + (date_lh['6'].min()).strftime('%Y%m%dT%H')+ "_" + (date_lh['6'].max()).strftime('%Y%m%dT%H') + "_K"+str(K)+"_"+str(fn_lim)+ "h_hist.png";
	fig.savefig(png_dir+ fle_out_png, dpi = 300, bbox_inches='tight')
	plt.close(fig)  
	
	####========================== Monthly Comparison=====================####
	add_wid = 0.1
	fig = plt.figure(1, figsize =(9, 8))
	fig.subplots_adjust(hspace=0.3)
	plt.subplot(2,1,1); plt.grid(which='major', axis='y', zorder=-1.0, alpha=0.3)
	plt.hlines(0, np.min(br1), np.max(br4), color='k')
	plt.bar(br1_dble_MN, bias_dble_MN[:,0], fill=False, hatch='/////', color = case_cld['HIT']['color'], width = (barWidth+add_wid)*freq_dble_MN[:,0], edgecolor =case_cld['HIT']['color'], label =case_cld['HIT']['name']) 
	plt.bar(br2_dble_MN, bias_dble_MN[:,1], fill=False, hatch='/////', color = case_cld['FALSE ALARM']['color'], width = (barWidth+add_wid)*freq_dble_MN[:,1], edgecolor = case_cld['FALSE ALARM']['color'], label =case_cld['FALSE ALARM']['name']) 
	plt.bar(br3_dble_MN, bias_dble_MN[:,2], fill=False, hatch='/////', color = case_cld['MISSED']['color'], width = (barWidth+add_wid)*freq_dble_MN[:,2], edgecolor = case_cld['MISSED']['color'], label =case_cld['MISSED']['name']) 
	plt.bar(br4_dble_MN, bias_dble_MN[:,3], fill=False, hatch='/////', color = case_cld['CLEAR SKY']['color'], width = (barWidth+add_wid)*freq_dble_MN[:,3], edgecolor = case_cld['CLEAR SKY']['color'], label =case_cld['CLEAR SKY']['name']) 
	
	plt.bar(br1_750_MN, bias_750_MN[:,0], fill=False, hatch='\\\\\\\\', color = case_cld['HIT m']['color'], width = (barWidth+add_wid)*freq_750_MN[:,0], edgecolor =case_cld['HIT m']['color'], label =case_cld['HIT m']['name']) 
	plt.bar(br2_750_MN, bias_750_MN[:,1], fill=False, hatch='\\\\\\\\', color = case_cld['FALSE ALARM m']['color'], width = (barWidth+add_wid)*freq_750_MN[:,1], edgecolor =case_cld['FALSE ALARM m']['color'], label =case_cld['FALSE ALARM m']['name']) 
	plt.bar(br3_750_MN, bias_750_MN[:,2], fill=False, hatch='\\\\\\\\', color = case_cld['MISSED m']['color'], width = (barWidth+add_wid)*freq_750_MN[:,2], edgecolor =case_cld['MISSED m']['color'], label =case_cld['MISSED m']['name']) 
	plt.bar(br4_750_MN, bias_750_MN[:,3], fill=False, hatch='\\\\\\\\', color = case_cld['CLEAR SKY m']['color'], width = (barWidth+add_wid)*freq_750_MN[:,3], edgecolor =case_cld['CLEAR SKY m']['color'], label =case_cld['CLEAR SKY m']['name']) 
	
	plt.title('(a) Bias & Relative Frequency of Incoming Short Wave Radiation ' +tlte, fontweight ='bold', fontsize = 15) 
	#plt.xlabel('Month', fontweight ='bold', fontsize = 15) 
	plt.ylabel('Bias ($Wm^{-2}$)', fontweight ='bold', fontsize = 15); plt.ylim([bias_lim[0], bias_lim[1]])
	plt.xticks([r + barWidth for r in range(len(xax_dble_MN))], xax_dble_MN, rotation=15);  plt.xlim([-0.5, 12.75]);
	plt.legend(loc=2, fontsize = 'x-small', ncol=2)
	
	plt.subplot(2,1,2);  plt.grid(which='major', axis='y', zorder=-1.0, alpha=0.3)
	plt.hlines(0, np.min(br1), np.max(br4), color='k')
	plt.bar(br1_dble_MN, SDE_dble_MN[:,0], fill=False, hatch='/////', color = case_cld['HIT']['color'], width = barWidth, edgecolor =case_cld['HIT']['color'], label =case_cld['HIT']['name']) 
	plt.bar(br2_dble_MN, SDE_dble_MN[:,1], fill=False, hatch='/////', color = case_cld['FALSE ALARM']['color'], width = barWidth, edgecolor = case_cld['FALSE ALARM']['color'], label =case_cld['FALSE ALARM']['name']) 
	plt.bar(br3_dble_MN, SDE_dble_MN[:,2], fill=False, hatch='/////', color = case_cld['MISSED']['color'], width = barWidth, edgecolor = case_cld['MISSED']['color'], label =case_cld['MISSED']['name']) 
	plt.bar(br4_dble_MN, SDE_dble_MN[:,3], fill=False, hatch='/////', color = case_cld['CLEAR SKY']['color'], width = barWidth, edgecolor = case_cld['CLEAR SKY']['color'], label =case_cld['CLEAR SKY']['name']) 
	
	plt.bar(br1_750_MN, SDE_750_MN[:,0], fill=False, hatch='\\\\\\\\', color = case_cld['HIT m']['color'], width = barWidth, edgecolor =case_cld['HIT m']['color'], label =case_cld['HIT m']['name']) 
	plt.bar(br2_750_MN, SDE_750_MN[:,1], fill=False, hatch='\\\\\\\\', color = case_cld['FALSE ALARM m']['color'], width = barWidth, edgecolor =case_cld['FALSE ALARM m']['color'], label =case_cld['FALSE ALARM m']['name']) 
	plt.bar(br3_750_MN, SDE_750_MN[:,2], fill=False, hatch='\\\\\\\\', color = case_cld['MISSED m']['color'], width = barWidth, edgecolor =case_cld['MISSED m']['color'], label =case_cld['MISSED m']['name']) 
	plt.bar(br4_750_MN, SDE_750_MN[:,3], fill=False, hatch='\\\\\\\\', color = case_cld['CLEAR SKY m']['color'], width = barWidth, edgecolor =case_cld['CLEAR SKY m']['color'], label =case_cld['CLEAR SKY m']['name']) 
	
	plt.title('(b) Standard Deviation of errors', fontweight ='bold', fontsize = 15) 
	plt.xlabel('Month', fontweight ='bold', fontsize = 15) 
	plt.ylabel('SDE ($Wm^{-2}$)', fontweight ='bold', fontsize = 15);
	plt.xticks([r + barWidth for r in range(len(xax_dble_MN))], xax_dble_MN, rotation=15); plt.xlim([-0.5, 12.75]);
	#plt.legend()
	fle_out_png = suite+'_Comparison_monthly_'+PNT+"_" + run_UT + "_SWD_Bias_" + (date_lh['6'].min()).strftime('%Y%m%dT%H')+ "_" + (date_lh['6'].max()).strftime('%Y%m%dT%H') + "_K"+str(K)+"_"+str(fn_lim)+ "h_hist.png";
	fig.savefig(png_dir+ fle_out_png, dpi = 300, bbox_inches='tight')
	plt.close(fig)  

	fig = plt.figure(2, figsize =(9, 8))
	fig.subplots_adjust(hspace=0.3)
	plt.subplot(2,1,1);  plt.grid(which='major', axis='y', zorder=-1.0, alpha=0.3)
	plt.hlines(0, np.min(br1), np.max(br4), color='k')
	plt.bar(br1_dble_MN, freq_dble_MN[:,0], fill=False, hatch='/////', color = case_cld['HIT']['color'], width = barWidth, edgecolor =case_cld['HIT']['color'], label =case_cld['HIT']['name']) 
	plt.bar(br2_dble_MN, freq_dble_MN[:,1], fill=False, hatch='/////', color = case_cld['FALSE ALARM']['color'], width = barWidth, edgecolor = case_cld['FALSE ALARM']['color'], label =case_cld['FALSE ALARM']['name']) 
	plt.bar(br3_dble_MN, freq_dble_MN[:,2], fill=False, hatch='/////', color = case_cld['MISSED']['color'], width = barWidth, edgecolor = case_cld['MISSED']['color'], label =case_cld['MISSED']['name']) 
	plt.bar(br4_dble_MN, freq_dble_MN[:,3], fill=False, hatch='/////', color = case_cld['CLEAR SKY']['color'], width = barWidth, edgecolor = case_cld['CLEAR SKY']['color'], label =case_cld['CLEAR SKY']['name']) 
	
	plt.bar(br1_750_MN, freq_750_MN[:,0], fill=False, hatch='\\\\\\\\', color = case_cld['HIT m']['color'], width = barWidth, edgecolor =case_cld['HIT m']['color'], label =case_cld['HIT m']['name']) 
	plt.bar(br2_750_MN, freq_750_MN[:,1], fill=False, hatch='\\\\\\\\', color = case_cld['FALSE ALARM m']['color'], width = barWidth, edgecolor =case_cld['FALSE ALARM m']['color'], label =case_cld['FALSE ALARM m']['name']) 
	plt.bar(br3_750_MN, freq_750_MN[:,2], fill=False, hatch='\\\\\\\\', color = case_cld['MISSED m']['color'], width = barWidth, edgecolor =case_cld['MISSED m']['color'], label =case_cld['MISSED m']['name']) 
	plt.bar(br4_750_MN, freq_750_MN[:,3], fill=False, hatch='\\\\\\\\', color = case_cld['CLEAR SKY m']['color'], width = barWidth, edgecolor =case_cld['CLEAR SKY m']['color'], label =case_cld['CLEAR SKY m']['name']) 
	
	plt.title('Normalized Frequency', fontweight ='bold', fontsize = 15) 
	plt.xlabel('Month', fontweight ='bold', fontsize = 15) 
	plt.ylabel('Frequency', fontweight ='bold', fontsize = 15);
	plt.xticks([r + barWidth for r in range(len(xax_dble_MN))], xax_dble_MN, rotation=15); plt.xlim([-0.5, 12.75]);
	plt.legend(loc=1, fontsize = 5, ncol=2)
	fle_out_png = suite+'_Comparison_monthly_'+PNT+"_" + run_UT + "_Frequency_" + (date_lh['6'].min()).strftime('%Y%m%dT%H')+ "_" + (date_lh['6'].max()).strftime('%Y%m%dT%H') + "_K"+str(K)+"_"+str(fn_lim)+ "h_hist.png";
	fig.savefig(png_dir+ fle_out_png, dpi = 300, bbox_inches='tight')
	plt.close(fig)  
	
