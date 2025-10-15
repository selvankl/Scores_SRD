#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####################################
# 0. Init
####################################
  #------------------------------
  # 0.1 Import modules
  #------------------------------



import os
cmd = "module load python/3.7.6"
os.system(cmd)
cmd = "module use ~mary/public/modulefiles"
os.system(cmd)
cmd = "module load epygram"
os.system(cmd)
os.environ['MTOOL_STEP_CACHE']="/scratch/mtool/selvarajd/cache"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *

import pandas as pd
from itertools import repeat
import numpy as np
import matplotlib.colors as colors  
import epygram
from epygram.extra import usevortex as vtx
from bronx.stdtypes.date import daterangex as rangex
import pyproj
from datetime import datetime as dt

import netCDF4 as nc
from netCDF4 import Dataset
from datetime import datetime, timedelta
import sys; sys.path.insert(0,'/scratch/work/selvarajd/python/SWD/tools/')
from plot_AROME import zone_M_N_lon_lat_opt
from zenith_angle import zenith_angle_ephem
import math
import time
from concurrent.futures import ProcessPoolExecutor as pool

def fn_int(x):
    return int(x)

intV = np.vectorize(fn_int)

def f_1(x1,x2,x3,x4):
     return dt(int(x1),int(x2),int(x3), int(x4))
datehr_v = np.vectorize(f_1)

#------------------------------
  # 0.2 Env init
#------------------------------
epygram.init_env()

####################################
# 1. Setup
####################################
  #------------------------------
  # 1.1 Parameters
  #------------------------------

def extract_time(f_field):
	f_al = f_field.as_lists() 
	date_a = int(f_al['dates'][3])
	hr_a = int(f_al['times'][3]/100)
	date_int = dt(int(str(int(date_a))[0:4]),int(str(int(date_a))[4:6]),int(str(int(date_a))[6:8]), int(str(int(hr_a))))
	return date_int
extract_time_V = np.vectorize(extract_time)


def Array_lon_lat(f_field, x_0, y_1):
	f_al = f_field.as_lists() 
	lonA_list = f_al['longitudes']
	latA_list = f_al['latitudes']
	lonA = np.reshape(lonA_list, (x_0,y_1) )
	latA = np.reshape(latA_list, (x_0,y_1) )
	return lonA, latA

def AROME_SAT_cumul_zone(arome, date, lon, lat, M, N, K, fn_lim):
  nebul_h	= np.sum(arome[1].data[M-K:M+K+1,N-K:N+K+1])
  nebul_h_1 = np.sum(arome[0].data[M-K:M+K+1,N-K:N+K+1])
						
	#print("Temps de boucle csd_a zone", time.time()-debut_boucle)
  csd = 0
  nebul_tot = ((nebul_h - nebul_h_1)/(2*K+1)**2)/3600
  #print(nebul_tot)
  if nebul_tot>fn_lim:
    csd = 1
  return  csd, nebul_tot

def AROME_SAT_instant_zone_nest(arome, M, N, K):
  M = int(M)
  N = int(N)
  K = int(K)
  fn_lim = 0.02
  #print(arome.shape, arome)
  nebul_h	= np.sum(arome[M-K:M+K+1,N-K:N+K+1]/100)
  #print(arome.shape, M, N, K, fn_lim, nebul_h)
	#print("Temps de boucle csd_a zone", time.time()-debut_boucle)
  csd = 0
  nebul_tot = ((nebul_h)/(2*K+1)**2)
  #print(nebul_tot)
  if nebul_tot>fn_lim:
    csd = 1
  return  csd, nebul_tot

AROME_SAT_instant_zone_nest_vec = np.vectorize(AROME_SAT_instant_zone_nest)


def AROME_SAT_instant_zone(arome, date, lon, lat, M, N, K, fn_lim):
  nebul_h	= np.sum(arome[M-K:M+K+1,N-K:N+K+1]/100)
	#print("Temps de boucle csd_a zone", time.time()-debut_boucle)
  csd = 0
  nebul_tot = ((nebul_h)/(2*K+1)**2)
  #print(nebul_tot)
  if nebul_tot>fn_lim:
    csd = 1
  return  csd, nebul_tot

def epygram_param(ech, suite, reseau, mb, cutoff, vconf):
    if suite=="dble":
        geom = 'EURW1S40'
    elif suite=="oper":
        geom = 'EURW1S40'
    elif suite=="GZLD":
        geom = 'PARIS1S100'
    elif suite=="GZQB":
        geom = 'PARIS1S100'
    elif suite=="H1EK":
        geom = 'PARIS1S100'
    elif suite=="H1SN":
        geom = 'PARIS1S100'
    elif suite=="H3Z3":
        geom = 'PARIS1S100'
    else:
        print('Wrong geometry is defined')

    print(suite, reseau, ech, mb)
    resource = vtx.get_resources(experiment=suite, date=reseau, term=ech,\
                                 getmode='epygram',model='arome',origin='hst',\
                                  kind='gridpoint',block='forecast', cutoff=cutoff,\
                                    vapp='arome', vconf=vconf,geometry=geom,\
                                      namespace='vortex.multi.fr', nativefmt = 'grib',\
                                        member=mb, uselocalcache=False, shouldfly=False)
    
    fieldAll = resource
    
    return fieldAll

def epygram_readfield(fields, param_G):
    #print(param_G)
    return fields.readfield(param_G)

epygram_readfield_V = np.vectorize(epygram_readfield)

#def epygram_readfield_2(fields, param_G):
#   f_data = (fields.readfield(param_G)).getdata()
#   return f_data.tolist()
#epygram_readfield_V2 = np.vectorize(epygram_readfield_2)

def read_dble(date_v,run_UT,suite_key,N_mem,forecast_time_Mx,param_name):
    exp_date = date_v
    
    myDates = exp_date.strftime("%Y%m%dT")+run_UT ####--------'20240809T0300P' OR #-2021091321-PT24H'

    suite   = suite_key   # On va chercher les fichiers de l'archive "oper" OR "dble", suite="G6CN"
    cutoff  = "production"
    vconf   = "pefrance"
    if forecast_time_Mx==1:
       forecast_time_hr=[forecast_time_Mx]
    else:
       forecast_time_hr=list(range(1,forecast_time_Mx+1))
    if N_mem==0:
       mb=N_mem
    else:
       mb=list(range(N_mem+1))
    param   = param_name #"snow" #"wind" #"rr" "tpw850"  #Must be defined in GribID dict    ==> le parametre meteo demande
    out_dir = "."
    lag     = 12                       # duree cumul precip

    reseaux = rangex(myDates)

    #------------------------------
    # 1.2 GRIB ID
    #------------------------------
    GribID = {}
    GribID["Lw"] = {"discipline": 0, "parameterCategory": 1, "parameterNumber": 65, "typeOfFirstFixedSurface": 1, "level": 0, "typeOfSecondFixedSurface": 255, "tablesVersion": 15, "productDefinitionTemplateNumber": 11}
    GribID["Sw"] = {"discipline": 0, "parameterCategory": 1, "parameterNumber": 66, "typeOfFirstFixedSurface": 1, "level": 0, "typeOfSecondFixedSurface": 255, "tablesVersion": 15, "productDefinitionTemplateNumber": 11}
    GribID["Gw"] = {"discipline": 0, "parameterCategory": 1, "parameterNumber": 75, "typeOfFirstFixedSurface": 1, "level": 0, "typeOfSecondFixedSurface": 255, "tablesVersion": 15, "productDefinitionTemplateNumber": 11}
    
    GribID["tpw850"] = {"discipline":0,"parameterCategory":0,"parameterNumber":3,"tabesVersion":15,"level":850}

    GribID["2t"] = {"discipline" : 0,"parameterCategory": 0 , "parameterNumber" : 0, "level":2,"scaledValueOfFirstFixedSurface": 2,"typeOfFirstFixedSurface": 103,"productDefinitionTemplateNumber":1}

    GribID['ssrd'] = {"discipline" : 0,"parameterCategory" : 4, "parameterNumber" : 7,"typeOfFirstFixedSurface": 1, "level": 0,"typeOfSecondFixedSurface": 255, "productDefinitionTemplateNumber" : 11, "typeOfStatisticalProcessing" : 1}
    GribID['srfNtot'] = {"discipline" : 0, "parameterCategory" : 6, "parameterNumber" : 1, "tablesVersion" : 15, "typeOfFirstFixedSurface" : 1, "scaleFactorOfFirstFixedSurface" : 0, "scaledValueOfFirstFixedSurface" : 0, "typeOfSecondFixedSurface" : 255}
    GribID['AtmNtot'] = {"discipline" : 0, "parameterCategory" : 6, "parameterNumber" : 1, "productDefinitionTemplateNumber" : 8, "tablesVersion" : 15}

    GribID["u"]={"discipline": 0, "parameterCategory" : 2, "parameterNumber" : 2, "scaledValueOfFirstFixedSurface" : 10, "typeOfFirstFixedSurface" :103, "level":10}
    GribID["v"]={"discipline" : 0 , "parameterCategory" : 2, "parameterNumber": 3, "scaledValueOfFirstFixedSurface" : 10, "typeOfFirstFixedSurface" : 103, "level":10}

    GribID["td2m"] = {"discipline":0,"parameterCategory":0,"parameterNumber":6,"typeOfFirstFixedSurface":103,"scaleFactorOfFirstFixedSurface":0,"scaledValueOfFirstFixedSurface":2}
    GribID["tpw"]  = {"discipline":0,"parameterCategory":0,"parameterNumber":3}


    GribID["td"] = {"discipline":0,"parameterCategory":0,"parameterNumber":6}
    GribID["TV"] = {"discipline":0,"parameterCategory":0,"parameterNumber":1}
    GribID["theta"] = {"discipline":0,"parameterCategory":0,"parameterNumber":2}
    
    crs=None
    print(reseaux, forecast_time_hr, N_mem)
    tic = time.time(); 
    fields_l = epygram_param(forecast_time_hr, suite, reseaux, mb, cutoff, vconf)  
    print(len(fields_l))
    fields_param_V = epygram_readfield_V(fields_l, GribID[param]) 
    ##--------REMOVING THE shouldfly-* files of vortex from the working directory-----
    cmd_should = 'rm -rf shouldfly-*'
    os.system(cmd_should)

    toc = time.time();
    print('Done in {:.4f} seconds'.format(toc-tic));    
    
    return fields_param_V

def read_dble_2_param(date_v,run_UT,suite_key,N_mem,forecast_time_Mx,param_name_1,param_name_2):
    exp_date = date_v
    
    myDates = exp_date.strftime("%Y%m%dT")+run_UT ####--------'20240809T0300P' OR #-2021091321-PT24H'

    suite   = suite_key   # On va chercher les fichiers de l'archive "oper" OR "dble", suite="G6CN"
    cutoff  = "production"
    vconf   = "pefrance"

    Missing_mem_GZLD = dt(2024, 10, 10, 0, 0)
    Missing_mem_H1EK1 = dt(2024, 8, 27, 0, 0)
    Missing_mem_H1EK2 = dt(2024, 8, 29, 0, 0)
    Missing_mem_H1EK3 = dt(2024, 8, 30, 0, 0)
    Missing_mem_H1EK4 = dt(2024, 8, 31, 0, 0)
    Missing_mem_dble1 = dt(2024, 3, 10, 0, 0)
    Missing_mem_dble2 = dt(2024, 3, 11, 0, 0)
    Missing_mem_H3Z3 = dt(2024, 5, 18, 0, 0)
    Missing_mem2_H3Z3 = dt(2024, 3, 11, 0, 0)
    Missing_mem3_H3Z3 = dt(2024, 3, 19, 0, 0)
    Missing_mem4_H3Z3 = dt(2024, 5, 18, 0, 0)
    Missing_mem5_H3Z3 = dt(2024, 5, 31, 0, 0)
    Missing_mem6_H3Z3 = dt(2024, 6, 1, 0, 0)
    Missing_mem7_H3Z3 = dt(2024, 6, 8, 0, 0)

    if forecast_time_Mx==1:
       forecast_time_hr=[forecast_time_Mx]
    else:
       forecast_time_hr=list(range(1,forecast_time_Mx+1))
    if N_mem==0:
       mb=N_mem
    elif Missing_mem_dble1== date_v and suite == 'dble':
       mb=list(range(1,N_mem+1))
       print(suite, date_v, Missing_mem_dble1, mb)
    elif Missing_mem_dble2== date_v and suite == 'dble':
       mb=list(range(1,N_mem+1))
       print(suite, date_v, Missing_mem_dble1, mb)
    elif Missing_mem_GZLD== date_v and suite == 'GZLD':
       st=list(range(8+1))
       ed=list(range(10,N_mem+1))
       mb = st+ed
       print(suite, date_v, Missing_mem_GZLD, mb)
    elif Missing_mem_H1EK1== date_v and suite == 'H1EK':
       st=list(range(2+1))
       ed=list(range(4,N_mem+1))
       mb = st+ed
       print(suite, date_v, Missing_mem_H1EK1, mb)
    elif Missing_mem_H1EK2== date_v and suite == 'H1EK':
       st=list(range(2,8+1))
       ed=list(range(13,N_mem+1))
       mb = list(range(0,1)) + st+ list(range(10,11)) +ed
       print(suite, date_v, Missing_mem_H1EK1, mb)
    elif Missing_mem_H1EK3== date_v and suite == 'H1EK':
       st=list(range(5+1))
       ed=list(range(7,N_mem+1))
       mb = st+ed
       print(suite, date_v, Missing_mem_H1EK1, mb)
    elif Missing_mem_H1EK4== date_v and suite == 'H1EK':
       st=list(range(2,4+1))
       ed=list(range(6,N_mem+1))
       mb = list(range(0,1)) + st+ed
       print(suite, date_v, Missing_mem_H1EK1, mb)
    elif Missing_mem_H3Z3== date_v and suite == 'H3Z3':
       mb = list(range(0,23))
       print(suite, date_v, Missing_mem_H3Z3, mb)
    elif Missing_mem2_H3Z3== date_v and suite == 'H3Z3':
       mb = list(range(1,N_mem+1))
       print(suite, date_v, Missing_mem2_H3Z3, mb)
    elif Missing_mem3_H3Z3== date_v and suite == 'H3Z3':
       st=list(range(1+1))
       ed=list(range(3,N_mem+1))
       mb = st+ed
       print(suite, date_v, Missing_mem3_H3Z3, mb)
    elif Missing_mem4_H3Z3 == date_v and suite == 'H3Z3':
       st=list(range(21+1))
       ed=list(range(23,N_mem+1))
       mb = st+ed
       print(suite, date_v, Missing_mem4_H3Z3, mb)
       print(suite, date_v, Missing_mem4_H3Z3, mb)
    elif Missing_mem5_H3Z3== date_v and suite == 'H3Z3':
       mb = list(range(0,N_mem-1))
       print(suite, date_v, Missing_mem5_H3Z3, mb)
    elif Missing_mem6_H3Z3== date_v and suite == 'H3Z3':
       mb = list(range(0,N_mem-1))
       print(suite, date_v, Missing_mem6_H3Z3, mb)
    elif Missing_mem7_H3Z3== date_v and suite == 'H3Z3':
       mb = list(range(0,N_mem-1))
       print(suite, date_v, Missing_mem7_H3Z3, mb)
    else:
        mb=list(range(N_mem+1))
       


    reseaux = rangex(myDates)

    #------------------------------
    # 1.2 GRIB ID
    #------------------------------
    GribID = {}
    GribID["rr"] = {'parameterNumber': 65}   #RAIN GRIB2
    GribID["tpw850"] = {"discipline":0,"parameterCategory":0,"parameterNumber":3,"tabesVersion":15,"level":850}

    GribID["2t"] = {"discipline" : 0,"parameterCategory": 0 , "parameterNumber" : 0, "level":2,"scaledValueOfFirstFixedSurface": 2,"typeOfFirstFixedSurface": 103,"productDefinitionTemplateNumber":1}
    if suite=="dble":
        GribID['ssrd'] = {"discipline" : 0,"parameterCategory" : 4, "parameterNumber" : 7,"typeOfFirstFixedSurface": 1, "level": 0, "typeOfLevel":'surface', "typeOfSecondFixedSurface": 255, "productDefinitionTemplateNumber" : 11, "typeOfStatisticalProcessing" : 1}
    elif suite=="oper":
        GribID['ssrd'] = {"discipline" : 0,"parameterCategory" : 4, "parameterNumber" : 7,"typeOfFirstFixedSurface": 1, "level": 0,"typeOfSecondFixedSurface": 255, "productDefinitionTemplateNumber" : 11, "typeOfStatisticalProcessing" : 1}
    elif suite=="GZLD":
        GribID['ssrd'] = {"discipline" : 0,"parameterCategory" : 4, "parameterNumber" : 7,"typeOfFirstFixedSurface": 1, "level": 0,"typeOfSecondFixedSurface": 255, "productDefinitionTemplateNumber" : 8, "typeOfStatisticalProcessing" : 1}
    elif suite=="GZQB":
        GribID['ssrd'] = {"discipline" : 0,"parameterCategory" : 4, "parameterNumber" : 7,"typeOfFirstFixedSurface": 1, "level": 0,"typeOfSecondFixedSurface": 255, "productDefinitionTemplateNumber" : 8, "typeOfStatisticalProcessing" : 1}
    elif suite=="H1EK":
        GribID['ssrd'] = {"discipline" : 0,"parameterCategory" : 4, "parameterNumber" : 7,"typeOfFirstFixedSurface": 1, "level": 0,"typeOfSecondFixedSurface": 255, "productDefinitionTemplateNumber" : 8, "typeOfStatisticalProcessing" : 1}
    elif suite=="H1SN":
        GribID['ssrd'] = {"discipline" : 0,"parameterCategory" : 4, "parameterNumber" : 7,"typeOfFirstFixedSurface": 1, "level": 0,"typeOfSecondFixedSurface": 255, "productDefinitionTemplateNumber" : 8, "typeOfStatisticalProcessing" : 1}
    elif suite=="H3Z3":
        GribID['ssrd'] = {"discipline" : 0,"parameterCategory" : 4, "parameterNumber" : 7,"typeOfFirstFixedSurface": 1, "level": 0, "typeOfLevel":0, "typeOfSecondFixedSurface": 255, "productDefinitionTemplateNumber" : 8, "typeOfStatisticalProcessing" : 1}
    else:
        print('Product definition needs to be checked')
        
    if suite=="dble":
        GribID['srfNtot'] = {"discipline" : 0, "parameterCategory" : 6, "parameterNumber" : 1, "tablesVersion" : 15, "typeOfFirstFixedSurface" : 1, "scaleFactorOfFirstFixedSurface" : 0, "scaledValueOfFirstFixedSurface" : 0, "typeOfSecondFixedSurface" : 255}
    elif suite=="oper":
        GribID['srfNtot'] = {"discipline" : 0, "parameterCategory" : 6, "parameterNumber" : 1, "tablesVersion" : 15, "typeOfFirstFixedSurface" : 1, "scaleFactorOfFirstFixedSurface" : 0, "scaledValueOfFirstFixedSurface" : 0, "typeOfSecondFixedSurface" : 255}
    elif suite=="GZLD":
        GribID['srfNtot'] = {"discipline" : 0, "parameterCategory" : 6, "parameterNumber" : 1, "tablesVersion" : 15, "typeOfFirstFixedSurface" : 1, "scaleFactorOfFirstFixedSurface" : 0, "scaledValueOfFirstFixedSurface" : 0, "typeOfSecondFixedSurface" : 255}
    elif suite=="GZQB":
        GribID['srfNtot'] = {"discipline" : 0, "parameterCategory" : 6, "parameterNumber" : 1, "tablesVersion" : 15, "typeOfFirstFixedSurface" : 1, "scaleFactorOfFirstFixedSurface" : 0, "scaledValueOfFirstFixedSurface" : 0, "typeOfSecondFixedSurface" : 255}
    elif suite=="H1EK":
        GribID['srfNtot'] = {"discipline" : 0, "parameterCategory" : 6, "parameterNumber" : 1, "tablesVersion" : 15, "typeOfFirstFixedSurface" : 1, "scaleFactorOfFirstFixedSurface" : 0, "scaledValueOfFirstFixedSurface" : 0, "typeOfSecondFixedSurface" : 255}
    elif suite=="H1SN":
        GribID['srfNtot'] = {"discipline" : 0, "parameterCategory" : 6, "parameterNumber" : 1, "tablesVersion" : 15, "typeOfFirstFixedSurface" : 1, "scaleFactorOfFirstFixedSurface" : 0, "scaledValueOfFirstFixedSurface" : 0, "typeOfSecondFixedSurface" : 255}
    elif suite=="H3Z3":
        GribID['srfNtot'] = {"discipline" : 0, "parameterCategory" : 6, "parameterNumber" : 1, "tablesVersion" : 15, "typeOfFirstFixedSurface" : 1, "scaleFactorOfFirstFixedSurface" : 0, "scaledValueOfFirstFixedSurface" : 0, "typeOfSecondFixedSurface" : 255}
    else:
        print('Product definition needs to be checked')
    
    GribID['AtmNtot'] = {"discipline" : 0, "parameterCategory" : 6, "parameterNumber" : 1, "productDefinitionTemplateNumber" : 8, "tablesVersion" : 15}

    GribID["u"]={"discipline": 0, "parameterCategory" : 2, "parameterNumber" : 2, "scaledValueOfFirstFixedSurface" : 10, "typeOfFirstFixedSurface" :103, "level":10}
    GribID["v"]={"discipline" : 0 , "parameterCategory" : 2, "parameterNumber": 3, "scaledValueOfFirstFixedSurface" : 10, "typeOfFirstFixedSurface" : 103, "level":10}

    GribID["td2m"] = {"discipline":0,"parameterCategory":0,"parameterNumber":6,"typeOfFirstFixedSurface":103,"scaleFactorOfFirstFixedSurface":0,"scaledValueOfFirstFixedSurface":2}
    GribID["tpw"]  = {"discipline":0,"parameterCategory":0,"parameterNumber":3}


    GribID["td"] = {"discipline":0,"parameterCategory":0,"parameterNumber":6}
    GribID["TV"] = {"discipline":0,"parameterCategory":0,"parameterNumber":1}
    GribID["theta"] = {"discipline":0,"parameterCategory":0,"parameterNumber":2}
    
    crs=None
    print(reseaux, forecast_time_hr, N_mem)
    tic = time.time(); 
    fields_l = epygram_param(forecast_time_hr, suite, reseaux, mb, cutoff, vconf)  
    print(len(fields_l))
    fields_param_1_V = epygram_readfield_V(fields_l, GribID[param_name_1]) 
    fields_param_2_V = epygram_readfield_V(fields_l, GribID[param_name_2]) 
    ##--------REMOVING THE shouldfly-* files of vortex from the working directory-----
    cmd_should = 'rm -rf shouldfly-*'
    os.system(cmd_should)

    toc = time.time();
    print('Reading PEAROME data is done in {:.4f} seconds'.format(toc-tic));    
    
    return fields_param_1_V, fields_param_2_V

def read_dble_6_param(date_v,run_UT,suite_key,N_mem,forecast_time_Mx,param_name_1,param_name_2,param_name_3,param_name_4,param_name_5,param_name_6):
    exp_date = date_v
    
    myDates = exp_date.strftime("%Y%m%dT")+run_UT ####--------'20240809T0300P' OR #-2021091321-PT24H'

    suite   = suite_key   # On va chercher les fichiers de l'archive "oper" OR "dble", suite="G6CN"
    cutoff  = "production"
    vconf   = "pefrance"
    
    Missing_mem_dble_10102024 = dt(2024, 10, 10, 0, 0)

    Missing_mem_GZLD = dt(2024, 10, 10, 0, 0)
    Missing_mem_H1EK1 = dt(2024, 8, 27, 0, 0)
    Missing_mem_H1EK2 = dt(2024, 8, 29, 0, 0)
    Missing_mem_H1EK3 = dt(2024, 8, 30, 0, 0)
    Missing_mem_H1EK4 = dt(2024, 8, 31, 0, 0)
    Missing_mem_H3Z3 = dt(2024, 5, 18, 0, 0)
    Missing_mem2_H3Z3 = dt(2024, 3, 11, 0, 0)
    Missing_mem3_H3Z3 = dt(2024, 3, 19, 0, 0)


    if forecast_time_Mx==1:
       forecast_time_hr=[forecast_time_Mx]
    else:
       forecast_time_hr=list(range(1,forecast_time_Mx+1))
    if N_mem==0:
       mb=N_mem
    elif Missing_mem_dble_10102024== date_v and suite == 'dble':
       st=list(range(13+1))
       ed=list(range(16,N_mem+1))
       mb = st+ed
       print(suite, date_v, Missing_mem_GZLD, mb)
    elif Missing_mem_GZLD== date_v and suite == 'GZLD':
       st=list(range(8+1))
       ed=list(range(10,N_mem+1))
       mb = st+ed
       print(suite, date_v, Missing_mem_GZLD, mb)
    elif Missing_mem_H1EK1== date_v and suite == 'H1EK':
       st=list(range(2+1))
       ed=list(range(4,N_mem+1))
       mb = st+ed
       print(suite, date_v, Missing_mem_H1EK1, mb)
    elif Missing_mem_H1EK2== date_v and suite == 'H1EK':
       st=list(range(2,8+1))
       ed=list(range(13,N_mem+1))
       mb = list(range(0,1)) + st+ list(range(10,11)) +ed
       print(suite, date_v, Missing_mem_H1EK1, mb)
    elif Missing_mem_H1EK3== date_v and suite == 'H1EK':
       st=list(range(5+1))
       ed=list(range(7,N_mem+1))
       mb = st+ed
       print(suite, date_v, Missing_mem_H1EK1, mb)
    elif Missing_mem_H1EK4== date_v and suite == 'H1EK':
       st=list(range(2,4+1))
       ed=list(range(6,N_mem+1))
       mb = list(range(0,1)) + st+ed
       print(suite, date_v, Missing_mem_H1EK1, mb)
    elif Missing_mem_H3Z3== date_v and suite == 'H3Z3':
       mb = list(range(0,23))
       print(suite, date_v, Missing_mem_H3Z3, mb)
    elif Missing_mem2_H3Z3== date_v and suite == 'H3Z3':
       mb = list(range(1,N_mem+1))
       print(suite, date_v, Missing_mem2_H3Z3, mb)
    elif Missing_mem3_H3Z3== date_v and suite == 'H3Z3':
       st=list(range(1+1))
       ed=list(range(3,N_mem+1))
       mb = st+ed
       print(suite, date_v, Missing_mem3_H3Z3, mb)
    else:
        mb=list(range(N_mem+1))
       


    reseaux = rangex(myDates)

    #------------------------------
    # 1.2 GRIB ID
    #------------------------------
    GribID = {}
    GribID["Lw"] = {"discipline": 0, "parameterCategory": 1, "parameterNumber": 65, "typeOfFirstFixedSurface": 1, "level": 0, "typeOfSecondFixedSurface": 255, "tablesVersion": 15, "productDefinitionTemplateNumber": 11}
    GribID["Sw"] = {"discipline": 0, "parameterCategory": 1, "parameterNumber": 66, "typeOfFirstFixedSurface": 1, "level": 0, "typeOfSecondFixedSurface": 255, "tablesVersion": 15, "productDefinitionTemplateNumber": 11}
    GribID["Gw"] = {"discipline": 0, "parameterCategory": 1, "parameterNumber": 75, "typeOfFirstFixedSurface": 1, "level": 0, "typeOfSecondFixedSurface": 255, "tablesVersion": 15, "productDefinitionTemplateNumber": 11}
    
    GribID["10u"] = {"discipline": 0, "parameterCategory": 2, "parameterNumber": 2, "typeOfFirstFixedSurface": 103, "level": 10, "typeOfSecondFixedSurface": 255, "tablesVersion": 15, "productDefinitionTemplateNumber": 1}
    GribID["10v"] = {"discipline": 0, "parameterCategory": 2, "parameterNumber": 3, "typeOfFirstFixedSurface": 103, "level": 10, "typeOfSecondFixedSurface": 255, "tablesVersion": 15, "productDefinitionTemplateNumber": 1}
    GribID["2t"] = {"discipline" : 0,"parameterCategory": 0 , "parameterNumber" : 0, "level":2,"scaledValueOfFirstFixedSurface": 2,"typeOfFirstFixedSurface": 103,"productDefinitionTemplateNumber":1}

    GribID["tpw850"] = {"discipline":0,"parameterCategory":0,"parameterNumber":3,"tabesVersion":15,"level":850}

    if suite=="dble":
        GribID['ssrd'] = {"discipline" : 0,"parameterCategory" : 4, "parameterNumber" : 7,"typeOfFirstFixedSurface": 1, "level": 0,"typeOfSecondFixedSurface": 255, "productDefinitionTemplateNumber" : 11, "typeOfStatisticalProcessing" : 1}
    elif suite=="oper":
        GribID['ssrd'] = {"discipline" : 0,"parameterCategory" : 4, "parameterNumber" : 7,"typeOfFirstFixedSurface": 1, "level": 0,"typeOfSecondFixedSurface": 255, "productDefinitionTemplateNumber" : 11, "typeOfStatisticalProcessing" : 1}
    elif suite=="GZLD":
        GribID['ssrd'] = {"discipline" : 0,"parameterCategory" : 4, "parameterNumber" : 7,"typeOfFirstFixedSurface": 1, "level": 0,"typeOfSecondFixedSurface": 255, "productDefinitionTemplateNumber" : 8, "typeOfStatisticalProcessing" : 1}
    elif suite=="GZQB":
        GribID['ssrd'] = {"discipline" : 0,"parameterCategory" : 4, "parameterNumber" : 7,"typeOfFirstFixedSurface": 1, "level": 0,"typeOfSecondFixedSurface": 255, "productDefinitionTemplateNumber" : 8, "typeOfStatisticalProcessing" : 1}
    elif suite=="H1EK":
        GribID['ssrd'] = {"discipline" : 0,"parameterCategory" : 4, "parameterNumber" : 7,"typeOfFirstFixedSurface": 1, "level": 0,"typeOfSecondFixedSurface": 255, "productDefinitionTemplateNumber" : 8, "typeOfStatisticalProcessing" : 1}
    elif suite=="H1SN":
        GribID['ssrd'] = {"discipline" : 0,"parameterCategory" : 4, "parameterNumber" : 7,"typeOfFirstFixedSurface": 1, "level": 0,"typeOfSecondFixedSurface": 255, "productDefinitionTemplateNumber" : 8, "typeOfStatisticalProcessing" : 1}
    elif suite=="H3Z3":
        GribID['ssrd'] = {"discipline" : 0,"parameterCategory" : 4, "parameterNumber" : 7,"typeOfFirstFixedSurface": 1, "level": 0,"typeOfSecondFixedSurface": 255, "productDefinitionTemplateNumber" : 8, "typeOfStatisticalProcessing" : 1}
    else:
        print('Product definition needs to be checked')
        
    if suite=="dble":
        GribID['srfNtot'] = {"discipline" : 0, "parameterCategory" : 6, "parameterNumber" : 1, "tablesVersion" : 15, "typeOfFirstFixedSurface" : 1, "scaleFactorOfFirstFixedSurface" : 0, "scaledValueOfFirstFixedSurface" : 0, "typeOfSecondFixedSurface" : 255}
    elif suite=="oper":
        GribID['srfNtot'] = {"discipline" : 0, "parameterCategory" : 6, "parameterNumber" : 1, "tablesVersion" : 15, "typeOfFirstFixedSurface" : 1, "scaleFactorOfFirstFixedSurface" : 0, "scaledValueOfFirstFixedSurface" : 0, "typeOfSecondFixedSurface" : 255}
    elif suite=="GZLD":
        GribID['srfNtot'] = {"discipline" : 0, "parameterCategory" : 6, "parameterNumber" : 1, "tablesVersion" : 15, "typeOfFirstFixedSurface" : 1, "scaleFactorOfFirstFixedSurface" : 0, "scaledValueOfFirstFixedSurface" : 0, "typeOfSecondFixedSurface" : 255}
    elif suite=="GZQB":
        GribID['srfNtot'] = {"discipline" : 0, "parameterCategory" : 6, "parameterNumber" : 1, "tablesVersion" : 15, "typeOfFirstFixedSurface" : 1, "scaleFactorOfFirstFixedSurface" : 0, "scaledValueOfFirstFixedSurface" : 0, "typeOfSecondFixedSurface" : 255}
    elif suite=="H1EK":
        GribID['srfNtot'] = {"discipline" : 0, "parameterCategory" : 6, "parameterNumber" : 1, "tablesVersion" : 15, "typeOfFirstFixedSurface" : 1, "scaleFactorOfFirstFixedSurface" : 0, "scaledValueOfFirstFixedSurface" : 0, "typeOfSecondFixedSurface" : 255}
    elif suite=="H1SN":
        GribID['srfNtot'] = {"discipline" : 0, "parameterCategory" : 6, "parameterNumber" : 1, "tablesVersion" : 15, "typeOfFirstFixedSurface" : 1, "scaleFactorOfFirstFixedSurface" : 0, "scaledValueOfFirstFixedSurface" : 0, "typeOfSecondFixedSurface" : 255}
    elif suite=="H3Z3":
        GribID['srfNtot'] = {"discipline" : 0, "parameterCategory" : 6, "parameterNumber" : 1, "tablesVersion" : 15, "typeOfFirstFixedSurface" : 1, "scaleFactorOfFirstFixedSurface" : 0, "scaledValueOfFirstFixedSurface" : 0, "typeOfSecondFixedSurface" : 255}
    else:
        print('Product definition needs to be checked')
    
    GribID['AtmNtot'] = {"discipline" : 0, "parameterCategory" : 6, "parameterNumber" : 1, "productDefinitionTemplateNumber" : 8, "tablesVersion" : 15}

    GribID["u"]={"discipline": 0, "parameterCategory" : 2, "parameterNumber" : 2, "scaledValueOfFirstFixedSurface" : 10, "typeOfFirstFixedSurface" :103, "level":10}
    GribID["v"]={"discipline" : 0 , "parameterCategory" : 2, "parameterNumber": 3, "scaledValueOfFirstFixedSurface" : 10, "typeOfFirstFixedSurface" : 103, "level":10}

    GribID["td2m"] = {"discipline":0,"parameterCategory":0,"parameterNumber":6,"typeOfFirstFixedSurface":103,"scaleFactorOfFirstFixedSurface":0,"scaledValueOfFirstFixedSurface":2}
    GribID["tpw"]  = {"discipline":0,"parameterCategory":0,"parameterNumber":3}


    GribID["td"] = {"discipline":0,"parameterCategory":0,"parameterNumber":6}
    GribID["TV"] = {"discipline":0,"parameterCategory":0,"parameterNumber":1}
    GribID["theta"] = {"discipline":0,"parameterCategory":0,"parameterNumber":2}
    
    crs=None
    print(reseaux, forecast_time_hr, N_mem)
    tic = time.time(); 
    fields_l = epygram_param(forecast_time_hr, suite, reseaux, mb, cutoff, vconf)  
    print(len(fields_l))
    fields_param_1_V = epygram_readfield_V(fields_l, GribID[param_name_1]) 
    fields_param_2_V = epygram_readfield_V(fields_l, GribID[param_name_2]) 
    fields_param_3_V = epygram_readfield_V(fields_l, GribID[param_name_3]) 
    fields_param_4_V = epygram_readfield_V(fields_l, GribID[param_name_4]) 
    fields_param_5_V = epygram_readfield_V(fields_l, GribID[param_name_5]) 
    fields_param_6_V = epygram_readfield_V(fields_l, GribID[param_name_6]) 
    ##--------REMOVING THE shouldfly-* files of vortex from the working directory-----
    cmd_should = 'rm -rf shouldfly-*'
    os.system(cmd_should)

    toc = time.time();
    print('Reading PEAROME data is done in {:.4f} seconds'.format(toc-tic));    
    
    return fields_param_1_V, fields_param_2_V, fields_param_3_V, fields_param_4_V, fields_param_5_V, fields_param_6_V


def csd_nebul_BDclim_arome_instant_zone(lon, lat, classe, num_station, alti, M, N, Z_cems, nebul_cems_h, run_date, min_tme, srf_m, K, fn_lim):
	cos_zen =  zenith_angle_ephem(lat, lon, run_date+timedelta(minutes = -30)) #Only when zenith angle is lower than a threshold
	if classe<4 and num_station!='6088001' and alti<1000 and cos_zen > 0.1 : #Selection of the stations
		nebul_cems_h = nebul_cems_h + Z_cems
		if min_tme=='00': #on identifie pour AROME une seule fois par heure
					###---csd_a, nebul = AROME_SAT_instant_zone(f_srf[run_date_ID], run_date, lon[i], lat[i], int(M[i]), int(N[i]), int(K), fn_lim)
					srf_cloud_mem = srf_m[0]
					#print(len(srf_m), len(srf_cloud_mem))
					csd_neb_arr = []
					print(srf_cloud_mem, K)
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



def BDClim_station_h(BD_data_dir, num_station, run_date):
	m_str = str(run_date.month).zfill(2)
	d_str = str(run_date.day).zfill(2)
	h_str = str(run_date.hour).zfill(2)
	fichier = BD_data_dir+'BDClim_%s_%s_%s_%s' %(run_date.year, m_str, d_str, h_str ) 
	#On importe les valeurs
	data_val_str = pd.read_csv(fichier, sep=" ", header=None)
	data_val_str.columns = 'num_poste', 'dates', 'times', 'glo', 'glo2', 'rr1', 'fxy', 'dxy', 't', 'dir', 'dir2', 'dif', 'dif2', 'n1', 'n2', 'n3', 'n4', 'n','hneigef', 'neigetot', 'hneigefi1';

	#print(data_val_str)
	
	data_rr = data_val_str['rr1'].values
	data_ff = data_val_str['fxy'].values
	data_dd = data_val_str['dxy'].values
	data_t = data_val_str['t'].values
	
	station_id = np.where(data_val_str['num_poste'].values==num_station)[0][0]
	
	return  np.array([run_date, num_station, data_rr[station_id], data_ff[station_id], data_dd[station_id], data_t[station_id] ])

def BDclim_Arome_zone_M_N_lon_lat_opt(BD_data_dir, num_station, swd_m_l, lon_a, lat_a, M, N, K, run_date):
	toc_h = time.time();
	swd_m = swd_m_l[0]
	#print(len(swd_m), swd_m[0].shape)
  #--------------------------------------------------
	#---------Statistiques calculation----------------------------
	#------------------------------------------
	#print(lat, lon, run_date)

	#Recover information each hour and each station
	BD_param = BDClim_station_h(BD_data_dir, num_station, run_date) #inf si pas present
		
	#Calculate GHI in AROME with the spatial tolerance method
	###---Z, lon_zone, lat_zone = zone_M_N_lon_lat_opt(swd_m[run_date_ID], lon_a, lat_a, M[i],N[i], date_v, h,K)
	tic = time.time()
	with pool(max_workers=len(swd_m)-1) as P:
		Z_zone = P.map(Arome_zone_M_N_lon_lat, swd_m, repeat(lon_a), repeat(lat_a), repeat(M), repeat(N), repeat(K))
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
	return [BD_param, Z]

def detect_hourly(swd_l, h_term_l, dates_swd, date_v_l):
	swd_h = swd_l
	h_term = h_term_l
	dates_l = dates_swd
	date_v = date_v_l
	run_date = date_v+timedelta(hours = h_term) 
	run_date_ID = np.where(dates_l==run_date)[0][0]
	print(run_date_ID)
	return run_date_ID


#Pareil que dessus mais optimiser pour l'ouverture du fichier netcdf
def Arome_zone_M_N_lon_lat(arome, lon_a, lat_a, i,j, K):
	
	#print(arome.shape)
	#var  = "SURFRAYT_SOLA_DE"
	i = int(i)
	j = int(j)
	K = int(K)


	#print(t, K, i, j)
	#diff = arome.variables[var][t-1,i-K:i+K+1,j-K:j+K+1]
	Z = arome[i-K:i+K+1,j-K:j+K+1]
	lon = lon_a[i-K:i+K+1,j-K:j+K+1]
	lat= lat_a[i-K:i+K+1,j-K:j+K+1]
	#Z = (Z-diff)/3600

	return Z, lon, lat


def read_dble_45_param(date_v,run_UT,suite_key,N_mem,forecast_time_Mx,param_var):

    #print(param_var); param_name_1,param_name_2,param_name_3,param_name_4,param_name_5,param_name_6,param_name_7,param_name_8,param_name_9,param_name_10,param_name_11,param_name_12,param_name_13,param_name_14,param_name_15,param_name_16,param_name_17,param_name_18,param_name_19,param_name_20,param_name_21,param_name_22,param_name_23,param_name_24,param_name_25,param_name_26,param_name_27,param_name_28,param_name_29,param_name_30,param_name_31,param_name_32,param_name_33,param_name_34,param_name_35,param_name_36,param_name_37,param_name_38,param_name_39,param_name_40,param_name_41,param_name_42,param_name_43,param_name_44,param_name_45 #,param_name_46
    exp_date = date_v; 
    
    myDates = exp_date.strftime("%Y%m%dT")+run_UT ####--------'20240809T0300P' OR #-2021091321-PT24H'

    suite   = suite_key   # On va chercher les fichiers de l'archive "oper" OR "dble", suite="G6CN"
    cutoff  = "production"
    vconf   = "pefrance"
    
    Missing_mem_dble_10102024 = dt(2024, 10, 10, 0, 0)

    Missing_mem_GZLD = dt(2024, 10, 10, 0, 0)
    Missing_mem_H1EK1 = dt(2024, 8, 27, 0, 0)
    Missing_mem_H1EK2 = dt(2024, 8, 29, 0, 0)
    Missing_mem_H1EK3 = dt(2024, 8, 30, 0, 0)
    Missing_mem_H1EK4 = dt(2024, 8, 31, 0, 0)
    Missing_mem_H3Z3 = dt(2024, 5, 18, 0, 0)
    Missing_mem2_H3Z3 = dt(2024, 3, 11, 0, 0)
    Missing_mem3_H3Z3 = dt(2024, 3, 19, 0, 0)


    if forecast_time_Mx==1:
       forecast_time_hr=[forecast_time_Mx]
    else:
       forecast_time_hr=list(range(1,forecast_time_Mx+1))
    if N_mem==0:
       mb=N_mem
    elif Missing_mem_dble_10102024== date_v and suite == 'dble':
       st=list(range(13+1))
       ed=list(range(16,N_mem+1))
       mb = st+ed
       print(suite, date_v, Missing_mem_GZLD, mb)
    elif Missing_mem_GZLD== date_v and suite == 'GZLD':
       st=list(range(8+1))
       ed=list(range(10,N_mem+1))
       mb = st+ed
       print(suite, date_v, Missing_mem_GZLD, mb)
    elif Missing_mem_H1EK1== date_v and suite == 'H1EK':
       st=list(range(2+1))
       ed=list(range(4,N_mem+1))
       mb = st+ed
       print(suite, date_v, Missing_mem_H1EK1, mb)
    elif Missing_mem_H1EK2== date_v and suite == 'H1EK':
       st=list(range(2,8+1))
       ed=list(range(13,N_mem+1))
       mb = list(range(0,1)) + st+ list(range(10,11)) +ed
       print(suite, date_v, Missing_mem_H1EK1, mb)
    elif Missing_mem_H1EK3== date_v and suite == 'H1EK':
       st=list(range(5+1))
       ed=list(range(7,N_mem+1))
       mb = st+ed
       print(suite, date_v, Missing_mem_H1EK1, mb)
    elif Missing_mem_H1EK4== date_v and suite == 'H1EK':
       st=list(range(2,4+1))
       ed=list(range(6,N_mem+1))
       mb = list(range(0,1)) + st+ed
       print(suite, date_v, Missing_mem_H1EK1, mb)
    elif Missing_mem_H3Z3== date_v and suite == 'H3Z3':
       mb = list(range(0,23))
       print(suite, date_v, Missing_mem_H3Z3, mb)
    elif Missing_mem2_H3Z3== date_v and suite == 'H3Z3':
       mb = list(range(1,N_mem+1))
       print(suite, date_v, Missing_mem2_H3Z3, mb)
    elif Missing_mem3_H3Z3== date_v and suite == 'H3Z3':
       st=list(range(1+1))
       ed=list(range(3,N_mem+1))
       mb = st+ed
       print(suite, date_v, Missing_mem3_H3Z3, mb)
    else:
        mb=list(range(N_mem+1))
       


    reseaux = rangex(myDates)

    #------------------------------
    # 1.2 GRIB ID
    #------------------------------
    GribID = {}

    if suite=="dble":
         GribID['H_COULIM'] = {"discipline" : 0,"parameterCategory" : 3, "parameterNumber" : 18,"typeOfFirstFixedSurface": 1, "scaledValueOfFirstFixedSurface": 0, "tablesVersion": 15}
         GribID['SRD'] = {"discipline" : 0,"parameterCategory" : 4, "parameterNumber" : 7,"typeOfFirstFixedSurface": 1, "tablesVersion": 15, "productDefinitionTemplateNumber" : 11}
         GribID['NEBUL'] = {"discipline" : 0,"parameterCategory" : 6, "parameterNumber" : 1,"typeOfFirstFixedSurface": 1,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}
         #GribID['NEBCON'] = {"discipline" 0: ,"parameterCategory" : 6, "parameterNumber" : 2,"typeOfFirstFixedSurface": , "level": ,"typeOfSecondFixedSurface": ,"tablesVersion": 15, "productDefinitionTemplateNumber" :}
         GribID['NEBBAS'] = {"discipline" : 0,"parameterCategory" : 6, "parameterNumber" : 3,"typeOfFirstFixedSurface": 1, "level": 0,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}
         GribID['NEBMOY'] = {"discipline" : 0,"parameterCategory" : 6, "parameterNumber" : 4,"typeOfFirstFixedSurface": 1, "level": 0,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}
         GribID['NEBHAUT'] = {"discipline" : 0,"parameterCategory" : 6, "parameterNumber" : 5,"typeOfFirstFixedSurface": 1, "level": 0,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}
         GribID['B_NUAGE'] = {"discipline" : 0,"parameterCategory" : 6, "parameterNumber" : 11,"typeOfFirstFixedSurface": 1, "level": 0,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}
         GribID['CC10'] = {"discipline" : 0,"parameterCategory" : 6, "parameterNumber" : 32,"typeOfFirstFixedSurface": 103, "level": 10,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}
         GribID['CC20'] = {"discipline" : 0,"parameterCategory" : 6, "parameterNumber" : 32,"typeOfFirstFixedSurface": 103, "level": 20,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}
         GribID['VISI'] = {"discipline" : 0,"parameterCategory" : 19, "parameterNumber" : 0,"typeOfFirstFixedSurface": 1, "level": 0,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 11, "typeOfGeneratingProcess" : 4}

         GribID['T850'] = {"discipline" : 0,"parameterCategory" : 0, "parameterNumber" : 0,"typeOfFirstFixedSurface": 100, "level": 850,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}
         GribID['T700'] = {"discipline" : 0,"parameterCategory" : 0, "parameterNumber" : 0,"typeOfFirstFixedSurface": 100, "level": 700,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}
         GribID['T500'] = {"discipline" : 0,"parameterCategory" : 0, "parameterNumber" : 0,"typeOfFirstFixedSurface": 100, "level": 500,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}
         GribID['T300'] = {"discipline" : 0,"parameterCategory" : 0, "parameterNumber" : 0,"typeOfFirstFixedSurface": 100, "level": 300,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}
         GribID['T200'] = {"discipline" : 0,"parameterCategory" : 0, "parameterNumber" : 0,"typeOfFirstFixedSurface": 100, "level": 200,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}

         GribID['HU850'] = {"discipline" : 0,"parameterCategory" : 1, "parameterNumber" : 1,"typeOfFirstFixedSurface": 100, "level": 850,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}
         GribID['HU700'] = {"discipline" : 0,"parameterCategory" : 1, "parameterNumber" : 1,"typeOfFirstFixedSurface": 100, "level": 700,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}
         GribID['HU500'] = {"discipline" : 0,"parameterCategory" : 1, "parameterNumber" : 1,"typeOfFirstFixedSurface": 100, "level": 500,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}
         GribID['HU300'] = {"discipline" : 0,"parameterCategory" : 1, "parameterNumber" : 1,"typeOfFirstFixedSurface": 100, "level": 300,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}
         GribID['HU200'] = {"discipline" : 0,"parameterCategory" : 1, "parameterNumber" : 1,"typeOfFirstFixedSurface": 100, "level": 200,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}

         GribID['U850'] = {"discipline" : 0,"parameterCategory" : 2, "parameterNumber" : 2,"typeOfFirstFixedSurface": 100, "level": 850,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}
         GribID['U700'] = {"discipline" : 0,"parameterCategory" : 2, "parameterNumber" : 2,"typeOfFirstFixedSurface": 100, "level": 700,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}
         GribID['U500'] = {"discipline" : 0,"parameterCategory" : 2, "parameterNumber" : 2,"typeOfFirstFixedSurface": 100, "level": 500,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}
         GribID['U300'] = {"discipline" : 0,"parameterCategory" : 2, "parameterNumber" : 2,"typeOfFirstFixedSurface": 100, "level": 300,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}
         GribID['U200'] = {"discipline" : 0,"parameterCategory" : 2, "parameterNumber" : 2,"typeOfFirstFixedSurface": 100, "level": 200,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}

         GribID['V850'] = {"discipline" : 0,"parameterCategory" : 2, "parameterNumber" : 3,"typeOfFirstFixedSurface": 100, "level": 850,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}
         GribID['V700'] = {"discipline" : 0,"parameterCategory" : 2, "parameterNumber" : 3,"typeOfFirstFixedSurface": 100, "level": 700,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}
         GribID['V500'] = {"discipline" : 0,"parameterCategory" : 2, "parameterNumber" : 3,"typeOfFirstFixedSurface": 100, "level": 500,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}
         GribID['V300'] = {"discipline" : 0,"parameterCategory" : 2, "parameterNumber" : 3,"typeOfFirstFixedSurface": 100, "level": 300,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}
         GribID['V200'] = {"discipline" : 0,"parameterCategory" : 2, "parameterNumber" : 3,"typeOfFirstFixedSurface": 100, "level": 200,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}

         GribID['W850'] = {"discipline" : 0,"parameterCategory" : 2, "parameterNumber" : 8,"typeOfFirstFixedSurface": 100, "level": 850,"typeOfSecondFixedSurface": 255, "tablesVersion": 15,"productDefinitionTemplateNumber" : 1}
         GribID['W700'] = {"discipline" : 0,"parameterCategory" : 2, "parameterNumber" : 8,"typeOfFirstFixedSurface": 100, "level": 700,"typeOfSecondFixedSurface": 255, "tablesVersion": 15,"productDefinitionTemplateNumber" : 1}
         GribID['W500'] = {"discipline" : 0,"parameterCategory" : 2, "parameterNumber" : 8,"typeOfFirstFixedSurface": 100, "level": 500,"typeOfSecondFixedSurface": 255, "tablesVersion": 15,"productDefinitionTemplateNumber" : 1}
         GribID['W300'] = {"discipline" : 0,"parameterCategory" : 2, "parameterNumber" : 8,"typeOfFirstFixedSurface": 100, "level": 300,"typeOfSecondFixedSurface": 255, "tablesVersion": 15,"productDefinitionTemplateNumber" : 1}
         GribID['W200'] = {"discipline" : 0,"parameterCategory" : 2, "parameterNumber" : 8,"typeOfFirstFixedSurface": 100, "level": 200,"typeOfSecondFixedSurface": 255, "tablesVersion": 15,"productDefinitionTemplateNumber" : 1}
         
         GribID['P2K'] = {"discipline" : 0,"parameterCategory" : 3, "parameterNumber" : 0,"typeOfFirstFixedSurface": 103, "level": 2000,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}
         GribID['P2K25'] = {"discipline" : 0,"parameterCategory" : 3, "parameterNumber" : 0,"typeOfFirstFixedSurface": 103, "level": 2250,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}
         GribID['P2K5'] = {"discipline" : 0,"parameterCategory" : 3, "parameterNumber" : 0,"typeOfFirstFixedSurface": 103, "level": 2500,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}
         GribID['P2K75'] = {"discipline" : 0,"parameterCategory" : 3, "parameterNumber" : 0,"typeOfFirstFixedSurface": 103, "level": 2750,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}
         GribID['P3K'] = {"discipline" : 0,"parameterCategory" : 3, "parameterNumber" : 0,"typeOfFirstFixedSurface": 103, "level": 3000,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}

         GribID['TKE1K'] = {"discipline" : 0,"parameterCategory" : 19, "parameterNumber" : 11,"typeOfFirstFixedSurface": 103, "level": 1000,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}
         GribID['TKE1K5'] = {"discipline" : 0,"parameterCategory" : 19, "parameterNumber" : 11,"typeOfFirstFixedSurface": 103, "level": 1500,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}
         GribID['TKE2K'] = {"discipline" : 0,"parameterCategory" : 19, "parameterNumber" : 11,"typeOfFirstFixedSurface": 103, "level": 2000,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}
         GribID['TKE2K5'] = {"discipline" : 0,"parameterCategory" : 19, "parameterNumber" : 11,"typeOfFirstFixedSurface": 103, "level": 2500,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}
         GribID['TKE3K'] = {"discipline" : 0,"parameterCategory" : 19, "parameterNumber" : 11,"typeOfFirstFixedSurface": 103, "level": 3000,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 1}
         ###--GribID['NEBCON'] = {"discipline" 0: ,"parameterCategory" : 6, "parameterNumber" : 2,"typeOfFirstFixedSurface": , "level": ,"typeOfSecondFixedSurface": ,"tablesVersion": 15, "productDefinitionTemplateNumber" :}
    elif suite=="H3Z3":
         GribID['H_COULIM'] = {"discipline" : 0,"parameterCategory" : 3, "parameterNumber" : 18,"typeOfFirstFixedSurface": 1, "scaledValueOfFirstFixedSurface": 0, "typeOfSecondFixedSurface":255, "tablesVersion": 15, "productDefinitionTemplateNumber": 0}
         GribID['SRD'] = {"discipline" : 0,"parameterCategory" : 4, "parameterNumber" : 7,"typeOfFirstFixedSurface": 1, "tablesVersion": 15, "productDefinitionTemplateNumber" : 8}
         GribID['NEBUL'] = {"discipline" : 0,"parameterCategory" : 6, "parameterNumber" : 1,"typeOfFirstFixedSurface": 1, "typeOfSecondFixedSurface": 255, "tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         #GribID['NEBCON'] = {"discipline" 0: ,"parameterCategory" : 6, "parameterNumber" : 2,"typeOfFirstFixedSurface": , "level": ,"typeOfSecondFixedSurface": ,"tablesVersion": 15, "productDefinitionTemplateNumber" :}
         GribID['NEBBAS'] = {"discipline" : 0,"parameterCategory" : 6, "parameterNumber" : 3,"typeOfFirstFixedSurface": 1, "level": 0,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['NEBMOY'] = {"discipline" : 0,"parameterCategory" : 6, "parameterNumber" : 4,"typeOfFirstFixedSurface": 1, "level": 0,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['NEBHAUT'] = {"discipline" : 0,"parameterCategory" : 6, "parameterNumber" : 5,"typeOfFirstFixedSurface": 1, "level": 0,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['B_NUAGE'] = {"discipline" : 0,"parameterCategory" : 6, "parameterNumber" : 11,"typeOfFirstFixedSurface": 1, "level": 0,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}

         GribID['CC10'] = {"discipline" : 0,"parameterCategory" : 6, "parameterNumber" : 32,"typeOfFirstFixedSurface": 103, "level": 10,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['CC20'] = {"discipline" : 0,"parameterCategory" : 6, "parameterNumber" : 32,"typeOfFirstFixedSurface": 103, "level": 20,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['VISI'] = {"discipline" : 0,"parameterCategory" : 19, "parameterNumber" : 0,"typeOfFirstFixedSurface": 1, "level": 0,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 8, "typeOfGeneratingProcess" : 2}

         GribID['T850'] = {"discipline" : 0,"parameterCategory" : 0, "parameterNumber" : 0,"typeOfFirstFixedSurface": 100, "level": 850,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['T700'] = {"discipline" : 0,"parameterCategory" : 0, "parameterNumber" : 0,"typeOfFirstFixedSurface": 100, "level": 700,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['T500'] = {"discipline" : 0,"parameterCategory" : 0, "parameterNumber" : 0,"typeOfFirstFixedSurface": 100, "level": 500,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['T300'] = {"discipline" : 0,"parameterCategory" : 0, "parameterNumber" : 0,"typeOfFirstFixedSurface": 100, "level": 300,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['T200'] = {"discipline" : 0,"parameterCategory" : 0, "parameterNumber" : 0,"typeOfFirstFixedSurface": 100, "level": 200,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}

         GribID['HU850'] = {"discipline" : 0,"parameterCategory" : 1, "parameterNumber" : 1,"typeOfFirstFixedSurface": 100, "level": 850,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['HU700'] = {"discipline" : 0,"parameterCategory" : 1, "parameterNumber" : 1,"typeOfFirstFixedSurface": 100, "level": 700,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['HU500'] = {"discipline" : 0,"parameterCategory" : 1, "parameterNumber" : 1,"typeOfFirstFixedSurface": 100, "level": 500,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['HU300'] = {"discipline" : 0,"parameterCategory" : 1, "parameterNumber" : 1,"typeOfFirstFixedSurface": 100, "level": 300,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['HU200'] = {"discipline" : 0,"parameterCategory" : 1, "parameterNumber" : 1,"typeOfFirstFixedSurface": 100, "level": 200,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}

         GribID['U850'] = {"discipline" : 0,"parameterCategory" : 2, "parameterNumber" : 2,"typeOfFirstFixedSurface": 100, "level": 850,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['U700'] = {"discipline" : 0,"parameterCategory" : 2, "parameterNumber" : 2,"typeOfFirstFixedSurface": 100, "level": 700,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['U500'] = {"discipline" : 0,"parameterCategory" : 2, "parameterNumber" : 2,"typeOfFirstFixedSurface": 100, "level": 500,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['U300'] = {"discipline" : 0,"parameterCategory" : 2, "parameterNumber" : 2,"typeOfFirstFixedSurface": 100, "level": 300,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['U200'] = {"discipline" : 0,"parameterCategory" : 2, "parameterNumber" : 2,"typeOfFirstFixedSurface": 100, "level": 200,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}

         GribID['V850'] = {"discipline" : 0,"parameterCategory" : 2, "parameterNumber" : 3,"typeOfFirstFixedSurface": 100, "level": 850,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['V700'] = {"discipline" : 0,"parameterCategory" : 2, "parameterNumber" : 3,"typeOfFirstFixedSurface": 100, "level": 700,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['V500'] = {"discipline" : 0,"parameterCategory" : 2, "parameterNumber" : 3,"typeOfFirstFixedSurface": 100, "level": 500,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['V300'] = {"discipline" : 0,"parameterCategory" : 2, "parameterNumber" : 3,"typeOfFirstFixedSurface": 100, "level": 300,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['V200'] = {"discipline" : 0,"parameterCategory" : 2, "parameterNumber" : 3,"typeOfFirstFixedSurface": 100, "level": 200,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}

         GribID['W850'] = {"discipline" : 0,"parameterCategory" : 2, "parameterNumber" : 8,"typeOfFirstFixedSurface": 100, "level": 850,"typeOfSecondFixedSurface": 255, "tablesVersion": 15,"productDefinitionTemplateNumber" : 0}
         GribID['W700'] = {"discipline" : 0,"parameterCategory" : 2, "parameterNumber" : 8,"typeOfFirstFixedSurface": 100, "level": 700,"typeOfSecondFixedSurface": 255, "tablesVersion": 15,"productDefinitionTemplateNumber" : 0}
         GribID['W500'] = {"discipline" : 0,"parameterCategory" : 2, "parameterNumber" : 8,"typeOfFirstFixedSurface": 100, "level": 500,"typeOfSecondFixedSurface": 255, "tablesVersion": 15,"productDefinitionTemplateNumber" : 0}
         GribID['W300'] = {"discipline" : 0,"parameterCategory" : 2, "parameterNumber" : 8,"typeOfFirstFixedSurface": 100, "level": 300,"typeOfSecondFixedSurface": 255, "tablesVersion": 15,"productDefinitionTemplateNumber" : 0}
         GribID['W200'] = {"discipline" : 0,"parameterCategory" : 2, "parameterNumber" : 8,"typeOfFirstFixedSurface": 100, "level": 200,"typeOfSecondFixedSurface": 255, "tablesVersion": 15,"productDefinitionTemplateNumber" : 0}
         
         GribID['P2K'] = {"discipline" : 0,"parameterCategory" : 3, "parameterNumber" : 0,"typeOfFirstFixedSurface": 103, "level": 2000,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['P2K25'] = {"discipline" : 0,"parameterCategory" : 3, "parameterNumber" : 0,"typeOfFirstFixedSurface": 103, "level": 2250,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['P2K5'] = {"discipline" : 0,"parameterCategory" : 3, "parameterNumber" : 0,"typeOfFirstFixedSurface": 103, "level": 2500,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['P2K75'] = {"discipline" : 0,"parameterCategory" : 3, "parameterNumber" : 0,"typeOfFirstFixedSurface": 103, "level": 2750,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['P3K'] = {"discipline" : 0,"parameterCategory" : 3, "parameterNumber" : 0,"typeOfFirstFixedSurface": 103, "level": 3000,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}

         GribID['TKE1K'] = {"discipline" : 0,"parameterCategory" : 19, "parameterNumber" : 11,"typeOfFirstFixedSurface": 103, "level": 1000,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['TKE1K5'] = {"discipline" : 0,"parameterCategory" : 19, "parameterNumber" : 11,"typeOfFirstFixedSurface": 103, "level": 1500,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['TKE2K'] = {"discipline" : 0,"parameterCategory" : 19, "parameterNumber" : 11,"typeOfFirstFixedSurface": 103, "level": 2000,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['TKE2K5'] = {"discipline" : 0,"parameterCategory" : 19, "parameterNumber" : 11,"typeOfFirstFixedSurface": 103, "level": 2500,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['TKE3K'] = {"discipline" : 0,"parameterCategory" : 19, "parameterNumber" : 11,"typeOfFirstFixedSurface": 103, "level": 3000,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         ###--GribID['NEBCON'] = {"discipline" 0: ,"parameterCategory" : 6, "parameterNumber" : 2,"typeOfFirstFixedSurface": , "level": ,"typeOfSecondFixedSurface": ,"tablesVersion": 15, "productDefinitionTemplateNumber" :}
    elif suite=="GYOP":
         GribID['H_COULIM'] = {"discipline" : 0,"parameterCategory" : 3, "parameterNumber" : 18,"typeOfFirstFixedSurface": 1, "level": 0,"typeOfSecondFixedSurface": 255, "tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['SRD'] = {"discipline" : 0,"parameterCategory" : 4, "parameterNumber" : 7,"typeOfFirstFixedSurface": 1, "level": 0,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 8, "typeOfStatisticalProcessing" : 1}
         GribID['NEBUL'] = {"discipline" : 0,"parameterCategory" : 6, "parameterNumber" : 1,"typeOfFirstFixedSurface": 1, "level": 0,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         #GribID['NEBCON'] = {"discipline" 0: ,"parameterCategory" : 6, "parameterNumber" : 2,"typeOfFirstFixedSurface": , "level": ,"typeOfSecondFixedSurface": ,"tablesVersion": 15, "productDefinitionTemplateNumber" :}
         GribID['NEBBAS'] = {"discipline" : 0,"parameterCategory" : 6, "parameterNumber" : 3,"typeOfFirstFixedSurface": 1, "level": 0,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['NEBMOY'] = {"discipline" : 0,"parameterCategory" : 6, "parameterNumber" : 4,"typeOfFirstFixedSurface": 1, "level": 0,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['NEBHAUT'] = {"discipline" : 0,"parameterCategory" : 6, "parameterNumber" : 5,"typeOfFirstFixedSurface": 1, "level": 0,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['B_NUAGE'] = {"discipline" : 0,"parameterCategory" : 6, "parameterNumber" : 11,"typeOfFirstFixedSurface": 1, "level": 0,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['CC10'] = {"discipline" : 0,"parameterCategory" : 6, "parameterNumber" : 32,"typeOfFirstFixedSurface": 103, "level": 10,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['CC20'] = {"discipline" : 0,"parameterCategory" : 6, "parameterNumber" : 32,"typeOfFirstFixedSurface": 103, "level": 20,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 0}
         GribID['VISI'] = {"discipline" : 0,"parameterCategory" : 19, "parameterNumber" : 0,"typeOfFirstFixedSurface": 1, "level": 0,"typeOfSecondFixedSurface": 255,"tablesVersion": 15, "productDefinitionTemplateNumber" : 8, "typeOfStatisticalProcessing" :  3}

         #GribID['T850'] = {"discipline" : ,"parameterCategory" : , "parameterNumber" : ,"typeOfFirstFixedSurface": , "level": ,"typeOfSecondFixedSurface": ,"tablesVersion": 15, "productDefinitionTemplateNumber" : }
         GribID['T700'] = 0
         GribID['T500'] = 0
         GribID['T300'] = 0
         GribID['T200'] = 0

         #GribID['HU850'] = {"discipline" : ,"parameterCategory" : , "parameterNumber" : ,"typeOfFirstFixedSurface": , "level": ,"typeOfSecondFixedSurface": ,"tablesVersion": 15, "productDefinitionTemplateNumber" : }
         GribID['HU700'] = 0
         GribID['HU500'] = 0
         GribID['HU300'] = 0
         GribID['HU200'] = 0

         #GribID['U850'] = {"discipline" : ,"parameterCategory" : , "parameterNumber" : ,"typeOfFirstFixedSurface": , "level": ,"typeOfSecondFixedSurface": ,"tablesVersion": 15, "productDefinitionTemplateNumber" : }
         GribID['U700'] = 0
         GribID['U500'] = 0
         GribID['U300'] = 0
         GribID['U200'] = 0

         #GribID['V850'] = {"discipline" : ,"parameterCategory" : , "parameterNumber" : ,"typeOfFirstFixedSurface": , "level": ,"typeOfSecondFixedSurface": ,"tablesVersion": 15, "productDefinitionTemplateNumber" : }
         GribID['V700'] = 0
         GribID['V500'] = 0
         GribID['V300'] = 0
         GribID['V200'] = 0

         #GribID['W850'] = {"discipline" : ,"parameterCategory" : , "parameterNumber" : ,"typeOfFirstFixedSurface": , "level": ,"typeOfSecondFixedSurface": , "tablesVersion": 15,"productDefinitionTemplateNumber" : }
         GribID['W700'] = 0
         GribID['W500'] = 0
         GribID['W300'] = 0
         GribID['W200'] = 0
         
         #GribID['P2K'] = {"discipline" : ,"parameterCategory" : , "parameterNumber" : ,"typeOfFirstFixedSurface": , "level": ,"typeOfSecondFixedSurface": ,"tablesVersion": 15, "productDefinitionTemplateNumber" : }
         GribID['P2K25'] = 0
         GribID['P2K5'] = 0
         GribID['P2K75'] =0 
         GribID['P3K'] = 0

         #GribID['TKE2K'] = {"discipline" : ,"parameterCategory" : , "parameterNumber" : ,"typeOfFirstFixedSurface": , "level": ,"typeOfSecondFixedSurface": ,"tablesVersion": 15, "productDefinitionTemplateNumber" : }
         GribID['TKE2K25'] = 0
         GribID['TKE2K5'] = 0
         GribID['TKE2K75'] = 0
         GribID['TKE3K'] = 0
    else:
        print("suite is not defined; check the name of the suite")
  
    
    crs=None
    print(reseaux, forecast_time_hr, N_mem)
    tic = time.time(); 
    fields_l = epygram_param(forecast_time_hr, suite, reseaux, mb, cutoff, vconf)  
    print(len(fields_l))
    
    fields_param_var = []
    for var_n in param_var:
        fields_param_var.append( epygram_readfield_V(fields_l, GribID[var_n]) )
    

    ##--------REMOVING THE shouldfly-* files of vortex from the working directory-----
    cmd_should = 'rm -rf shouldfly-*'
    os.system(cmd_should)

    toc = time.time();
    print('Reading PEAROME data is done in {:.4f} seconds'.format(toc-tic));    
    
    return fields_param_var


