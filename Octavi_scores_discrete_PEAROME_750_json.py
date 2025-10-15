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
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *

import numpy as np
import netCDF4 as nc
from netCDF4 import Dataset
import datetime as dt
from scipy.spatial import cKDTree
import matplotlib.colors as colors  
import epygram
from epygram.extra import usevortex as vtx
from bronx.stdtypes.date import daterangex as rangex
import pyproj
import pandas as pd
import h5py
import glob as glb

def point_plus_proche(LON, LAT, lon, lat):
	coord = list(zip(LAT.flatten(),LON.flatten()))
	tree = cKDTree(coord)
	d, i = tree.query((lat,lon), k = 1, n_jobs=-1)
	nlon = len(LON[0,:])
	nlat = len(LAT)
	i_lat = i//nlon
	i_lon = i%nlon
	return i_lat, i_lon, LAT[i_lat,i_lon], LON[i_lat,i_lon]

def fn_int(x):
    return int(x)

intV = np.vectorize(fn_int)

#--------BDClim file location--------
BDdir = '/home/gmap/mrmn/selvarajd/SAVE/BDClim_site/'
fle_site = 'BDClim_recup_station_9999_v2'
BD_dat = []
with open(BDdir+fle_site) as my_file: 
    fle_dat = my_file.readlines()
for l_dat in fle_dat:
    dat_str = l_dat.replace(' ',',')
    BD_dat.append([float(n) for n in dat_str.split(",")])

BD_dat = np.array(BD_dat)
BD_dat[:,0] = intV(BD_dat[:,0])

BD_data = pd.DataFrame(data=BD_dat, index=None, columns=['ID', 'lon', 'lat', 'Alt_m', 'Quality', 'M_mod', 'N_mod', 'M_sat', 'N_sat'])
BD_data['ID']=intV(BD_data['ID'].values)
BD_data['M_mod']=intV(BD_data['M_mod'].values)
BD_data['N_mod']=intV(BD_data['N_mod'].values)
BD_data['M_sat']=intV(BD_data['M_sat'].values)
BD_data['N_sat']=intV(BD_data['N_sat'].values)
BD_data['lon_mod']=0.0
BD_data['lat_mod']=0.0
BD_data['lon_sat']=0.0
BD_data['lat_sat']=0.0

del BD_dat, dat_str, fle_dat;

### ===========Satellite==============
Sat_lat_long = "/home/gmap/mrmn/selvarajd/SAVE/BDClim_site/latlon_+000.0.h5"
Sat_data = h5py.File(Sat_lat_long,'r')
Sat_lat = Sat_data['latitudes'][:]    
Sat_lon = Sat_data['longitudes'][:]    
Missing_Sat = np.where(BD_data['M_sat'].values == 9999)[0]
Mx_lat = 51; Mx_lat_id = np.where(Sat_lat<Mx_lat)
Mn_lat = 47; Mn_lat_id = np.where(Sat_lat<Mn_lat)
Mx_lon = 5; Mx_lon_id = np.where(Sat_lon>Mx_lon)
Mn_lon = 0; Mn_lon_id = np.where(Sat_lon>Mn_lon)

Mn_slt = Mx_lon_id[0][0]
Mx_slt = 1961
Sat_lat_slt = Sat_lat[ Mn_slt:Mx_slt, Mn_slt:Mx_slt ]
Sat_lon_slt = Sat_lon[ Mn_slt:Mx_slt, Mn_slt:Mx_slt ] 

### --------M & N from Satellite for the missing stations
for i_ID in range(len(BD_data['lon'])): #Missing_Sat:
	lon_BD, lat_BD = BD_data['lon'].values[i_ID], BD_data['lat'].values[i_ID]
	i_lat, i_lon, Sat_lat_slt[i_lat,i_lon], Sat_lon_slt[i_lat,i_lon] = point_plus_proche(Sat_lon_slt, Sat_lat_slt, lon_BD, lat_BD)
	print([i_lat, i_lon, Sat_lat_slt[i_lat,i_lon], lat_BD, Sat_lon_slt[i_lat,i_lon], lon_BD])

	BD_data['M_sat'].values[i_ID] = int(i_lat+Mn_slt)
	BD_data['N_sat'].values[i_ID] = int(i_lon+Mn_slt)
	BD_data['lat_sat'].values[i_ID] = Sat_lat[i_lat+Mn_slt,i_lon+Mn_slt]
	BD_data['lon_sat'].values[i_ID] = Sat_lon[i_lat+Mn_slt,i_lon+Mn_slt]
	del i_lat, i_lon, lon_BD, lat_BD;

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
#myDates = '2021102609-2021102615-PT1H' # Start-End-Step with YYYYMMDDHH format HH= reseau pour PEARO c'est 03 09 15 ou 21
#myDates = '2022010203-2022010321-PT6H'

myDates = '20240809T0300P' #-2021091321-PT24H'
suite   = "dble"           # On va chercher les fichiers de l'archive oper, suite="G6CN" : on va chercher les fichiers de l'experience de recherche G6CN
cutoff  = "production"
vconf   = "pefrance"
Nmem = 1
members = list(range(0,Nmem,1))  # les 16 membres PEARO
echeances = [18] #30           # echeance de la prevision, n'importe quelle valeur entre 0 et 45h est disponible (par pas de 1h)
param   = "ssrd" #"snow" #"wind" #"rr" "tpw850"  #Must be defined in GribID dict    ==> le parametre meteo demande
out_dir = "."
lag     = 12                       # duree cumul precip

reseaux = rangex(myDates)

  #------------------------------
  # 1.2 GRIB ID
  #------------------------------
###GRIB2 help : http://intra.cnrm.meteo.fr/gws/wtg/ --> Concept --> enter name and copy dict to clipboard
GribID = {}
GribID["rr"] = {'parameterNumber': 65}   #RAIN GRIB2
GribID["tpw850"] = {"discipline":0,"parameterCategory":0,"parameterNumber":3,"tablesVersion":15,"level":850}

GribID["2t"] = {"discipline" : 0,"parameterCategory": 0 , "parameterNumber" : 0, "level":2,"scaledValueOfFirstFixedSurface": 2,"typeOfFirstFixedSurface": 103,"productDefinitionTemplateNumber":1}

GribID['ssrd'] = {"discipline" : 0,"parameterCategory": 4 , "parameterNumber" : 7, "level":0,"typeOfFirstFixedSurface": 1,"typeOfSecondFixedSurface": 255,"productDefinitionTemplateNumber":11}

GribID["u"]={"discipline": 0, "parameterCategory" : 2, "parameterNumber" : 2, "scaledValueOfFirstFixedSurface" : 10, "typeOfFirstFixedSurface" :103, "level":10}
GribID["v"]={"discipline" : 0 , "parameterCategory" : 2, "parameterNumber": 3, "scaledValueOfFirstFixedSurface" : 10, "typeOfFirstFixedSurface" : 103, "level":10}

GribID["td2m"] = {"discipline":0,"parameterCategory":0,"parameterNumber":6,"typeOfFirstFixedSurface":103,"scaleFactorOfFirstFixedSurface":0,"scaledValueOfFirstFixedSurface":2}
GribID["tpw"]  = {"discipline":0,"parameterCategory":0,"parameterNumber":3}


GribID["td"] = {"discipline":0,"parameterCategory":0,"parameterNumber":6}
GribID["TV"] = {"discipline":0,"parameterCategory":0,"parameterNumber":1}
GribID["theta"] = {"discipline":0,"parameterCategory":0,"parameterNumber":2}


GribID["snow"] = {'parameterNumber': 66}   #RAIN GRIB2

  #------------------------------
  # 1.2 Colormaps
  #------------------------------
Colormaps = {}
Colormaps['rr'] = 'rr24h'
Colormaps['snow'] = 'rr1h'
Colormaps['tpw850'] = 'viridis'
Colormaps['t2m'] = 'viridis'
Colormaps['wind'] = 'viridis'
Colormaps['2t'] = 'cubehelix'
Colormaps['ssrd'] = 'cubehelix'

  #------------------------------
  # 1.3 Projection
  #------------------------------
crs=None

####################################
# 2. Plot members
####################################
  #------------------------------
  # 2.1 Get data for each reseau/each member
  #------------------------------
T_lim = intV( np.round(np.linspace(270,310,21)*100)/100 )
increaseFactor=1.2

for reseau in reseaux:
  for ech in echeances:
    tab=[]
    axs=[]
    #ii = [0,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
    #jj = [0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]
  
    for mb in members:
        resource = vtx.get_resources(experiment=suite, date=reseau, term=ech,\
                                        getmode='epygram',model='arome',origin='hst',\
                                        kind='gridpoint',block='forecast', cutoff=cutoff,\
                                        vapp='arome', vconf=vconf,geometry='EURW1S40',\
                                        namespace='vortex.archive.fr', nativefmt = 'grib',\
                                        member=mb, uselocalcache=False, shouldfly=False)
        
        field = resource[0].readfield(GribID[param])
        fieldAll =   field.extract_zoom(dict(lonmin=0.67,lonmax=4.02,latmin=47.75,latmax=49.95))
        fieldAll.geometry
        
  #------------------------------
  # 2.3 Finding the locations that match with BDClim lat and lon
  #------------------------------
        if mb==0:
            geod = pyproj.Geod(ellps='WGS84') 
            ##-------------------------All fields------------------------###
            fldAll_data = fieldAll.getdata()
            print(fldAll_data.shape[0], fldAll_data.shape[1]);
            fldAll_list = fieldAll.as_lists()
            lonA_list = fldAll_list['longitudes']
            latA_list = fldAll_list['latitudes']
            lonA = np.reshape(lonA_list, (fldAll_data.shape[0],fldAll_data.shape[1]) )
            latA = np.reshape(latA_list, (fldAll_data.shape[0],fldAll_data.shape[1]) )

            for i_ID in range(BD_data.shape[0]):
                lon_BD, lat_BD = BD_data['lon'].values[i_ID], BD_data['lat'].values[i_ID]
                ##-------------------M & N from PEAROME
                i_lat, i_lon, latA[i_lat,i_lon], lonA[i_lat,i_lon] = point_plus_proche(lonA, latA, lon_BD, lat_BD)
                BD_data['M_mod'].values[i_ID] = int(i_lat)
                BD_data['N_mod'].values[i_ID] = int(i_lon)
                BD_data['lat_mod'].values[i_ID] = latA[i_lat,i_lon]
                BD_data['lon_mod'].values[i_ID] = lonA[i_lat,i_lon]
                del i_lat, i_lon;

            BD_data.to_csv(BDdir+'BDClim_recup_station_zoom_dble_v2.csv', header=True, index=None, sep=' ', mode='a')
            BD_data.to_hdf(BDdir+'BDClim_recup_station_zoom_dble_v2.h5', key="cy48t1", mode="w")  
