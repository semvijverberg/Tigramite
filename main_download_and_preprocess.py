#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 17:48:31 2018

@author: semvijverberg
"""
import os
os.chdir('/Users/semvijverberg/surfdrive/Scripts/Tigramite')
script_dir = os.getcwd()
import numpy as np
import pandas as pd
import functions_pp
import matplotlib.pyplot as plt
from datetime import datetime
import xarray as xr
import cartopy.crs as ccrs

retrieve_ERA_i_field = functions_pp.retrieve_ERA_i_field
Variable = functions_pp.Variable
import_array = functions_pp.import_array

# *****************************************************************************
# *****************************************************************************
# Part 1 Downloading, preprocessing, choosing general experiment settings
# *****************************************************************************
# *****************************************************************************
# If you already have your ncdfs skip to step 2

# this will be your basepath, all raw_input and output will stored in subfolder 
# which will be made when running the code
base_path = "/Users/semvijverberg/surfdrive/Data_ERAint/"
path_raw = os.path.join(base_path, 'input_raw')
if os.path.isdir(path_raw) == False : os.makedirs(path_raw) 
# *****************************************************************************
# Step 1 Download netcdf
# *****************************************************************************
# Information needed to download ncdf from ECMWF Public Datasets.
# See http://apps.ecmwf.int/datasets/. 

exp = dict(
     {'dataset'     :       'ERA-i',
     'grid_res'     :       2.5,
     'startyear'    :       1979,
     'endyear'      :       2017,
     #'vars'        :       [['name_RV','name_actor'],['ECMWF_var_codes'],['ECMWF levtypes'], ['vertical levels']]
     'vars'         :       [['t2m', 'sst'],['167.128', '34.128'],['sfc', 'sfc'],[0, 0]],
     'base_path'    :       base_path}
     )

# Information needed to pre-process, select temporal frequency of data (must 
# be an even number, otherwise you will split one day in half when taking 
# temporal average), see comments in functions for more details.
exp['tfreq'] = 14
# the s(elect)startdate and enddate refer to the period/season you want to investigate.
exp['sstartdate'] = '{}-5-1 09:00:00'.format(exp['startyear'])
exp['senddate']   = '{}-8-31 09:00:00'.format(exp['startyear'])
exp['exp_pp'] = '{}_m{}-{}_dt{}'.format("_".join(exp['vars'][0]), 
                exp['sstartdate'].split('-')[1], exp['senddate'].split('-')[1], exp['tfreq'])
exp['path_exp_pp'] = os.path.join(base_path, exp['exp_pp'])
if os.path.isdir(exp['path_exp_pp']) == False : os.makedirs(exp['path_exp_pp'])
#%%
# assign instance
for idx in range(len(exp['vars'][0]))[:]:
    # =============================================================================
    # Make a class for each variable, this class contains variable specific information,
    # needed to download and post process the data. Along the way, information is added 
    # to class based on decisions that you make. 
    # =============================================================================
    var_class = Variable(exp, name=exp['vars'][0][idx], var_cf_code=exp['vars'][1][idx], 
                         levtype=exp['vars'][2][idx], lvllist=exp['vars'][3][idx], stream='oper') 
    exp[exp['vars'][0][idx]] = var_class
    # =============================================================================
    # Downloading data from Era-interim,     
    # =============================================================================
    retrieve_ERA_i_field(var_class)
    # =============================================================================
    # If you have already have your netcdfs, then comment retrieve_ERA_i_field and
    # rename them according to format:
    # 'varname'_'startyear'_'endyear'_1_12_'daily'_'grid_res'deg.nc
    # it should be a daily mean time series. save the ncdfs in subfolder 'input_raw'
# *****************************************************************************
# Step 2 Preprocess data (this function uses cdo and nco)
# *****************************************************************************
    # First time: Read Docstring by typing 'functions_pp.preprocessing_ncdf?' in console
    # Strongly recommended, see the function description to read what is does.
    functions_pp.preprocessing_ncdf(var_class, exp)
# *****************************************************************************
# Step 3 Select Response Variable period (which period of the year you want to predict)
# *****************************************************************************
# RV should always be the first variable of the vars list in exp
RV = exp[exp['vars'][0][0]]
marray, RV = functions_pp.import_array(RV, path='pp')
one_year = RV.dates_np.where(RV.dates_np.year == RV.startyear+1).dropna()
months = [7,8] # Selecting the timesteps of 14 day mean ts that fall in juli and august
RV_period = []
for mon in months:
    # append the indices of each year corresponding to your RV period
    RV_period.insert(-1, np.where(RV.dates_np.month == mon)[0] )
RV_period = [x for sublist in RV_period for x in sublist]
RV_period.sort()
exp['RV_period'] = RV_period
exp['RV_oneyr'] = RV.dates_np[RV_period].where(RV.dates_np[RV_period].year == RV.startyear+1).dropna()
# *****************************************************************************
# Step 4 Select spatial mask for the Response Variable (e.g. a SREX region)
# *****************************************************************************
exp_mask_region = '13Jul-24Aug_ward' # create this subfolder in 
exp['path_exp_mask_region'] = os.path.join('/Users/semvijverberg/surfdrive/Data_ERAint/t2m_sst_m5-8_dt14/', exp_mask_region)
cluster = 0
exp['clus_anom_std'] = 1
clusters = np.squeeze(xr.Dataset.from_dict(np.load(os.path.join(exp['path_exp_mask_region'], 'clusters_dic.npy')).item()).to_array())
cluster_out = clusters.sel(cluster=cluster)
exp['RV_masked'] = np.ma.masked_where((cluster_out < exp['clus_anom_std']*cluster_out.std()), cluster_out).mask
#%% 
# *****************************************************************************
# *****************************************************************************
# Part 2 Configure RGCPD/Tigramite settings
# *****************************************************************************
# *****************************************************************************

exp['alpha'] = 0.01 # set significnace level for correlation maps
exp['alpha_fdr'] = 2*exp['alpha'] # conservative significance level
exp['FDR_control'] = False # Do you want to use the conservative alpha_fdr or normal alpha?
exp['lag_min'] = 3 # Lag time(s) of interest
exp['lag_max'] = 4 
# If your pp data is not a full year, there is Maximum meaningful lag given by: 
#exp['lag_max'] = dates[dates.year == 1979].size - exp['RV_oneyr'].size
exp['alpha_level_tig'] = 0.2 # Alpha level for final regression analysis by Tigrimate
exp['la_min'] = -89 # select domain of correlation analysis
exp['la_max'] = 89
exp['lo_min'] = -180
exp['lo_max'] = 360
exp['pcA_sets'] = dict({   # dict of sets of pc_alpha values
      'pcA_set1a' : [ 0.05], # 0.05 0.01 
      'pcA_set1b' : [ 0.01], # 0.05 0.01 
      'pcA_set1c' : [ 0.1], # 0.05 0.01 
      'pcA_set2'  : [ 0.2, 0.1, 0.05, 0.01, 0.001], # set2
      'pcA_set3'  : [ 0.1, 0.05, 0.01], # set3
      'pcA_set4'  : [ 0.5, 0.4, 0.3, 0.2, 0.1], # set4
      'pcA_none'  : None # default
      })
exp['pcA_set'] = ['pcA_set1a'] 
exp['map_proj'] = ccrs.LambertCylindrical(central_longitude=int(cluster_out.longitude.mean()))
exp['path_output'] = os.path.join(exp['path_exp_mask_region'], 'output_tigr_SST_T2m/')
# =============================================================================
# Save Experiment design
# =============================================================================
filename_exp_design = os.path.join(exp['path_exp_mask_region'], 'input_tig_dic.npy')
np.save(filename_exp_design, exp)
#%%
# *****************************************************************************
# *****************************************************************************
# Part 3 Run RGCPD python script with settings
# *****************************************************************************
# *****************************************************************************
args = ['python {}'.format(os.path.join(script_dir, 'main_RGCPD_tig3.py')) +' '+ filename_exp_design]
functions_pp.kornshell_with_input(args)



#%%


#
## save to github
#import os
#import subprocess
#runfile = os.path.join(script_dir, 'saving_repository_to_Github.sh')
#subprocess.call(runfile)
#
#
#
#
## depricated
##%%
#idx = 0
#temperature = ( Variable(name='t2m', dataset='ERA-i', var_cf_code=exp['vars'][1][idx], 
#                   levtype=exp['vars'][2][idx], lvllist=exp['vars'][3][idx], startyear=exp['startyear'], 
#                   endyear=exp['endyear'], startmonth=exp['dstartmonth'], endmonth=exp['dendmonth'], 
#                   grid=exp['grid_res'], stream='oper') )
## Download variable to input_raw
#retrieve_ERA_i_field(temperature)
## preprocess variable to input_pp_exp'expnumber'
##functions_pp.preprocessing_ncdf(temperature, exp['grid_res'], exp['tfreq'], exp['exp'])
##marray, temperature = functions_pp.import_array(temperature, path='pp')
##marray
##%%
#idx = 1
#sst = ( Variable(name='sst', dataset='ERA-i', var_cf_code=exp['vars'][1][idx], 
#                   levtype=exp['vars'][2][idx], lvllist=exp['vars'][3][idx], startyear=exp['startyear'], 
#                   endyear=exp['endyear'], startmonth=exp['dstartmonth'], endmonth=exp['dendmonth'], 
#                   grid=exp['grid_res'], stream='oper') )
## Download variable
#retrieve_ERA_i_field(sst)
#functions_pp.preprocessing_ncdf(sst, exp['grid_res'], exp['tfreq'], exp['exp'])
#
#
##%%
#
## =============================================================================
## Simple example of cdo commands within python by calling bash script
## =============================================================================
#import functions_pp
#infile = os.path.join(var_class.base_path, 'input_raw', var_class.filename)
#outfile = os.path.join(var_class.base_path, 'input_pp', 'output.nc')
##tmp = os.path.join(temperature.base_path, 'input_raw', temperature.filename)
#args1 = 'cdo timmean {} {}'.format(infile, outfile)
##args2 = 'cdo setreftime,1900-01-01,0,1h -setcalendar,gregorian {} {} '.format(infile, outfile)
#args = [args1]
#
#functions_pp.kornshell_with_input(args)