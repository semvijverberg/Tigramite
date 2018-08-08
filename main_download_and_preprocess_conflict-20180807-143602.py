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
import_array = functions_pp.import_array
#%%
# *****************************************************************************
# *****************************************************************************
# Part 1 Downloading, preprocessing, choosing general experiment settings
# *****************************************************************************
# *****************************************************************************
# If you already have your ncdfs skip to step 2 of part 1

# this will be your basepath, all raw_input and output will stored in subfolder 
# which will be made when running the code
base_path = "/Users/semvijverberg/surfdrive/Data_ERAint/"
path_raw = os.path.join(base_path, 'input_raw')
path_pp  = os.path.join(base_path, 'input_pp')
if os.path.isdir(path_raw) == False : os.makedirs(path_raw) 
if os.path.isdir(path_pp) == False: os.makedirs(path_pp)
ECMWFdownload = True

# *****************************************************************************
# Step 1 Create dictionary and variable class (and optionally download ncdfs)
# *****************************************************************************
# The dictionary is used as a container with all information for the experiment
# The dic is saved at intermediate steps, so you can continue the experiment 
# from these break points. It also stored as a log in the final output.
exp = dict(
     {'dataset'     :       'ERA-i',
      'vars'        :       [['t2m', 'u']],     # ['name_RV','name_actor1', ...]
     'grid_res'     :       2.5,
     'startyear'    :       1979,
     'endyear'      :       2017,
     'base_path'    :       base_path,
     'path_raw'     :       path_raw,
     'path_pp'     :       path_pp}
     )
# =============================================================================
# Info to download ncdf from ECMWF, atm only analytical fields (no forecasts)
# =============================================================================
# You need the ecmwf-api-client package for this option.
if ECMWFdownload == True:
    # See http://apps.ecmwf.int/datasets/. 
    # 'vars'        :       [['t2m', 'sst'],['167.128','34.128'],['sfc', 'sfc'],[0, 0]],
    # 'vars'        :       [['t2m', 'u'],['167.128', '131.128'],['sfc', 'pl'],[0, '500']],
    # 'vars'        :       [['t2m', 'sst', 'u'],['167.128', '34.128', '131.128'],['sfc', 'sfc', 'pl'],[0, 0, '500']],
    exp['vars']     =   [
                        ['t2m', 'u'],              # ['name_RV','name_actor', ...]
                        ['167.128', '131.128'],    # ECMWF param ids
                        ['sfc', 'pl'],             # Levtypes
                        [0, 200],                  # Vertical levels
                        ]
else:
    exp['own_nc_names'] = ['hgt.200mb.daily.1979-2016.del29feb.nc',
                           'prcp_GLB_daily_1979-2016-del29feb.75-88E_18-25N.nc']
    
# =============================================================================
# Make a class for each variable, this class contains variable specific information,
# needed to download and post process the data. Along the way, information is added 
# to class based on decisions that you make. 
# =============================================================================
for idx in range(len(exp['vars'][0]))[:]:
    var_class = functions_pp.Variable(exp, idx, ECMWFdownload) 
    exp[exp['vars'][0][idx]] = var_class

# =============================================================================
# Downloading data from Era-interim?  
# =============================================================================
if ECMWFdownload == True:
    for var in exp['vars'][0]:
        var_class = exp[var]
        retrieve_ERA_i_field(var_class)

# *****************************************************************************
# Step 2 Preprocess data (this function uses cdo and nco)
# *****************************************************************************
# Information needed to pre-process, 
# Select temporal frequency:
exp['tfreq'] = 14
# s(elect)startdate and enddate create the period/season you want to investigate:
exp['sstartdate'] = '{}-5-1 09:00:00'.format(exp['startyear'])
exp['senddate']   = '{}-8-31 09:00:00'.format(exp['startyear'])
exp['exp_pp'] = '{}_m{}-{}_dt{}'.format("_".join(exp['vars'][0]), 
                exp['sstartdate'].split('-')[1], exp['senddate'].split('-')[1], exp['tfreq'])
exp['path_exp'] = os.path.join(base_path, exp['exp_pp'])
if os.path.isdir(exp['path_exp']) == False : os.makedirs(exp['path_exp'])
exp['mask'] = 'ward'

# =============================================================================
# Preprocess data (this function uses cdo/nco and requires execution rights of
# the created bash script)
# =============================================================================
# First time: Read Docstring by typing 'functions_pp.preprocessing_ncdf?' in console
# Solve permission error by giving bash script execution right, read Docstring
for var in exp['vars'][0]:
    var_class = exp[var]
    outfile, datesstr, var_class = functions_pp.datestr_for_preproc(var_class, exp)
    if os.path.isfile(outfile) == True: 
        print('looks like you already have done the pre-processing,\n'
              'to save time let\'s not do it twice..')
        pass
    else:    
        functions_pp.preprocessing_ncdf(outfile, datesstr, var_class, exp)

#%%  
# *****************************************************************************
# Step 3 Preprocess Response Variable (RV) 
# *****************************************************************************  
# =============================================================================
# 3.1 Select RV period (which period of the year you want to predict)
# =============================================================================
# RV should always be the first variable of the vars list in exp
RV = exp[exp['vars'][0][0]]
ncdfarray, RV = functions_pp.import_array(RV)
one_year = RV.dates.where(RV.dates.year == RV.startyear+1).dropna()
months = [7,8] # Selecting the timesteps of 14 day mean ts that fall in juli and august
RV_period = []
for mon in months:
    # append the indices of each year corresponding to your RV period
    RV_period.insert(-1, np.where(RV.dates.month == mon)[0] )
RV_period = [x for sublist in RV_period for x in sublist]
RV_period.sort()
exp['RV_period'] = RV_period
exp['RV_oneyr'] = RV.dates[RV_period].where(RV.dates[RV_period].year == RV.startyear+1).dropna()
RV_name_range = '{}{}-{}{}_'.format(exp['RV_oneyr'].min().day, exp['RV_oneyr'].min().month_name()[:3], 
                 exp['RV_oneyr'].max().day, exp['RV_oneyr'].max().month_name()[:3] )
# =============================================================================
# 3.2 Select spatial mask to create 1D timeseries (e.g. a SREX region)
# =============================================================================
# create this subfolder in exp['path_exp'] for RV_period and spatial mask and
# save your spatial mask in there
exp_mask_region = '13Jul-24Aug_' + exp['mask'] 
exp['path_exp_mask_region'] = os.path.join(exp['path_exp'], exp_mask_region)
if os.path.isdir(exp['path_exp_mask_region']) != True : os.makedirs(exp['path_exp_mask_region'])
# Load in clustering output
clusters = np.squeeze(xr.Dataset.from_dict(np.load(os.path.join(exp['path_exp_mask_region'], 'clusters_dic.npy')).item()).to_array())
# some settings
cluster = 0
exp['clus_anom_std'] = 1
# Retrieve mask
cluster_out = clusters.sel(cluster=cluster)
exp['RV_masked'] = np.ma.masked_where((cluster_out < exp['clus_anom_std']*cluster_out.std()), cluster_out).mask
# retrieve region used for clustering from ncdfarray
clusregarray, region_coords = functions_pp.find_region(ncdfarray, region='U.S.')
RV_region = np.ma.masked_array(data=clusregarray, 
                               mask=np.reshape(np.repeat(exp['RV_masked'], ncdfarray.time.size), 
                               clusregarray.shape))
RV.RV1D = np.ma.mean(RV_region, axis = (1,2)) # take spatial mean with mask loaded in beginning
RV.RV_ts = RV.RV1D[exp['RV_period']] # extract specific months of MT index 
# Store added information in RV class to the exp dictionary
exp[exp['vars'][0][0]] = RV
print('\n\t**\n\tOkay, end of Part 1!\n\t**' )
filename_exp_design1 = os.path.join(exp['path_exp_mask_region'], 'input_tig_dic_part_1.npy')
print('\nNext time, you can choose to start with part 2 by loading in '
      'part 1 settings from dictionary \n{}.'.format(filename_exp_design1))
np.save(filename_exp_design1, exp)
#%% 
# *****************************************************************************
# *****************************************************************************
# Part 2 Configure RGCPD/Tigramite settings
# *****************************************************************************
# *****************************************************************************
exp = np.load(filename_exp_design1).item()
exp['lag_min'] = 3 # Lag time(s) of interest
exp['lag_max'] = 4 
exp['alpha'] = 0.01 # set significnace level for correlation maps
exp['alpha_fdr'] = 2*exp['alpha'] # conservative significance level
exp['FDR_control'] = False # Do you want to use the conservative alpha_fdr or normal alpha?
# If your pp data is not a full year, there is Maximum meaningful lag given by: 
#exp['lag_max'] = dates[dates.year == 1979].size - exp['RV_oneyr'].size
exp['alpha_level_tig'] = 0.1 # Alpha level for final regression analysis by Tigrimate
exp['pcA_sets'] = dict({   # dict of sets of pc_alpha values
      'pcA_set1a' : [ 0.05], # 0.05 0.01 
      'pcA_set1b' : [ 0.01], # 0.05 0.01 
      'pcA_set1c' : [ 0.1], # 0.05 0.01 
      'pcA_set2'  : [ 0.2, 0.1, 0.05, 0.01, 0.001], # set2
      'pcA_set3'  : [ 0.1, 0.05, 0.01], # set3
      'pcA_set4'  : [ 0.5, 0.4, 0.3, 0.2, 0.1], # set4
      'pcA_none'  : None # default
      })
exp['pcA_set'] = 'pcA_set1a' 
exp['la_min'] = -89 # select domain of correlation analysis
exp['la_max'] = 89
exp['lo_min'] = -180
exp['lo_max'] = 360
# Some output settings
exp['file_type1'] = ".pdf"
exp['file_type2'] = ".png"
exp['includeRV'] = 1 # if 1, then RV is also considered as possible actor (autocorr)
exp['SaveTF'] = True # if false, output will be printed in console
exp['plotin1fig'] = True 
exp['showplot'] = True
central_lon_plots = int(cluster_out.longitude.mean())
map_proj = ccrs.LambertCylindrical(central_longitude=central_lon_plots)
# output paths
exp['path_output'] = os.path.join(exp['path_exp_mask_region'], 'output_tigr_SST_T2m/')
exp['fig_path'] = os.path.join(exp['path_output'], 'lag{}to{}/'.format(exp['lag_min'],exp['lag_max']))
exp['params_combination'] = 'aCorr{}'.format(exp['alpha'])
if os.path.isdir(exp['fig_path']) != True : os.makedirs(exp['fig_path'])
exp['fig_subpath'] = os.path.join(exp['fig_path'], '{}_{}_SIGN{}_subinfo/'.format(exp['params_combination'],
                                       exp['pcA_set'], exp['alpha_level_tig']))
if os.path.isdir(exp['fig_subpath']) != True : os.makedirs(exp['fig_subpath'])                                  
# =============================================================================
# Save Experiment design
# =============================================================================
filename_exp_design2 = os.path.join(exp['fig_subpath'], 'input_tig_dic_part_2.npy')
np.save(filename_exp_design2, exp)
print('\n\t**\n\tOkay, end of Part 2!\n\t**' )
#%%
# *****************************************************************************
# *****************************************************************************
# Part 3 Start your experiment by running RGCPD python script with settings
# *****************************************************************************
# *****************************************************************************
import main_RGCPD_tig3

exp, outdic_actors = main_RGCPD_tig3.calculate_corr_maps(filename_exp_design2, map_proj)
#%%
main_RGCPD_tig3.run_PCMCI(exp, outdic_actors, map_proj)

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
# =============================================================================
# How to run python script from python script:
# =============================================================================
#import subprocess
#script_path = os.path.join(script_dir, 'main_RGCPD_tig3.py')
#bash_and_args = ['python', script_path]
#p = subprocess.Popen(bash_and_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
#out = p.communicate(filename_exp_design2)
#print(out[0])