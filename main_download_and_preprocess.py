#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 17:48:31 2018

@author: semvijverberg
"""
import os, sys
os.chdir('/Users/semvijverberg/surfdrive/Scripts/Tigramite')
script_dir = os.getcwd()
import numpy as np
import pandas as pd
import functions_pp
import matplotlib.pyplot as plt
from datetime import datetime
import xarray as xr
import cartopy.crs as ccrs
#import pickle
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
# True if you want to download ncdfs through ECMWF MARS, only analytical fields 
ECMWFdownload = True
importRVts = False

# *****************************************************************************
# Step 1 Create dictionary and variable class (and optionally download ncdfs)
# *****************************************************************************
# The dictionary is used as a container with all information for the experiment
# The dic is saved at intermediate steps, so you can continue the experiment 
# from these break points. It also stored as a log in the final output.
ex = dict(
     {'dataset'     :       'ERA-i',
      'vars'        :       [['t2m', 'u']],     # ['name_RV','name_actor1', ...]
     'grid_res'     :       2.5,
     'startyear'    :       1979,
     'endyear'      :       2017,
     'base_path'    :       base_path,
     'path_raw'     :       path_raw,
     'path_pp'     :       path_pp}
     )

# own ncdfs must have same period, daily data and on same grid
ex['own_actor_nc_names'] = [[]]
ex['own_RV_nc_name'] = ['t2mmax', 't2mmax_1979-2017_1_12_daily_2.5deg.nc']

# =============================================================================
# Info to download ncdf from ECMWF, atm only analytical fields (no forecasts)
# =============================================================================
# You need the ecmwf-api-client package for this option.
if ECMWFdownload == True:
    # See http://apps.ecmwf.int/datasets/. 
    ex['vars']      =       [['t2m', 'sst'],['167.128','34.128'],['sfc', 'sfc'],[0, 0]]
#    ex['vars']      =       [['t2m', 'u'],['167.128', '131.128'],['sfc', 'pl'],[0, '500']]
#    ex['vars']      =       [['t2m', 'sst', 'u'],['167.128', '34.128', '131.128'],['sfc', 'sfc', 'pl'],[0, 0, '500']]
#    ex['vars']      =       [['t2m', 'sst', 'u', 't100'],
#                            ['167.128', '34.128', '131.128', '130.128'],
#                            ['sfc', 'sfc', 'pl', 'pl'],[0, 0, '500', '100']]
#    ex['vars']     =   [
#                        ['t2m', 'u'],              # ['name_RV','name_actor', ...]
#                        ['167.128', '131.128'],    # ECMWF param ids
#                        ['sfc', 'pl'],             # Levtypes
#                        [0, 200],                  # Vertical levels
#                        ]



# =============================================================================
# Make a class for each variable, this class contains variable specific information,
# needed to download and post process the data. Along the way, information is added 
# to class based on decisions that you make. 
# =============================================================================
if ECMWFdownload == True:
    for idx in range(len(ex['vars'][0]))[:]:
        # class for ECMWF downloads
        var_class = functions_pp.Variable(ex, idx, ECMWFdownload) 
        ex[ex['vars'][0][idx]] = var_class
if len(ex['own_actor_nc_names'][0]) != 0:
    print ex['own_actor_nc_names'][0][0]
    for idx in range(len(ex['own_actor_nc_names'])):
        ECMWFdownload = False
        var_class = functions_pp.Variable(ex, idx, ECMWFdownload) 
        ex[ex['vars'][0][idx]] = var_class
if len(ex['own_RV_nc_name']) != 0:
    ECMWFdownload = False
    var_class = functions_pp.Variable(ex, idx, ECMWFdownload) 
    ex[ex['vars'][0][0]] = var_class

# =============================================================================
# Downloading data from Era-interim?  
# =============================================================================
if ECMWFdownload == True:
    for var in ex['vars'][0]:
        var_class = ex[var]
        retrieve_ERA_i_field(var_class)

# *****************************************************************************
# Step 2 Preprocess data (this function uses cdo and nco)
# *****************************************************************************
# Information needed to pre-process, 
# Select temporal frequency:
ex['tfreq'] = 14
# s(elect)startdate and enddate create the period/season you want to investigate:
ex['sstartdate'] = '{}-6-1 09:00:00'.format(ex['startyear'])
ex['senddate']   = '{}-8-31 09:00:00'.format(ex['startyear'])
ex['exp_pp'] = '{}_m{}-{}_dt{}'.format("_".join(ex['vars'][0]), 
                ex['sstartdate'].split('-')[1], ex['senddate'].split('-')[1], ex['tfreq'])
ex['path_exp'] = os.path.join(base_path, ex['exp_pp'])
if os.path.isdir(ex['path_exp']) == False : os.makedirs(ex['path_exp'])
ex['mask'] = 'ward'

# =============================================================================
# Preprocess data (this function uses cdo/nco and requires execution rights of
# the created bash script)
# =============================================================================
# First time: Read Docstring by typing 'functions_pp.preprocessing_ncdf?' in console
# Solve permission error by giving bash script execution right, read Docstring
for var in ex['vars'][0]:
    var_class = ex[var]
    outfile, datesstr, var_class = functions_pp.datestr_for_preproc(var_class, ex)
    if os.path.isfile(outfile) == True: 
        print('looks like you already have done the pre-processing,\n'
              'to save time let\'s not do it twice..')
        pass
    else:    
        functions_pp.preprocessing_ncdf(outfile, datesstr, var_class, ex)
#%%  
# *****************************************************************************
# Step 3 Preprocess Response Variable (RV) 
# *****************************************************************************  
# =============================================================================
# 3.1 Select RV period (which period of the year you want to predict)
# =============================================================================
if importRVts == True:
    dicRV = np.load( os.path.join(ex['path_pp'],"T95ts_1982-2015_m6-8.pkl") ).item()
#    dicRV = pickle.load( open(os.path.join(ex['path_pp'],"T95ts_1982-2015_m6-8.pkl"), "rb") ) 
    class RV_seperateclass:
        dates = dicRV['dates']
    RV = RV_seperateclass()
    RV.startyear = RV.dates.year[0]
    RV.endyear = RV.dates.year[-1]
    if RV.startyear != ex['startyear']:
        print('make sure the dates of the RV match with the actors')

elif importRVts == False:
    # RV should always be the first variable of the vars list in ex
    RV = ex[ex['vars'][0][0]]
    ncdfarray, RV = functions_pp.import_array(RV)
    
one_year = RV.dates.where(RV.dates.year == RV.startyear+1).dropna()
months = [7,8] # Selecting the timesteps of 14 day mean ts that fall in juli and august
RV_period = []
for mon in months:
    # append the indices of each year corresponding to your RV period
    RV_period.insert(-1, np.where(RV.dates.month == mon)[0] )
RV_period = [x for sublist in RV_period for x in sublist]
RV_period.sort()
ex['RV_period'] = RV_period
ex['RV_oneyr'] = RV.dates[RV_period].where(RV.dates[RV_period].year == RV.startyear+1).dropna()
months = dict( {1:'jan',2:'feb',3:'mar',4:'apr',5:'may',6:'jun',
                7:'jul',8:'aug',9:'sep',10:'okt',11:'nov',12:'dec' } )
RV_name_range = '{}{}-{}{}_'.format(ex['RV_oneyr'].min().day, months[ex['RV_oneyr'].min().month], 
                 ex['RV_oneyr'].max().day, months[ex['RV_oneyr'].max().month] )

# =============================================================================
# 3.2 Select spatial mask to create 1D timeseries (e.g. a SREX region)
# =============================================================================
# create this subfolder in ex['path_exp'] for RV_period and spatial mask and
# ex['path_exp_periodmask'] = os.path.join(ex['path_exp'], exp_folder_periodmask)

ex['path_exp_periodmask'] = os.path.join(ex['path_exp'], '13Jul-24Aug_' + ex['mask'] )
if os.path.isdir(ex['path_exp_periodmask']) != True : os.makedirs(ex['path_exp_periodmask'])
if importRVts == True:
    RV.RVfullts = dicRV['RVfullts']
elif importRVts == False:
    # save your spatial mask in input_pp (ex['path_pp'])
    ex['path_exp_mask_region'] = ex['path_pp']
    # Load in clustering output
    try:
        clusters = np.squeeze(xr.Dataset.from_dict(np.load(os.path.join(ex['path_exp_mask_region'], 'clusters_dic.npy')).item()).to_array())
    except IOError, e:
        print('\n**\nSpatial mask not found.\n'
              'Place your spatial mask in folder: \n{}\n'
              'and rerun this section.\n**'.format(ex['path_exp_mask_region']))
        raise(e)
        
    # some settings
    cluster = 0
    ex['clus_anom_std'] = 1
    # Retrieve mask
    cluster_out = clusters.sel(cluster=cluster)
    ex['RV_masked'] = np.ma.masked_where((cluster_out < ex['clus_anom_std']*cluster_out.std()), cluster_out).mask
    # retrieve region used for clustering from ncdfarray
    clusregarray, region_coords = functions_pp.find_region(ncdfarray, region='U.S.')
    RV_region = np.ma.masked_array(data=clusregarray, 
                                   mask=np.reshape(np.repeat(ex['RV_masked'], ncdfarray.time.size), 
                                   clusregarray.shape))
    RV.RVfullts = np.ma.mean(RV_region, axis = (1,2)) # take spatial mean with mask loaded in beginning

RV.RV_ts = RV.RVfullts[ex['RV_period']] # extract specific months of MT index 
# Store added information in RV class to the exp dictionary
ex[ex['vars'][0][0]] = RV
print('\n\t**\n\tOkay, end of Part 1!\n\t**' )
filename_exp_design1 = os.path.join(ex['path_exp_periodmask'], 'input_dic_part_1.npy')
print('\nNext time, you can choose to start with part 2 by loading in '
      'part 1 settings from dictionary \n{}\n.'.format(filename_exp_design1))
np.save(filename_exp_design1, ex)
#%% 
# *****************************************************************************
# *****************************************************************************
# Part 2 Configure RGCPD/Tigramite settings
# *****************************************************************************
# *****************************************************************************
ex = np.load(filename_exp_design1).item()
ex['lag_min'] = 3 # Lag time(s) of interest
ex['lag_max'] = 4 
ex['alpha'] = 0.01 # set significnace level for correlation maps
ex['alpha_fdr'] = 2*ex['alpha'] # conservative significance level
ex['FDR_control'] = False # Do you want to use the conservative alpha_fdr or normal alpha?
# If your pp data is not a full year, there is Maximum meaningful lag given by: 
#ex['lag_max'] = dates[dates.year == 1979].size - ex['RV_oneyr'].size
ex['alpha_level_tig'] = 0.2 # Alpha level for final regression analysis by Tigrimate
ex['pcA_sets'] = dict({   # dict of sets of pc_alpha values
      'pcA_set1a' : [ 0.05], # 0.05 0.01 
      'pcA_set1b' : [ 0.01], # 0.05 0.01 
      'pcA_set1c' : [ 0.1], # 0.05 0.01 
      'pcA_set2'  : [ 0.2, 0.1, 0.05, 0.01, 0.001], # set2
      'pcA_set3'  : [ 0.1, 0.05, 0.01], # set3
      'pcA_set4'  : [ 0.5, 0.4, 0.3, 0.2, 0.1], # set4
      'pcA_none'  : None # default
      })
ex['pcA_set'] = 'pcA_set1a' 
ex['la_min'] = -89 # select domain of correlation analysis
ex['la_max'] = 89
ex['lo_min'] = -180
ex['lo_max'] = 360
# Some output settings
ex['file_type1'] = ".pdf"
ex['file_type2'] = ".png" 
ex['excludeRV'] = 0 # if 0, then RV is also considered as possible actor (autocorr)
ex['SaveTF'] = True # if false, output will be printed in console
ex['plotin1fig'] = False 
ex['showplot'] = True
central_lon_plots = int(cluster_out.longitude.mean())
map_proj = ccrs.LambertCylindrical(central_longitude=central_lon_plots)
# output paths
ex['path_output'] = os.path.join(ex['path_exp_periodmask'], 'output_tigr_SST_T2m/')
ex['fig_path'] = os.path.join(ex['path_output'], 'lag{}to{}/'.format(ex['lag_min'],ex['lag_max']))
ex['params'] = '{}_ac{}_at{}'.format(ex['pcA_set'], ex['alpha'],
                                                  ex['alpha_level_tig'])
if os.path.isdir(ex['fig_path']) != True : os.makedirs(ex['fig_path'])
ex['fig_subpath'] = os.path.join(ex['fig_path'], '{}_subinfo'.format(ex['params']))
if os.path.isdir(ex['fig_subpath']) != True : os.makedirs(ex['fig_subpath'])                                  
# =============================================================================
# Save Experiment design
# =============================================================================
filename_exp_design2 = os.path.join(ex['fig_subpath'], 'input_dic_{}.npy'.format(ex['params']))
np.save(filename_exp_design2, ex)
print('\n\t**\n\tOkay, end of Part 2!\n\t**' )
print('\nNext time, you\'re able to redo the experiment by loading in '
      'part 2 settings from dictionary \n{}.\n'.format(filename_exp_design2))
#%%
# *****************************************************************************
# *****************************************************************************
# Part 3 Start your experiment by running RGCPD python script with settings
# *****************************************************************************
# *****************************************************************************
import main_RGCPD_tig3
# =============================================================================
# Find precursor fields (potential precursors)
# =============================================================================
ex, outdic_actors = main_RGCPD_tig3.calculate_corr_maps(filename_exp_design2, map_proj)

#%% 
# =============================================================================
# Run tigramite to extract causal precursors
# =============================================================================
copy_stdout = sys.stdout
parents_RV, var_names = main_RGCPD_tig3.run_PCMCI(ex, outdic_actors, map_proj)
#%%
# =============================================================================
# Plot final results
# =============================================================================
main_RGCPD_tig3.plottingfunction(ex, parents_RV, var_names, outdic_actors, map_proj)
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
#temperature = ( Variable(name='t2m', dataset='ERA-i', var_cf_code=ex['vars'][1][idx], 
#                   levtype=ex['vars'][2][idx], lvllist=ex['vars'][3][idx], startyear=ex['startyear'], 
#                   endyear=ex['endyear'], startmonth=ex['dstartmonth'], endmonth=ex['dendmonth'], 
#                   grid=ex['grid_res'], stream='oper') )
## Download variable to input_raw
#retrieve_ERA_i_field(temperature)
## preprocess variable to input_pp_exp'expnumber'
##functions_pp.preprocessing_ncdf(temperature, ex['grid_res'], ex['tfreq'], ex['exp'])
##marray, temperature = functions_pp.import_array(temperature, path='pp')
##marray
##%%
#idx = 1
#sst = ( Variable(name='sst', dataset='ERA-i', var_cf_code=ex['vars'][1][idx], 
#                   levtype=ex['vars'][2][idx], lvllist=ex['vars'][3][idx], startyear=ex['startyear'], 
#                   endyear=ex['endyear'], startmonth=ex['dstartmonth'], endmonth=ex['dendmonth'], 
#                   grid=ex['grid_res'], stream='oper') )
## Download variable
#retrieve_ERA_i_field(sst)
#functions_pp.preprocessing_ncdf(sst, ex['grid_res'], ex['tfreq'], ex['exp'])
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