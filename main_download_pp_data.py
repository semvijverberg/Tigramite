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
import what_variable_pp
import retrieve_ERA_i
import functions_pp
import matplotlib.pyplot as plt
from datetime import datetime
Variable = what_variable_pp.Variable
retrieve_ERA_i_field = retrieve_ERA_i.retrieve_ERA_i_field
import_array = functions_pp.import_array




# homogenize netcdf
exp = dict(
     {'dataset'     :       'ERA-i',
     'grid_res'     :       2.5,
     'tfreq'        :       14,
     'startyear'    :       1979,
     'endyear'      :       2017,
     'dstartmonth'  :       1,
     'dendmonth'    :       12,
     'vars'         :      [['t2m', 'sst'],['167.128', '34.128'],['sfc', 'sfc'],[0, 0]]}
     )

exp['sstartdate'] = '{}-5-1 09:00:00'.format(exp['startyear'])
exp['senddate']   = '{}-8-31 09:00:00'.format(exp['startyear'])
exp['exp'] = '{}_m{}-{}_dt{}'.format("_".join(exp['vars'][0]), 
   exp['sstartdate'].split('-')[1], exp['senddate'].split('-')[1], exp['tfreq'])

#%%
# assign instance
for idx in range(len(exp['vars'][0]))[:]:
    print idx    
    var_class =  Variable(name=exp['vars'][0][idx], dataset='ERA-i', var_cf_code=exp['vars'][1][idx], 
                       levtype=exp['vars'][2][idx], lvllist=exp['vars'][3][idx], startyear=exp['startyear'], 
                       endyear=exp['endyear'], startmonth=exp['dstartmonth'], endmonth=exp['dendmonth'], 
                       grid=exp['grid_res'], stream='oper') 
    exp[exp['vars'][0][idx]] = var_class
    retrieve_ERA_i_field(var_class)
#    if os.path.isfile(outfilename):
#        pass
#    else:
#        functions_pp.preprocessing_ncdf(var_class, exp)
    functions_pp.preprocessing_ncdf(var_class, exp)

np.save(os.path.join(var_class.path_pp, exp['exp']+'_pp_dic.npy'), exp)
test = np.load(os.path.join(var_class.path_pp, exp['exp']+'_dic.npy')).item()

#%%
# Select reponse variable period
RV = exp[exp['vars'][0][1]]
marray, RV = functions_pp.import_array(RV, path='pp')
one_year = RV.dates_np.where(RV.dates_np.year == RV.startyear+1).dropna()
months = [7,8]
RV_period = []
for mon in months:
    RV_period.insert(-1, np.where(RV.dates_np.month == mon)[0] )
RV_period = [x for sublist in RV_period for x in sublist]
RV_period.sort()
exp['RV_period'] = RV_period

#%%  
# =============================================================================
# Save Experiment design
# =============================================================================
RV = exp[exp['vars'][0][0]]






#%% save to github
import os
import subprocess
runfile = os.path.join(script_dir, 'saving_repository_to_Github.sh')
subprocess.call(runfile)
#%% check if days of year are alligned to be able to calculate a meaningful cdo ydaymean
RV = exp['sst']
marray, RV = functions_pp.import_array(RV, path='pp')
time = marray.time.values
for yr in range(exp['startyear'],exp['endyear']+1):
    
    one_year = RV.dates_np.where(RV.dates_np.year == yr).dropna()
    print yr, len(one_year)
    print one_year.dayofyear
#%% Check if multi year mean is indeed 0
#print exp['RV_period']
plt.figure()
one_date = marray.sel(time=RV.dates_np[RV_period[::4]])
one_date.mean(dim='time')[0].plot.contourf()
# depricated
#%%
idx = 0
temperature = ( Variable(name='t2m', dataset='ERA-i', var_cf_code=exp['vars'][1][idx], 
                   levtype=exp['vars'][2][idx], lvllist=exp['vars'][3][idx], startyear=exp['startyear'], 
                   endyear=exp['endyear'], startmonth=exp['dstartmonth'], endmonth=exp['dendmonth'], 
                   grid=exp['grid_res'], stream='oper') )
# Download variable to input_raw
retrieve_ERA_i_field(temperature)
# preprocess variable to input_pp_exp'expnumber'
#functions_pp.preprocessing_ncdf(temperature, exp['grid_res'], exp['tfreq'], exp['exp'])
#marray, temperature = functions_pp.import_array(temperature, path='pp')
#marray
#%%
idx = 1
sst = ( Variable(name='sst', dataset='ERA-i', var_cf_code=exp['vars'][1][idx], 
                   levtype=exp['vars'][2][idx], lvllist=exp['vars'][3][idx], startyear=exp['startyear'], 
                   endyear=exp['endyear'], startmonth=exp['dstartmonth'], endmonth=exp['dendmonth'], 
                   grid=exp['grid_res'], stream='oper') )
# Download variable
retrieve_ERA_i_field(sst)
functions_pp.preprocessing_ncdf(sst, exp['grid_res'], exp['tfreq'], exp['exp'])


#%%

# =============================================================================
# Simple example of cdo commands within python by calling bash script
# =============================================================================
infile = os.path.join(temperature.base_path, 'input_raw', temperature.filename)
outfile = os.path.join(temperature.base_path, 'input_pp', 'output.nc')
#tmp = os.path.join(temperature.base_path, 'input_raw', temperature.filename)
args1 = 'cdo timmean {} {}'.format(infile, outfile)
#args2 = 'cdo setreftime,1900-01-01,0,1h -setcalendar,gregorian {} {} '.format(infile, outfile)
args = [args1]

functions_pp.kornshell_with_input(args)