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
import what_variable_pp
import retrieve_ERA_i
import functions_pp
Variable = what_variable_pp.Variable
retrieve_ERA_i_field = retrieve_ERA_i.retrieve_ERA_i_field
import_array = functions_pp.import_array

# homogenize netcdf
exp = dict(
     {'exp'          :       'exp1',
     'dataset'       :       'ERA-i',
     'grid_res'      :       2.5,
     'tfreq'         :       20,
     'startyear'     :       1979,
     'endyear'       :       2017,
     'startmonth'    :       3,
     'endmonth'      :       9,
     'vars'          :      [['t2m', 'sst'],['167.128', '34.128'],['sfc', 'sfc'],[0, 0]]}
     )


#%%
# assign instance
for idx in range(len(exp['vars'][0])):
    print idx    
    var_class = ( Variable(name=exp['vars'][0][idx], dataset='ERA-i', var_cf_code=exp['vars'][1][idx], 
                       levtype=exp['vars'][2][idx], lvllist=exp['vars'][3][idx], startyear=exp['startyear'], 
                       endyear=exp['endyear'], startmonth=exp['startmonth'], endmonth=exp['endmonth'], 
                       grid=exp['grid_res'], stream='oper') )
    exp[exp['vars'][0][idx]] = var_class
    retrieve_ERA_i_field(var_class)
    functions_pp.preprocessing_ncdf(var_class, exp['grid_res'], exp['tfreq'], exp['exp'])
    


#%%  
# =============================================================================
# Save Experiment design
# =============================================================================
RV = exp[exp['vars'][0][0]]
np.save(os.path.join(RV.path_pp, exp['exp']+'_dic.npy'), exp)
#list = np.load(os.path.join(temperature.path_pp, 'exp1_dic.npy')).item()
#np.save(os.path.join(temperature.path_pp, 'exp1_dic.npy'), exp1_dic)
test = np.load(os.path.join(RV.path_pp, exp['exp']+'_dic.npy')).item()





#%% save to github
import os
import subprocess
runfile = os.path.join(script_dir, 'saving_repository_to_Github.sh')
subprocess.call(runfile)
#%%




 depricated
#%%
temperature = Variable(name=exp['vars'][0][0], dataset='ERA-i', var_cf_code='167.128', levtype='sfc', lvllist=0,
                       startyear=1979, endyear=2017, startmonth=3, endmonth=9, grid=exp['grid_res'], stream='oper', units='K')
# Download variable to input_raw
retrieve_ERA_i_field(temperature)
# preprocess variable to input_pp_exp'expnumber'
functions_pp.preprocessing_ncdf(temperature, exp['grid_res'], exp['tfreq'], exp['exp'])
marray, temperature = functions_pp.import_array(temperature, path='pp')
#marray
#%%
# assign instance
sst = Variable(name='sst', dataset='ERA-i', var_cf_code='34.128', levtype='sfc', lvllist=0,
                       startyear=1979, endyear=2017, startmonth=3, endmonth=8, grid=exp['grid_res'], stream='oper')
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