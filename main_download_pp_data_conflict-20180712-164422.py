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
dict(
     {exp           :       'exp1'},
     {dataset       :       'ERA-i'},
     {grid_res      :       2.5},
     {tfreq         :       20},
     {startyear     :       1979},
     {endyear       :       2017},
     {startmonth    :       3},
     {endmonth      :       9})
exp = 'exp1'
grid_res = 2.5
days = 20
temporal_freq = np.timedelta64(days, 'D') 

#%%
# assign instance
temperature = Variable(name='t2m', dataset='ERA-i', var_cf_code='167.128', levtype='sfc', lvllist=0,
                       startyear=1979, endyear=2017, startmonth=3, endmonth=9, grid='2.5/2.5', stream='oper', units='K')
# Download variable to input_raw
retrieve_ERA_i_field(temperature)
# preprocess variable to input_pp_exp'expnumber'
functions_pp.preprocessing_ncdf(temperature, grid_res, temporal_freq, exp)
marray, temperature = functions_pp.import_array(temperature, path='pp')
#marray
#%%
# assign instance
sst = Variable(name='sst', dataset='ERA-i', var_cf_code='34.128', levtype='sfc', lvllist=0,
                       startyear=1979, endyear=2017, startmonth=3, endmonth=9, grid='2.5/2.5', stream='oper', units='K')
# Download variable
retrieve_ERA_i_field(sst)
functions_pp.preprocessing_ncdf(sst, grid_res, temporal_freq, exp)

#%%
# =============================================================================
# Experiment design
# =============================================================================
exp1_dic = dict( {'predictant':temperature} )

np.save(os.path.join(temperature.path_pp, 'exp1_dic.npy'), [temperature, sst])
np.save(os.path.join(temperature.path_pp, 'exp1_dic.npy'), exp1_dic)






#%% save to github
import os
import subprocess
runfile = os.path.join(script_dir, 'saving_repository_to_Github.sh')
subprocess.call(runfile)
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