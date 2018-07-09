#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 17:48:31 2018

@author: semvijverberg
"""
import os
os.chdir('/Users/semvijverberg/surfdrive/Scripts/Tigramite')
script_dir = os.getcwd()
import what_variable_pp
import retrieve_ERA_i
import functions_pp
Variable = what_variable_pp.Variable
retrieve_ERA_i_field = retrieve_ERA_i.retrieve_ERA_i_field
import_array = functions_pp.import_array

#%%
# assign instance
sst = Variable(name='SST', dataset='ERA-i', var_cf_code='34.128', levtype='sfc', lvllist=0,
                       startyear=1979, endyear=2017, startmonth=1, endmonth=12, grid='2.5/2.5', stream='oper', units='K')
# Download variable
retrieve_ERA_i_field(sst)
# homogenize netcdf
grid_res = 2.5
temporal_freq = 5*4 #days
functions_pp.preprocessing_ncdf(sst, grid_res, temporal_freq)
#%%
# assign instance
temperature = Variable(name='2_metre_temperature', dataset='ERA-i', var_cf_code='167.128', levtype='sfc', lvllist=0,
                       startyear=1979, endyear=2017, startmonth=1, endmonth=12, grid='2.5/2.5', stream='moda', units='K')
# Download variable
retrieve_ERA_i_field(temperature)
# homogenize netcdf
functions_pp.preprocessing_ncdf(temperature, grid_res, temporal_freq)
#%% assign instance
temperature = Variable(name='2_metre_temperature', dataset='ERA-i', var_cf_code='167.128', levtype='sfc', lvllist=0,
                       startyear=1979, endyear=2017, startmonth=6, endmonth=8, grid='2.5/2.5', stream='oper', units='K')
# Download variable
retrieve_ERA_i_field(temperature)
# homogenize netcdf
functions_pp.preprocessing_ncdf(temperature, grid_res, temporal_freq)

#%% load in array
marray, temperature = import_array(temperature, decode_cf=True, decode_coords=True)


#%% save to github
import os
import subprocess
runfile = os.path.join(script_dir, 'saving_repository_to_Github.sh')
subprocess.call(runfile)

exit()











