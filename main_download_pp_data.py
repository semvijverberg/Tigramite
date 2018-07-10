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
grid_res = 2.5
temporal_freq = np.timedelta64(5, 'D') 

#%%
# assign instance
sst = Variable(name='SST', dataset='ERA-i', var_cf_code='34.128', levtype='sfc', lvllist=0,
                       startyear=1979, endyear=2017, startmonth=3, endmonth=9, grid='2.5/2.5', stream='oper', units='K')
# Download variable
retrieve_ERA_i_field(sst)
sst.filename = functions_pp.preprocessing_ncdf(sst, grid_res, temporal_freq)
#%%
# assign instance
temperature = Variable(name='2_metre_temperature', dataset='ERA-i', var_cf_code='167.128', levtype='sfc', lvllist=0,
                       startyear=1979, endyear=2017, startmonth=3, endmonth=9, grid='2.5/2.5', stream='oper', units='K')
# Download variable
retrieve_ERA_i_field(temperature)
temperature.filename = functions_pp.preprocessing_ncdf(temperature, grid_res, temporal_freq)





#%% save to github
import os
import subprocess
runfile = os.path.join(script_dir, 'saving_repository_to_Github.sh')
subprocess.call(runfile)

exit()











