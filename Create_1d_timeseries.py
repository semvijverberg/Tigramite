#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 14:59:49 2018

@author: semvijverberg
"""
#%%
import os, sys
import pandas as pd
import xarray as xr
import pickle

filepath = '/Users/semvijverberg/Downloads/PEP-master/datafiles'
data = pd.read_csv(os.path.join(filepath, 'PEP-T95TimeSeries.txt'))
print data
datelist = []
values = []
for r in data.values:
    year = int(r[0][:4])
    month = int(r[0][5:7])
    day = int(r[0][7:11])
    string = '{}-{}-{}'.format(year, month, day)
    values.append(float(r[0][10:]))

    datelist.append( pd.Timestamp(string) )

RVts = xr.DataArray(values, coords=[pd.to_datetime(datelist)], dims=['time'])

filepath = '/Users/semvijverberg/surfdrive/Data_ERAint/input_pp'
pickle.dump( RVts, open(os.path.join(filepath,"T95ts_1982-2015_m6-8.pkl"), "wb") ) 
