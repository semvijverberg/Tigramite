#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 11:51:50 2018

@author: semvijverberg
"""

class Variable:

    from datetime import datetime, timedelta
    def __init__(self, name, dataset, startyear, endyear, startmonth, endmonth, grid, tfreq, exp):
        import os
        # self is the instance of the employee class
        # below are listed the instance variables
        self.name = name
        self.startyear = startyear
        self.endyear = endyear
        self.startmonth = startmonth
        self.endmonth = endmonth
        self.grid = grid
        self.tfreq = tfreq
        self.dataset = dataset
        self.base_path = '/Users/semvijverberg/surfdrive/Data_ERAint/'
        print exp
        self.path_pp = os.path.join(self.base_path, 'input_pp'+'_'+exp)
        self.path_raw = os.path.join(self.base_path, 'input_raw')
        if os.path.isdir(self.path_pp):
            pass
        else:
            print("{}\n\npath input does not exist".format(self.path_pp))
        filename_pp = '{}_{}-{}_{}_{}_dt-{}days_{}'.format(self.name, self.startyear, 
                    self.endyear, self.startmonth, self.endmonth, self.tfreq, 
                    self.grid).replace(' ', '_').replace('/','x')
        self.filename_pp = filename_pp +'.nc'
        print("Variable function selected {} \n".format(self.filename_pp))
        
def import_array(cls):
    import os
    import xarray as xr
    from netCDF4 import num2date
    import pandas as pd
    file_path = os.path.join(cls.path_pp, cls.filename_pp)
    ncdf = xr.open_dataset(file_path, decode_cf=True, decode_coords=True, decode_times=False)
    marray = ncdf.to_array(file_path).rename(({file_path: cls.name.replace(' ', '_')}))
    marray.name = cls.name
    numtime = marray['time']
    dates = num2date(numtime, units=numtime.units, calendar=numtime.attrs['calendar'])
    dates_np = pd.to_datetime(dates)
    print('temporal frequency \'dt\' is: \n{}'.format(dates_np[1]- dates_np[0]))
    marray['time'] = dates_np
    return marray

def find_region(data, region='EU'):
    if region == 'EU':
        west_lon = -30; east_lon = 40; south_lat = 35; north_lat = 65

    elif region ==  'U.S.':
        west_lon = -120; east_lon = -70; south_lat = 20; north_lat = 50

    region_coords = [west_lon, east_lon, south_lat, north_lat]
    import numpy as np
    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return int(idx)
    if west_lon <0 and east_lon > 0:
        # left_of_meridional = np.array(data.sel(latitude=slice(north_lat, south_lat), longitude=slice(0, east_lon)))
        # right_of_meridional = np.array(data.sel(latitude=slice(north_lat, south_lat), longitude=slice(360+west_lon, 360)))
        # all_values = np.concatenate((np.reshape(left_of_meridional, (np.size(left_of_meridional))), np.reshape(right_of_meridional, np.size(right_of_meridional))))
        lon_idx = np.concatenate(( np.arange(find_nearest(data['longitude'], 360 + west_lon), len(data['longitude'])),
                              np.arange(0,find_nearest(data['longitude'], east_lon), 1) ))
        lat_idx = np.arange(find_nearest(data['latitude'],north_lat),find_nearest(data['latitude'],south_lat),1)
        all_values = data.sel(latitude=slice(north_lat, south_lat), longitude=(data.longitude > 360 + west_lon) | (data.longitude < east_lon))
    if west_lon < 0 and east_lon < 0:
        all_values = data.sel(latitude=slice(north_lat, south_lat), longitude=slice(360+west_lon, 360+east_lon))
        lon_idx = np.arange(find_nearest(data['longitude'], 360 + west_lon), find_nearest(data['longitude'], 360+east_lon))
        lat_idx = np.arange(find_nearest(data['latitude'],north_lat),find_nearest(data['latitude'],south_lat),1)

    return all_values, region_coords

def calc_anomaly(marray, cls, q = 0.95):
    import xarray as xr
    import numpy as np
    print("calc_anomaly called for {}".format(cls.name, marray.shape))
    clim = marray.groupby('time.month').mean('time', keep_attrs=True)
    clim.name = 'clim_' + marray.name
    anom = marray.groupby('time.month') - clim
    anom['time_multi'] = anom['time']
    anom['time_date'] = anom['time']
    anom = anom.set_index(time_multi=['time_date','month'])
    anom.attrs = marray.attrs
#    substract = lambda x, y: (x - y)
#    anom = xr.apply_ufunc(substract, marray, np.tile(clim,(1,(cls.endyear+1-cls.startyear),1,1)), keep_attrs=True)
    anom.name = 'anom_' + marray.name
    std = anom.groupby('time.month').reduce(np.percentile, dim='time', keep_attrs=True, q=q)
#    std = anom.groupby('time.month').reduce(np.percentile, dim='time', keep_attrs=True, q=q)
    std.name = 'std_' + marray.name
    return clim, anom, std
# =============================================================================
# Plotting functions
# =============================================================================

def xarray_plot(data, path='default', saving=False):
    # from plotting import save_figure
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import numpy as np
    plt.figure()
    data = np.squeeze(data)
    if len(data.longitude[np.where(data.longitude > 180)[0]]) != 0:
        data = convert_longitude(data)
    else:
        pass
    if data.ndim != 2:
        print "number of dimension is {}, printing first element of first dimension".format(np.squeeze(data).ndim)
        data = data[0]
    else:
        pass
    proj = ccrs.Orthographic(central_longitude=data.longitude.mean().values, central_latitude=data.latitude.mean().values)
    ax = plt.axes(projection=proj)
    ax.coastlines()
    # ax.set_global()
    plot = data.plot.pcolormesh(ax=ax, cmap=plt.cm.RdBu_r,
                             transform=ccrs.PlateCarree(), add_colorbar=True)
    if saving == True:
        save_figure(data, path=path)
    plt.show()
    
def convert_longitude(data):
    import numpy as np
    import xarray as xr
    lon_above = data.longitude[np.where(data.longitude > 180)[0]]
    lon_normal = data.longitude[np.where(data.longitude <= 180)[0]]
    # roll all values to the right for len(lon_above amount of steps)
    data = data.roll(longitude=len(lon_above))
    # adapt longitude values above 180 to negative values
    substract = lambda x, y: (x - y)
    lon_above = xr.apply_ufunc(substract, lon_above, 360)
    convert_lon = xr.concat([lon_above, lon_normal], dim='longitude')
    data['longitude'] = convert_lon
    return data

def save_figure(data, path):
    import os
    import matplotlib.pyplot as plt
#    if 'path' in locals():
#        pass
#    else:
#        path = '/Users/semvijverberg/Downloads'
    if path == 'default':
        path = '/Users/semvijverberg/Downloads'
    else:
        path = path
    import datetime
    today = datetime.datetime.today().strftime("%d-%m-%y_%H'%M")
    if data.name != '':
        name = data.name.replace(' ', '_')
    if 'name' in locals():
        print 'input name is: {}'.format(name)
        name = name + '_' + today + '.jpeg'
        pass
    else:
        name = 'fig_' + today + '.jpeg'
    print('{} to path {}'.format(name, path))
    plt.savefig(os.path.join(path,name), format='jpeg', bbox_inches='tight')