#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 17:48:31 2018

@author: semvijverberg
"""
class Variable:
    from datetime import datetime, timedelta
    import pandas as pd
    """Levtypes: \n surface  :   sfc \n model level  :   ml (1 to 137) \n pressure levels (1000, 850.. etc)
    :   pl \n isentropic level    :   pt
    \n
    Monthly Streams:
    Monthly mean of daily mean  :   moda
    Monthly mean of analysis timesteps (synoptic monthly means)  :   mnth
    Daily Streams:
    Operational (for surface)   :   oper
    """
    ecmwf_website = 'http://apps.ecmwf.int/codes/grib/param-db'
    def __init__(self, ex, idx, ECMWFdown):
        import calendar
        import os
        # self is the instance of the employee class
        # below are listed the instance variables
        self.name = ex['vars'][0][idx]
        self.startyear = ex['startyear']
        self.endyear = ex['endyear']
        self.startmonth = 1
        self.endmonth = 12
        self.grid = ex['grid_res']
        self.dataset = ex['dataset']
        self.base_path = ex['base_path']
        self.path_raw = ex['path_raw']
        self.path_pp = ex['path_pp']
        if ECMWFdown == True:
            self.var_cf_code = ex['vars'][1][idx]
            self.levtype = ex['vars'][2][idx]
            self.lvllist = ex['vars'][3][idx]
            self.stream = 'oper'
#            if stream == 'oper':
            time_ana = "00:00:00/06:00:00/12:00:00/18:00:00"
#            else:
#                time_ana = "00:00:00"
            self.time_ana = time_ana 
            
            days_in_month = dict( {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31} )
            days_in_month_leap = dict( {1:31, 2:29, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31} )
            start = Variable.datetime(self.startyear, self.startmonth, 1)
            
            # creating list of dates that we want to download given the startyear/startmonth to endyear/endmonth
            datelist_str = [start.strftime('%Y-%m-%d')]
            if self.stream == 'oper':
                end = Variable.datetime(self.endyear, self.endmonth, days_in_month[self.endmonth])
                while start < end:          
                    start += Variable.timedelta(days=1)
                    datelist_str.append(start.strftime('%Y-%m-%d'))
                    if start.month == end.month and start.day == days_in_month[self.endmonth] and start.year != self.endyear:
                        start = Variable.datetime(start.year+1, self.startmonth, 1)
                        datelist_str.append(start.strftime('%Y-%m-%d'))  
            elif self.stream == 'moda' or 'mnth':
                end = Variable.datetime(self.endyear, self.endmonth, 1)
                while start < end:          
                    days = days_in_month[start.month] if calendar.isleap(start.year)==False else days_in_month_leap[start.month]
                    start += Variable.timedelta(days=days)
                    datelist_str.append(start.strftime('%Y-%m-%d'))
                    if start.month == end.month and start.year != self.endyear:
                        start = Variable.datetime(start.year+1, self.startmonth, 1)
                        datelist_str.append(start.strftime('%Y-%m-%d'))             
            self.datelist_str = datelist_str

            # Convert to datetime datelist
#            self.dates_dt = [Variable.datetime.strptime(date, '%Y-%m-%d').date() for date in datelist_str]
            self.dates = Variable.pd.to_datetime(datelist_str)
    
            self.filename = '{}_{}-{}_{}_{}_{}_{}deg.nc'.format(self.name, self.startyear, self.endyear, self.startmonth, self.endmonth, 'daily', self.grid).replace(' ', '_')
        elif ECMWFdown == False:
            if len(ex['own_actor_nc_names'][0]) != 0:
                self.name = ex['own_actor_nc_names'][idx][0]
                self.filename = ex['own_actor_nc_names'][idx][1]
                ex['vars'][0].append(self.name)
                print 't'
            if len(ex['own_RV_nc_name'][0]) != 0:
                self.name = ex['own_RV_nc_name'][0]
                self.filename = ex['own_RV_nc_name'][1]
                ex['vars'][0].insert(0, self.name)

        print('\n\t**\n\t{} {}-{} on {} grid\n\t**\n'.format(self.name, self.startyear, self.endyear, self.grid))
#        print("Variable function selected {} \n".format(self.filename))

def retrieve_ERA_i_field(cls):
#    from functions_pp import kornshell_with_input
    from ecmwfapi import ECMWFDataServer
    import os
    server = ECMWFDataServer()
    file_path = os.path.join(cls.path_raw, cls.filename)
    file_path_raw = file_path.replace('daily','oper')
    datestring = "/".join(cls.datelist_str)
#    if cls.stream == "mnth" or cls.stream == "oper":
#        time = "00:00:00/06:00:00/12:00:00/18:00:00"
#    elif cls.stream == "moda":
#        time = "00:00:00"
#    else:
#        print("stream is not available")


    if os.path.isfile(path=file_path) == True:
        print("You have already download the variable")
        print("to path: {} \n ".format(file_path))
        pass
    else:
        print("You WILL download variable {} \n stream is set to {} \n".format \
            (cls.name, cls.stream))
        print("to path: \n \n {} \n \n".format(file_path_raw))
        # !/usr/bin/python
        if cls.levtype == 'sfc':
            server.retrieve({
                "dataset"   :   "interim",
                "class"     :   "ei",
                "expver"    :   "1",
                "date"      :   datestring,
                "grid"      :   '{}/{}'.format(cls.grid,cls.grid),
                "levtype"   :   cls.levtype,
                # "levelist"  :   cls.lvllist,
                "param"     :   cls.var_cf_code,
                "stream"    :   cls.stream,
                 "time"      :  cls.time_ana,
                "type"      :   "an",
                "format"    :   "netcdf",
                "target"    :   file_path_raw,
                })
        elif cls.levtype == 'pl':
            server.retrieve({
                "dataset"   :   "interim",
                "class"     :   "ei",
                "expver"    :   "1",
                "date"      :   datestring,
                "grid"      :   '{}/{}'.format(cls.grid,cls.grid),
                "levtype"   :   cls.levtype,
                "levelist"  :   cls.lvllist,
                "param"     :   cls.var_cf_code,
                "stream"    :   cls.stream,
                 "time"      :  cls.time_ana,
                "type"      :   "an",
                "format"    :   "netcdf",
                "target"    :   file_path_raw,
                })
        print("convert operational 6hrly data to daily means")
        args = ['cdo daymean {} {}'.format(file_path_raw, file_path)]
        kornshell_with_input(args)
    return
        
def kornshell_with_input(args):
    '''some kornshell with input '''
    import os
    import subprocess
    cwd = os.getcwd()
    # Writing the bash script:
    new_bash_script = os.path.join(cwd,'bash_scripts', "bash_script.sh")
#    arg_5d_mean = 'cdo timselmean,5 {} {}'.format(infile, outfile)
    #arg1 = 'ncea -d latitude,59.0,84.0 -d longitude,-95,-10 {} {}'.format(infile, outfile)
    
#    bash_and_args = [new_bash_script, arg_5d_mean]
    bash_and_args = [new_bash_script]
    [bash_and_args.append(arg) for arg in args]
    with open(new_bash_script, "w") as file:
        file.write("#!/bin/sh\n")
        file.write("echo starting bash script\n")
        for No_args in range(len(bash_and_args)):
            if No_args != 0:
                file.write("${}\n".format(No_args)) 
    p = subprocess.Popen(bash_and_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = p.communicate()
    print out[0].decode()
    return

def datestr_for_preproc(cls, ex):
    ''' 
    The cdo timselmean that is used in the preprocessing_ncdf() will keep calculating 
    a mean over 10 days and does not care about the date of the years (also logical). 
    This means that after 36 timesteps you have averaged 360 days into steps of 10. 
    The last 5/6 days of the year do not fit the 10 day mean. It will just continuing 
    doing timselmean operations, meaning that the second year the first timestep will 
    be e.g. the first of januari (instead of the fifth of the first year). To ensure 
    the months and days in each year correspond, we need to adjust the dates that were 
    given by ex['sstartdate'] - ex['senddate'].
    '''
    import os
    from netCDF4 import Dataset
    from netCDF4 import num2date
    import pandas as pd
    import numpy as np
    import calendar
    # check temporal frequency raw data
    file_path = os.path.join(cls.path_raw, cls.filename)
    ncdf = Dataset(file_path)
    numtime = ncdf.variables['time']
    dates = pd.to_datetime(num2date(numtime[:], units=numtime.units, calendar=numtime.calendar))
# =============================================================================
#     select dates
# =============================================================================
    # selday_pp is the period you aim to study
    seldays_pp = pd.DatetimeIndex(start=ex['sstartdate'], end=ex['senddate'], 
                                freq=(dates[1] - dates[0]))
    end_day = seldays_pp.max() 
    # after time averaging over 'tfreq' number of days, you want that each year 
    # consists of the same day. For this to be true, you need to make sure that
    # the selday_pp period exactly fits in a integer multiple of 'tfreq'
    temporal_freq = np.timedelta64(ex['tfreq'], 'D') 
    fit_steps_yr = (end_day - seldays_pp.min())  / temporal_freq
    # line below: The +1 = include day 1 in counting
    start_day = (end_day - (temporal_freq * np.round(fit_steps_yr, decimals=0))) + 1 
    curr_yr = pd.DatetimeIndex(start=start_day, end=end_day, 
                                freq=(dates[1] - dates[0]))
    # create datestring for MARS download request
    datesstr = [str(date).split('.', 1)[0] for date in curr_yr.values]
    nyears = (dates.year[-1] - dates.year[0])+1
    for yr in range(1,nyears):
        if calendar.isleap(yr+dates.year[0]) == True:
            next_yr = curr_yr + pd.Timedelta('{}d'.format(366))
        elif calendar.isleap(yr+dates.year[0]) == False:
            next_yr = curr_yr + pd.Timedelta('{}d'.format(365))
        curr_yr = next_yr
        nextstr = [str(date).split('.', 1)[0] for date in next_yr.values]
        datesstr = datesstr + nextstr
    # store the adjusted dates in np.datetime64 format in var_class
    cls.dates = pd.to_datetime(datesstr)
# =============================================================================
#   # give appropriate name to output file    
# =============================================================================
    outfilename = cls.filename[:-3]+'.nc'
    outfilename = outfilename.replace('daily', 'dt-{}days'.format(ex['tfreq']))
    months = dict( {1:'jan',2:'feb',3:'mar',4:'apr',5:'may',6:'jun',7:'jul',
                         8:'aug',9:'sep',10:'okt',11:'nov',12:'dec' } )
    
    outfilename = outfilename.replace('_{}_'.format(1),'_{}{}_'.format(start_day.day, months[start_day.month]))
    outfilename = outfilename.replace('_{}_'.format(12),'_{}{}_'.format(end_day.day, months[end_day.month]))
    cls.filename_pp = outfilename
    cls.path_pp = ex['path_pp']
    outfile = os.path.join(ex['path_pp'], outfilename)
    print 'output file of pp will be saved as: \n' + outfile + '\n'
    return outfile, datesstr, cls

def preprocessing_ncdf(outfile, datesstr, cls, ex):
    ''' 
    This function does some python manipulation based on your experiment 
    to create the corresponding cdo commands. 
    A kornshell script is created in the folder bash_scripts. First time you
    run it, it will give execution rights error. Open terminal -> go to 
    bash_scrips folder -> type 'chmod 755 bash_script.sh' to give exec rights.
    - Select time period of interest from daily mean time series
    - Do timesel mean based on your ex['tfreq']
    - Make sure the calenders are the same, dates are used to select data by xarray
    - Gridpoint detrending
    - Calculate anomalies (w.r.t. multi year daily means)
    - deletes time bonds from the variables
    - stores new relative time axis converted to numpy.datetime64 format in var_class
    '''
    import os
    from netCDF4 import Dataset
    from netCDF4 import num2date
    import pandas as pd
    import numpy as np
    # Final input and output files
    infile = os.path.join(cls.path_raw, cls.filename)
    # convert to inter annual daily mean to make this faster
    tmpfile = os.path.join(cls.path_raw, 'tmpfiles', 'tmp.nc')
    # check temporal frequency raw data
    file_path = os.path.join(cls.path_raw, cls.filename)
    ncdf = Dataset(file_path)
    numtime = ncdf.variables['time']
    dates = pd.to_datetime(num2date(numtime[:], units=numtime.units, calendar=numtime.calendar))
    timesteps = int(np.timedelta64(ex['tfreq'], 'D')  / (dates[1] - dates[0]))
    temporal_freq = np.timedelta64(ex['tfreq'], 'D') 
# =============================================================================
#   # commands to convert select days and temporal frequency 
# =============================================================================
    sel_dates = 'cdo select,date={} {} {}'.format(datesstr, infile, tmpfile)
    sel_dates = sel_dates.replace(', ',',').replace('\'','').replace('[','').replace(']','')
    convert_temp_freq = 'cdo timselmean,{} {} {}'.format(timesteps, tmpfile, outfile)
# =============================================================================
#    # problem with remapping, ruins the time coordinates
# =============================================================================
#    gridfile = os.path.join(cls.path_raw, 'grids', 'landseamask_{}deg.nc'.format(ex['grid_res']))
#    convert_grid = 'ncks -O --map={} {} {}'.format(gridfile, outfile, outfile)
#    cdo_gridfile = os.path.join(cls.path_raw, 'grids', 'lonlat_{}d_grid.txt'.format(grid_res))
#    convert_grid = 'cdo remapnn,{} {} {}'.format(cdo_gridfile, outfile, outfile)
# =============================================================================
#   # other unused cdo commands
# =============================================================================
#    del_days = 'cdo delete,month=12,11 -delete,month=10,day=31,30,29,28 -delete,month=2,day=29 {} {}'.format(infile, tmpfile)
#    selmon = 'cdo -selmon,{}/{} {} {}'.format(ex['sstartmonth'],ex['sendmonth'], outfile, outfile)#.replace('[','').replace(']','').replace(', ',',')
#    echo_selmon = 'echo '+selmon
#    overwrite_taxis =   'cdo settaxis,{},1month {} {}'.format(starttime.strftime('%Y-%m-%d,%H:%M'), tmpfile, tmpfile)
#    del_leapdays = 'cdo delete,month=2,day=29 {} {}'.format(infile, tmpfile)
# =============================================================================
#   # commands to homogenize data
# =============================================================================
    convert_time_axis = 'cdo setreftime,1900-01-01,00:00:00 -setcalendar,gregorian {} {}'.format(outfile, outfile)
    rm_timebnds = 'ncks -O -C -x -v time_bnds {} {}'.format(outfile, outfile)
    rm_res_timebnds = 'ncpdq -O {} {}'.format(outfile, outfile)
    variables = ncdf.variables.keys()
    [var for var in variables if var not in 'longitude time latitude']    
    add_path_raw = 'ncatted -a path_raw,global,c,c,{} {}'.format(str(ncdf.filepath()), outfile) 
    add_units = 'ncatted -a units,global,c,c,{} {}'.format(ncdf.variables[var].units, outfile) 
    detrend = 'cdo -b 32 detrend {} {}'.format(outfile, tmpfile)
#    clim = 'cdo ydaymean {} {}'.format(outfile, tmpfile)
    anom = 'cdo -b 32 ydaysub {} -ydayavg {} {}'.format(tmpfile, tmpfile, outfile) 
    echo_end = "echo data is detrended and are anomaly versus muli-year daily mean\n"
    echo_test = 'echo test'
    ncdf.close()
    # ORDER of commands, --> is important!
    args = [sel_dates, convert_temp_freq, convert_time_axis, rm_timebnds, rm_res_timebnds, 
            add_path_raw, add_units, detrend, anom, echo_end, echo_test] 
    kornshell_with_input(args)
# =============================================================================
#     # update class (more a check if dates are indeed correct)
# =============================================================================
    file_path = os.path.join(cls.path_pp, cls.filename_pp)
    ncdf = Dataset(file_path)
    numtime = ncdf.variables['time']
    dates = pd.to_datetime(num2date(numtime[:], units=numtime.units, calendar=numtime.calendar))
    cls.dates = dates
    cls.temporal_freq = '{}days'.format(temporal_freq.astype('timedelta64[D]').astype(int))
    return 
    
def import_array(cls, path='pp'):
    import os
    import xarray as xr
    from netCDF4 import num2date
    import pandas as pd
    import numpy as np
    if path == 'raw':
        file_path = os.path.join(cls.path_raw, cls.filename)

    else:
        file_path = os.path.join(cls.path_pp, cls.filename_pp)        
    ncdf = xr.open_dataset(file_path, decode_cf=True, decode_coords=True, decode_times=False)
    marray = np.squeeze(ncdf.to_array(file_path).rename(({file_path: cls.name.replace(' ', '_')})))
    numtime = marray['time']
    dates = num2date(numtime, units=numtime.units, calendar=numtime.attrs['calendar'])
    dates = pd.to_datetime(dates)
#    print('temporal frequency \'dt\' is: \n{}'.format(dates[1]- dates[0]))
    marray['time'] = dates
    cls.dates = dates
    return marray, cls

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
        all_values = data.sel(latitude=slice(north_lat, south_lat), 
                              longitude=(data.longitude > 360 + west_lon) | (data.longitude < east_lon))
    if west_lon < 0 and east_lon < 0:
        all_values = data.sel(latitude=slice(north_lat, south_lat), longitude=slice(360+west_lon, 360+east_lon))
        lon_idx = np.arange(find_nearest(data['longitude'], 360 + west_lon), find_nearest(data['longitude'], 360+east_lon))
        lat_idx = np.arange(find_nearest(data['latitude'],north_lat),find_nearest(data['latitude'],south_lat),1)

    return all_values, region_coords