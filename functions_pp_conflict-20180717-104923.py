#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 17:48:31 2018

@author: semvijverberg
"""

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


def preprocessing_ncdf(cls, exp):
    import os
    from netCDF4 import Dataset
    from netCDF4 import num2date
    import pandas as pd
    import numpy as np
    import calendar
    # Final input and output files
    infile = os.path.join(cls.path_raw, cls.filename)
    # convert to inter annual daily mean to make this faster
    tmpfile = os.path.join(cls.path_raw, 'tmpfiles', 'tmp.nc')
    # check temporal frequency raw data
    file_path = os.path.join(cls.path_raw, cls.filename)
    ncdf = Dataset(file_path)
    numtime = ncdf.variables['time']
    dates = pd.to_datetime(num2date(numtime[:], units=numtime.units, calendar=numtime.calendar))
    temporal_freq = np.timedelta64(exp['tfreq'], 'D') 
    timesteps = int(np.timedelta64(exp['tfreq'], 'D')  / (dates[1] - dates[0]))
# =============================================================================
#     select dates
# =============================================================================
    seldays_pp = pd.DatetimeIndex(start=exp['sstartdate'], end=exp['senddate'], 
                                freq=(dates[1] - dates[0]))
    end_day = seldays_pp.max() 
    fit_steps_yr = (end_day - seldays_pp.min())  / temporal_freq
    start_day = (seldays_pp.max() - (temporal_freq * np.round(fit_steps_yr, decimals=0))) + 1 #include day 1 in tfreq mean
    curr_yr = pd.DatetimeIndex(start=start_day, end=end_day, 
                                freq=(dates[1] - dates[0]))
    datesstr = [str(date).split('.', 1)[0] for date in curr_yr.values]
    nyears = (dates.year[-1] - dates.year[0])+1
    for yr in range(1,nyears):
        if calendar.isleap(yr+dates.year[0]) == True:
            next_yr = curr_yr + pd.Timedelta('{}d'.format(366))
#            print yr+dates.year[0]
#            print curr_yr[0]
#            print next_yr[0]
        elif calendar.isleap(yr+dates.year[0]) == False:
            next_yr = curr_yr + pd.Timedelta('{}d'.format(365))
        curr_yr = next_yr
        nextstr = [str(date).split('.', 1)[0] for date in next_yr.values]
        datesstr = datesstr + nextstr
# =============================================================================
#   # give appropriate name to output file    
# =============================================================================
    outfilename = cls.filename[:-3]+'.nc'
    outfilename = outfilename.replace('daily', 'dt-{}days'.format(exp['tfreq']))
    outfilename = outfilename.replace('_{}_'.format(exp['dstartmonth']),'_{}{}_'.format(start_day.day, start_day.month_name()[:3]))
    outfilename = outfilename.replace('_{}_'.format(exp['dendmonth']),'_{}{}_'.format(end_day.day, end_day.month_name()[:3]))
    cls.path_pp = os.path.join(cls.base_path, 'input_pp_'+exp['exp'])
    if os.path.isdir(cls.path_pp):
        pass
    else:
        os.makedirs(cls.path_pp)
    outfile = os.path.join(cls.path_pp, outfilename)
    print infile + '\n'
    print outfile + '\n'
    
# =============================================================================
#   # commands to convert select days and temporal frequency 
# =============================================================================
    sel_dates = 'cdo select,date={} {} {}'.format(datesstr, infile, tmpfile)
    sel_dates = sel_dates.replace(', ',',').replace('\'','').replace('[','').replace(']','')
    convert_temp_freq = 'cdo timselmean,{} {} {}'.format(timesteps, tmpfile, outfile)
# =============================================================================
#    # problem with remapping, ruins the time coordinates
# =============================================================================
#    gridfile = os.path.join(cls.path_raw, 'grids', 'landseamask_{}deg.nc'.format(exp['grid_res']))
#    convert_grid = 'ncks -O --map={} {} {}'.format(gridfile, outfile, outfile)
#    cdo_gridfile = os.path.join(cls.path_raw, 'grids', 'lonlat_{}d_grid.txt'.format(grid_res))
#    convert_grid = 'cdo remapnn,{} {} {}'.format(cdo_gridfile, outfile, outfile)
# =============================================================================
#   # other unused cdo commands
# =============================================================================
#    del_days = 'cdo delete,month=12,11 -delete,month=10,day=31,30,29,28 -delete,month=2,day=29 {} {}'.format(infile, tmpfile)
#    selmon = 'cdo -selmon,{}/{} {} {}'.format(exp['sstartmonth'],exp['sendmonth'], outfile, outfile)#.replace('[','').replace(']','').replace(', ',',')
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
#     # update class
# =============================================================================
    cls.filename_pp = outfilename
    file_path = os.path.join(cls.path_pp, cls.filename_pp)
    ncdf = Dataset(file_path)
    numtime = ncdf.variables['time']
    dates = pd.to_datetime(num2date(numtime[:], units=numtime.units, calendar=numtime.calendar))
    cls.dates_np = dates
    cls.temporal_freq = '{}days'.format(temporal_freq.astype('timedelta64[D]').astype(int))
    return 
    
def import_array(cls, path='pp'):
    import os
    import xarray as xr
    from netCDF4 import num2date
    import pandas as pd
    if path == 'raw':
        file_path = os.path.join(cls.path_raw, cls.filename)

    else:
        file_path = os.path.join(cls.path_pp, cls.filename_pp)        
    ncdf = xr.open_dataset(file_path, decode_cf=True, decode_coords=True, decode_times=False)
    marray = ncdf.to_array(file_path).rename(({file_path: cls.name.replace(' ', '_')}))
    numtime = marray['time']
    dates = num2date(numtime, units=numtime.units, calendar=numtime.attrs['calendar'])
    dates_np = pd.to_datetime(dates)
    print('temporal frequency \'dt\' is: \n{}'.format(dates_np[1]- dates_np[0]))
    marray['time'] = dates_np
    cls.dates_np = dates_np
    return marray, cls

