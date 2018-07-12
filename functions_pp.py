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


def preprocessing_ncdf(cls, grid_res, tfreq, exp):
    import os
    from netCDF4 import Dataset
    from netCDF4 import num2date
    import pandas as pd
    import numpy as np
    # Final input and output files
    infile = os.path.join(cls.path_raw, cls.filename)
    tmpfile = os.path.join(cls.path_raw, 'tmpfiles', 'tmp.nc')
    outfilename = cls.filename[:-3]+'.nc'
    outfilename = outfilename.replace('oper', 'dt-{}days'.format(tfreq))
    cls.path_pp = os.path.join(cls.base_path, 'input_pp_'+exp)
    outfile = os.path.join(cls.path_pp, outfilename)
    print infile + '\n'
    print outfile + '\n'
    # check temporal frequency raw data
    file_path = os.path.join(cls.path_raw, cls.filename)
    ncdf = Dataset(file_path)
    numtime = ncdf.variables['time']
    dates = pd.to_datetime(num2date(numtime[:2], units=numtime.units, calendar=numtime.calendar))
    temporal_freq = np.timedelta64(tfreq, 'D') 
    timesteps = int(temporal_freq / (dates[1] - dates[0]))
    
    variables = ncdf.variables.keys()
    [var for var in variables if var not in 'longitude time latitude']    
    # commands to homogenize data
    convert_temp_freq = 'cdo timselmean,{} {} {}'.format(timesteps, infile, outfile)
    echo_tfreq = 'echo '+convert_temp_freq
    # problem with remapping, ruins the time coordinates
#    gridfile = os.path.join(cls.path_raw, 'grids', 'landseamask_{}deg.nc'.format(grid_res))
#    convert_grid = 'ncks -O --map={} {} {}'.format(gridfile, outfile, outfile)
#    cdo_gridfile = os.path.join(cls.path_raw, 'grids', 'lonlat_{}d_grid.txt'.format(grid_res))
#    convert_grid = 'cdo remapnn,{} {} {}'.format(cdo_gridfile, outfile, outfile)
#    selmon = 'cdo selmon,3,9 
#    overwrite_taxis =   'cdo settaxis,{},1month {} {}'.format(starttime.strftime('%Y-%m-%d,%H:%M'), tmpfile, tmpfile)
    convert_time_axis = 'cdo setreftime,1900-01-01,00:00:00 -setcalendar,gregorian {} {}'.format(outfile, outfile)
    rm_timebnds = 'ncks -O -C -x -v time_bnds {} {}'.format(outfile, outfile)
    rm_res_timebnds = 'ncpdq -O {} {}'.format(outfile, outfile)
    add_path_raw = 'ncatted -a path_raw,global,c,c,{} {}'.format(str(ncdf.filepath()), outfile) 
    add_units = 'ncatted -a units,global,c,c,{} {}'.format(ncdf.variables[var].units, outfile) 
    detrend = 'cdo -b 32 detrend {} {}'.format(outfile, outfile)
    clim = 'cdo ydaymean {} {}'.format(outfile, tmpfile)
    anom = 'cdo -b 32 ydaysub {} {} {}'.format(outfile, tmpfile, outfile) 
    ncdf.close()
    # ORDER of commands, --> is important!
#    args = [echo_tfreq, convert_temp_freq , convert_grid]
    args = [echo_tfreq, convert_temp_freq, convert_time_axis, rm_timebnds, rm_res_timebnds, 
            add_path_raw, add_units, detrend, clim, anom] 
    kornshell_with_input(args)
    # update class
    cls.filename_pp = outfilename
    file_path = os.path.join(cls.path_pp, cls.filename_pp)
    ncdf = Dataset(file_path)
    numtime = ncdf.variables['time']
    dates = pd.to_datetime(num2date(numtime[:], units=numtime.units, calendar=numtime.calendar))
    cls.dates_np = dates
    cls.tfreq = '{}days'.format(temporal_freq.astype('timedelta64[D]').astype(int))
    return 
    
def import_array(cls, path='pp'):
    import os
    import xarray as xr
    from netCDF4 import num2date
    import pandas as pd
    if path == 'pp':
        file_path = os.path.join(cls.path_pp, cls.filename_pp)
    else:
        file_path = os.path.join(cls.path_raw, cls.filename)
    ncdf = xr.open_dataset(file_path, decode_cf=True, decode_coords=True, decode_times=False)
    marray = ncdf.to_array(file_path).rename(({file_path: cls.name.replace(' ', '_')}))
    numtime = marray['time']
    dates = num2date(numtime, units=numtime.units, calendar=numtime.attrs['calendar'])
    dates_np = pd.to_datetime(dates)
    print('temporal frequency \'dt\' is: \n{}'.format(dates_np[1]- dates_np[0]))
    marray['time'] = dates_np
    cls.dates_np = dates_np
    return marray, cls

