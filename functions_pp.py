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
    
    out = p.communicate()[0]
    print out.decode()


def preprocessing_ncdf(cls, grid_res, temporal_freq):
    import os
    from netCDF4 import Dataset
    from netCDF4 import num2date
    import pandas as pd
    # Final input and output files
    infile = os.path.join(cls.path_input, cls.filename)
    tmpfile = os.path.join(cls.path_input, 'tmpfiles', 'tmp.nc')
    outfilename = cls.filename[:-3]+'.nc'.replace(cls.grid.replace('/','x'), '{}-{}'.format(grid_res,grid_res))
    outfilename = outfilename.replace('oper', 'dt-{}days'.format(temporal_freq.astype('timedelta64[D]').astype(int)))
    outfile = os.path.join(cls.base_path, 'input_pp', outfilename)
    print infile + '\n'
    print outfile + '\n'
    # check temporal frequency raw data
    file_path = os.path.join(cls.path_input, cls.filename)
    ncdf = Dataset(file_path)
    numtime = ncdf.variables['time']
    dates = pd.to_datetime(num2date(numtime[:2], units=numtime.units, calendar=numtime.calendar))
    timesteps = int(temporal_freq / (dates[1] - dates[0]))
    ncdf.close()
    # commands to homogenize data
    convert_temp_freq = 'cdo timselmean,{} {} {}'.format(timesteps, infile, tmpfile)
    convert_time_axis = 'cdo setreftime,1900-01-01,0,1h -setcalendar,gregorian {} {}'.format(tmpfile, tmpfile)
    gridfile = os.path.join(cls.path_input, 'grids', 'lonlat_{}d_grid.txt'.format(grid_res))
    convert_grid = 'cdo remapbil,{} {} {}'.format(gridfile, tmpfile, outfile)
    args = [convert_temp_freq, convert_time_axis, convert_grid]
    kornshell_with_input(args)
    return outfilename
    
def import_array(cls, path='pp'):
    import os
    import xarray as xr
    from netCDF4 import num2date
    import pandas as pd
    if path == 'pp':
        file_path = os.path.join(cls.base_path, 'input_pp', cls.filename)
    else:
        file_path = os.path.join(cls.path_input, cls.filename)
    ncdf = xr.open_dataset(file_path, decode_cf=True, decode_coords=True, decode_times=False)
    marray = ncdf.to_array(file_path).rename(({file_path: cls.name.replace(' ', '_')}))
    numtime = marray['time']
    dates = num2date(numtime, units=numtime.units, calendar=numtime.attrs['calendar'])
    dates_np = pd.to_datetime(dates)
    print('temporal frequency \'dt\' is: \n{}'.format(dates_np[1]- dates_np[0]))
    marray['time'] = dates_np
    cls.dates_np = dates_np
    return marray, cls

