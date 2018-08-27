



class Variable:
    import os
    """Levtypes: \n surface  :   sfc \n model level  :   ml (1 to 137) \n pressure levels (1000, 850.. etc)
    :   pl \n isentropic level    :   pt
    \n
    Monthly Streams:
    Monthly mean of daily mean  :   moda
    Monthly mean of analysis timesteps (synoptic monthly means)  :   mnth
    Daily Streams:
    Operational (for surface)   :   oper

    """
    from datetime import datetime, timedelta
    import pandas as pd
    
    # below is a class variable

    ecmwf_website = 'http://apps.ecmwf.int/codes/grib/param-db'
    base_path = "/Users/semvijverberg/surfdrive/Data_ERAint/"
    path_raw = os.path.join(base_path, 'input_raw')
    if os.path.isdir(path_raw):
        pass
    else:
        os.makedirs(path_raw)
    def __init__(self, name, dataset, var_cf_code, levtype, lvllist, startyear, endyear, startmonth, endmonth, grid, stream):
        # self is the instance of the employee class
        # below are listed the instance variables
        self.name = name
        self.var_cf_code = var_cf_code
        self.lvllist = lvllist
        self.levtype = levtype
        self.startyear = startyear
        self.endyear = endyear
        self.startmonth = startmonth
        self.endmonth = endmonth
        self.grid = grid
        self.stream = stream
        self.dataset = dataset
        if stream == 'oper':
            time_ana = "00:00:00/06:00:00/12:00:00/18:00:00"
        else:
            time_ana = "00:00:00"
        self.time_ana = time_ana 
        
        days_in_month = dict( {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31} )
        days_in_month_leap = dict( {1:31, 2:29, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31} )
        start = Variable.datetime(self.startyear, self.startmonth, 1)
        
        import calendar
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
#        self.dates_dt = [Variable.datetime.strptime(date, '%Y-%m-%d').date() for date in datelist_str]
        self.dates_np = Variable.pd.to_datetime(datelist_str)

        filename = '{}_{}-{}_{}_{}_{}_{}deg'.format(self.name, self.startyear, self.endyear, self.startmonth, self.endmonth, 'daily', self.grid).replace(' ', '_')
        self.filename = filename +'.nc'
        print(("Variable function selected {} \n".format(self.filename)))

#
#
# class Variable:
#     """Levtypes: \n surface  :   sfc \n model level  :   ml (1 to 137) \n pressure levels (1000, 850.. etc)
#     :   pl \n isentropic level    :   pt
#     \n
#     Monthly Streams:
#     Monthly mean of daily mean  :   moda
#     Monthly mean of analysis timesteps (synoptic monthly means)  :   mnth
#     Daily Streams:
#     Operational (for surface)   :   oper
#
#     """
#     from datetime import datetime, timedelta
#     # below is a class variable
#
#     ecmwf_website = 'http://apps.ecmwf.int/codes/grib/param-db'
#     base_path = "/Users/semvijverberg/surfdrive/Output_ERA/"
#     def __init__(self, name, var_cf_code, levtype, lvllist, startyear, endyear, startmonth, endmonth, grid, stream):
#         # self is the instance of the employee class
#         # below are listed the instance variables
#         self.name = name
#         self.var_cf_code = var_cf_code
#         self.lvllist = lvllist
#         self.levtype = levtype
#         self.startyear = startyear
#         self.endyear = endyear
#         self.startmonth = startmonth
#         self.endmonth = endmonth
#         self.grid = grid
#         self.stream = stream
#
#
#         start = Variable.datetime(self.startyear, self.startmonth, 1)
#         end = Variable.datetime(self.endyear, self.endmonth, 1)
#         datelist = [start.strftime('%Y-%m-%d')]
#         while start <= end:
#             if start.month < end.month:
#                 start += Variable.timedelta(days=31)
#                 datelist.append(Variable.datetime(start.year, start.month, 1).strftime('%Y-%m-%d'))
#             else:
#                 start = Variable.datetime(start.year+1, self.startmonth, 1)
#                 datelist.append(Variable.datetime(start.year, start.month, 1).strftime('%Y-%m-%d'))
#         self.datelist = datelist
#
#         filename = '{}_{}-{}_{}_{}_{}_{}'.format(self.name, self.startyear, self.endyear, self.startmonth, self.endmonth, self.stream, self.grid).replace(' ', '_').replace('/','x')
#         self.filename = filename +'.nc'
#         print("Variable function selected {} \n".format(self.filename))
#
