

def retrieve_ERA_i_field(cls):
    from functions_pp import kornshell_with_input
    from ecmwfapi import ECMWFDataServer
    import os
    server = ECMWFDataServer()
    file_path = os.path.join(cls.path_raw, cls.filename)
    datestring = "/".join(cls.datelist_str)
    if cls.stream == "mnth" or cls.stream == "oper":
        time = "00:00:00/06:00:00/12:00:00/18:00:00"
    elif cls.stream == "moda":
        time = "00:00:00"
    else:
        print("stream is not available")


    if os.path.isfile(path=file_path) == True:
        print(("You have already download the variable {} from {} to {} on grid {}d ".format(cls.name, cls.startyear, cls.endyear, cls.grid)))
        print(("\n to path: {} \n ".format(file_path)))
        pass
    else:
        print((" You WILL download variable {} \n stream is set to {} \n".format \
            (cls.name, cls.stream)))
        print(("\n to path: \n \n {} \n \n".format(file_path)))
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
                "target"    :   file_path,
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
                "target"    :   file_path,
                })
        print("convert operational 6hrly data to daily means")
        args = ['cdo daymean {} {}'.format(file_path.replace('daily','oper'), file_path)]
        kornshell_with_input(args)


    return

#
# def retrieve_ERA_i_field(cls):
#     from ecmwfapi import ECMWFDataServer
#     import os
#     server = ECMWFDataServer()
#     file_path = os.path.join(cls.base_path, cls.filename)
#     datestring = "/".join(cls.datelist)
#     if cls.stream == "mnth" or cls.stream == "oper":
#         time = "00:00:00/06:00:00/12:00:00/18:00:00"
#     elif cls.stream == "moda":
#         time = "00:00:00"
#     else:
#         print("stream is not available")
#
#
#     if os.path.isfile(path=file_path) == True:
#         print("You have already download the variable {} from {} to {} on grid {} ".format(cls.name, cls.startyear, cls.endyear, cls.grid))
#         print("\n to path: {} \n ".format(file_path))
#         pass
#     else:
#         print(" You WILL download variable {} \n stream is set to {} \n all dates: {} \n".format \
#             (cls.name, cls.stream, datestring))
#         print("\n to path: \n \n {} \n \n".format(file_path))
#         # !/usr/bin/python
#         server.retrieve({
#             "dataset"   :   "interim",
#             "class"     :   "ei",
#             "expver"    :   "1",
#             "date"      :   datestring,
#             "grid"      :   cls.grid,
#             "levtype"   :   cls.levtype,
#             # "levelist"  :   cls.lvllist,
#             "param"     :   cls.var_cf_code,
#             "stream"    :   cls.stream,
#             # "time"      :   time,
#             "type"      :   "an",
#             "format"    :   "netcdf",
#             "target"    :   file_path,
#             })
#
#     return