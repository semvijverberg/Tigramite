# -*- coding: utf-8 -*-
#%%
script_dir = '/Users/semvijverberg/surfdrive/Scripts/Tigramite'
import sys, os
os.chdir(script_dir)
sys.path.append('/Users/semvijverberg/surfdrive/Scripts/Tigramite')
#import numpy
import matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"
from mpl_toolkits.basemap import Basemap#, shiftgrid, cm
from netCDF4 import Dataset
from netCDF4 import num2date
import matplotlib.pyplot as plt
import seaborn as sns
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
import RGCPD_functions_version_04 as rgcpd
#from tigramite import plotting as tp
#from tigramite.models import LinearMediation, Prediction
#import statsmodels.api as sm
#from sklearn import preprocessing
#from datetime import datetime, date, timedelta
#from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
#import scipy
#from scipy import signal
#from matplotlib import gridspec
#import sklearn
#import tigramite

import subprocess
import functions_tig 
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
from inspect import getsourcelines
def seefunction(func):
    print getsourcelines(func)
# =============================================================================
#  saving to github
# =============================================================================
#runfile = os.path.join(script_dir, 'saving_repository_to_Github.sh')
#subprocess.call(runfile)

# =============================================================================
# Load 'exp' dictionairy with information of pre-processed data (variables, paths, filenames, etcetera..)
# and add RGCPD/Tigrimate experiment settings
# =============================================================================
exp_clus = '13Jul-24Aug_ward'
path_exp_clus = os.path.join('/Users/semvijverberg/surfdrive/Data_ERAint/t2m_sst_m5-8_dt14/', exp_clus)
exp = np.load(os.path.join(path_exp_clus, 'raw_input_tig_dic.npy')).item()
# Response Variable is what we want to predict
RV = exp['t2m']
file_path = os.path.join(RV.path_pp, RV.filename_pp)
ncdf = Dataset(file_path)
numtime = ncdf.variables['time']
dates = pd.to_datetime(num2date(numtime[:], units=numtime.units, calendar=numtime.calendar))
print("\nBelow printing dates accross years, should be all same day (otherwise ydaymean not appropriate")
print(dates[::dates[dates.year == 1979].size])
exp['clus_anom_std'] = 1
exp['alpha'] = 0.01 # set significnace level for correlation maps
exp['alpha_fdr'] = 2*exp['alpha'] # conservative significance level
exp['FDR_control'] = False # Do you want to use the conservative alpha_fdr or normal alpha?
exp['lag_min'] = 3 # Lag time(s) of interest
exp['lag_max'] = 4 
# If your pp data is not a full year, there is Maximum meaningful lag given by: 
#exp['lag_max'] = dates[dates.year == 1979].size - exp['RV_oneyr'].size
exp['alpha_level_tig'] = 0.5 # Alpha level for final regression analysis by Tigrimate
exp['la_min'] = -89 # select domain of correlation analysis
exp['la_max'] = 89
exp['lo_min'] = -180
exp['lo_max'] = 360
exp['time_cycle'] = dates[dates.year == 1979].size # time-cycle of data. total timesteps in one year
pcA_sets = dict({   # dict of sets of pc_alpha values
      'pcA_set1a' : [ 0.05], # 0.05 0.01 
      'pcA_set1b' : [ 0.01], # 0.05 0.01 
      'pcA_set1c' : [ 0.1], # 0.05 0.01 
      'pcA_set2'  : [ 0.2, 0.1, 0.05, 0.01, 0.001], # set2
      'pcA_set3'  : [ 0.1, 0.05, 0.01], # set3
      'pcA_set4'  : [ 0.5, 0.4, 0.3, 0.2, 0.1], # set4
      'pcA_none'  : None # default
      })
exp['pcA_set'] = ['pcA_set1a'] 
params_combination = 'lag{}to{}_aCorr{}'.format(exp['lag_min'],
                         exp['lag_max'], exp['alpha'])
exp['path_output'] = os.path.join(exp['path_exp_clus'], 'output_tigr_SST_T2m/')
#=====================================================================================
# Information on period taken for response-variable, already decided in main_download_and_pp
#=====================================================================================
n_years = dates.year.max() - dates.year.min() 
timeperiod = '{}-{}'.format(RV.startyear, RV.endyear)
time_range_all = [0, RV.dates_np.size]
RV_indices = exp['RV_period']
# =============================================================================
# Mask for Response Variable
# =============================================================================
cluster = 0
clusters = np.squeeze(xr.Dataset.from_dict(np.load(os.path.join(exp['path_exp_clus'], 'clusters_dic.npy')).item()).to_array())
cluster_out = clusters.sel(cluster=cluster)
RV_masked = np.ma.masked_where((cluster_out < exp['clus_anom_std']*cluster_out.std()), cluster_out).mask
# =============================================================================
# Figure/Plotting settings
# =============================================================================
map_proj = ccrs.LambertCylindrical(central_longitude=int(cluster_out.longitude.mean()))
file_type1 = ".pdf"
file_type2 = ".png"
SaveTF = True
showplot = False
plotin1fig = True
fig_path = os.path.join(exp['path_output'], 'lag{}to{}/'.format(exp['lag_min'],exp['lag_max']))
fig_subpath_form = os.path.join(fig_path, '{}_pcA_SIGN{}_subinfo/'.format(params_combination, 
                           exp['alpha_level_tig']))
if os.path.isdir(fig_path) != True : os.makedirs(fig_path)

#
#==================================================================================
# 1) Define index of interest (here: Polar vortex index based on gph)
#==================================================================================
#====================================================================
# load preprocessed predictant 
RV_array = np.squeeze(functions_tig.import_array(RV))
RV_array, region_coords = functions_tig.find_region(RV_array, region='U.S.')
RV_array.shape
#shear.shape     
time , nlats, nlons = RV_array.shape # [months , lat, lon]
#=====================================================================================
# mean over longitude and latitude
functions_tig.xarray_plot(cluster_out)
plt.imshow(RV_masked)
# fix this
RV_region = np.ma.masked_array(data=RV_array, mask=np.reshape(np.repeat(RV_masked, RV_array.time.size), RV_array.shape))
RV1D = np.ma.mean(RV_region, axis = (1,2)) # take spatial mean with mask loaded in beginning
RV_ts = RV1D[RV_indices] # extract specific months of MT index 
#

# =============================================================================
# 2) DEFINE PRECURSOS COMMUNITIES:
# =============================================================================
# - calculate and plot pattern correltion for differnt fields
# - create time-series over these regions 

#=====================================================================================
outd = dict()
class act:
    def __init__(self, name, Corr_Coeff, lat_grid, lon_grid, actbox, tsCorr, n_reg_perlag):
        self.name = var
        self.Corr_Coeff = Corr_Coeff
        self.lat_grid = lat_grid
        self.lon_grid = lon_grid
        self.actbox = actbox
        self.tsCorr = tsCorr
        self.n_reg_perlag = n_reg_perlag

# map for plotting the correlation maps
m = Basemap(projection='hammer',lon_0 = int(cluster_out.longitude.mean()) ,resolution='c')

allvar = exp['vars'][0] # list of all variable names
for var in allvar: # loop over all variables
    actor = exp[var]
    #===========================================
    # 3c) Precursor field = sst
    #===========================================
    ncdf = Dataset(os.path.join(actor.path_pp, actor.filename_pp), 'r')
    array = ncdf.variables[var][:,:,:].squeeze()    
    time , nlats, nlons = array.shape # [months , lat, lon]
    box = [exp['la_min'], exp['la_max'], exp['lo_min'], exp['lo_max']]
    # =============================================================================
    # Calculate correlation 
    # =============================================================================
    Corr_Coeff, lat_grid, lon_grid = rgcpd.calc_corr_coeffs_new(ncdf, array, box, RV_ts, time_range_all, exp['lag_min'], 
                                                          exp['lag_max'], exp['time_cycle'], RV_indices, 
                                                          exp['alpha_fdr'], FDR_control=exp['FDR_control'])
    Corr_Coeff = np.ma.array(data = Corr_Coeff[:,:], mask = Corr_Coeff.mask[:,:])
    # =============================================================================
    # Convert regions in time series  
    # =============================================================================
    actbox = rgcpd.extract_data(ncdf, array, time_range_all, box)
    actbox = np.reshape(actbox, (actbox.shape[0], -1))
    tsCorr, n_reg_perlag = rgcpd.calc_actor_ts_and_plot(Corr_Coeff, actbox, 
                            exp['lag_min'], lat_grid, lon_grid, m, var)
    outd[var] = act(var, Corr_Coeff, lat_grid, lon_grid, actbox, tsCorr, n_reg_perlag)
    # =============================================================================
    # Plot    
    # =============================================================================
    if plotin1fig == False and showplot == True:
        fig = functions_tig.xarray_plot_region([var], outd, exp['lag_min'],  exp['lag_max'], 
                                               map_proj, exp['tfreq'])
        fig_filename = '{}_vs_{}_{}'.format(allvar[0], var, params_combination) + file_type2
        plt.savefig(os.path.join(fig_path, fig_filename), bbox_inches='tight', dpi=200)
                                       

if plotin1fig == True and showplot == True:
    variables = outd.keys()
    fig = functions_tig.xarray_plot_region(variables, outd, exp['lag_min'],  exp['lag_max'], 
                                           map_proj, exp['tfreq'])
    fig_filename = '{}_vs_{}_{}'.format(allvar[0], var, params_combination) + file_type2
    plt.savefig(os.path.join(fig_path, fig_filename), bbox_inches='tight', dpi=200)

                                                         
#%%
#=====================================================================================
#
# 4) PCMCI-algorithm
#
#=====================================================================================
# ======================================================================================================================
# Choose
# ======================================================================================================================
# alpha level for multiple linear regression model (last step)
alpha_level = exp['alpha_level_tig']

# p lets you choose the pcA alpha values
for pc_alpha_name in exp['pcA_set']:
        
    print( ' run tigramite 3, run.pcmci')
    print(pc_alpha_name)
    '''
    save output
    '''
    pc_alpha = pcA_sets[pc_alpha_name]   
    insertpca = fig_subpath_form.split('pcA')
    fig_subpath = insertpca[0] + '{}'.format(pc_alpha_name) + insertpca[1]
    if os.path.isdir(fig_subpath):
        pass
    else:
        os.makedirs(fig_subpath)
    if SaveTF == True:
        orig_stdout = sys.stdout
        sys.stdout = f = open(''.join([fig_subpath,'old.txt']), 'a')  
    
    # create list with all actors, these will be merged into the fulldata array
    actorlist = []
    for var in allvar[1:]:
        print var
        actor = outd[var]
        actorlist.append(actor.tsCorr)        
        # create array which numbers the regions
        idx = allvar.index(var) - 1
        actor.var_info = [[i+1, var, idx] for i in range(actor.tsCorr.shape[1])]
    # stack actor time-series together:
    fulldata = np.concatenate(tuple(actorlist), axis = 1)
    print('There are {} regions in total'.format(fulldata.shape[1]))
    # add the full 1D time series of interest as first entry:
    fulldata = np.column_stack((RV1D, fulldata))
    # save fulldata
    file_name = 'fulldata_{}'.format(params_combination)#,'.pdf' ])
    fulldata.dump(os.path.join(fig_subpath, file_name+'.pkl'))   
    # Array of corresponing regions with var_names (first entry is RV)
    var_names = [[0, allvar[0]]] + actor.var_info 
    file_name = 'list_actors_{}'.format(params_combination)
    var_names_np = np.asanyarray(var_names)
    var_names_np.dump(os.path.join(fig_subpath, file_name+'.pkl'))  
    # ======================================================================================================================
    # tigramite 3
    # ======================================================================================================================
    data = fulldata
    print(data.shape)    
    # RV mask False for period that I want to analyse
    idx_start_RVperiod = int(np.where(dates[dates.year == exp['RV_oneyr'].year[0]] == exp['RV_oneyr'][0])[0])
    data_mask = np.ones(data.shape, dtype='bool') # true for actor months, false for RV months
    for i in range(exp['RV_oneyr'].size): # total timesteps RV period, 12 is 
        data_mask[idx_start_RVperiod+i:: exp['time_cycle'],:] = False 
    T, N = data.shape # Time, Regions
    # ======================================================================================================================
    # Initialize dataframe object (needed for tigramite functions)
    # ======================================================================================================================
    dataframe = pp.DataFrame(data=data, mask=data_mask) 
    # Create 'time axis' and variable names
    datatime = np.arange(len(data))
    # ======================================================================================================================
    # pc algorithm: only parents for selected_variables are calculated
    # ======================================================================================================================
    parcorr = ParCorr(significance='analytic', 
                      use_mask =True, 
                      mask_type='y', 
                      verbosity=2)
    #==========================================================================
    # multiple testing problem:
    #==========================================================================
    pcmci   = PCMCI(dataframe=dataframe,  
                    cond_ind_test=parcorr,
                    var_names=var_names,
                    selected_variables=None, # consider precursor only of RV variable, then selected_variables should be None
                    verbosity=2)
         
    # ======================================================================================================================
    #selected_links = dictionary/None
    results = pcmci.run_pcmci(tau_max=exp['lag_max'], pc_alpha = pc_alpha, tau_min = exp['lag_min'], max_combinations=1) 
    
    q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], fdr_method='fdr_bh')
    
    pcmci._print_significant_links(p_matrix = results['p_matrix'], 
                                   q_matrix = q_matrix, 
                                   val_matrix = results['val_matrix'],
                                   alpha_level = alpha_level)
            
    sig = pcmci._return_significant_parents(pq_matrix=q_matrix,
                                            val_matrix=results['val_matrix'], 
                                            alpha_level=alpha_level)
                               
    
    all_parents = sig['parents']
    link_matrix = sig['link_matrix']
    #%%
    """
    what's this?
    med = LinearMediation(dataframe=dataframe,            
                use_mask =True,
                mask_type ='y',
                data_transform = None)   # False or None for sklearn.preprocessing.StandardScaler = 
    
    med.fit_model(all_parents=all_parents, tau_max= tau_max)
    """
    
    # parents of index of interest:
    # parents_neighbors = all_parents, estimates, iterations 		
    parents_RV = all_parents[0]
    
    # combine all variables in one list
    precursor_fields = exp['vars'][0][1:]
    Corr_Coeff_list = []
    for var in allvar[1:]:
        actor = outd[var]
        Corr_Coeff_list.append(actor.Corr_Coeff)
    Corr_precursor_ALL = Corr_Coeff_list
    print 'You have {} precursor(s) with {} lag(s)'.format(np.array(Corr_precursor_ALL).shape[0],
                                                            np.array(Corr_precursor_ALL).shape[2])
    
    n_parents = len(parents_RV)   
    for i in range(n_parents):
        link_number = parents_RV[i][0]
        lag = np.abs(parents_RV[i][1])-1
        index_in_fulldata = parents_RV[i][0]
        if index_in_fulldata>0:
       
            according_varname = var_names[index_in_fulldata][1]
            according_number = var_names[index_in_fulldata][0]
            according_field_number = var_names[index_in_fulldata][2] # what is this, there is no third index
            print("index_in_fulldata")
            print(index_in_fulldata)
            print("according_varname")
            print(according_varname)
            print("according_number")
            print(according_number)
            print("according_field_number")
            print(according_field_number)
            # *********************************************************
            # print and save only significant regions
            # *********************************************************
            according_fullname = str(according_number) + according_varname   
            Corr_precursor = Corr_precursor_ALL[according_field_number]
#            xr.DataArray(data=Corr_precursor, coords=[actor.lat_grid, actor.lon_grid], dims=('latitude','longitude'))
            rgcpd.print_particular_region(according_number, Corr_precursor[:, :], actor.lat_grid, actor.lon_grid, m, according_fullname)
            fig_file = '{}_{}_{}_sign{}_{}{}'.format(according_fullname,
                        params_combination,pc_alpha_name,alpha_level,str(parents_RV[i][1]),file_type2)
            plt.savefig(os.path.join(fig_subpath, fig_file), dpi=250)   
            plt.close()
               
            print('                                        ')
            # *********************************************************                                
            # save data
            # *********************************************************
            according_fullname = str(according_number) + according_varname
            name = ''.join([str(index_in_fulldata),'_',according_fullname])
            print(fulldata[:,index_in_fulldata].size)
            print(name)
        else :
            print 'Index itself is also causal parent -> skipped' 
            print('*******************              ***************************                ******************')

# =============================================================================
#   Plotting all fields significant at alpha_level_tig
# =============================================================================
    # stack all variables and lags togther to plot them in same basemap
#    Corr_Coeff_all_r_l = np.ma.concatenate(Corr_precursor_ALL[:],axis=1)
#    all_regions = np.ma.masked_all(Corr_Coeff_all_r_l[:,0].shape)
#    all_regions_tig = np.ma.masked_all(Corr_Coeff_all_r_l[:,0].shape)
#    for i in range(Corr_Coeff_all_r_l.shape[1]):
#        regions_i = rgcpd.define_regions_and_rank_new(Corr_Coeff_all_r_l[:,i], actor.lat_grid, actor.lon_grid)
#        regions_i = np.nan_to_num(regions_i)
#
#        all_regnumbers_tig = [reg[0] for reg in parents_RV]
#        indices_regs = np.where(regions_i >1)[0]
#        for i in indices_regs:
#            all_regions[i] = regions_i[i]
#            if regions_i[i] in all_regnumbers_tig:
#                all_regions_tig[i] = regions_i[i]
#    all_regions = all_regions.reshape((actor.lat_grid.size, actor.lon_grid.size))
#    all_regions_tig = all_regions_tig.reshape((actor.lat_grid.size, actor.lon_grid.size))
#
#    
#    m.drawcoastlines(color='gray', linewidth=0.35)
#    m.drawmapboundary(fill_color='white', color='white')
#    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
#    m.contourf(lon_mesh,lat_mesh, all_regions_tig, latlon = True)
#    plt.title('{}-{} {} tfreq{}days'.format(RV.name, RV.dataset, timeperiod, exp['tfreq']))
#    file_name = '{}_{}_SIGN{}_all'.format(params_combination, pc_alpha_name, alpha_level)
#    plt.savefig(os.path.join(fig_subpath, file_name + file_type1))
#    plt.savefig(os.path.join(fig_path, file_name + file_type2),dpi=250)  
# =============================================================================
#                 
# =============================================================================
#    if plot_all == True:
##        m = Basemap(projection='hammer',lon_0 = 0 ,resolution='c')     #300       
#        #plt.plot()
#        count = 0
#        #fig = plt.figure(figsize=(6, 4))
#        fig = plt.subplots(figsize=(6, 4))
#         	
#        for i in range(n_parents):
#            link_number = parents_RV[i][0]
#            lag = np.abs(parents_RV[i][1])-1
#            #==========================================================================
#            # save precursos indices form fulldata, to be stored and used to find the 
#            # precursors of the precursors
#            #==========================================================================
#            index_in_fulldata = parents_RV[i][0]
#            if index_in_fulldata>0:
#               
#                according_varname = var_names[index_in_fulldata][1]
#                according_number = var_names[index_in_fulldata][0]
#                according_field_number = var_names[index_in_fulldata][2]
#                # *********************************************************
#                # print and save only sign regions
#                # *********************************************************
#                according_fullname = str(according_number) + according_varname   
#                Corr_precursor = Corr_precursor_ALL[according_field_number]
#               
#                #oad.print_particular_region(according_number, Corr_precursor[:, exp['lag_min'] - 1:], lat_grid_v200, lon_grid_v200, m, according_fullname)
#                #rgcpd.print_particular_region(according_number, Corr_precursor[:, :], lat_grid_v200, lon_grid_v200, m, according_fullname)
#                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                number_region = according_number
#                Corr_Coeff = Corr_precursor[:, :]
#                
#                title  = according_fullname
#                # preparations
#                lag_steps = Corr_Coeff.shape[1]
#                lat_len = actor.lat_grid.shape[0]
#                lon_len = actor.lon_grid.shape[0]
#                
#                 	
#                cmap_regions = matplotlib.colors.ListedColormap(sns.color_palette("Set2"))
#                cmap_regions.set_bad('w')
#                               
##                x = 0
##                vmax = 50
##                for i in range(lag_steps):
##                    Regions_lag_i = rgcpd.define_regions_and_rank_new(Corr_Coeff[:,i], actor.lat_grid, actor.lon_grid)
##                    n_regions_lag_i = int(Regions_lag_i.max())
##                		
##                    x_reg = np.max(Regions_lag_i)	
##                    levels = np.arange(x, x + x_reg +1)+.5
#                    
##                    A_r = A_r + x
##                				
##                    x = A_r.max() 
##                		
##                    print x
##                    if x >= number_region:
##                        A_number_region = np.zeros(A_r.shape)
##                			
##                        A_number_region[A_r == number_region]=1
##                        A_number_region[A_r != number_region]=np.nan
##                        #values[ values==0 ] = np.nan
##                        print('count')
##                        print(count)
##                        if count == 0:
##                            cs =  m.contourf(lon_mesh,lat_mesh, A_number_region, latlon = True, colors = ["deepskyblue", "black", "white"]) #SLP
##                        elif count == 1:
##                            cs1 =  m.contourf(lon_mesh,lat_mesh, A_number_region, latlon = True, colors = ["royalblue" , "black", "white"]) #V200Japan
##                        elif count == 2:
##                            cs2 =  m.contourf(lon_mesh,lat_mesh, A_number_region, latlon = True, colors = ["lightsalmon","black", "white"]) #V200Panama
##                        elif count == 3:
##                            cs3 =  m.contourf(lon_mesh,lat_mesh, A_number_region, latlon = True,colors = ["crimson", "black", "white"]) #V200Arctic
##                        elif count == 4:
##                            cs4 =  m.contourf(lon_mesh,lat_mesh, A_number_region, latlon = True,colors = ["sandybrown", "black", "white"])
##                        elif count == 5:
##                            cs5 =  m.contourf(lon_mesh,lat_mesh, A_number_region, latlon = True,colors = ["royalblue", "black", "white"])
##                            
##                        elif count == 6:
##                            cs6 =  m.contourf(lon_mesh,lat_mesh, A_number_region, latlon = True,colors = ["khaki", "black", "white"])
#                		
#                		  #if colors should be different for each subplot:
#                        #m.contourf(lon_mesh,lat_mesh, A_r, levels, latlon = True, cmap = cmap_regions, vmin = 1, cmax = vmax)
#                        #m.colorbar(location="bottom")
#                        
##                        m.drawcoastlines(color='gray', linewidth=0.35)
##                        m.drawmapboundary(fill_color='white', color='gray')
##                    
##                        plt.title('{}-{} {} tfreq{}days'.format(RV.name, RV.dataset, timeperiod, exp['tfreq']))
##                        file_name = '{}_{}_SIGN{}_all'.format(params_combination, pc_alpha_name, alpha_level)
##                        plt.savefig(os.path.join(fig_subpath, file_name + file_type1))
##                        plt.savefig(os.path.join(fig_path, file_name + file_type2),dpi=250)
#                # *********************************************************
#                according_fullname = str(according_number) + according_varname
#                name = ''.join([str(index_in_fulldata),'_',according_fullname])
#                count = count +1
#                print("count")
#                print(count)
#            else :
#                print 'Index itself is also causal parent -> skipped' 
#                print('*******************              ***************************                ******************')
#        
#        else:
#            print("vars not sign at alpha = 0.05")
            
    
#    m.drawcoastlines(color='gray', linewidth=0.35)
#    m.drawmapboundary(fill_color='white', color='gray')
#
#    plt.title('{}-{} {} tfreq{}days'.format(RV.name, RV.dataset, timeperiod, exp['tfreq']))
#    file_name = '{}_{}_SIGN{}_all'.format(params_combination, pc_alpha_name, alpha_level)
#    plt.savefig(os.path.join(fig_subpath, file_name + file_type1))
#    plt.savefig(os.path.join(fig_path, file_name + file_type2),dpi=250)
    
    # *****************************************************************************
    # save output if SaveTF == True
    # *****************************************************************************
    
    if SaveTF == True:
        sys.stdout = orig_stdout
        f.close()    
        # reopen the file to reorder the lines
        in_file=open(''.join([fig_subpath,'old.txt']),"rb")     
        contents = in_file.read()
        in_file.close()    
        cont_split = contents.splitlines()
        # save a new file    
        in_file=open(''.join([fig_subpath, params_combination,'pcA',pc_alpha_name,'_SIGN',str(alpha_level),'.txt']),"wb")
        for i in range(0,len(cont_split)):
            in_file.write(cont_split[i]+'\r\n')
        in_file.close()
        # delete old file
        import os
        os.remove(''.join([fig_subpath,'old.txt']))  
          
            
