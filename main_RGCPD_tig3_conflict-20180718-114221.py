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
import RGCPD_functions_version_03 as rgcpd
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
from inspect import getsourcelines
def seefunction(func):
    print getsourcelines(func)
#%% saving to github
#runfile = os.path.join(script_dir, 'saving_repository_to_Github.sh')
#subprocess.call(runfile)
#%%
exp_clus = '13Jul-24Aug_ward'
path_exp_clus = os.path.join('/Users/semvijverberg/surfdrive/Data_ERAint/t2m_sst_m5-8_dt14/', exp_clus)
if os.path.isdir(fig_path):
    pass
else:
    os.makedirs(fig_path)
exp = np.load(os.path.join(path_exp_clus, 'raw_input_tig_dic.npy')).item()
# =============================================================================
# # Expand dictionary for experiment
# =============================================================================
RV = exp['t2m']
file_path = os.path.join(RV.path_pp, RV.filename_pp)
ncdf = Dataset(file_path)
numtime = ncdf.variables['time']
dates = pd.to_datetime(num2date(numtime[:], units=numtime.units, calendar=numtime.calendar))
print("\nCheck if dates are correct, should be all same day acros years")
dates[::dates[dates.year == 1979].size]
exp['clus_anom_std'] = 1
exp['alpha'] = 0.01 # significnace level for correlation maps
exp['alpha_fdr'] = 2*exp['alpha'] # conservative significance level
exp['lag_min'] = 3
exp['lag_max'] = 4#dates[dates.year == 1979].size - exp['RV_oneyr'].size
exp['alpha_level_tig'] = 0.5 # meaning of this one?
exp['FDR_control'] = False
exp['la_min'] = -89 # select domain
exp['la_max'] = 89
exp['lo_min'] = -180
exp['lo_max'] = 360
exp['time_cycle'] = dates[dates.year == 1979].size # time-cycle of data. total timesteps in one year
exp['pcA_range'] = [0,] 
exp['path_lags'] = os.path.join(exp['path_exp_clus'], 'lag{}to{}'.format(exp['lag_min'],exp['lag_max']))
fig_path = os.path.join(exp['path_lags'], 'output_tigr_SST_T2m/')
# sets of pc_alpha values
pcA_set1a = [ 0.05] # 0.05 0.01 
pcA_set1b = [ 0.01] # 0.05 0.01 
pcA_set1c = [ 0.1] # 0.05 0.01 
pcA_set2 = [ 0.2, 0.1, 0.05, 0.01, 0.001] # set2
pcA_set3 = [ 0.1, 0.05, 0.01] # set3
pcA_set4 = [ 0.5, 0.4, 0.3, 0.2, 0.1] # set4
pcA_none = None # default


#%%
file_type1 = ".pdf"
file_type2 = ".png"
SaveTF = True
plot_all = False
#=====================================================================================
# Calculates the indices which are taken for response-variable
#=====================================================================================
n_years = dates.year.max() - dates.year.min() 
timeperiod = '{}-{}'.format(RV.startyear, RV.endyear)
time_range_all = [0, RV.dates_np.size]
RV_indices = exp['RV_period']
params_combination = 'lag{}to{}_aCorr{}'.format(exp['lag_min'],
                         exp['lag_max'], exp['alpha'])

#%%
#==================================================================================
# 1) Define index of interest (here: Polar vortex index based on gph)
#==================================================================================
# Important: 
# index is defined over all time-steps. 
# Later in step 2 only the specific steps will be extracted
#====================================================================
# load preprocessed predictant 
RV_array = np.squeeze(functions_tig.import_array(RV))
RV_array, region_coords = functions_tig.find_region(RV_array, region='U.S.')
RV_array.shape
#shear.shape     
time , nlats, nlons = RV_array.shape # [months , lat, lon]
#=====================================================================================
# mean over longitude and latitude
cluster = 0
clusters = np.squeeze(xr.Dataset.from_dict(np.load(os.path.join(exp['path_exp_clus'], 'clusters_dic.npy')).item()).to_array())
cluster_out = clusters.sel(cluster=cluster)
RV_masked = np.ma.masked_where((cluster_out < exp['clus_anom_std']*cluster_out.std()), cluster_out).mask
functions_tig.xarray_plot(cluster_out)
plt.imshow(RV_masked)
# fix this
RV_region = np.ma.masked_array(data=RV_array, mask=np.reshape(np.repeat(RV_masked, RV_array.time.size), RV_array.shape))
RV1D = np.ma.mean(RV_region, axis = (1,2))
#=====================================================================================
# 2) extract specific months of MT index 
#=====================================================================================
RV_index = RV1D[RV_indices]
#=====================================================================================

#%%

# 3) DEFINE PRECURSOS COMMUNITIES:
# - calculate and plot pattern correltion for differnt fields
# - create time-series over these regions 

#=====================================================================================
outd = dict()
class act:
    def __init__(self, name, Corr_mask, lat_grid, lon_grid, actbox, tsCorr, n_reg_perlag, fig):
        self.name = var
        self.Corr_mask = Corr_mask
        self.lat_grid = lat_grid
        self.lon_grid = lon_grid
        self.actbox = actbox
        self.tsCorr = tsCorr
        self.n_reg_perlag = n_reg_perlag
        self.fig = fig

# map for plotting the correlation maps
#m = Basemap(llcrnrlon=-180,llcrnrlat=-89,urcrnrlon=180,urcrnrlat=85, projection='mill')
#m = Basemap(llcrnrlon=-180,llcrnrlat=-89,urcrnrlon=180,urcrnrlat=85, projection='cyl')
m = Basemap(projection='hammer',lon_0 = int(cluster_out.longitude.mean()) ,resolution='c')
#m = Basemap(llcrnrlon=-180,llcrnrlat=-89,urcrnrlon=180,urcrnrlat=85, projection='cea')
allvar = exp['vars'][0]
for var in allvar:
    print var
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
    Corr_mask, lat_grid, lon_grid = rgcpd.calc_corr_coeffs_new(ncdf, array, box, RV_index, time_range_all, exp['lag_min'], 
                                                          exp['lag_max'], exp['time_cycle'], RV_indices, 
                                                          exp['alpha_fdr'], FDR_control=exp['FDR_control'])
    # =============================================================================
    # Plot    
    # =============================================================================
    fig_corr_act1 = rgcpd.plot_corr_coeffs(Corr_mask, m, exp['lag_min'], lat_grid, lon_grid,\
                                            title=var, Corr_mask=False)
    fig_filename = '{}_vs_{}_{}'.format(allvar[0], var, params_combination) + file_type2
    plt.savefig(os.path.join(fig_path, fig_filename), dpi=250)  
    # =============================================================================
    # what happens here?    
    # =============================================================================
    actbox = rgcpd.extract_data(ncdf, array, time_range_all, box)
    actbox = np.reshape(actbox, (actbox.shape[0], -1))
    tsCorr, n_reg_perlag, fig = rgcpd.calc_actor_ts_and_plot(Corr_mask, actbox, 
                            exp['lag_min'], lat_grid, lon_grid, m, var+' actors')
    outd[var] = act(var, Corr_mask, lat_grid, lon_grid, actbox, tsCorr, n_reg_perlag, fig)

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
for p in exp['pcA_range']:#range(7):
        
    print( ' run tigramite 3, run.pcmci')
    print(p)
    '''
    save output
    '''
    if SaveTF == True:
        orig_stdout = sys.stdout
        sys.stdout = f = open(''.join([fig_path,'old.txt']), 'a')
    
    #for only_one_lag in [0, ]:#range(lag_max):
    # check with  different lags
    # used combinations of parameters: 3-4, 3-5
    tau_min = exp['lag_min']
    tau_max = exp['lag_max'] 
    
    actorlist = []
    for var in allvar[1:]:
        print var
        actor = outd[var]
        actorlist.append(actor.tsCorr)
        
        # create array which contains number of region and variable name for each entry in fulldata:
        idx = allvar.index(var) - 1
        actor.var_info = [[i+1, var, idx] for i in range(actor.tsCorr.shape[1])]
    # stack actor time-series together:
    fulldata = np.concatenate(tuple(actorlist), axis = 1)
    # add index of interest as first entry (here PCH):
    fulldata = np.column_stack((RV1D, fulldata))
    # save fulldata
    file_name = 'fulldata_{}'.format(params_combination)#,'.pdf' ])
    fulldata.dump(os.path.join(fig_path, file_name+'.pkl'))
    
    # first entry is index of interest
    var_names = [[0, allvar[0]]] + actor.var_info 
    file_name = 'list_actors_{}'.format(params_combination)
    var_names_np = np.asanyarray(var_names)
    var_names_np.dump(os.path.join(fig_path, file_name+'.pkl'))  
    ## ======================================================================================================================
    
    # ======================================================================================================================
    # tigramite 3
    # ======================================================================================================================
    data = fulldata
    
    # ======================================================================================================================
    # new mask
    # ======================================================================================================================
    print(data.shape)
    
    # RV mask False for period that I want to analyse
    idx_start_RVperiod = int(np.where(dates[dates.year == 1980] == exp['RV_oneyr'][0])[0])
    data_mask = np.ones(data.shape, dtype='bool') # true for actor months, false for RV months
    for i in range(exp['RV_oneyr'].size): # total timesteps RV period, 12 is 
        data_mask[idx_start_RVperiod+i:: exp['time_cycle'],:] = False # [22+i:: 52,:]
#    for i in range(n_steps): # take into account 4 months starting from june=5
#        data_mask[start_day+i:: exp['time_cycle'],:] = False # [22+i:: 52,:]
    ##
    T, N = data.shape
    
    # ======================================================================================================================
    # Initialize dataframe object (needed for tigramite functions)
    # ======================================================================================================================
    dataframe = pp.DataFrame(data=data, mask=data_mask)
    
    # Specify time axis and variable names
    datatime = np.arange(len(data))
    
    # ======================================================================================================================
    # pc algorithm: only parents for selected_variables are calculated (here entry[0] = PoV)
    # ======================================================================================================================
    
    
    parcorr = ParCorr(significance='analytic', 
                      use_mask =True, 
                      mask_type='y', 
                      verbosity=2)
    pcmci = PCMCI(
        dataframe=dataframe, 
        cond_ind_test=parcorr,
        var_names=var_names,
        selected_variables=None, # consider precursor only of RV variable, then selected_variables should be None
        verbosity=2)
        
    # ======================================================================================================================
    # results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha = pc_alpha, tau_min = tau_min, max_combinations=1  )
    # ======================================================================================================================
    
    if p == 0:
        pc_alpha = pcA_set1a 
        pc_alpha_name = str(pcA_set1a)
    elif p == 1:
        pc_alpha = pcA_set1b
        pc_alpha_name = str(pcA_set1b)
    elif p == 2:
        pc_alpha = pcA_set1c
        pc_alpha_name = str(pcA_set1c)
    elif p == 3:
        pc_alpha = pcA_set2
        pc_alpha_name = 'set2'
    elif p == 4:
        pc_alpha = pcA_set3
        pc_alpha_name = 'set3'
    elif p == 5:
        pc_alpha = pcA_set4
        pc_alpha_name = 'set4'
    elif p == 6:
        pc_alpha = pcA_none
        pc_alpha_name = 'none'
        
    # ======================================================================================================================
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha = pc_alpha, tau_min = tau_min, max_combinations=1  ) #selected_links = dictionary/None
    #results = pcmci.run_pcmci(selected_links =None, tau_max=tau_max, pc_alpha = pc_alpha, tau_min = tau_min,save_iterations=False,  max_conds_dim=None, max_combinations=1, max_conds_py=None, max_conds_px=None) #selected_links = dictionary/None
    #results = pcmci.run_pcmci(selected_links =  dictionary, tau_max=tau_max, pc_alpha = pc_alpha, tau_min = tau_min,save_iterations=False,  max_conds_dim=None, max_combinations=1, max_conds_py=None, max_conds_px=None) #selected_links = dictionary/None
    
    q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], fdr_method='fdr_bh')
    
    pcmci._print_significant_links(
            p_matrix = results['p_matrix'], 
            q_matrix = q_matrix, #results['p_matrix']
        
            val_matrix = results['val_matrix'],
            alpha_level = alpha_level)
            
    sig = pcmci._return_significant_parents(
                                pq_matrix=q_matrix,
                                val_matrix=results['val_matrix'], 
                                alpha_level=alpha_level)
    
    all_parents = sig['parents']
    link_matrix = sig['link_matrix']
    
    # print all_parents
    # print all_parents
    # print link_matrix
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
    
     #==========================================================================
    # multiple testing problem:
    #==========================================================================
    precursor_fields = exp['vars'][0][1:]
    Corr_precursor_ALL = [actor.Corr_mask]
    
    n_parents = len(parents_RV)
    a_plot = alpha_level
    
    #indices_parents_PoV = [pvalues_PoV.index(i) for i in parents_PoV]
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
            # print and save only sign regions
            # *********************************************************
            according_fullname = str(according_number) + according_varname   
            Corr_precursor = Corr_precursor_ALL[according_field_number]
           
            rgcpd.print_particular_region(according_number, Corr_precursor[:, :], actor.lat_grid, actor.lon_grid, m, according_fullname)
            fig_file = '{}par)_{}_{}_pcA{}_sign{}_{}{}'.format(i,according_fullname,
                        params_combination,pc_alpha_name,a_plot,str(parents_RV[i][1]),file_type1)
#                                    'pcA',pc_alpha_name,'_SIGN',str(a_plot),'_lag',str( parents_RV[i][1]), file_type1])#,'.pdf' ])
            plt.savefig(os.path.join(fig_path, fig_file))   
               
            print('                                        ')
            # *********************************************************                                
            # save data
            # *********************************************************
            according_fullname = str(according_number) + according_varname
            name = ''.join([str(index_in_fulldata),'_',according_fullname])
            #fulldata[:,index_in_fulldata].dump(''.join(['/home/dicapua/GOTHAM/Data/output/RGCPD/monthly/precurs_',name,'_2Act.monthly.lag', params_combination,
            #'_t_min_',str(tau_min),'_t_max_', str(tau_max),'pc_alpha',str(pc_alpha),'.lag',str( parents_RV[i][1]), 'sign_lev',str(round(adjusted_pvalues[link_number, lag],4)) ]))
            print(fulldata[:,index_in_fulldata].size)
            print(name)
        else :
            print 'Index itself is also causal parent -> skipped' 
            print('*******************              ***************************                ******************')
    
    if plot_all == True:
        m = Basemap(projection='hammer',lon_0 = 0 ,resolution='c')     #300       
        #plt.plot()
        count = 0
        #fig = plt.figure(figsize=(6, 4))
        fig = plt.subplots(figsize=(6, 4))
         	
        for i in range(n_parents):
            link_number = parents_RV[i][0]
            lag = np.abs(parents_RV[i][1])-1
            #==========================================================================
            # save precursos indices form fulldata, to be stored and used to find the 
            # precursors of the precursors
            #==========================================================================
            index_in_fulldata = parents_RV[i][0]
            if index_in_fulldata>0:
               
                according_varname = var_names[index_in_fulldata][1]
                according_number = var_names[index_in_fulldata][0]
                according_field_number = var_names[index_in_fulldata][2]
                # *********************************************************
                # print and save only sign regions
                # *********************************************************
                according_fullname = str(according_number) + according_varname   
                Corr_precursor = Corr_precursor_ALL[according_field_number]
               
                #oad.print_particular_region(according_number, Corr_precursor[:, tau_min - 1:], lat_grid_v200, lon_grid_v200, m, according_fullname)
                #rgcpd.print_particular_region(according_number, Corr_precursor[:, :], lat_grid_v200, lon_grid_v200, m, according_fullname)
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                number_region = according_number
                Corr_Coeff = Corr_precursor[:, :]
                
                title  = according_fullname
                # preparations
                lag_steps = Corr_Coeff.shape[1]
                lat_len = actor.lat_grid.shape[0]
                lon_len = actor.lon_grid.shape[0]
                lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
                 	
                cmap_regions = matplotlib.colors.ListedColormap(sns.color_palette("Set2"))
                cmap_regions.set_bad('w')
                
                
                x = 0
                vmax = 50
                for i in range(lag_steps):
                    Regions_lag_i = rgcpd.define_regions_and_rank_new(Corr_Coeff[:,i], actor.lat_grid, actor.lon_grid)
                    n_regions_lag_i = int(Regions_lag_i.max())
                		
                    x_reg = np.max(Regions_lag_i)	
                    levels = np.arange(x, x + x_reg +1)+.5
                
                    A_r = np.reshape(Regions_lag_i, (lat_len, lon_len))
                    A_r = A_r + x
                				
                    x = A_r.max() 
                		
                    print x
                    if x >= number_region:
                        A_number_region = np.zeros(A_r.shape)
                			
                        A_number_region[A_r == number_region]=1
                        A_number_region[A_r != number_region]=np.nan
                        #values[ values==0 ] = np.nan
                        print('count')
                        print(count)
                        if count == 0:
                            cs =  m.contourf(lon_mesh,lat_mesh, A_number_region, latlon = True, colors = ["deepskyblue", "black", "white"]) #SLP
                        elif count == 1:
                            cs1 =  m.contourf(lon_mesh,lat_mesh, A_number_region, latlon = True, colors = ["royalblue" , "black", "white"]) #V200Japan
                        elif count == 2:
                            cs2 =  m.contourf(lon_mesh,lat_mesh, A_number_region, latlon = True, colors = ["lightsalmon","black", "white"]) #V200Panama
                        elif count == 3:
                            cs3 =  m.contourf(lon_mesh,lat_mesh, A_number_region, latlon = True,colors = ["crimson", "black", "white"]) #V200Arctic
                        elif count == 4:
                            cs4 =  m.contourf(lon_mesh,lat_mesh, A_number_region, latlon = True,colors = ["sandybrown", "black", "white"])
                        elif count == 5:
                            cs5 =  m.contourf(lon_mesh,lat_mesh, A_number_region, latlon = True,colors = ["royalblue", "black", "white"])
                            
                        elif count == 6:
                            cs6 =  m.contourf(lon_mesh,lat_mesh, A_number_region, latlon = True,colors = ["khaki", "black", "white"])
                		
                		  #if colors should be different for each subplot:
                        #m.contourf(lon_mesh,lat_mesh, A_r, levels, latlon = True, cmap = cmap_regions, vmin = 1, cmax = vmax)
                        #m.colorbar(location="bottom")
                        break
                # *********************************************************
                according_fullname = str(according_number) + according_varname
                name = ''.join([str(index_in_fulldata),'_',according_fullname])
                count = count +1
                print("count")
                print(count)
            else :
                print 'Index itself is also causal parent -> skipped' 
                print('*******************              ***************************                ******************')
        
        else:
            print("vars not sign at alpha = 0.05")
            
    
    m.drawcoastlines(color='gray', linewidth=0.35)
    m.drawmapboundary(fill_color='white', color='gray')
    
    plt.title('{}-{} {} tfreq{}days'.format(RV.name, RV.dataset, timeperiod, exp['tfreq']))
    file_name = '{}_pcA{}_SIGN{}_all{}'.format(params_combination, pc_alpha_name, a_plot)
    plt.savefig(os.path.join(fig_path, file_name + file_type1),dpi=250)
    plt.savefig(os.path.join(fig_path, file_name + file_type2))
    
    # *****************************************************************************
    # save output if SaveTF == True
    # *****************************************************************************
    
    if SaveTF == True:
        sys.stdout = orig_stdout
        f.close()    
        # reopen the file to reorder the lines
        in_file=open(''.join([fig_path,'old.txt']),"rb")     
        contents = in_file.read()
        in_file.close()    
        cont_split = contents.splitlines()
        # save a new file    
        in_file=open(''.join([fig_path, params_combination,'pcA',pc_alpha_name,'_SIGN',str(a_plot),'.txt']),"wb")
        for i in range(0,len(cont_split)):
            in_file.write(cont_split[i]+'\r\n')
        in_file.close()
        # delete old file
        import os
        os.remove(''.join([fig_path,'old.txt']))  
          
            
