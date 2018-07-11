# -*- coding: utf-8 -*-
#%%
import os
os.chdir('/Users/semvijverberg/surfdrive/VU_Amsterdam/Scripts/Tigramite')
script_dir = os.getcwd()
import subprocess
runfile = os.path.join(script_dir, 'saving_repository_to_Github.sh')
subprocess.call(runfile)
#%%

import matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"

from pylab import *
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
from netCDF4 import Dataset
from netcdftime import utime
from datetime import datetime, date, timedelta
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
import pandas
from pandas import DataFrame
import scipy
from scipy import signal
from datetime import datetime 
import datetime
from matplotlib.patches import Polygon
import seaborn as sns
import numpy as np

from matplotlib import gridspec

import sklearn
import sys, os

# add my own python functions
sys.path.append('/Users/semvijverberg/surfdrive/VU_Amsterdam/Scripts/Tigramite')
#%%
#import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.models import LinearMediation, Prediction
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb

# import Marlene's RGCPD functions
import RGCPD_functions_version_03 as rgcpd

import statsmodels.api as sm
from sklearn import preprocessing
import statsmodels.api as sm

# paths

file_type1 = ".pdf"
file_type2 = ".png"

SaveTF = False
plot_all = True

fig_path = '/Users/semvijverberg/surfdrive/Data_ERAint/output/output_tigr_SST_T2m'

#=====================================================================================
# 0) Parameters which must be specified
#=====================================================================================
n_years = 38
timeperiod = '1979-2019'
# time-cycle of data
# 12 for 7days, 365 for daily etc...
time_cycle = 52

# if complete time-series should be considered, n_steps = time-cycle
# if only specific time-steps are considered (e.g. DJF then n-step = 3), n_steps says how many:

n_steps = 17 # days of the summer season # means that 5 months out of the 12 are considered

 
# Start month is index of first relevant time-step
# start_day = time-cycle, if alls values of the years should be considered
start_day = 22 # means start with November (= 10ther entry in array), pythonic counting

seas_indices = range(n_steps)

# significnace level for correlation maps
alpha = 0.01 #  try different values 0.001, 0.005, 0.05, 0.1
alpha_fdr = 2*alpha

alpha_level_vec = [0.1, 0.3, 0.9, 0.9]
# which is the minimum lag whoch should be considered


lag_min = 1
lag_max = 2 # should be <time_cycle
# select domain
la_min = -89
la_max = 89
lo_min = -180
lo_max = 360

params_combination = ''.join([str(lag_min),'-',str(lag_max), '_months', str(start_day), '-', str(start_day+n_steps-1), '_alphaCorr',str(alpha)])



# map for plotting the correlation maps
#m = Basemap(llcrnrlon=-180,llcrnrlat=-89,urcrnrlon=180,urcrnrlat=85, projection='mill')
#m = Basemap(llcrnrlon=-180,llcrnrlat=-89,urcrnrlon=180,urcrnrlat=85, projection='cyl')
m = Basemap(projection='hammer',lon_0 = 0 ,resolution='c')
#m = Basemap(llcrnrlon=-180,llcrnrlat=-89,urcrnrlon=180,urcrnrlat=85, projection='cea')


#=====================================================================================
# Calculates the indices which are taken for response-variable
#=====================================================================================
lag_steps = lag_max - lag_min +1
time_range_all = [0, time_cycle * n_years]
RV_indices = []

# case 1) complete time-series
# cut-off the first year
if n_steps == time_cycle:
	RV_indices = range(time_cycle*n_years)[time_cycle:]

# case 2) starts with Jan or lag_max is in previous year:
# cutoff the first year
elif start_day - lag_max < 0: #5- 5 = 0 
	for i in range(n_steps):
		a = range(time_cycle*n_years)[time_cycle:][start_day +i::time_cycle]
		RV_indices = RV_indices + a

#case 3) winter overlap DJ
# cut-off last year except winter
elif start_day + n_steps >=time_cycle: #5+4 =9
	for i in range(n_steps):
		a = range(time_cycle*n_years)[:-(start_day + n_steps -time_cycle)][start_day +i::time_cycle]
		RV_indices = RV_indices + a

#case 4) all good.
else:
	for i in range(n_steps):
		a = range(time_cycle*n_years)[start_day +i::time_cycle]
		RV_indices = RV_indices + a

	
RV_indices.sort()



#==================================================================================
# 1) Define index of interest (here: Polar vortex index based on gph)
#==================================================================================

#===================================================================
# Important: 
# index is defined over all time-steps. 
# Later in step 2 only the specific steps will be extracted
#====================================================================
# load preprocessed predictant
#np.ma.make_mask(clusters.sel(cluster=0)>1*clusters.sel(cluster=0).std())
path_input = '/Users/semvijverberg/surfdrive/Circulation-Regimes/input/'
mt_rain =  Dataset('/p/projects/gotham/giorgia/Rainfall/prcp_GLB_daily_1979-2016_del29feb.75-88E_18-25N.7days.nc')
MT_rain = mt_rain.variables['prcp'][:,:,:].squeeze()
MT_rain.shape
#shear.shape     
months , nlats, nlons = MT_rain.shape # [months , lat, lon]

#=====================================================================================
# mean over longitude and latitude

MT_rain1D = numpy.mean(MT_rain, axis = (1,2))

#=====================================================================================
# # detrend in each time-step	
for j in range(time_cycle):
    MT_rain1D[j::time_cycle] = scipy.signal.detrend(MT_rain1D[j::time_cycle], axis = 0)    
    #MT_rain1D1[j::time_cycle] = scipy.signal.detrend(MT_rain1D1[j::time_cycle], axis = 0)

#calculate anomalies 
for i in range(int(time_cycle)):
    MT_rain1D[i::int(time_cycle)] = (MT_rain1D[i::int(time_cycle)] - np.mean(MT_rain1D[:time_cycle*n_years][i::int(time_cycle)], axis = 0))	
    #MT_rain1D1[i::int(time_cycle)] = (MT_rain1D1[i::int(time_cycle)] - np.mean(MT_rain1D1[:time_cycle*n_years][i::int(time_cycle)], axis = 0))	

#=====================================================================================
# 2) extract specific months of MT index 
#=====================================================================================

MT_index = MT_rain1D[RV_indices]



#=====================================================================================
#
# 3) DEFINE PRECURSOS COMMUNITIES:
# - calculate and plot pattern correltion for differnt fields
# - create time-series over these regions 
#
#=====================================================================================




#===========================================
# 3c) Precursor field = v200
#===========================================
v200 = Dataset('/p/projects/gotham/giorgia/ERA.interim/WIND.data/U_V200mb.1979-2016.daily.del29feb.7days.nc', 'r')
V200 = v200.variables['var132'][:,:,:].squeeze() #/100 ??
V200 = V200
# # # detrend in each time-step	
for j in range(time_cycle):
	V200[j::time_cycle] = scipy.signal.detrend(V200[j::time_cycle], axis = 0)
		
#calculate anomalies
for i in range(int(time_cycle)):
    V200[i::int(time_cycle)] = (V200[i::int(time_cycle)] - np.mean(V200[:time_cycle*n_years][i::int(time_cycle)], axis = 0))


box = [la_min, la_max, lo_min, lo_max]

# add rgcpd
Corr_V200, lat_grid_v200, lon_grid_v200 = rgcpd.calc_corr_coeffs_new(v200, V200, box, MT_index,time_range_all, lag_min, lag_max,
                                                                  time_cycle, RV_indices, alpha_fdr, FDR_control=False)

#plot
fig_corr_V200 = rgcpd.plot_corr_coeffs(Corr_V200, m, lag_min, lat_grid_v200, lon_grid_v200,\
		title= 'V200', Corr_mask=False)
  #plot
#fig_corr_SLP = rgcpd.plot_corr_coeffs(Corr_SLP, m, lag_min, lat_grid_slp, lon_grid_slp,\#
#		title= 'SLP', Corr_mask=True)

fig_file = ''.join(['2Act_V200_',timeperiod,'.7days.corr_maps.lag', params_combination, file_type1])#,'.pdf' ])
plt.savefig(''.join([fig_path, fig_file]))   
fig_file = ''.join(['2Act_V200_',timeperiod,'.7days.corr_maps.lag', params_combination, file_type2])#,'.pdf' ])
plt.savefig(''.join([fig_path, fig_file]))   


V200_box = rgcpd.extract_data(v200, V200, time_range_all, box)
# reshape
V200_box = np.reshape(V200_box, (V200_box.shape[0], -1))

#m = Basemap(llcrnrlon=0,llcrnrlat=-20,urcrnrlon=360,urcrnrlat=85,projection='mill')
Actors_V200, n_reg_perlag_V200, fig_V200 = rgcpd.calc_actor_ts_and_plot(Corr_V200, V200_box, lag_min, lat_grid_v200, lon_grid_v200, m, 'V200 actors')





#=====================================================================================
#
# 4) PCMCI-algorithm
#
#=====================================================================================
# ======================================================================================================================
# Choose
# ======================================================================================================================
alpha_level = alpha_level_vec[l]

# set of pc_alpha values
pcA_none = None# default
pcA_set1a = [ 0.05] # 0.05 0.01 
pcA_set1b = [ 0.01] # 0.05 0.01 
pcA_set1c = [ 0.1] # 0.05 0.01 
pcA_set2 = [ 0.2, 0.1, 0.05, 0.01, 0.001] # set2
pcA_set3 = [ 0.1, 0.05, 0.01] # set3
pcA_set4 = [ 0.5, 0.4, 0.3, 0.2, 0.1] # set4

for p in [0,]:#range(7):
        
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
    tau_min = lag_min
    tau_max = lag_min 
    
    Act_v200 = Actors_V200[:,:]
    #Act_slp = Actors_SLP[:,int(np.sum(n_reg_perlag_SLP[:tau_min - lag_min ])):]
    
    # stack actor time-series together:
    fulldata = Act_v200#numpy.concatenate((Act_v200), axis = 1)
    
    # add index of interest as first entry (here PCH):
    fulldata = numpy.column_stack((MT_rain1D, fulldata))
    # save fulldata
    file_name = ''.join([ params_combination, '_fulldata'])#,'.pdf' ])
    fulldata.dump(''.join([fig_path, file_name]))  
    
    # create arry which contains number of region and variable name for each entry in fulldata:
    var_V200 = [[i+1, 'V200', 0] for i in range(Act_v200.shape[1])]
    #var_SLP = [[i+1, 'SLP', 1] for i in range(Act_slp.shape[1])]
    
    # first entry is index of interest
    var_names = [[0, 'MT']] + var_V200 
        
    
    file_name = ''.join(['_maps.lag', params_combination, '_var_name'])#,'.pdf' ])
    var_names_np = numpy.asanyarray(var_names)
    var_names_np.dump(''.join([fig_path, file_name]))  
       
      
    ## ======================================================================================================================
    
    # ======================================================================================================================
    # tigramite 3
    # ======================================================================================================================
    data = fulldata
    
    # ======================================================================================================================
    # new mask
    # ======================================================================================================================
    print(data.shape)
    
    data_mask = np.ones(data.shape, dtype='bool')
    for i in range(4): # take into account 4 months starting from june=5
        data_mask[5+i:: 12,:] = False # [22+i:: 52,:]
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
        selected_variables=None, #[0], # only parents for the monsoon trough rainfall
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
    parents_MT = all_parents[0]
    
     #==========================================================================
    # multiple testing problem:
    #==========================================================================
    precursor_fields = [ 'V200']
    Corr_precursor_ALL = [ Corr_V200 ]
    
    n_parents = len(parents_MT)
    a_plot = alpha_level
    
    #indices_parents_PoV = [pvalues_PoV.index(i) for i in parents_PoV]
    for i in range(n_parents):
        link_number = parents_MT[i][0]
        lag = np.abs(parents_MT[i][1])-1
        index_in_fulldata = parents_MT[i][0]
        if index_in_fulldata>0:
       
            according_varname = var_names[index_in_fulldata][1]
            according_number = var_names[index_in_fulldata][0]
            acording_field_number = var_names[index_in_fulldata][2]
            print("index_in_fulldata")
            print(index_in_fulldata)
            print("according_varname")
            print(according_varname)
            print("according_number")
            print(according_number)
            print("acording_field_number")
            print(acording_field_number)
            # *********************************************************
            # print and save only sign regions
            # *********************************************************
            according_fullname = str(according_number) + according_varname   
            Corr_precursor = Corr_precursor_ALL[acording_field_number]
           
            rgcpd.print_particular_region(according_number, Corr_precursor[:, :], lat_grid_v200, lon_grid_v200, m, according_fullname)
            fig_file = ''.join([str(i),'_par',according_fullname,'_lag', params_combination,'_tau',str(tau_min),'-',str(tau_max),
                                    'pc_A',pc_alpha_name,'_SIGN',str(a_plot),'_lag',str( parents_MT[i][1]), file_type1])#,'.pdf' ])
            plt.savefig(''.join([fig_path, fig_file]))   
               
            print('                                        ')
            # *********************************************************                                
            # save data
            # *********************************************************
            according_fullname = str(according_number) + according_varname
            name = ''.join([str(index_in_fulldata),'_',according_fullname])
            #fulldata[:,index_in_fulldata].dump(''.join(['/home/dicapua/GOTHAM/Data/output/RGCPD/monthly/precurs_',name,'_2Act.monthly.lag', params_combination,
            #'_t_min_',str(tau_min),'_t_max_', str(tau_max),'pc_alpha',str(pc_alpha),'.lag',str( parents_MT[i][1]), 'sign_lev',str(round(adjusted_pvalues[link_number, lag],4)) ]))
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
            link_number = parents_MT[i][0]
            lag = np.abs(parents_MT[i][1])-1
            #==========================================================================
            # save precursos indices form fulldata, to be stored and used to find the 
            # precursors of the precursors
            #==========================================================================
            index_in_fulldata = parents_MT[i][0]
            if index_in_fulldata>0:
               
                according_varname = var_names[index_in_fulldata][1]
                according_number = var_names[index_in_fulldata][0]
                acording_field_number = var_names[index_in_fulldata][2]
                # *********************************************************
                # print and save only sign regions
                # *********************************************************
                according_fullname = str(according_number) + according_varname   
                Corr_precursor = Corr_precursor_ALL[acording_field_number]
               
                #oad.print_particular_region(according_number, Corr_precursor[:, tau_min - 1:], lat_grid_v200, lon_grid_v200, m, according_fullname)
                #rgcpd.print_particular_region(according_number, Corr_precursor[:, :], lat_grid_v200, lon_grid_v200, m, according_fullname)
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                number_region = according_number
                Corr_GPH = Corr_precursor[:, :]
                lat_grid_gph= lat_grid_v200
                lon_grid_gph = lon_grid_v200
                
                title  = according_fullname
                # preparations
                lag_steps = Corr_GPH.shape[1]
                la_gph = lat_grid_gph.shape[0]
                lo_gph = lon_grid_gph.shape[0]
                lons_gph, lats_gph = numpy.meshgrid(lon_grid_gph, lat_grid_gph)
                 	
                cmap_regions = matplotlib.colors.ListedColormap(sns.color_palette("Set2"))
                cmap_regions.set_bad('w')
                
                
                x = 0
                vmax = 50
                for i in range(lag_steps):
                    Regions_lag_i = rgcpd.define_regions_and_rank_new(Corr_GPH[:,i], lat_grid_gph, lon_grid_gph)
                    n_regions_lag_i = int(Regions_lag_i.max())
                		
                    x_reg = numpy.max(Regions_lag_i)	
                    levels = numpy.arange(x, x + x_reg +1)+.5
                
                    A_r = numpy.reshape(Regions_lag_i, (la_gph, lo_gph))
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
                            cs =  m.contourf(lons_gph,lats_gph, A_number_region, latlon = True, colors = ["deepskyblue", "black", "white"]) #SLP
                        elif count == 1:
                            cs1 =  m.contourf(lons_gph,lats_gph, A_number_region, latlon = True, colors = ["royalblue" , "black", "white"]) #V200Japan
                        elif count == 2:
                            cs2 =  m.contourf(lons_gph,lats_gph, A_number_region, latlon = True, colors = ["lightsalmon","black", "white"]) #V200Panama
                        elif count == 3:
                            cs3 =  m.contourf(lons_gph,lats_gph, A_number_region, latlon = True,colors = ["crimson", "black", "white"]) #V200Arctic
                        elif count == 4:
                            cs4 =  m.contourf(lons_gph,lats_gph, A_number_region, latlon = True,colors = ["sandybrown", "black", "white"])
                        elif count == 5:
                            cs5 =  m.contourf(lons_gph,lats_gph, A_number_region, latlon = True,colors = ["royalblue", "black", "white"])
                            
                        elif count == 6:
                            cs6 =  m.contourf(lons_gph,lats_gph, A_number_region, latlon = True,colors = ["khaki", "black", "white"])
                		
                		  #if colors should be different for each subplot:
                        #m.contourf(lons_gph,lats_gph, A_r, levels, latlon = True, cmap = cmap_regions, vmin = 1, cmax = vmax)
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
    
    plt.title(''.join(['CPCrain-ERA-I MT ',timeperiod]))
    plt.savefig(''.join([fig_path, 'lag', params_combination,'pc_A',pc_alpha_name,'_SIGN',str(a_plot),'all', file_type1]))
    plt.savefig(''.join([fig_path, 'lag', params_combination,'pc_A',pc_alpha_name,'_SIGN',str(a_plot),'all', file_type2]))
    
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
        in_file=open(''.join([fig_path,'lag', params_combination,'pc_A',pc_alpha_name,'_SIGN',str(a_plot),'.txt']),"wb")
        for i in range(0,len(cont_split)):
            in_file.write(cont_split[i]+'\r\n')
        in_file.close()
        # delete old file
        import os
        os.remove(''.join([fig_path,'old.txt']))  
          
            
