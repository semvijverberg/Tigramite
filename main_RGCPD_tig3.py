# -*- coding: utf-8 -*-
#%%
# import socket
# socket.gethostbyname('')
script_dir = '/Users/semvijverberg/surfdrive/Scripts/Tigramite'
import sys, os
os.chdir(script_dir)
sys.path.append('/Users/semvijverberg/surfdrive/Scripts/Tigramite')
#import numpy
#import matplotlib
#matplotlib.rcParams['backend'] = "Qt4Agg"
from mpl_toolkits.basemap import Basemap#, shiftgrid, cm
from netCDF4 import Dataset
from netCDF4 import num2date
import matplotlib.pyplot as plt
#import seaborn as sns
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
import RGCPD_functions_version_04 as rgcpd
import functions_tig
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
# =============================================================================
#  saving to github
# =============================================================================
#runfile = os.path.join(script_dir, 'saving_repository_to_Github.sh')
#subprocess.call(runfile)


#%%
def calculate_corr_maps(filename_exp_design2, map_proj):
    # =============================================================================
    # Load 'exp' dictionairy with information of pre-processed data (variables, paths, filenames, etcetera..)
    # and add RGCPD/Tigrimate experiment settings
    # =============================================================================
    exp = np.load(str(filename_exp_design2)).item()
    # Response Variable is what we want to predict
    RV_name = exp['vars'][0][0]
    RV = exp[RV_name]
    file_path = os.path.join(RV.path_pp, RV.filename_pp)
    ncdf = Dataset(file_path)
    numtime = ncdf.variables['time']
    dates = pd.to_datetime(num2date(numtime[:], units=numtime.units, calendar=numtime.calendar))
    exp['time_cycle'] = dates[dates.year == 1979].size # time-cycle of data. total timesteps in one year
    #=====================================================================================
    # Information on period taken for response-variable, already decided in main_download_and_pp
    #=====================================================================================
    time_range_all = [0, RV.dates_np.size]
    #==================================================================================
    # Start of experiment
    #==================================================================================
    #==================================================================================
    # 1) Define index of interest 
    #==================================================================================
    # load preprocessed predictant 
    RV_array = np.squeeze(functions_tig.import_array(RV))
    RV_array, region_coords = functions_tig.find_region(RV_array, region='U.S.')
    RV_array.shape
    #shear.shape     
    time , nlats, nlons = RV_array.shape # [months , lat, lon]
    #=====================================================================================
    # mean over longitude and latitude
    plt.imshow(exp['RV_masked'])
    # fix this
    RV_region = np.ma.masked_array(data=RV_array, mask=np.reshape(np.repeat(exp['RV_masked'], RV_array.time.size), RV_array.shape))
    RV1D = np.ma.mean(RV_region, axis = (1,2)) # take spatial mean with mask loaded in beginning
    RV_ts = RV1D[exp['RV_period']] # extract specific months of MT index 
     

    # =============================================================================
    # 2) DEFINE PRECURSOS COMMUNITIES:
    # =============================================================================
    # - calculate and plot pattern correltion for differnt fields
    # - create time-series over these regions 
    #=====================================================================================
    outdic_actors = dict()
    class act:
        def __init__(self, name, Corr_Coeff, lat_grid, lon_grid, actbox, tsCorr, n_reg_perlag):
            self.name = var
            self.Corr_Coeff = Corr_Coeff
            self.lat_grid = lat_grid
            self.lon_grid = lon_grid
            self.actbox = actbox
            self.tsCorr = tsCorr
            self.n_reg_perlag = n_reg_perlag
    
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
                                                              exp['lag_max'], exp['time_cycle'], exp['RV_period'], 
                                                              exp['alpha_fdr'], FDR_control=exp['FDR_control'])
        Corr_Coeff = np.ma.array(data = Corr_Coeff[:,:], mask = Corr_Coeff.mask[:,:])
        # =============================================================================
        # Convert regions in time series  
        # =============================================================================
        actbox = rgcpd.extract_data(ncdf, array, time_range_all, box)
        actbox = np.reshape(actbox, (actbox.shape[0], -1))
        tsCorr, n_reg_perlag = rgcpd.calc_actor_ts_and_plot(Corr_Coeff, actbox, 
                                exp['lag_min'], lat_grid, lon_grid, var)
        outdic_actors[var] = act(var, Corr_Coeff, lat_grid, lon_grid, actbox, tsCorr, n_reg_perlag)
        # =============================================================================
        # Plot    
        # =============================================================================
        if exp['plotin1fig'] == False and exp['showplot'] == True:
            fig = functions_tig.xarray_plot_region([var], outdic_actors, exp['lag_min'],  exp['lag_max'], 
                                                   map_proj, exp['tfreq'])
            fig_filename = '{}_vs_{}_{}'.format(allvar[0], var, exp['params_combination']) + exp['file_type2']
            plt.savefig(os.path.join(exp['fig_path'], fig_filename), bbox_inches='tight', dpi=200)
                                           
    
    if exp['plotin1fig'] == True and exp['showplot'] == True:
        variables = outdic_actors.keys()
        fig = functions_tig.xarray_plot_region(variables, outdic_actors, exp['lag_min'],  exp['lag_max'],
                                               map_proj, exp['tfreq'])
        fig_filename = '{}_vs_{}_{}'.format(allvar[0], var, exp['params_combination']) + exp['file_type2']
        plt.savefig(os.path.join(exp['fig_path'], fig_filename), bbox_inches='tight', dpi=200)

    return exp, RV1D, outdic_actors





def run_PCMCI(exp, RV1D, outdic_actors, map_proj):
    #%%
    import os, sys
    #=====================================================================================
    #
    # 4) PCMCI-algorithm
    #
    #=====================================================================================
    # alpha level for multiple linear regression model (last step)
    alpha_level = exp['alpha_level_tig']  
    print('run tigramite 3, run.pcmci')
    print('alpha level(s) for MCI tests: {}'.format(exp['pcA_set']))
    print('alpha level for multiple linear regression model (last step): {}'.format(exp['alpha_level_tig']))
    '''
    save output
    '''
    pc_alpha = exp['pcA_sets'][exp['pcA_set']]   
    if exp['SaveTF'] == True:
        orig_stdout = sys.stdout
        sys.stdout = f = open(''.join([exp['fig_subpath'],'old.txt']), 'a')  
    
    # load Response Variable class
    RV_name = exp['vars'][0][0]
    RV = exp[RV_name]
    # create list with all actors, these will be merged into the fulldata array
    allvar = exp['vars'][0]
    var_names = [[0, allvar[0]]]
    actorlist = []
    for var in allvar[1:]:
        print var
        actor = outdic_actors[var]
        actorlist.append(actor.tsCorr)        
        # create array which numbers the regions
        idx = allvar.index(var) - 1
        actor.var_info = [[i+1, var, idx] for i in range(actor.tsCorr.shape[1])]
        # Array of corresponing regions with var_names (first entry is RV)
        var_names = var_names + actor.var_info 
    # stack actor time-series together:
    fulldata = np.concatenate(tuple(actorlist), axis = 1)
    print('There are {} regions in total'.format(fulldata.shape[1]))
    # add the full 1D time series of interest as first entry:
    fulldata = np.column_stack((RV1D, fulldata))
    # save fulldata
    file_name = 'fulldata_{}'.format(exp['params_combination'])#,'.pdf' ])
    fulldata.dump(os.path.join(exp['fig_subpath'], file_name+'.pkl'))   

    file_name = 'list_actors_{}'.format(exp['params_combination'])
    var_names_np = np.asanyarray(var_names)
    var_names_np.dump(os.path.join(exp['fig_subpath'], file_name+'.pkl'))  
    # ======================================================================================================================
    # tigramite 3
    # ======================================================================================================================
    data = fulldata
    print(data.shape)    
    # RV mask False for period that I want to analyse
    idx_start_RVperiod = int(np.where(RV.dates_np[RV.dates_np.year == exp['RV_oneyr'].year[0]] == exp['RV_oneyr'][0])[0])
    data_mask = np.ones(data.shape, dtype='bool') # true for actor months, false for RV months
    for i in range(exp['RV_oneyr'].size): # total timesteps RV period, 12 is 
        data_mask[idx_start_RVperiod+i:: exp['time_cycle'],:] = False 
    T, N = data.shape # Time, Regions
    # ======================================================================================================================
    # Initialize dataframe object (needed for tigramite functions)
    # ======================================================================================================================
    dataframe = pp.DataFrame(data=data, mask=data_mask) 
    # Create 'time axis' and variable names
#    datatime = np.arange(len(data))
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
    precursor_fields = allvar[1:]
    Corr_Coeff_list = []
    for var in precursor_fields:
        actor = outdic_actors[var]
        Corr_Coeff_list.append(actor.Corr_Coeff)
    Corr_precursor_ALL = Corr_Coeff_list
    print 'You have {} precursor(s) with {} lag(s)'.format(np.array(Corr_precursor_ALL).shape[0],
                                                            np.array(Corr_precursor_ALL).shape[2])
    
    n_parents = len(parents_RV)   
    for i in range(n_parents):
        link_number = parents_RV[i][0]
        lag = np.abs(parents_RV[i][1])-1 # is the minus 1 really correct?
        index_in_fulldata = parents_RV[i][0]
        if index_in_fulldata>0:
       
            according_varname = var_names[index_in_fulldata][1]
            according_number = var_names[index_in_fulldata][0]
            according_var_idx = var_names[index_in_fulldata][2] # what is this, there is no third index
            print("index_in_fulldata")
            print(index_in_fulldata)
            print("according_varname")
            print(according_varname)
            print("according_number")
            print(according_number)
            print("according_var_idx")
            print(according_var_idx)
            # *********************************************************
            # print and save only significant regions
            # *********************************************************
            according_fullname = str(according_number) + according_varname   
            Corr_precursor = Corr_precursor_ALL[according_var_idx]
    #            xr.DataArray(data=Corr_precursor, coords=[actor.lat_grid, actor.lon_grid], dims=('latitude','longitude'))
            actor = outdic_actors[according_varname]
            fig_region, A_number_region = rgcpd.print_particular_region(according_number, Corr_precursor[:, :], actor, map_proj, 
                                                            according_fullname)
            fig_file = '{}_{}_{}_sign{}_{}{}'.format(according_fullname,
                        exp['params_combination'],exp['pcA_set'],alpha_level,str(parents_RV[i][1]),exp['file_type2'])
            plt.savefig(os.path.join(exp['fig_subpath'], fig_file), dpi=250)   
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
    print('Plotting all fields significant at alpha_level_tig, while conditioning on parents'
          'that were found in the PC step')
    # =============================================================================
    all_idx_tig = [actoridx[0] for actoridx in parents_RV]
    all_reg_tig = [var_names[i][:2] for i in all_idx_tig]
#    Corr_Coeff_all_r_l = np.ma.concatenate(Corr_precursor_ALL[:],axis=1)
    Corr_Coeff_all_r_l = np.ma.array(Corr_precursor_ALL)
    lags = Corr_Coeff_all_r_l.shape[2]
    for var in allvar[1:]:       
        varidx = allvar.index(var) -1 # minus the first idx of the RV
        # stack all variables and lags togther to plot them in same basemap
        all_regions_corr = np.zeros((actor.lat_grid.size * actor.lon_grid.size))
        all_regions_tig = np.zeros((actor.lat_grid.size * actor.lon_grid.size))
        all_regions_del = np.zeros((actor.lat_grid.size * actor.lon_grid.size))
        for i in range(lags):
            regions_i = rgcpd.define_regions_and_rank_new(Corr_Coeff_all_r_l[varidx,:,i], actor.lat_grid, actor.lon_grid)
            regions_i = np.nan_to_num(regions_i)
        
            indices_regs = np.where( regions_i >= 0.5 )[0]
            for i in indices_regs:
                all_regions_corr[i] = regions_i[i]
                all_regions_del[i]  = regions_i[i]
                if [regions_i[i], var] in all_reg_tig:            
                    all_regions_tig[i] = regions_i[i]
                    all_regions_del[i]      = 0
        all_regions_corr = all_regions_corr.reshape((actor.lat_grid.size, actor.lon_grid.size))
        all_regions_tig = all_regions_tig.reshape((actor.lat_grid.size, actor.lon_grid.size))
        all_regions_del = all_regions_del.reshape((actor.lat_grid.size, actor.lon_grid.size))
        array = np.concatenate( (all_regions_corr[None,:,:], all_regions_tig[None,:,:]), axis=0)
        array = np.ma.masked_equal(array, 0)
        all_regions_del[all_regions_del > 0] = 1
        names = [var +' all regions significant Corr', var +' all regions significant Tig']
        lat = actor.lat_grid
        lon = actor.lon_grid
        cmap = 'tab20c'
        
        xrdata = xr.DataArray(data=array, coords=[names, lat, lon], 
                            dims=['names','latitude','longitude'], name='Corr Coeff')
        clevels = np.arange(int(xrdata.min()), int(xrdata.max())+1)
        g = xr.plot.FacetGrid(xrdata, col='names', subplot_kws={'projection': map_proj},
                          aspect= (lon.size) / lat.size, size=3)
        
        plotdata = xrdata.sel(names=names[0])
        im = plotdata.plot.contourf(ax=g.axes[0,0], transform=ccrs.PlateCarree(),
                                            cmap=cmap, levels=clevels, alpha=1.,
                                            subplot_kws={'projection':map_proj},
                                            add_colorbar=False)
        plotdata = xrdata.sel(names=names[0]) 
        plotdata.values = all_regions_del
        im = plotdata.plot.contour(ax=g.axes[0,0], transform=ccrs.PlateCarree(),
                                            colors=['black'], 
                                            subplot_kws={'projection':map_proj})
        g.axes[0,0].set_title(names[0], fontsize='x-large')
        g.axes[0,0].text(0.5, -0.1, 'Black contours are not significant after MCI',
                          horizontalalignment='center', fontsize='x-large',
                          verticalalignment='center', transform=g.axes[0,0].transAxes)
        plotdata = xrdata.sel(names=names[1])
    #    im = plotdata.plot.contourf(ax=g.axes[0,1], transform=ccrs.PlateCarree(),
    #                                        colors=['black'], 
    #                                        subplot_kws={'projection':map_proj})
        im = plotdata.plot.contour(ax=g.axes[0,1], transform=ccrs.PlateCarree(),
                                            cmap=cmap, levels=clevels,
                                            subplot_kws={'projection':map_proj},
                                            add_colorbar=False)
        g.axes[0,1].set_title(names[1], fontsize='x-large')
        for ax in g.axes.flat:
            ax.coastlines(color='grey')
        
        plt.tight_layout()
        cbar_ax = g.fig.add_axes([0.25, 0.0, 0.5, 0.05])
        plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
        file_name = 'new_{}_{}_SIGN{}_all'.format(exp['params_combination'], exp['pcA_set'], alpha_level)
        plt.savefig(os.path.join(exp['fig_path'], file_name + exp['file_type2']),dpi=250)
    #    plt.savefig(os.path.join(exp['fig_subpath'], file_name + exp['file_type1']))
    
    # *****************************************************************************
    # save output if exp['SaveTF'] == True
    # *****************************************************************************
    
    if exp['SaveTF'] == True:
        sys.stdout = orig_stdout
        f.close()    
        # reopen the file to reorder the lines
        in_file=open(''.join([exp['fig_subpath'],'old.txt']),"rb")     
        contents = in_file.read()
        in_file.close()    
        cont_split = contents.splitlines()
        # save a new file    
        in_file=open(''.join([exp['fig_subpath'], exp['params_combination'],'pcA',exp['pcA_set'],'_SIGN',str(alpha_level),'.txt']),"wb")
        for i in range(0,len(cont_split)):
            in_file.write(cont_split[i]+'\r\n')
        in_file.close()
        # delete old file
        os.remove(''.join([exp['fig_subpath'],'old.txt']))  
          
            
