# -*- coding: utf-8 -*-
#%%
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
    #%%
    # =============================================================================
    # Load 'exp' dictionairy with information of pre-processed data (variables, paths, filenames, etcetera..)
    # and add RGCPD/Tigrimate experiment settings
    # =============================================================================
    ex = np.load(str(filename_exp_design2)).item()
    # Response Variable is what we want to predict
    RV_name = ex['vars'][0][0]
    RV = ex[RV_name]
    ex['time_cycle'] = RV.dates[RV.dates.year == RV.startyear].size # time-cycle of data. total timesteps in one year
    #=====================================================================================
    # Information on period taken for response-variable, already decided in main_download_and_pp
    #=====================================================================================
    time_range_all = [0, RV.dates.size]
    #==================================================================================
    # Start of experiment
    #==================================================================================


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
    
    allvar = ex['vars'][0] # list of all variable names
    for var in allvar[ex['excludeRV']:]: # loop over all variables
        actor = ex[var]
        #===========================================
        # 3c) Precursor field = sst
        #===========================================
        ncdf = Dataset(os.path.join(actor.path_pp, actor.filename_pp), 'r')
        array = ncdf.variables[var][:,:,:].squeeze()    
        time , nlats, nlons = array.shape # [months , lat, lon]
        box = [ex['la_min'], ex['la_max'], ex['lo_min'], ex['lo_max']]
        # =============================================================================
        # Calculate correlation 
        # =============================================================================
        Corr_Coeff, lat_grid, lon_grid = rgcpd.calc_corr_coeffs_new(ncdf, array, box, RV.RV_ts, time_range_all, ex['lag_min'], 
                                                              ex['lag_max'], ex['time_cycle'], ex['RV_period'], 
                                                              ex['alpha_fdr'], FDR_control=ex['FDR_control'])
        Corr_Coeff = np.ma.array(data = Corr_Coeff[:,:], mask = Corr_Coeff.mask[:,:])
        # =============================================================================
        # Convert regions in time series  
        # =============================================================================
        actbox = rgcpd.extract_data(ncdf, array, time_range_all, box)
        actbox = np.reshape(actbox, (actbox.shape[0], -1))
        # tsCorr is total time series (.shape[0]) and .shape[1] are the correlated regions
        # stacked on top of each other (from lag_min to lag_max)
        tsCorr, n_reg_perlag = rgcpd.calc_actor_ts_and_plot(Corr_Coeff, actbox, 
                                ex, lat_grid, lon_grid, var)
        # Order of regions: strongest to lowest correlation strength
        outdic_actors[var] = act(var, Corr_Coeff, lat_grid, lon_grid, actbox, tsCorr, n_reg_perlag)
        # =============================================================================
        # Plot    
        # =============================================================================
        if ex['plotin1fig'] == False and ex['showplot'] == True:
            functions_tig.xarray_plot_region([var], outdic_actors, ex, map_proj)
            fig_filename = '{}_corr_{}_vs_{}'.format(ex['params'], allvar[0], var) + ex['file_type2']
            plt.savefig(os.path.join(ex['fig_path'], fig_filename), bbox_inches='tight', dpi=200)
                                           
    
    if ex['plotin1fig'] == True and ex['showplot'] == True:
        variables = outdic_actors.keys()
        functions_tig.xarray_plot_region(variables, outdic_actors, ex['lag_min'],  ex['lag_max'],
                                               map_proj, ex['tfreq'])
        fig_filename = '{}_corr_all'.format(ex['params'], allvar[0], var) + ex['file_type2']
        plt.savefig(os.path.join(ex['fig_path'], fig_filename), bbox_inches='tight', dpi=200)
#%%
    return ex, outdic_actors



def run_PCMCI(ex, outdic_actors, map_proj):
    #=====================================================================================
    #
    # 4) PCMCI-algorithm
    #
    #=====================================================================================
#%%
    # save output
    if ex['SaveTF'] == True:
        orig_stdout = sys.stdout
        sys.stdout = f = open(os.path.join(ex['fig_subpath'], 'old.txt'), 'a')
    # alpha level for independence test within the pc procedure (finding parents)
    pc_alpha = ex['pcA_sets'][ex['pcA_set']]
    # alpha level for multiple linear regression model while conditining on parents of
    # parents
    alpha_level = ex['alpha_level_tig']  
    print('run tigramite 3, run.pcmci')
    print('alpha level(s) for independence tests within the pc procedure'
          '(finding parents): {}'.format(pc_alpha))
    print('alpha level for multiple linear regression model while conditining on parents of '
          'parents: {}'.format(ex['alpha_level_tig']))
   

 
    
    # load Response Variable class
    RV_name = ex['vars'][0][0]
    RV = ex[RV_name]
    # create list with all actors, these will be merged into the fulldata array
    allvar = ex['vars'][0]
    var_names = [[0, allvar[0]]]
    actorlist = []
    for var in allvar[ex['excludeRV']:]:
        print var
        actor = outdic_actors[var]
        actorlist.append(actor.tsCorr)        
        # create array which numbers the regions
        var_idx = allvar.index(var) - ex['excludeRV']
        n_regions = actor.tsCorr.shape[1]
        actor.var_info = [[i+1, var, var_idx] for i in range(n_regions)]
        # Array of corresponing regions with var_names (first entry is RV)
        var_names = var_names + actor.var_info 
    # stack actor time-series together:
    fulldata = np.concatenate(tuple(actorlist), axis = 1)
    print('There are {} regions in total'.format(fulldata.shape[1]))
    # add the full 1D time series of interest as first entry:
    fulldata = np.column_stack((RV.RV1D, fulldata))
    # save fulldata
    file_name = 'fulldata_{}'.format(ex['params'])#,'.pdf' ])
    fulldata.dump(os.path.join(ex['fig_subpath'], file_name+'.pkl'))   

    file_name = 'list_actors_{}'.format(ex['params'])
    var_names_np = np.asanyarray(var_names)
    var_names_np.dump(os.path.join(ex['fig_subpath'], file_name+'.pkl'))  
    # ======================================================================================================================
    # tigramite 3
    # ======================================================================================================================
    data = fulldata
    print(data.shape)    
    # RV mask False for period that I want to analyse
    idx_start_RVperiod = int(np.where(RV.dates[RV.dates.year == ex['RV_oneyr'].year[0]] == ex['RV_oneyr'][0])[0])
    data_mask = np.ones(data.shape, dtype='bool') # true for actor months, false for RV months
    for i in range(ex['RV_oneyr'].size): # total timesteps RV period, 12 is 
        data_mask[idx_start_RVperiod+i:: ex['time_cycle'],:] = False 
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
    results = pcmci.run_pcmci(tau_max=ex['lag_max'], pc_alpha = pc_alpha, tau_min = ex['lag_min'], max_combinations=1) 
    
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
#    """
#    what's this?
#    med = LinearMediation(dataframe=dataframe,            
#                use_mask =True,
#                mask_type ='y',
#                data_transform = None)   # False or None for sklearn.preprocessing.StandardScaler = 
#    
#    med.fit_model(all_parents=all_parents, tau_max= tau_max)
#    """
    
    # parents of index of interest:
    # parents_neighbors = all_parents, estimates, iterations 		
    parents_RV = all_parents[0]
    
    # combine all variables in one list
    precursor_fields = allvar[ex['excludeRV']:]
    Corr_Coeff_list = []
    for var in precursor_fields:
        actor = outdic_actors[var]
        Corr_Coeff_list.append(actor.Corr_Coeff)
    Corr_precursor_ALL = Corr_Coeff_list
#    print('You have {} precursor(s) with {} lag(s)'.format(np.array(Corr_precursor_ALL).shape[0],
#                                                            np.array(Corr_precursor_ALL).shape[2]))
    
    n_parents = len(parents_RV)   
    for i in range(n_parents):
        link_number = parents_RV[i][0]
        lag = np.abs(parents_RV[i][1]) #-1 There was a minus, but is it really correct?
        index_in_fulldata = parents_RV[i][0]
        if index_in_fulldata>0: 
       
            according_varname = var_names[index_in_fulldata][1]
            according_number = var_names[index_in_fulldata][0]
            according_var_idx = var_names[index_in_fulldata][2]
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
            according_fullname = '{}lag_{}_reg{}'.format(according_varname, 
                                  lag,str(according_number))   
            # !!! Corr_precursor_all !!!
            Corr_precursor = Corr_precursor_ALL[according_var_idx]
            actor = outdic_actors[according_varname]
            rgcpd.print_particular_region(according_number, Corr_precursor[:, :], 
                                          actor, map_proj, according_fullname)
            fig_file = '{}{}'.format(according_fullname, ex['file_type2'])
                        
            plt.savefig(os.path.join(ex['fig_subpath'], fig_file), dpi=250)   
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
            print('Index itself is also causal parent -> skipped')
            print('*******************              ***************************')
 
    if ex['SaveTF'] == True:
        f.close()    
        # reopen the file to reorder the lines
        in_file=open(os.path.join(ex['fig_subpath'], 'old.txt'),"rb")     
        contents = in_file.read()
        in_file.close()    
        cont_split = contents.splitlines()
        # save a new file    
        in_file=open(os.path.join(ex['fig_subpath'], ex['params']+'.txt'),"wb")
        for i in range(0,len(cont_split)):
            in_file.write(cont_split[i]+'\r\n')
        in_file.close()
        # pass output to original console again
        sys.stdout = orig_stdout
        # delete old file
        os.remove(os.path.join(ex['fig_subpath'],'old.txt')) 
        #%%
    return parents_RV, var_names
#%%

def plottingfunction(ex, parents_RV, var_names, outdic_actors, map_proj):
    #%%
    # =============================================================================
    print('\nPlotting all fields significant at alpha_level_tig, while conditioning on parents'
          'that were found in the PC step')
    # =============================================================================
    # i+1 below, assuming parents_RV pythonic counting of idx
    # !!!! Check with Marlene !!!!
    all_reg_tig = [[var_names[i][0],var_names[i][1],l] for i,l in parents_RV]
    allvar = ex['vars'][0]
    precursor_fields = allvar[ex['excludeRV']:]
    Corr_Coeff_list = []
    for var in precursor_fields:
        actor = outdic_actors[var]
        Corr_Coeff_list.append(actor.Corr_Coeff)
    Corr_precursor_ALL = Corr_Coeff_list
    # shape of Corr_Coeff_all_r_l = [prec_vars, gridpoints, lags]
    Corr_Coeff_all_r_l = np.ma.array(Corr_precursor_ALL)
    lags = range(ex['lag_min'],ex['lag_max']+1)
    prec_names = np.array(allvar[ex['excludeRV']:])
    if ex['plotin1fig'] == True:
      # Build array, keeping only correlated regions that are sign after tigramite step
        all_regions_corr = np.zeros((prec_names.size, actor.lat_grid.size*actor.lon_grid.size))
        all_regions_tig = np.zeros((prec_names.size, actor.lat_grid.size*actor.lon_grid.size))
        all_regions_del = np.zeros((prec_names.size, actor.lat_grid.size*actor.lon_grid.size))
        
        for var in allvar[ex['excludeRV']:]:       
            varidx = allvar.index(var) - ex['excludeRV']  # minus the first idx of the RV if exclRV = 1
            tomatch_reg_n_var_names = 0
            skip_inds_prev_lag = 0
            for lag in lags:
                lagidx = lags.index(lag)
                # Rangs regions of var and lag from 1 to n_regions according to corr strength
                regions_i = rgcpd.define_regions_and_rank_new(Corr_Coeff_all_r_l[varidx,:,lagidx], 
                                                              actor.lat_grid, actor.lon_grid)
                regions_i = np.nan_to_num(regions_i)
                
                
                tomatch_reg_n_var_names = tomatch_reg_n_var_names + skip_inds_prev_lag
                regions_i = regions_i + tomatch_reg_n_var_names

                indices_regs = np.where( regions_i >= 0.5 )[0]
                tdata = regions_i.data.reshape(71,144)
                
                
                for i in indices_regs:
                    number_region = regions_i[i]
                    all_regions_corr[varidx,i] = number_region 
                    all_regions_del[varidx,i]  = number_region
#                    if regions_i[i] == 10 or regions_i[i] == 44:
#                        print regions_i[i]
                    if [number_region, var, -lag] in all_reg_tig:
#                        print(number_region, var, -lag)
                        
#                        Corr_Coeff_lag_i = Corr_Coeff_all_r_l[varidx,:,:]
#                        actor = outdic_actors[var]
#                        title = '{} {} {}'.format(regions_i[i], var, -lag)
#                        rgcpd.print_particular_region(number_region, Corr_Coeff_lag_i, actor, map_proj, title)
#                        print True
                        all_regions_tig[varidx,i] = number_region
                        all_regions_del[varidx,i] = 0
                skip_inds_prev_lag = regions_i.max()
                        
        all_regions_corr = all_regions_corr.reshape((prec_names.size, actor.lat_grid.size, 
                                                     actor.lon_grid.size))
        all_regions_tig = all_regions_tig.reshape((prec_names.size, actor.lat_grid.size, 
                                                   actor.lon_grid.size))
        all_regions_del = all_regions_del.reshape((prec_names.size, actor.lat_grid.size, 
                                                   actor.lon_grid.size))
        array = np.concatenate( (all_regions_corr[None,:,:,:], all_regions_tig[None,:,:,:]), axis=0)
    elif ex['plotin1fig'] == False: 
        # Build array, keeping only correlated regions that are sign after tigramite step
        all_regions_corr = np.zeros((prec_names.size, len(lags), actor.lat_grid.size*actor.lon_grid.size))
        all_regions_tig = np.zeros((prec_names.size, len(lags), actor.lat_grid.size*actor.lon_grid.size))
        all_regions_del = np.zeros((prec_names.size, len(lags), actor.lat_grid.size*actor.lon_grid.size))
        
#        for var in allvar[ex['excludeRV']:]:     
        for var in allvar[ex['excludeRV']:]: 
            varidx = allvar.index(var) - ex['excludeRV']  # minus the first idx of the RV if exclRV = 1
            tomatch_reg_n_var_names = 0
            skip_inds_prev_lag = 0
            for lag in lags:
                lagidx = lags.index(lag)
                regions_i = rgcpd.define_regions_and_rank_new(Corr_Coeff_all_r_l[varidx,:,lagidx], 
                                                              actor.lat_grid, actor.lon_grid)
                regions_i = np.nan_to_num(regions_i)

                tomatch_reg_n_var_names = tomatch_reg_n_var_names + skip_inds_prev_lag
                regions_i = regions_i + tomatch_reg_n_var_names
            
                indices_regs = np.where( regions_i >= 0.5 )[0]
                for i in indices_regs:
                    number_region = regions_i[i]
                    all_regions_corr[varidx,lagidx,i] = number_region
                    all_regions_del[varidx,lagidx,i]  = number_region
                    if [number_region, var, -lag] in all_reg_tig: 
                        all_regions_tig[varidx,lagidx,i] = number_region
                        all_regions_del[varidx,lagidx,i] = 0
                skip_inds_prev_lag = regions_i.max()
        all_regions_corr = all_regions_corr.reshape((prec_names.size, len(lags), actor.lat_grid.size, 
                                                     actor.lon_grid.size))
        all_regions_tig = all_regions_tig.reshape((prec_names.size, len(lags), actor.lat_grid.size, 
                                                   actor.lon_grid.size))
        all_regions_del = all_regions_del.reshape((prec_names.size, len(lags), actor.lat_grid.size, 
                                                   actor.lon_grid.size))
        array = np.concatenate( (all_regions_corr[None,:,:,:,:], all_regions_tig[None,:,:,:,:]), axis=0)
   
    array = np.ma.masked_equal(array, 0)
    all_regions_del[all_regions_del > 0] = 1
    names_col = ['all regions significant Corr', ' all regions significant Tig']
    lat = actor.lat_grid
    lon = actor.lon_grid

    #%%    
        
    def finalfigure(xrdata, all_regions_deladj, file_name):
        g = xr.plot.FacetGrid(xrdata, col='names_col', row='names_row', subplot_kws={'projection': map_proj},
                          aspect= (lon.size) / lat.size, size=3)
        figwidth = g.fig.get_figwidth() ; figheight = g.fig.get_figheight()
        if xrdata.max() >= 2: 
            cmap = plt.cm.Dark2
            clevels = np.arange(int(xrdata.min()), int(xrdata.max())+1E-9, 1)
        else:
            cmap = plt.cm.Greens
            clevels = [0., 0.95, 1.0]
        
        
#        xrdata.data[xrdata.data > 0.5] = 2.
#        clevels = np.linspace(int(xrdata.data.min()), int(xrdata.max())+1E-9, 2)
        for row in xrdata.names_row.values:
            rowidx = list(xrdata.names_row.values).index(row)
            plotrow = xrdata.sel(names_row=row)
    #        for col in xrdata.names_col.values:
    #            colidx = list(xrdata.names_col.values).index(col)
            colidx = 0
            plotdatac = plotrow.sel(names_col=names_col[0])
            im = plotdatac.plot.contourf(ax=g.axes[rowidx,colidx], transform=ccrs.PlateCarree(),
                                                cmap=cmap, levels=clevels, alpha=1,
                                                subplot_kws={'projection':map_proj},
                                                add_colorbar=False)
            plotdatat = plotrow.sel(names_col=names_col[1])
            plotdatat.data = np.nan_to_num(plotdatat)
            plotdatat.plot.contour(ax=g.axes[rowidx,colidx], transform=ccrs.PlateCarree(),
                                                colors=['black'], linewidth=0.35,
                                                subplot_kws={'projection':map_proj},
                                                add_colorbar=False)
            colidx = 1
            plotdatat = plotrow.sel(names_col=names_col[1])
            if np.sum(plotdatat) == 0.0:
                g.axes[rowidx,colidx].text(0.5, 0.5, 'No regions significant',
                              horizontalalignment='center', fontsize='x-large',
                              verticalalignment='center', transform=g.axes[rowidx,colidx].transAxes)
            else:
                im = plotdatat.plot.contourf(ax=g.axes[rowidx,colidx], transform=ccrs.PlateCarree(),
                                                cmap=cmap, levels=clevels,
                                                subplot_kws={'projection':map_proj},
                                                add_colorbar=False)
                plotdatat.data = np.nan_to_num(plotdatat)
                plotdatat.plot.contour(ax=g.axes[rowidx,colidx], transform=ccrs.PlateCarree(),
                                                    colors=['black'], linewidth=0.35,
                                                    subplot_kws={'projection':map_proj},
                                                    add_colorbar=False)
            
            g.axes[rowidx,0].text(-figwidth/100, 0.5, row,
                      horizontalalignment='center', fontsize='x-large',
                      verticalalignment='center', transform=g.axes[rowidx,0].transAxes)
        for ax in g.axes.flat:
            ax.coastlines(color='grey', alpha=0.3)
            ax.set_title('')
        g.axes[0,1].set_title(names_col[1] + '\nat alpha={} with '
                      'pc_alpha(s)={}'.format(ex['alpha_level_tig']  , ex['pcA_sets'][ex['pcA_set']]), fontsize='x-large')
        g.axes[0,0].set_title(names_col[0] + '\nat Corr p-value={}'.format(ex['alpha']),
                      fontsize='x-large')
#        g.axes[rowidx,0].text(0.5, figwidth/100, 'Black contours are not significant after MCI',
#                      horizontalalignment='center', fontsize='x-large',
#                      verticalalignment='center', transform=g.axes[rowidx,0].transAxes)
        if ex['plotin1fig'] == False:
            cbar_ax = g.fig.add_axes([0.25, (figheight/25)/len(g.row_names), 
                                      0.5, (figheight/150)/len(g.row_names)])
            plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
#        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.subplots_adjust(wspace=0.0, hspace=-0.3)
        g.fig.savefig(os.path.join(ex['fig_path'], file_name + ex['file_type2']),dpi=250)
        return
    
    
    if ex['plotin1fig'] == True:
        names_row = [i +' all lags' for i in prec_names]
        xrdata = xr.DataArray(data=array, coords=[names_col, names_row, lat, lon], 
                            dims=['names_col','names_row','latitude','longitude'], name='Corr Coeff')
        xrdata.data = np.nan_to_num(xrdata.data)
        xrdata.data[xrdata.data > 0.5] = 1.
        all_regions_deladj = all_regions_del
        file_name = '{}_tigout_all'.format(ex['params'])
        finalfigure(xrdata, all_regions_deladj, file_name)
    if ex['plotin1fig'] == False:        
        for var in allvar[ex['excludeRV']:]:       
            varidx = allvar.index(var) - ex['excludeRV']
            onevar_array = array[:,varidx,:,:,:].copy()
            names_row = []
            for lag in lags:
                names_row.append(var + '\n-{} days'.format(lag * ex['tfreq']))
            
            xrdata = xr.DataArray(data=onevar_array, coords=[names_col, names_row, lat, lon], 
                                dims=['names_col','names_row','latitude','longitude'], name='Corr Coeff')           
            
            all_regions_deladj = all_regions_del[varidx]
            file_name = '{}_tigout_{}'.format(ex['params'], var)
            finalfigure(xrdata, all_regions_deladj, file_name)
 #%%
    return

         
