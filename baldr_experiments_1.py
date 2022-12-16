#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 21:53:27 2022

@author: bcourtne
"""

import pyzelda.zelda as zelda
import pyzelda.ztools as ztools
import pyzelda.utils.aperture as aperture
import pyzelda.utils.zernike as zernike
import pyzelda.utils.mft as mft

import copy
import argparse
from astropy.io import fits
import numpy as np
import scipy.signal as sig
from scipy.interpolate import interp1d 
from scipy.optimize import curve_fit
import pylab as plt
import pandas as pd
import os 
import glob
from astroquery.simbad import Simbad
from matplotlib import gridspec
import aotools
from scipy.stats import poisson
import json

os.chdir('/home2/bcourtne/baldr')

from functions import baldr_functions as baldr


"""
to copy from chapman machine to my local mac : 
scp chapman/file/path bcourtne@134.171.187.31://Users/bcourtne/etc
"""

#%% If we want to re-simulate some naomi phase screens 

#for wvl in np.linspace(1.2e-6 , 1.9e-6, 10):
naomi_parameter_dict_1 = {'seeing':1 , 'L0':25, 'D':1.8, 'D_pix':2**8, \
                        'wvl_0':0.658e-6,'n_modes':14,\
                            'lag':5e-3,'V_turb':50, 'pupil_geometry': 'AT',\
                                'dt': 0.3 * 1e-3}
    
sim_time = 0.1 #s  
wvl_list = np.linspace(0.9e-6,  2.0e-6,  20) # what wavelengths to evaluate the phase screens at

save_dir = '/home2/bcourtne/baldr/naomi_screens/' #'/Users/bcourtne/Documents/ANU_PHD2/heimdallr/naomi_simulated_phasescreens/
naomi_screens = baldr.naomi_simulation(naomi_parameter_dict_1, sim_time = sim_time, wvl_list=wvl_list, save = True, saveAs = save_dir + 'naomi_screens_sim_1.fits')

# see baldr_functions[old].py for some sanity checks




#%% baldr simulation grid search 


# load the phase screens from naomi
naomi_screens = fits.open( '/home2/bcourtne/baldr/naomi_screens/naomi_screens_sim_1.fits' )

save_dir = '/home2/bcourtne/baldr/baldr_screens/'  #f'/Users/bcourtne/Documents/ANU_PHD2/heimdallr/

wvls = np.array( [a.header['wvl'] for a in naomi_screens] )

# naomi reference strehls 
naomi_strehls_ts={}
for w,wvl in enumerate(wvls): 
    naomi_strehls_ts[wvl] = np.exp(-np.nanvar( naomi_screens[w].data,axis=(1,2)) )
naomi_mean_strehls = [np.mean(naomi_strehls_ts[wvl]) for wvl in naomi_strehls_ts]

#wvl_0 = 1.65e-6 #central wavelength (m) of baldr wfs 



baldr_dict = { 'baldr_wvl_0':1.65e-6, 'baldr_min_wvl': np.nan, 'baldr_max_wvl': np.nan, 'throughput':0.01, 'f_ratio':20, 'DIT':0.3e-3, 'processing_latency':0.2e-3, 'RON':2 }

#target_phase_shift = -np.pi/3 
# 
baldr_mean_strehls = {} # to hld strehls 
for target_phase_shift in np.linspace(0,-120,25): # np.linspace( -np.pi/2, -np.pi/10, 10 ):
    
    print('target_phase_shift = ', round(target_phase_shift,2) )
    
    baldr_mean_strehls[target_phase_shift] = {}
    
    
    for bandwidth in np.linspace(2,40,15): #percent
        
        print( bandwidth )
        baldr_mean_strehls[target_phase_shift][bandwidth] = {}
        
        for spec_density in np.logspace(10,20,15) :
            
            input_spectrum_dict = { 'input_spectrum_wvls':np.linspace(0.9e-6,2e-6,100), 'input_spectrum_ph/m2/wvl/s': float(spec_density) * np.ones(100) } 
        
            wvl_0 = baldr_dict['baldr_wvl_0']
            
            #baldr_dict['wvl_range_wfs'] = (wvl_0 - 1/2 * bandwidth/100 * wvl_0, wvl_0 + 1/2 * bandwidth/100 * wvl_0)
            
            baldr_dict['baldr_min_wvl'] = wvl_0 - 1/2 * bandwidth/100 * wvl_0
            baldr_dict['baldr_max_wvl'] = wvl_0 + 1/2 * bandwidth/100 * wvl_0                             
            
            # achromatic  (si02 on-axis, su8 off-axis)
            init_phase_mask_dict = {'T_off':1,'T_on':1,'d_off':np.nan, 'd_on':np.nan, 'target_phase_shift': target_phase_shift, 'glass_on':'sio2', 'glass_off':'su8', 'phase_mask_rad_at_wvl_0':1,'achromatic_diffraction':False}
            
            #optimie it  (note that for su8, sio2 combination optimize works best for negative desired phase shifts)
            phase_mask_dict = baldr.optimize_mask_depths( init_phase_mask_dict, wvls=np.linspace( baldr_dict['baldr_min_wvl'], baldr_dict['baldr_max_wvl'], 20 ), plot_results=True)
    
            # now process input_spectrum to extract relevant features for wfs bandwidth (# photons total, reness etc) 
            baldr_dict_w_input_spectrum = baldr.process_input_spectral_features( {**input_spectrum_dict, **baldr_dict} ) # could change this wto work on master dict 
            
            # put everything in master_dict
            parameter_dict = {**baldr_dict_w_input_spectrum , **phase_mask_dict}

            baldr_screens = baldr.baldr_simulation( naomi_screens , parameter_dict , save = True, saveAs = save_dir + f'baldr_screens_achromatic_phase_mask_target_phase{target_phase_shift}_bandwidth-{bandwidth}pc_SD-{spec_density:.3E}_sim_1.fits')
            #baldr_screens = baldr.baldr_simulation( naomi_screens , input_spectrum_dict, baldr_dict, phase_mask_dict, wvl_0 , save = True, saveAs = f'/Users/bcourtne/Documents/ANU_PHD2/heimdallr/baldr_screens_achromatic_phase_mask_bandwidth_{bandwidth}pc_sim_1.fits')
            wvls = np.array( [a.header['wvl'] for a in baldr_screens] )
                
            baldr_strehls_ts={}
            for w,wvl in enumerate(wvls): 
                baldr_strehls_ts[wvl] = np.exp(-np.nanvar( baldr_screens[w].data,axis=(1,2)) ) 
            
            baldr_mean_strehls[target_phase_shift][bandwidth][spec_density] = {wvl: np.mean(baldr_strehls_ts[wvl]) for wvl in baldr_strehls_ts}


with open( '/home2/bcourtne/baldr/baldr_mean_strehls_grid.json', "w") as outfile:
    json.dump(baldr_mean_strehls, outfile)






