#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 16:09:42 2023

@author: bcourtne
"""

import time
import multiprocessing 
import pyzelda.zelda as zelda
import pyzelda.ztools as ztools
import pyzelda.utils.aperture as aperture
import pyzelda.utils.zernike as zernike
import pyzelda.utils.mft as mft

import multiprocessing
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

os.chdir('/Users/bcourtne/Documents/ANU_PHD2/heimdallr')


from functions import baldr_functions as baldr

#%%  NAOMI sim

#for wvl in np.linspace(1.2e-6 , 1.9e-6, 10):
naomi_parameter_dict_1 = {'seeing':1 , 'L0':25, 'D':1.8, 'D_pix':2**8, \
                        'wvl_0':0.658e-6,'n_modes':14,\
                            'lag':5e-3,'V_turb':50, 'pupil_geometry': 'AT',\
                                'dt': 0.3 * 1e-3}
    
sim_time = 0.1 #s  
wvl_list = np.linspace(0.9e-6,  2.0e-6,  20) # what wavelengths to evaluate the phase screens at

save_dir = '/Users/bcourtne/Documents/ANU_PHD2/heimdallr/naomi_screens/' #'/Users/bcourtne/Documents/ANU_PHD2/heimdallr/naomi_simulated_phasescreens/
naomi_screens = baldr.naomi_simulation(naomi_parameter_dict_1, sim_time = sim_time, wvl_list=wvl_list, save = False, saveAs = save_dir + 'naomi_screens_sim_1_bright.fits')


#%%  baldr sim


#save_dir = '/Users/bcourtne/Documents/ANU_PHD2/heimdallr/'
# load the phase screens from naomi
#naomi_screens = fits.open( save_dir + 'naomi_screens_sim_1.fits' )

input_screens = copy.deepcopy(naomi_screens) 

# --- setting up parameters 
bandwidth = 20 # %
spec_density = 1e13 # ph/m2/s/wvl


baldr_dict = { 'baldr_wvl_0':1.65e-6, 'baldr_min_wvl': np.nan, 'baldr_max_wvl': np.nan, 'throughput':0.01, 'f_ratio':20, 'DIT':0.3e-3, 'processing_latency':0.2e-3, 'RON':2,'pixel_window': 2**5 }

mask_type='achromatic' #'achromatic' or 'chromatic'

mask_dict={'achromatic':{'glass_off':'su8'},'chromatic':{'glass_off':'sio2'}}

wvl_0 = baldr_dict['baldr_wvl_0']

baldr_dict['baldr_min_wvl'] = wvl_0 - 1/2 * bandwidth/100 * wvl_0
baldr_dict['baldr_max_wvl'] = wvl_0 + 1/2 * bandwidth/100 * wvl_0       

input_spectrum_dict = { 'input_spectrum_wvls':np.linspace(0.9e-6,2e-6,30), 'input_spectrum_ph/m2/wvl/s': float(spec_density) * np.ones(30) } 

init_phase_mask_dict = {'T_off':1,'T_on':1,'d_off':np.nan, 'd_on':np.nan, 'target_phase_shift': -90, 'glass_on':'sio2', 'glass_off':mask_dict[mask_type]['glass_off'], 'phase_mask_rad_at_wvl_0':1,'achromatic_diffraction':False}

#optimie it  (note that for su8, sio2 combination optimize works best for negative desired phase shifts)
phase_mask_dict = baldr.optimize_mask_depths( init_phase_mask_dict, wvls=np.linspace( baldr_dict['baldr_min_wvl'], baldr_dict['baldr_max_wvl'], 20 ), plot_results=True)


baldr_dict_w_input_spectrum = baldr.process_input_spectral_features( {**input_spectrum_dict, **baldr_dict} ) # could change this wto work on master dict 
# put everything in master_dict
parameter_dict = {**baldr_dict_w_input_spectrum , **phase_mask_dict}

baldr_screens = baldr.baldr_sim( input_screens, parameter_dict ,report_exec_times=False)


#%%