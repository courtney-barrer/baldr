#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 12:43:13 2023

@author: bcourtne

what impact does cold stop have on output intensity of ZWFS?  
"""

import numpy as np
import pylab as plt
import pandas as pd
import os
from scipy.interpolate import interp1d
import pyzelda.utils.zernike as zernike
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy 
import copy
import aotools
from astropy.io import fits
import json
import time

#import zelda
os.chdir('/Users/bcourtne/Documents/ANU_PHD2/baldr')

from functions import baldr_functions_2 as baldr
from functions import data_structure_functions as config

tel_config =  config.init_telescope_config_dict(use_default_values = True)
phasemask_config = config.init_phasemask_config_dict(use_default_values = True) 
DM_config = config.init_DM_config_dict(use_default_values = True) 
detector_config = config.init_detector_config_dict(use_default_values = True)

# define a hardware mode for the ZWFS 
mode_dict = config.create_mode_config_dict( tel_config, phasemask_config, DM_config, detector_config)

# init out Baldr ZWFS object with the desired mode 
zwfs = baldr.ZWFS(mode_dict) 

#%% Testing update of cold stop diameter and general focal plane phase mask size etc
# need way to update zwfs.FPM.cold_stop_mask with diameter

input_field_0 =  baldr.init_a_field( Hmag=4, mode='0', wvls=zwfs.wvls, pup_geometry=zwfs.mode['telescope']['pup_geometry'],\
             D_pix=zwfs.mode['telescope']['telescope_diameter_pixels'], \
                 dx=zwfs.mode['telescope']['telescope_diameter']/zwfs.mode['telescope']['telescope_diameter_pixels'], \
                     r0=0.1, L0=25, phase_scale_factor = 1)    
    
out_field = zwfs.FPM.get_output_field( input_field_0, wvl_lims=[-np.inf, np.inf], keep_intermediate_products=True, replace_nan_with=None)

plt.figure()    
plt.title('diffraction limited field in focal plane')
plt.imshow( abs( zwfs.FPM.Psi_B[5]) )

plt.figure() 
plt.title(f'phase shift region set to {round(zwfs.FPM.phase_shift_diameter /( zwfs.FPM.f_ratio * 1.6e-6  ),2)} lambda/D@1.6um')
plt.imshow( zwfs.FPM.phase_shift_region )

plt.figure() 
plt.title('cold stop region before update')
plt.imshow(zwfs.FPM.cold_stop_mask )

print( 'updating cold stop diameter' )
zwfs.FPM.update_cold_stop_parameters(cold_stop_diameter=2*zwfs.FPM.cold_stop_diameter)

plt.figure() 
plt.title('cold stop region after update')
plt.imshow(zwfs.FPM.cold_stop_mask )


#%% ZWFS intensity output for different cold stop radii

# init out Baldr ZWFS object with the desired mode 
zwfs = baldr.ZWFS(mode_dict) 

input_field_0 =  baldr.init_a_field( Hmag=5, mode='5', wvls=zwfs.wvls, pup_geometry=zwfs.mode['telescope']['pup_geometry'],\
             D_pix=zwfs.mode['telescope']['telescope_diameter_pixels'], \
                 dx=zwfs.mode['telescope']['telescope_diameter']/zwfs.mode['telescope']['telescope_diameter_pixels'], \
                     r0=0.1, L0=25, phase_scale_factor = 1)    

plt.figure()
plt.title('input pahse')
plt.imshow( input_field_0.phase[ zwfs.wvls[0] ] )

fig,ax = plt.subplots(3,1,figsize=(15,5))

zwfs.FPM.update_cold_stop_parameters(1/3*zwfs.FPM.cold_stop_diameter)
sig0 = zwfs.detection_chain( input_field_0 ,FPM_on=True, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=None)
ax[0].set_title( f'with cold stop diameter = {round(zwfs.FPM.cold_stop_diameter / (zwfs.FPM.f_ratio * 1.6e-6),1)} \lambda/D @ 1.6um' )
ax[0].imshow( sig0.signal )

zwfs.FPM.update_cold_stop_parameters(3*zwfs.FPM.cold_stop_diameter)
sig1 = zwfs.detection_chain( input_field_0, FPM_on=True, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=None )
ax[1].set_title( f'with cold stop diameter = {round(zwfs.FPM.cold_stop_diameter / (zwfs.FPM.f_ratio * 1.6e-6),1)} \lambda/D @ 1.6um' )
ax[1].imshow( sig1.signal )

zwfs.FPM.update_cold_stop_parameters(cold_stop_diameter=None)
sig2 = zwfs.detection_chain( input_field_0 ,FPM_on=True, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=None)
ax[2].set_title(f'with cold stop diameter = {zwfs.FPM.cold_stop_diameter}')
ax[2].imshow( sig2.signal )





