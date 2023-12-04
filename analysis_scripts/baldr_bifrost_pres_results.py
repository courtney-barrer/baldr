#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 16:43:59 2023

@author: bcourtne
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import zelda
os.chdir('/Users/bcourtne/Documents/ANU_PHD2/baldr')

from functions import baldr_functions_2 as baldr
from functions import data_structure_functions as config



#%% CONSTRUCTION ANIMATION OF IM CONSTRUCTION 
IM_path = '/Users/bcourtne/Documents/ANU_PHD2/baldr/figures/IM_construction_animation'

tel_config =  config.init_telescope_config_dict(use_default_values = True)
tel_config['pup_geometry']='AT'
phasemask_config = config.init_phasemask_config_dict(use_default_values = True) 
DM_config = config.init_DM_config_dict(use_default_values = True) 
detector_config = config.init_detector_config_dict(use_default_values = True)
detector_config['DIT']= 1e-3
# define a hardware mode for the ZWFS 
mode_dict = config.create_mode_config_dict( tel_config, phasemask_config, DM_config, detector_config)

# Serializing json
#json_object = json.dumps(mode_dict, indent=4)
# Writingfile
#with open("/Users/bcourtne/Documents/ANU_PHD2/baldr/first_stage_ao_screens/test.json", "w") as outfile:
#    outfile.write(json_object)
    
    
# init out Baldr ZWFS object with the desired mode 
zwfs = baldr.ZWFS(mode_dict) 

# define an internal calibration source 
calibration_source_config_dict = config.init_calibration_source_config_dict(use_default_values = True)
#calibration_source_config_dict['flux']=1e-12 # W/m2/nm
#define what modal basis, and how many how many modes to control, then use internal calibration source to create interaction matrices 
#and setup control parameters of ZWFS

#add control method using first 20 Zernike modes
zwfs.setup_control_parameters(  calibration_source_config_dict, N_controlled_modes=21, modal_basis='zernike', pokeAmp = 50e-9 , label='control_70_zernike_modes')

#%%

for i in range(20):
    fig = plt.figure(figsize=(16, 12))
        
    ax1 = fig.add_subplot(131)
    ax1.set_title('Solarstein field phase',fontsize=20)
    ax1.axis('off')
    im1 = ax1.imshow( zwfs.wvls[0]/(np.pi*2) * 1e9 *zwfs.control_variables['control_70_zernike_modes']['calibration_field'].phase[zwfs.wvls[0]] )
    
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    cbar = fig.colorbar( im1, cax=cax, orientation='horizontal')
    cbar.set_label( r'OPD [nm]', rotation=0)
    
    
    ax2 = fig.add_subplot(132)
    ax2.set_title('DM surface',fontsize=20)
    ax2.axis('off')
    im2 = ax2.imshow(1e9 * zwfs.control_variables['control_70_zernike_modes']['pokeAmp'] * zwfs.control_variables['control_70_zernike_modes']['control_basis'][i])
    
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    cbar = fig.colorbar(im2, cax=cax, orientation='horizontal')
    cbar.set_label( r'OPD [nm]', rotation=0)
    
    ax3 = fig.add_subplot(133)
    ax3.set_title('detector signal',fontsize=20)
    ax3.axis('off')
    # the IM is nromalize by the meta intensity to so to get to ADU we at leasthave to multiply by nph when mask is out on cal source
    im3 = ax3.imshow(  np.array(zwfs.control_variables['control_70_zernike_modes']['IM'][i]).reshape(zwfs.dm.N_act) )
    
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    cbar = fig.colorbar(im3, cax=cax, orientation='horizontal')
    cbar.set_label( 'meta intensity', rotation=0)
    
    plt.tight_layout()
    plt.savefig(IM_path+f'/constructing_IM_mode_{i}.png')
    
plt.figure()
plt.title('Interaction matrix (unfiltered)')
plt.imshow( zwfs.control_variables['control_70_zernike_modes']['IM'] )
plt.xlabel( 'pixels')
plt.ylabel( 'mode')
plt.savefig(IM_path+'/constructing_IM_FINAL_IM.png')




#%% Full closed loop , first-we simulate first stage ao and save
fname_id = 'Npix240_1.8m_disk_V50_r00.1_lag25.0_Nmodes7_it200'

input_screen_fits =f'/Users/bcourtne/Documents/ANU_PHD2/baldr/phase_screens_first_stage_ao/first_stage_AO_phasescreens_{fname_id}.fits'
ao_1_screens_fits = fits.open(input_screen_fits)

throughput = 0.01
Hmag = 0
#Hmag_at_vltiLab = Hmag  - 2.5*np.log10(throughput)
#flux_at_vltilab = baldr.star2photons('H',Hmag_at_vltiLab,airmass=1,k=0.18,ph_m2_s_nm=True) #ph/m2/s/nm

# setting up the hardware and software modes of our ZWFS
tel_config =  config.init_telescope_config_dict(use_default_values = True)
phasemask_config = config.init_phasemask_config_dict(use_default_values = True) 
DM_config = config.init_DM_config_dict(use_default_values = True) 
detector_config = config.init_detector_config_dict(use_default_values = True)

# the only thing we need to be compatible is the pupil geometry and Npix, Dpix 
tel_config['pup_geometry'] = 'AT'
#tel_config['pup_geometry']=ao_1_screens_fits[0].header['PUP_GEOM']
tel_config['pupil_nx_pixels']=ao_1_screens_fits[0].header['NPIX']
phasemask_config['nx_size_focal_plane']=ao_1_screens_fits[0].header['NPIX']

#phasemask_config['phasemask_diameter'] = phasemask_config['phasemask_diameter'] * 1.9

tel_config['telescope_diameter']=ao_1_screens_fits[0].header['HIERARCH diam[m]']
tel_config['telescope_diameter_pixels']=int(round( ao_1_screens_fits[0].header['HIERARCH diam[m]']/ao_1_screens_fits[0].header['dx[m]'] ) )
detector_config['pix_scale_det'] = ao_1_screens_fits[0].header['HIERARCH diam[m]']/detector_config['detector_npix']
detector_config['DIT']  = 0.5e-3 #s


 #s
# define a hardware mode for the ZWFS 
mode_dict = config.create_mode_config_dict( tel_config, phasemask_config, DM_config, detector_config)

#create our zwfs object
zwfs = baldr.ZWFS(mode_dict)

# define an internal calibration source 
calibration_source_config_dict = config.init_calibration_source_config_dict(use_default_values = True)
calibration_source_config_dict['temperature']=1900 #K (Thorlabs SLS202L/M - Stabilized Tungsten Fiber-Coupled IR Light Source )
calibration_source_config_dict['calsource_pup_geometry'] = 'AT'

#add control method using first 20 Zernike modes
#zwfs.setup_control_parameters(  calibration_source_config_dict, N_controlled_modes=20, modal_basis='zernike', pokeAmp = 50e-9 , label='control_20_zernike_modes')

#zwfs.setup_control_parameters(  calibration_source_config_dict, N_controlled_modes=20, modal_basis='KL', pokeAmp = 50e-9 , label='control_20_KL_modes')
#zwfs.setup_control_parameters(  calibration_source_config_dict, N_controlled_modes=40, modal_basis='KL', pokeAmp = 50e-9 , label='control_40_KL_modes')
zwfs.setup_control_parameters(  calibration_source_config_dict, N_controlled_modes=70, modal_basis='KL', pokeAmp = 50e-9 , label='control_70_KL_modes')

# do closed loop simulation 
#t_bald, asgard_field_kl20, err_kl20, DM_shape_kl20, detector_signal_kl20 = baldr.baldr_closed_loop(input_screen_fits, zwfs, control_key='control_20_KL_modes', Hmag=Hmag, throughput=throughput, Ku=1 , Nint=2 ,return_intermediate_products=True)
#t_bald, asgard_field_kl20, err_kl20 = baldr.baldr_closed_loop(input_screen_fits, zwfs, control_key='control_20_KL_modes', Hmag=0, throughput=0.01, Ku=1 , Nint=2 ,return_intermediate_products=False)
#t_bald, asgard_field_kl40, err_kl40 = baldr.baldr_closed_loop(input_screen_fits, zwfs, control_key='control_40_KL_modes', Hmag=0, throughput=0.01, Ku=1 , Nint=2 ,return_intermediate_products=False)
#t_bald, asgard_field_kl70, err_kl70 = baldr.baldr_closed_loop(input_screen_fits, zwfs, control_key='control_70_KL_modes', Hmag=0, throughput=0.01, Ku=1 , Nint=2 ,return_intermediate_products=False)

t_bald, asgard_field_kl70, err_kl70,  DM_shape_kl70, detector_signal_kl70 =  baldr.baldr_closed_loop(input_screen_fits, zwfs, control_key='control_70_KL_modes', Hmag=0, throughput=0.01, Ku=1 , Nint=2 ,return_intermediate_products=True)


#%%


closed_loop_ani_path = '/Users/bcourtne/Documents/ANU_PHD2/baldr/figures/closed_loop_animation'

t_1 = np.linspace(0, len( ao_1_screens_fits[0].data) * ao_1_screens_fits[0].header['dt[s]'],  len( ao_1_screens_fits[0].data) )
wvl_key = 1.21111111111111e-06
wvl_i = 7
i=0
t1_indx = [np.argmin( abs( t - t_1)) for t in t_bald] # check len(t_bald) == len( t1_indx)


strehl_before = np.array([np.exp(-np.nanvar( ao_1_screens_fits[wvl_i].data[t1_indx[i]][zwfs.pup>0.5])) for i in range(len(t_bald))] )
strehl_after = np.array([np.exp(-np.nanvar(asgard_field_kl70[i].phase[wvl_key][zwfs.pup>0.5])) for i in range(len(t_bald))] )
    
pup_tmp = np.array( zwfs.pup.copy(), dtype=float)
pup_tmp[zwfs.pup==0] = np.nan

outer = [['upper left',  'upper right'],
          ['lower left', 'lower right'],
          ['strehl','strehl']]

for i in np.arange( 0, len(t_bald) ):  
    #fig = plt.figure(figsize=(16, 12))
    
    fig, axd = plt.subplot_mosaic(outer ,figsize=(12,18))
    plt.subplots_adjust(hspace=.5,wspace=.5)
    

    #---------------------------
    axd['upper left'].set_title('Input Phase (J-Band)',fontsize=20)
    axd['upper left'].axis('off')
    im1 = axd['upper left'].imshow( 1e9*wvl_key/(2*np.pi) *pup_tmp*  ao_1_screens_fits[wvl_i].data[t1_indx[i]] ,vmin = -500, vmax = 500)
    
    divider = make_axes_locatable(axd['upper left'])
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    cbar = fig.colorbar( im1, cax=cax, orientation='horizontal')
    cbar.set_label( r'OPD [nm]', rotation=0,fontsize=15)
    
    #---------------------------
    axd['lower left'].set_title('Baldr DM',fontsize=20)
    axd['lower left'].axis('off')
    im2 = axd['lower left'].imshow( 1e8*(DM_shape_kl70[t_bald[i]]-np.mean(DM_shape_kl70[t_bald[i]])), vmin = -20, vmax = 20 )
    
    divider = make_axes_locatable(axd['lower left'])
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    cbar = fig.colorbar(im2, cax=cax, orientation='horizontal',format='%.0e')
    cbar.set_label( r'DM Command [V]', rotation=0,fontsize=15)
    
    #---------------------------
    axd['upper right'].set_title('Baldr Residual (J-Band)',fontsize=20)
    axd['upper right'].axis('off')
    im3 = axd['upper right'].imshow( 1e9*wvl_key/(2*np.pi) * pup_tmp  * ( asgard_field_kl70[i].phase[wvl_key]-np.mean(asgard_field_kl70[i].phase[wvl_key]) ) , vmin = -500, vmax = 500 )
    
    divider = make_axes_locatable(axd['upper right'])
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    cbar = fig.colorbar(im3, cax=cax, orientation='horizontal')
    cbar.set_label( r'OPD [nm]', rotation=0, fontsize=15)
    
    #---------------------------
    axd['lower right'].set_title('Baldr Detector',fontsize=20)
    axd['lower right'].axis('off')
    im4 = axd['lower right'].imshow( detector_signal_kl70[t_bald[i]] , vmin = 0, vmax = np.max(detector_signal_kl70[t_bald[-1]]) )
    
    divider = make_axes_locatable(axd['lower right'])
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    cbar = fig.colorbar(im4, cax=cax, orientation='horizontal')
    cbar.set_label( r'Intensity [adu]', rotation=0, fontsize=15)
    
    #---------------------------
    axd['strehl'].plot(np.diff(t_bald)[0] * np.arange(i), strehl_before[:i],label='before Baldr')
    axd['strehl'].plot(np.diff(t_bald)[0] * np.arange(i), strehl_after[:i],label='after Baldr')
    
    axd['strehl'].set_xlim([0, t_bald[-1] ])
    axd['strehl'].set_ylabel('J Strehl Ratio',fontsize=25)
    axd['strehl'].set_xlabel('time (s)',fontsize=25)
    axd['strehl'].legend(loc='upper right',fontsize=25)
    plt.tight_layout()
    plt.savefig(closed_loop_ani_path+f'/baldr_closed_loop_ani_{i}___{fname_id}.png')
  

#%%  Checking out put of first staage naomi AO 
# !!! run from bcourtne@chapman3.sc.eso.org:/home/bcourtne/baldr/first_stage_ao_simulation.py)

#naomi_grid = {'bright-medATM': {'lag':4.6e-3,'n_modes':14,'V':40,'r0':0.13},\
#    'bright-poorATM':{'lag':4.6e-3,'n_modes':14,'V':70,'r0':0.08},\
#    'faint-medATM':{'lag':24e-3,'n_modes':7,'V':40,'r0':0.13},
#    'faint-poorATM':{'lag':24e-3,'n_modes':7,'V':70,'r0':0.08}}
naomi_phasescreen_path = '/Users/bcourtne/Documents/ANU_PHD2/baldr/phase_screens_first_stage_ao'

naomi_screen_fnames_dict = {'bright-medATM':'Npix240_1.8m_disk_V40_r00.13_lag4.6_Nmodes14_it500',\
                           'bright-poorATM':'Npix240_1.8m_disk_V70_r00.08_lag4.6_Nmodes14_it500',\
                           'faint-medATM':'Npix240_1.8m_disk_V40_r00.13_lag24.0_Nmodes7_it500',\
                           'faint-poorATM':'Npix240_1.8m_disk_V70_r00.08_lag24.0_Nmodes7_it500'}
                               
mean_strehls = {}
std_strehls = {}
strehls={}
Hband_wvl_indx = 15
plt.figure()
kwargs={'fontsize':15}
colors = ['red', 'darkred', 'blue', 'darkblue']
for grid_pt, col in zip(naomi_screen_fnames_dict,colors):
    input_screen_fits =f'/Users/bcourtne/Documents/ANU_PHD2/baldr/phase_screens_first_stage_ao/first_stage_AO_phasescreens_{naomi_screen_fnames_dict[grid_pt]}.fits'
    ao_1_screens_fits = fits.open(input_screen_fits) 
    mean_strehls[grid_pt] = np.mean(  [np.exp(-np.nanvar( ao_1_screens_fits[Hband_wvl_indx].data[i] )) for i in range(len( ao_1_screens_fits[15].data )) ] ) 
    std_strehls[grid_pt] =  np.std(  [np.exp(-np.nanvar( ao_1_screens_fits[Hband_wvl_indx].data[i] )) for i in range(len( ao_1_screens_fits[15].data )) ] )        
    strehls[grid_pt] = np.array([np.exp(-np.nanvar( ao_1_screens_fits[15].data[i] )) for i in range(len( ao_1_screens_fits[15].data )) ])
        
    plt.hist(strehls[grid_pt] , alpha =0.4, label=grid_pt,color=col )

plt.xlabel('H-Band Strehl Ratio',**kwargs) 
plt.ylabel('counts',**kwargs) 
plt.gca().tick_params(labelsize=15)
plt.legend() 

plt.figure()
plt.errorbar( mean_strehls.keys(), mean_strehls.values() ,yerr=list( std_strehls.values()) )
plt.ylabel('H-Band Strehl Ratio',**kwargs) 
plt.xlabel( 'AT/Naomi category',**kwargs)
               