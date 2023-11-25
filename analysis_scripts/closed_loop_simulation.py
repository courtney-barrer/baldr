#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 10:38:07 2023

@author: bcourtne
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 22:40:50 2023

@author: bcourtne
"""
import numpy as np
import pylab as plt
import pandas as pd
import os
import pyzelda.utils.zernike as zernike
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy 
import copy
import aotools

#import zelda
os.chdir('/Users/bcourtne/Documents/ANU_PHD2/baldr')

from functions import baldr_functions_2 as baldr
from functions import data_structure_functions as config

    
# testing the new configuration formating 
tel_config =  config.init_telescope_config_dict(use_default_values = True)
phasemask_config = config.init_phasemask_config_dict(use_default_values = True) 
DM_config = config.init_DM_config_dict(use_default_values = True) 
detector_config = config.init_detector_config_dict(use_default_values = True)

# define a hardware mode for the ZWFS 
mode_dict = config.create_mode_config_dict( tel_config, phasemask_config, DM_config, detector_config)

# init out Baldr ZWFS object with the desired mode 
zwfs = baldr.ZWFS(mode_dict) 

# define an internal calibration source 
calibration_source_config_dict = config.init_calibration_source_config_dict(use_default_values = True)

#define what modal basis, and how many how many modes to control, then use internal calibration source to create interaction matrices 
#and setup control parameters of ZWFS

#add control method using first 20 Zernike modes
zwfs.setup_control_parameters(  calibration_source_config_dict, N_controlled_modes=20, modal_basis='zernike', pokeAmp = 50e-9 , label='control_20_zernike_modes')
zwfs.setup_control_parameters(  calibration_source_config_dict, N_controlled_modes=70, modal_basis='zernike', pokeAmp = 50e-9 , label='control_70_zernike_modes')
#add control method using first 20 KL modes
zwfs.setup_control_parameters(  calibration_source_config_dict, N_controlled_modes=20, modal_basis='KL', pokeAmp = 50e-9 , label='control_20_KL_modes')


#%% #---- now consider dectecting and correcting a turbulent input field in open loop 
input_field_0 = [ baldr.init_a_field( Hmag=4, mode='Kolmogorov', wvls=zwfs.wvls, pup_geometry=zwfs.mode['telescope']['pup_geometry'],\
             D_pix=zwfs.mode['telescope']['telescope_diameter_pixels'], \
                 dx=zwfs.mode['telescope']['telescope_diameter']/zwfs.mode['telescope']['telescope_diameter_pixels'], \
                     r0=0.1, L0=25, phase_scale_factor = 1) ]

"""
input_field = [ baldr.init_a_field( Hmag=4, mode='2', wvls=zwfs.wvls, pup_geometry=zwfs.mode['telescope']['pup_geometry'],\
             D_pix=zwfs.mode['telescope']['telescope_diameter_pixels'], \
                 dx=zwfs.mode['telescope']['telescope_diameter']/zwfs.mode['telescope']['telescope_diameter_pixels'], \
                     r0=0.1, L0=25, phase_scale_factor = 1e-1) ]"""
#idea to propagate easily 
"""aaa = [input_field.phase[zwfs.wvls[0]]]
for i in range(20):
    aaa.append( zwfs.pup*( 0.9 * aaa[-1] + 1e-2*np.random.randn(*aaa[-1].shape)))
    plt.figure()
    plt.imshow(aaa[-1])"""
    
#meth1 = 'control_method_1'
opd_fig, opd_ax = plt.subplots(1,1)
svd_fig, svd_ax = plt.subplots(1,1)
for meth1 in zwfs.control_variables:
    
    input_field = copy.copy(input_field_0) #make a copy of the original input field for each method
    
    #begin with flat DM
    flat_cmd=np.zeros(zwfs.dm.N_act).reshape(-1)
    zwfs.dm.update_shape( flat_cmd ) 
    
    sig_cal_on = zwfs.control_variables[meth1]['sig_on_ref'] #intensity measured on calibration source with phase mask in
    
    Nph_cal = zwfs.control_variables[meth1]['Nph_cal'] # sum of intensities (#photons) of calibration source with mask out 
    
    IM = zwfs.control_variables[meth1]['IM'] #interaction matrix from calibrationn source 
    CM = zwfs.control_variables[meth1]['CM'] #control matrix from calibrationn source 
    control_basis = zwfs.control_variables[meth1]['control_basis'] # DM vcontrol basis used to construct IM
    
    U,S,Vt = np.linalg.svd( IM )
    # look at eigenvalues of IM modes
    svd_ax.plot(S,label=meth1)

    
    #modal filtering for gains
    if '70' not in meth1:
        S_filt = np.array([ s if i<len(S)-2 else 0 for i, s in enumerate(S)  ]) #np.array([ s if s>2e-3 else 0 for s in S  ])
    else:
        S_filt = np.array([ s if i<len(S)-10 else 0 for i, s in enumerate(S)  ]) #np.array([ s if s>2e-3 else 0 for s in S  ])
    
    # just use at begining of observsation (don't update)
    sig_turb_off = zwfs.detection_chain( input_field[-1] , FPM_on=False) #intensity measured on sky with phase mask out
    
    Nph_obj = np.sum(sig_turb_off.signal) # sum of intensities (#photons) of on sky source with mask out 
    
    
    opd_rms = [np.std( input_field[-1].phase[zwfs.wvls[0]][zwfs.pup >0.5])]
    for i in range(40):
        
        cmd_tm1 = zwfs.dm.surface.reshape(-1) #prior cmd 
        # writing shorthand notations ...
        sig_turb = zwfs.detection_chain( input_field[-1] , FPM_on=True)  #intensity measured on sky with phase mask in
    
        #plt.imshow(sig_turb.signal)
        #plt.imshow(input_field.phase[zwfs.wvls[0]])
        
        
        # control_matrix @ 1/M * ( sig - M/N * ref_sig )
        modal_reco_list = CM.T @ (  1/Nph_obj * (sig_turb.signal - Nph_obj/Nph_cal * sig_cal_on.signal) ).reshape(-1) #list of amplitudes of the modes measured by the ZWFS
        modal_gains = -0.8 * S_filt  / np.max(S_filt) * zwfs.control_variables[meth1]['pokeAmp'] # -1 * zwfs.control_variables[meth1]['pokeAmp']* np.ones( len(modal_reco_list) ) # we set the gain by the poke amplitude 
        dm_reco = np.sum( np.array([ g * a * Z for g,a,Z in  zip(modal_gains,modal_reco_list, control_basis)]) , axis=0)
        
        #dm_reco = np.sum( np.array([ modal_gains[i] * a * Z for i,(a,Z) in enumerate( zip(modal_reco_list, control_basis))]) , axis=0)
        
        cmd_t = dm_reco.reshape(-1) #new command 
        zwfs.dm.update_shape( cmd_t ) #0.9 * cmd_t + 0.1 * cmd_tm1 )   #update our DM shape
        
        #sig_turb_after = zwfs.detection_chain( input_field[-1] , FPM_on=True)
        
        input_field.append( input_field[-1].applyDM(zwfs.dm) )
        
        opd_rms.append(np.std( input_field[-1].phase[zwfs.wvls[0]][zwfs.pup >0.5]) ) 
    
    
    #plt.imshow(input_field[-1].phase[zwfs.wvls[0]])
    #plt.imshow(zwfs.dm.surface)
    
    opd_ax.plot( np.exp( - np.array(opd_rms)**2 ), label=meth1 )
    
opd_ax.set_ylabel('Strehl ratio (H-Band)')
opd_ax.set_xlabel('iterations (1ms DIT)')
opd_ax.legend()
opd_ax.set_title('open loop convergence on stationary phase screen')

svd_ax.set_ylabel('singular values')
svd_ax.set_xlabel('mode #')
svd_ax.legend()

#plt.figure()
#plt.plot(opd_rms)


#%% Full closed loop 
r0 = 0.1 #m , defined at 500nm 
L0 = 25 #m
throughput = 0.01
Hmag = 3 

Hmag_at_vltiLab = Hmag  - 2.5*np.log10(throughput)

seed_phase_screen =  aotools.turbulence.infinitephasescreen.PhaseScreenVonKarman(nx_size = zwfs.mode['telescope']['telescope_diameter_pixels'], pixel_scale=zwfs.mode['telescope']['telescope_diameter']/zwfs.mode['telescope']['telescope_diameter_pixels'],\
                r0=r0 , L0=L0, n_columns=2,random_seed = 1)
seed_flux = baldr.star2photons('H', Hmag_at_vltiLab, airmass=1, k = 0.18, ph_m2_s_nm = True)

# add random noise on photometry that matches scinilation spectrum (can do empirically from IRIS data )

# for each method begin with phase_screen = copy.copy( seed_phase_screen ), then create field 

phase_screen = copy.copy( seed_phase_screen )

#for X in y:
phase_screen.add_row()
field_phase = [(500e-9/w)**(6/5) * phase_screen.scrn for w in zwfs.wvls]
field_flux = [ seed_flux * zwfs.pup  for w in zwfs.wvls] # could add noise here too


#%%

#plt.imshow( zwfs.dm.surface )

# estimate #photons 
#sig_off=baldr.detection_chain(input_field, dm, FPM_cal, det)
#sig_off.signal = np.mean( [baldr.detection_chain(input_field, dm, FPM_cal, det).signal for _ in range(10)]  , axis=0) # average over a few 
#Nph = np.sum(sig_off.signal)

# detect input field
sig_turb = baldr.detection_chain(input_field, zwfs.dm, zwfs.FPM, zwfs.det)

#important to scale reference field by ratio of phtons in calibrator vs target 
modal_reco_list = pinv_IM_modal.T @ (  1/Nph * (sig_turb.signal - Nph/Nph_cal * sig_on_ref.signal) ).reshape(-1) 
gains =  -pokeAmp * np.ones( len(modal_reco_list) ) #[0] + list(-25e-9 * np.ones( len(modal_reco_list)-1 )) +[0]

dm_reco = np.sum( np.array([gains[i] * a * Z for i,(a,Z) in enumerate( zip(modal_reco_list, control_basis))]) , axis=0)


cmd=dm_reco.reshape(-1)

dm.update_shape( cmd )   



#plt.imshow(zwfs.control_variables['control_method_1']['sig_on_ref'])

# if we want to reconstruct what the modes looked like on the calibration source
plt.figure()
plt.imshow(np.array( zwfs.control_variables['control_method_1']['IM'][2]).reshape(zwfs.mode['detector']['detector_npix'],zwfs.mode['detector']['detector_npix']))






#%%
#control_basis = baldr.create_control_basis(zwfs.dm, N_controlled_modes=20, basis_modes='zernike')

#IM, CM = baldr.build_IM( calibration_field , zwfs.dm, zwfs.FPM, zwfs.det,control_basis, pokeAmp=5e-8)


# detect the fields
calibration_sig = zwfs.detection_chain(calibration_field, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True)
calibration_sig_OFF = zwfs_OFF.detection_chain(calibration_field, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True)

fig,ax = plt.subplots(2,1)
ax[0].imshow(calibration_sig.signal)
ax[1].imshow(calibration_sig_OFF.signal)

# now apply aberration on DM and detect the fields - we should see the off just looks like the pupil while calibration_sig should follow aberration 
AMP = 1.6/15 * 1e-6
cmd = AMP * control_basis[6].reshape(-1) # apply a new mode on DM
zwfs.dm.update_shape(cmd) #
calibration_field = calibration_field.applyDM(zwfs.dm)

calibration_sig = zwfs.detection_chain(calibration_field, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True)
calibration_sig_OFF = zwfs_OFF.detection_chain(calibration_field, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True)

fig,ax = plt.subplots(2,1)
ax[0].imshow(calibration_sig.signal)
ax[1].imshow(calibration_sig_OFF.signal)





#np.max( calibration_field.flux[zwfs.wvls[0]] ) = 6e26!
# these following steps should be wrapped to attach objects to zwfs



# detection is turning QE to zero!! could be in the interpolation !
#sig = det4cal.detect_field( calibration_field, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True)
#plt.imshow(sig.signal)

# each wvls bin should contribute 
#np.diff(zwfs.wvls)[0] * (zwfs.mode['telescope']['telescope_diameter']/zwfs.mode['telescope']['telescope_diameter_pixels'])**2 * 1e-3






# some test s
plt.figure()
plt.imshow(zwfs.pup)

fig,ax = plt.subplots(3,3,figsize=(15,15) )

AMP = 1.6e-6/15
input_field = baldr.create_calibration_field_for_ZWFS( calibration_source_config_dict, zwfs)

ax[0,0].imshow( input_field.phase[zwfs.wvls[0]] ) 
ax[0,0].set_title('input phase')

cmd = 0 * control_basis[0].reshape(-1) #init flat DM
zwfs.dm.update_shape(cmd) #

ax[0,1].imshow( zwfs.dm.surface ) 
ax[0,1].set_title('DM surface 1')

input_field = input_field.applyDM(zwfs.dm) # apply our DM
outfield = zwfs.FPM.get_output_field( input_field, keep_intermediate_products=False) #get the outputfield from phase mask
outfield.define_pupil_grid(dx=zwfs.mode['telescope']['telescope_diameter']/zwfs.mode['telescope']['telescope_diameter_pixels'], D_pix=zwfs.mode['telescope']['telescope_diameter_pixels'])

ax[0,2].imshow( abs( outfield.flux[zwfs.wvls[0]] * np.exp(1j*outfield.phase[zwfs.wvls[0]] ) )**2 ) 
ax[0,2].set_title('output intensity 1')

# 2nd row 
input_field = baldr.create_calibration_field_for_ZWFS( calibration_source_config_dict, zwfs)
ax[1,0].imshow( input_field.phase[zwfs.wvls[0]] ) 
ax[1,0].set_title('input phase')

cmd = AMP * control_basis[6].reshape(-1) # apply a new mode on DM
zwfs.dm.update_shape(cmd) #

ax[1,1].imshow( zwfs.dm.surface ) 
ax[1,1].set_title('DM surface 2')

input_field = calibration_field.applyDM(zwfs.dm)
outfield  = zwfs.FPM.get_output_field( input_field  ,keep_intermediate_products=False)
outfield.define_pupil_grid(dx=zwfs.mode['telescope']['telescope_diameter']/zwfs.mode['telescope']['telescope_diameter_pixels'], D_pix=zwfs.mode['telescope']['telescope_diameter_pixels'])

ax[1,2].imshow( abs( outfield.flux[zwfs.wvls[0]] * np.exp(1j*outfield.phase[zwfs.wvls[0]] ) )**2 )
ax[1,2].set_title('output intensity 2')

# 3rd row 
input_field = baldr.create_calibration_field_for_ZWFS( calibration_source_config_dict, zwfs)
ax[2,0].imshow( input_field.phase[zwfs.wvls[0]] ) 
ax[2,0].set_title('input phase')

cmd = AMP * control_basis[6].reshape(-1)  # apply a new mode on DM
# add some abberations outside the pupil (null space of fibre coupler), make it so total phase shift applied at 4 points = 0. Then we can oscillate 
aaa= np.zeros(control_basis[0].shape)
aaa[zwfs.dm.N_act[0]//2,0] = 2*AMP * np.cos(0)
aaa[0,zwfs.dm.N_act[0]//2] = 2*AMP  * np.cos(np.pi/2)
aaa[zwfs.dm.N_act[0]//2,-1] = 2*AMP * np.cos(np.pi)
aaa[-1,zwfs.dm.N_act[0]//2] = 2*AMP * np.cos(3*np.pi/2)
cmd+=aaa.reshape(-1)
zwfs.dm.update_shape(cmd) #

ax[2,1].imshow( zwfs.dm.surface ) 
ax[2,1].set_title('DM surface 2')

input_field = calibration_field.applyDM(zwfs.dm)
outfield  = zwfs.FPM.get_output_field( input_field  ,keep_intermediate_products=False)
outfield.define_pupil_grid(dx=zwfs.mode['telescope']['telescope_diameter']/zwfs.mode['telescope']['telescope_diameter_pixels'], D_pix=zwfs.mode['telescope']['telescope_diameter_pixels'])

ax[2,2].imshow( abs( outfield.flux[zwfs.wvls[0]] * np.exp(1j*outfield.phase[zwfs.wvls[0]] ) )**2 )
ax[2,2].set_title('output intensity 2')

# from ABCD method could recouerate incoherent & coherent flux from central reference field and input field






#%%
def plot_ao_correction_process(phase_before, phase_reco, phase_after , title_list =None):
    """ everything should be input as  micrometer OPD """
    
    fig = plt.figure(figsize=(16, 12))
    
    ax1 = fig.add_subplot(131)
    ax1.set_title('phase before',fontsize=20)
    ax1.axis('off')
    im1 = ax1.imshow(  phase_before )
    
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    cbar = fig.colorbar( im1, cax=cax, orientation='horizontal')
    cbar.set_label( r'OPD $(\mu m)$', rotation=0)
    
    ax2 = fig.add_subplot(132)
    ax2.set_title('reconstructed phase',fontsize=20)
    ax2.axis('off')
    im2 = ax2.imshow( phase_reco )
    
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    cbar = fig.colorbar(im2, cax=cax, orientation='horizontal')
    cbar.set_label( r'OPD $(\mu m)$', rotation=0)
    
    ax3 = fig.add_subplot(133)
    ax3.set_title('Residual',fontsize=20)
    ax3.axis('off')
    im3 = ax3.imshow( phase_after ) 
    
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    cbar = fig.colorbar(im3, cax=cax, orientation='horizontal')
    cbar.set_label( r'OPD $(\mu m)$', rotation=0)


"""
# define a field

# create a DM

# create a FPM

# create a detector

# detect the field 

# send a command to the DM and then re-detect the field   
"""

# prelims 
dim = 12*20
stellar_dict = { 'Hmag':6,'r0':0.1,'L0':25,'V':50,'airmass':1.3,'extinction': 0.18 }
tel_dict = { 'dim':dim,'D':1.8,'D_pix':12*20,'pup_geometry':'AT' }
naomi_dict = { 'naomi_lag':4.5e-3, 'naomi_n_modes':14, 'naomi_lambda0':0.6e-6,\
              'naomi_Kp':1.1, 'naomi_Ki':0.93 }
FPM_dict = {'A':1, 'B':1, 'f_ratio':21, 'd_on':26.5e-6, 'd_off':26e-6,\
            'glass_on':'sio2', 'glass_off':'sio2','desired_phase_shift':60,\
                'rad_lam_o_D':1.2 ,'N_samples_across_phase_shift_region':10,\
                    'nx_size_focal_plane':dim}
det_dict={'DIT' : 1, 'ron' : 1, 'pw' : 20, 'QE' : 1}

baldr_dict={'baldr_lag':0.5e-3,'baldr_lambda0':1.6e-6,'baldr_Ki':0.1, 'baldr_Kp':9}

locals().update(stellar_dict)
locals().update(tel_dict)
locals().update(naomi_dict)
locals().update(FPM_dict)
locals().update(det_dict)
locals().update(baldr_dict)

dx = D/D_pix # grid spatial element (m/pix)
ph_flux_H = baldr.star2photons('H', Hmag, airmass=airmass, k = extinction, ph_m2_s_nm = True) # convert stellar mag to ph/m2/s/nm 

pup = baldr.pick_pupil(pupil_geometry='disk', dim=D_pix, diameter=D_pix) # baldr.AT_pupil(dim=D_pix, diameter=D_pix) #telescope pupil

basis = zernike.zernike_basis(nterms=20, npix=D_pix)


# define a field
wvls = np.linspace(1.4e-6,1.7e-6,10) # input wavelengths 
input_phases = [np.nan_to_num(basis[3]) * (500e-9/w)**(6/5) for w in wvls]
input_fluxes = [ph_flux_H * pup  for _ in wvls] # ph_m2_s_nm

input_field = baldr.field( phases = input_phases  , fluxes = input_fluxes  , wvls=wvls )

input_field.define_pupil_grid(dx=dx, D_pix=D_pix)


plt.figure()
plt.imshow( input_field.phase[wvls[0]] )


# create a DM
N_act=[12,12]
dm = baldr.DM(surface=100e-9 * np.eye(N_act[0]), gain=1, angle=0)


# create a FPM

phase_shift_diameter = rad_lam_o_D * f_ratio * wvls[0]   ##  f_ratio * wvls[0] = lambda/D  given f_ratio
"""#dx_focal_plane = phase_shift_diameter / N_samples_across_phase_shift_region  # dif elemetn in focal plane (m)



        if nx_size_focal_plane==None:
            nx_size_focal_plane = input_field.nx_size
            
        if dx_focal_plane==None:
            dx_focal_plane = self.phase_shift_diameter/20
"""

FPM = baldr.zernike_phase_mask(A,B,phase_shift_diameter,f_ratio,d_on,d_off,glass_on,glass_off)

FPM.optimise_depths(desired_phase_shift=desired_phase_shift, across_desired_wvls=wvls ,verbose=True)

#FPM_cal = baldr.zernike_phase_mask(A,B,phase_shift_diameter,f_ratio,d_on,d_off,glass_on,glass_off)
#FPM_cal.d_off = FPM_cal.d_on


# get output field with and without DM command

output_field = FPM.get_output_field( input_field, wvl_lims=[0,100], \
                                        nx_size_focal_plane = None , dx_focal_plane = None, keep_intermediate_products=True )
  
output_field.define_pupil_grid(dx=input_field.dx, D_pix=input_field.D_pix)

input_field_dm = input_field.applyDM(dm)
output_field_dm = FPM.get_output_field( input_field_dm, wvl_lims=[0,100], \
                                        nx_size_focal_plane = None , dx_focal_plane = None, keep_intermediate_products=True )
    
output_field_dm.define_pupil_grid(dx=input_field.dx, D_pix=input_field.D_pix)

# create a detector

npix_det = D_pix//pw # number of pixels across detector 
pix_scale_det = dx * pw # m/pix

det_dict['[calc]npix_det'] = npix_det 
det_dict['[calc]pix_scale_det'] = pix_scale_det

# set up detector object
det = baldr.detector(npix=npix_det, pix_scale = pix_scale_det , DIT= DIT, ron=ron, QE={w:QE for w in wvls})


# send a command to the DM and then re-detect the field  

sig1 = det.detect_field( output_field )

sig2 = det.detect_field( output_field_dm )

fig,ax = plt.subplots(2,1)
ax[0].imshow( sig1.signal )
ax[1].imshow( sig2.signal )




#%% Interaction matrix using I(90) - I_ref(90) ) / Nph following OLIVIER FAUVARQUE 2016

dim = 12*20
stellar_dict = { 'Hmag':6,'r0':0.1,'L0':25,'V':50,'airmass':1.3,'extinction': 0.18 }
tel_dict = { 'dim':dim,'D':1.8,'D_pix':12*20,'pup_geometry':'AT' }
naomi_dict = { 'naomi_lag':4.5e-3, 'naomi_n_modes':14, 'naomi_lambda0':0.6e-6,\
              'naomi_Kp':1.1, 'naomi_Ki':0.93 }
FPM_dict = {'A':1, 'B':1, 'f_ratio':21, 'd_on':26.5e-6, 'd_off':26e-6,\
            'glass_on':'sio2', 'glass_off':'sio2','desired_phase_shift':60,\
                'rad_lam_o_D':1.2 ,'N_samples_across_phase_shift_region':10,\
                    'nx_size_focal_plane':dim}
det_dict={'DIT' : 1, 'ron' : 1, 'pw' : 20, 'QE' : 1}

baldr_dict={'baldr_lag':0.5e-3,'baldr_lambda0':1.6e-6,'baldr_Ki':0.1, 'baldr_Kp':9}

locals().update(stellar_dict)
locals().update(tel_dict)
locals().update(naomi_dict)
locals().update(FPM_dict)
locals().update(det_dict)
locals().update(baldr_dict)

# apply a Zernike mode to the input phase 
wvls = np.linspace(1.4e-6,1.8e-6,10) # input wavelengths 

dx = D/D_pix # grid spatial element (m/pix)
ph_flux_H = baldr.star2photons('H', Hmag, airmass=airmass, k = extinction, ph_m2_s_nm = True) # convert stellar mag to ph/m2/s/nm 
ph_flux_H_cal = baldr.star2photons('H', Hmag, airmass=airmass, k = extinction, ph_m2_s_nm = True) # convert stellar mag to ph/m2/s/nm 

pup = baldr.pick_pupil(pupil_geometry='disk', dim=D_pix, diameter=D_pix) # baldr.AT_pupil(dim=D_pix, diameter=D_pix) #telescope pupil

N_controlled_modes = 20 
basis = zernike.zernike_basis(nterms=N_controlled_modes, npix=D_pix)



## ==== CREATE calibration field, DM, phase masks, detectors, 

# calibration field 
calibration_phases = [np.nan_to_num(basis[0])  for w in wvls]
calibration_fluxes = [ph_flux_H_cal * pup  for _ in wvls] # ph_m2_s_nm

calibration_field = baldr.field( phases = calibration_phases, fluxes = calibration_fluxes  , wvls=wvls )
calibration_field.define_pupil_grid( dx=dx, D_pix=D_pix )

# 
N_act=[12,12]
pup_inner = baldr.pick_pupil(pupil_geometry='disk', dim=D_pix, diameter=D_pix-D_pix//N_act[0]) # define DM pupil a little smaller 
dm = baldr.DM(surface=np.zeros(N_act), gain=1, angle=0,surface_type = 'continuous') 

phase_shift_diameter = rad_lam_o_D * f_ratio * wvls[0]   ##  f_ratio * wvls[0] = lambda/D  given f_ratio
FPM = baldr.zernike_phase_mask(A,B,phase_shift_diameter,f_ratio,d_on,d_off,glass_on,glass_off)

FPM.optimise_depths(desired_phase_shift=desired_phase_shift, across_desired_wvls=wvls ,verbose=True)

FPM_cal = baldr.zernike_phase_mask(A,B,phase_shift_diameter,f_ratio,d_on,d_off,glass_on,glass_off)
FPM_cal.d_off = FPM_cal.d_on

# set up detector object
det = baldr.detector(npix=npix_det, pix_scale = pix_scale_det , DIT= DIT, ron=ron, QE={w:QE for w in wvls})
mask = baldr.pick_pupil(pupil_geometry='disk', dim=det.npix, diameter=det.npix) 




## ==== CREATE INTERACTION MATRIX
# modal IM  (modal)  
cmd = np.zeros( dm.surface.reshape(-1).shape ) 
dm.update_shape(cmd) #zero dm first


# get the reference signal from calibration field with phase mask in
sig_on_ref = baldr.detection_chain(calibration_field, dm, FPM, det)
sig_on_ref.signal = np.mean( [baldr.detection_chain(calibration_field, dm, FPM, det).signal for _ in range(10)]  , axis=0) # average over a few 

# estimate #photons of in calibration field by removing phase mask (zero phase shift)   
sig_off_ref = baldr.detection_chain(calibration_field, dm, FPM_cal, det)
sig_off_ref.signal = np.mean( [baldr.detection_chain(calibration_field, dm, FPM_cal, det).signal for _ in range(10)]  , axis=0) # average over a few 
Nph_cal = np.sum(sig_off_ref.signal)

# Put modes on DM and measure signals from calibration field
pokeAmp = 50e-9

# CREATE THE CONTROL BASIS FOR OUR DM
control_basis = baldr.create_control_basis(dm=dm, N_controlled_modes=N_controlled_modes, basis_modes='zernike')

# BUILD OUR INTERACTION AND CONTROL MATRICESFROM THE CALIBRATION SOURCE AND OUR ZWFS SETUP
IM_modal, pinv_IM_modal = baldr.build_IM(calibration_field=calibration_field, dm=dm, FPM=FPM, det=det, control_basis=control_basis, pokeAmp=pokeAmp)

U,S,Vt = np.linalg.svd( IM_modal )
plt.figure()
plt.loglog(S)
plt.ylabel('singular values')
plt.xlabel('mode #')



#%% ==== OPEN LOOP CORRCTION 
#now create our input field & correct it with WFS ut_phases = [np.nan_to_num(basis[7]) * (500e-9/w)**(6/5) for w in wvls]


#scrn_seed = baldr.PhaseScreen_PostAO(nx_size=dim, pixel_scale=dx, r0=0.15, L0=20, D=1.8, sigma2_ao =  -np.log( 0.5 ) , N_act=144)
#for i in range(1000): scrn_seed.add_row()



input_fluxes = [10 * ph_flux_H * pup  for _ in wvls] # ph_m2_s_nm
input_phases = [np.nan_to_num(basis[4]) * (500e-9/w)**(6/5) for w in wvls] #particular mode
#aaa = 1 * np.sum([ a * np.nan_to_num(basis[i]) for a, i in zip(np.random.rand(10), range(10))],axis=0)
#input_phases = [aaa * (500e-9/w)**(6/5) for w in wvls] 
#input_phases = [  pup * scrn_seed.scrn * (500e-9/wvls[-1])**(6/5) for w in wvls]

input_field = baldr.field( phases = input_phases  , fluxes = input_fluxes  , wvls=wvls )

input_field.define_pupil_grid(dx=dx, D_pix=D_pix)

"""
sig_on = baldr.detection_chain(input_field, dm, FPM, det)
sig_on_ref = baldr.detection_chain(calibration_field, dm, FPM, det)

sig_off = baldr.detection_chain(input_field, dm, FPM_cal, det)
sig_off.signal = np.mean( [baldr.detection_chain(input_field, dm, FPM_cal, det).signal for _ in range(10)]  , axis=0) # average over a few 

Nph = np.sum(sig_off.signal)

# check it makes sense
fig,ax = plt.subplots(1,2,figsize=(8,4))
ax[0].imshow( input_field.phase[wvls[-1]] )
ax[0].set_title('field input phase')
ax[1].imshow( 1/Nph * (sig_on.signal - sig_on_ref.signal) ) # * mask)
ax[1].set_title('normalized detector signal ')
"""

#zero dm first
cmd = np.zeros( dm.surface.reshape(-1).shape ) 
dm.update_shape(cmd) 

# estimate #photons 
sig_off=baldr.detection_chain(input_field, dm, FPM_cal, det)
sig_off.signal = np.mean( [baldr.detection_chain(input_field, dm, FPM_cal, det).signal for _ in range(10)]  , axis=0) # average over a few 
Nph = np.sum(sig_off.signal)

# detect input field
sig_turb = baldr.detection_chain(input_field, dm, FPM, det)

#important to scale reference field by ratio of phtons in calibrator vs target 
modal_reco_list = pinv_IM_modal.T @ (  1/Nph * (sig_turb.signal - Nph/Nph_cal * sig_on_ref.signal) ).reshape(-1) 
gains =  -pokeAmp * np.ones( len(modal_reco_list) ) #[0] + list(-25e-9 * np.ones( len(modal_reco_list)-1 )) +[0]

dm_reco = np.sum( np.array([gains[i] * a * Z for i,(a,Z) in enumerate( zip(modal_reco_list, control_basis))]) , axis=0)


cmd=dm_reco.reshape(-1)

dm.update_shape( cmd )   


plt.plot( np.linspace(-1,1,len(dm.surface)), dm.surface[len(dm.surface)//2,:] ); plt.plot(np.linspace(-1,1,len(input_field.phase[wvls[0]])), input_field.phase[wvls[0]][len(input_field.phase[wvls[0]])//2,:] )

plt.figure()
plt.imshow(dm.surface) 



input_field_dm = input_field.applyDM(dm)


print(f'opd before = {np.std( input_field.phase[wvls[0]][pup_inner>0] )}nm rms')
print(f'opd after = {np.std( input_field_dm.phase[wvls[0]][pup_inner>0] )}nm rms')
#sig_after = baldr.detection_chain(input_field, dm, FPM, det)

i=0
plot_ao_correction_process( phase_before= 1e6 *wvls[i]/(2*np.pi)* input_field.phase[wvls[i]] , phase_reco = 1e6 *dm.surface, phase_after = 1e6 *wvls[i]/(2*np.pi)*input_field_dm.phase[wvls[i]] , title_list =None)



# %%==== CLOSED LOOP CORRCTION 


# init phase screen from first stage AO  (define Strehl, r0 at wvls[-1])
#scrn_seed = baldr.PhaseScreen_PostAO(nx_size=dim, pixel_scale=dx, r0=0.1, L0=23, D=1.8, sigma2_ao =  -np.log( 0.2 ) , N_act=44)
#for i in range(1000): scrn_seed.add_row() # run it to get rid of transiet behaviour 
# baldr.PhaseScreen_PostAO( seems to always evolve to very high strehls independent of sigma2_ao ??

scrn_seed = baldr.PhaseScreenKolmogorov(nx_size=dim, pixel_scale=dx, r0=0.1, L0=23)
AscalefactoR =  15e-2 # to scale input phase screen by 


# keep input fluxes constant 
input_fluxes = [10 * ph_flux_H * pup  for _ in wvls] # ph_m2_s_nm

# init phase screens (note we definer0 etc at wvls[-1])
input_phases = [pup * AscalefactoR  *  scrn_seed.scrn * (wvls[-1]/w)**(6/5) for w in wvls] #(500e-9/w)**(6/5) for w in wvls]
input_field = baldr.field( phases = input_phases  , fluxes = input_fluxes  , wvls=wvls )
input_field.define_pupil_grid(dx=dx, D_pix=D_pix)

#zero dm first
cmd = np.zeros( dm.surface.reshape(-1).shape ) 
dm.update_shape(cmd) 

# initial estimate #photons 
sig_off.signal = np.mean( [baldr.detection_chain(input_field, dm, FPM_cal, det).signal for _ in range(10)]  , axis=0) # average over a few 
Nph = np.sum(sig_off.signal)

#baldr PI parameters 
Kp= 0.1 #0.05 #0.9
Ki= 0.9 #0.95  #0.5


# init performance tracking lists 
opd_before = [ np.std(input_field.phase[wvls[0]][pup>0.5] ) ]
opd_after= [ np.std(input_field.phase[wvls[0]][pup>0.5] ) ]
strehl_before = [ np.exp(-np.var(input_field.phase[wvls[0]][pup>0.5] ) ) ]
strehl_after = [ np.exp(-np.var(input_field.phase[wvls[0]][pup>0.5] ) ) ]
for it in range(100):
    
    print(it) 
    # roll phase screen 
    new_phases = {w:pup * AscalefactoR * scrn_seed.add_row() * (500e-9/w)**(6/5) for w in wvls}
    
    # append to field 
    input_field.phases = new_phases 

    
    if 0: #np.mod(it,5)==0:  # every so often we take out phase mask to measure #photons 
        # estimate #photons 
        sig_off.signal = np.mean( [baldr.detection_chain(input_field, dm, FPM_cal, det).signal for _ in range(10)]  , axis=0) # average over a few 
        Nph = np.sum(sig_off.signal)


    
    # detect input field
    sig_turb = baldr.detection_chain(input_field, dm, FPM, det)

    # reconstruct the phase 
    #important to scale reference field by ratio of phtons in calibrator vs target 
    modal_reco_list = pinv_IM_modal.T @ (  1/Nph * (sig_turb.signal - Nph/Nph_cal * sig_on_ref.signal) ).reshape(-1) 
    gains =  -pokeAmp * np.ones( len(modal_reco_list) ) #[0] + list(-25e-9 * np.ones( len(modal_reco_list)-1 )) +[0]
    err_signal = np.sum( np.array([gains[i] * a * Z for i,(a,Z) in enumerate( zip(modal_reco_list, control_basis))]) , axis=0)

    # get the error signal
    new_cmd = err_signal.reshape(-1)
    
    # applyy PI control 
    cmd = Kp * new_cmd + Ki * cmd
    
    # update DM shape
    dm.update_shape(cmd) 
    
    # apply new DM shape to input field 
    input_field = input_field.applyDM(dm)
    
    opd_before.append( wvls[0]/(np.pi*2) * np.std(AscalefactoR * scrn_seed.scrn[pup>0.5] * (wvls[-1]/wvls[0])**(6/5)) )
    
    opd_after.append( wvls[0]/(np.pi*2) * np.std(input_field.phase[wvls[0]][pup>0.5] ) )
    
    strehl_before.append( np.exp(-np.var(AscalefactoR * scrn_seed.scrn[pup>0.5] * (wvls[-1]/wvls[0])**(6/5)) ) )
    
    strehl_after.append( np.exp(-np.var(input_field.phase[wvls[0]][pup>0.5] ) ) )
    
    
"""
plt.plot( opd_before, label='1st STAGE AO')
plt.plot(opd_after, label='1st STAGE AO + BALDR')
plt.legend()
"""
plt.plot( strehl_before, label='1st STAGE AO')
plt.plot( strehl_after, label='1st STAGE AO + BALDR')
plt.legend()











#%% OLD STUFF BELOW

# Now put a mode on input with two masks (0 and 90 deg phase shift)and see if we can correct with DM from inverse IM 

# np.linalg.pinv(IM)



def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    
    python code from here https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite 
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False




# prelims 
dim = 12*20
stellar_dict = { 'Hmag':6,'r0':0.1,'L0':25,'V':50,'airmass':1.3,'extinction': 0.18 }
tel_dict = { 'dim':dim,'D':1.8,'D_pix':12*20,'pup_geometry':'AT' }
naomi_dict = { 'naomi_lag':4.5e-3, 'naomi_n_modes':14, 'naomi_lambda0':0.6e-6,\
              'naomi_Kp':1.1, 'naomi_Ki':0.93 }
FPM_dict = {'A':1, 'B':1, 'f_ratio':21, 'd_on':26.5e-6, 'd_off':26e-6,\
            'glass_on':'sio2', 'glass_off':'sio2','desired_phase_shift':60,\
                'rad_lam_o_D':1.2 ,'N_samples_across_phase_shift_region':10,\
                    'nx_size_focal_plane':dim}
det_dict={'DIT' : 1, 'ron' : 1, 'pw' : 20, 'QE' : 1}

baldr_dict={'baldr_lag':0.5e-3,'baldr_lambda0':1.6e-6,'baldr_Ki':0.1, 'baldr_Kp':9}

locals().update(stellar_dict)
locals().update(tel_dict)
locals().update(naomi_dict)
locals().update(FPM_dict)
locals().update(det_dict)
locals().update(baldr_dict)

dx = D/D_pix # grid spatial element (m/pix)
ph_flux_H = baldr.star2photons('H', Hmag, airmass=airmass, k = extinction, ph_m2_s_nm = True) # convert stellar mag to ph/m2/s/nm 

pup = baldr.pick_pupil(pupil_geometry='disk', dim=D_pix, diameter=D_pix) # baldr.AT_pupil(dim=D_pix, diameter=D_pix) #telescope pupil

basis = zernike.zernike_basis(nterms=20, npix=D_pix)



# apply a Zernike mode to the input phase 
wvls = np.linspace(1.4e-6,1.7e-6,10) # input wavelengths 
input_phases = [np.nan_to_num(basis[7]) * (500e-9/w)**(6/5) for w in wvls]
calibration_phases = [np.nan_to_num(basis[0])  for w in wvls]

input_fluxes = [ph_flux_H * pup  for _ in wvls] # ph_m2_s_nm

input_field = baldr.field( phases = input_phases  , fluxes = input_fluxes  , wvls=wvls )

input_field.define_pupil_grid(dx=dx, D_pix=D_pix)

calibration_field = baldr.field( phases = calibration_phases, fluxes = input_fluxes  , wvls=wvls )
calibration_field.define_pupil_grid( dx=dx, D_pix=D_pix )

N_act=[12,12]
dm = baldr.DM(surface=np.zeros(N_act), gain=1, angle=0,surface_type = 'continuous') 

phase_shift_diameter = rad_lam_o_D * f_ratio * wvls[0]   ##  f_ratio * wvls[0] = lambda/D  given f_ratio
FPM = baldr.zernike_phase_mask(A,B,phase_shift_diameter,f_ratio,d_on,d_off,glass_on,glass_off)

FPM.optimise_depths(desired_phase_shift=desired_phase_shift, across_desired_wvls=wvls ,verbose=True)

FPM_cal = baldr.zernike_phase_mask(A,B,phase_shift_diameter,f_ratio,d_on,d_off,glass_on,glass_off)
FPM_cal.d_off = FPM_cal.d_on

# set up detector object
det = baldr.detector(npix=npix_det, pix_scale = pix_scale_det , DIT= DIT, ron=ron, QE={w:QE for w in wvls})
mask = baldr.pick_pupil(pupil_geometry='disk', dim=det.npix, diameter=det.npix) 

sig = baldr.detection_chain(input_field, dm, FPM, det)
sig_cal = baldr.detection_chain(input_field, dm, FPM_cal, det)
sig_cal.signal = np.mean( [baldr.detection_chain(input_field, dm, FPM_cal, det).signal for _ in range(10)]  ,axis=0) # average over a few 

# check it makes sense
fig,ax = plt.subplots(1,2,figsize=(8,4))
ax[0].imshow( input_field.phase[wvls[-1]] )
ax[0].set_title('field input phase')
ax[1].imshow( sig.signal/sig_cal.signal * mask)
ax[1].set_title('normalized detector signal ')



# Read chapter 6 Control techniques of Roddiers AO textbook
# check linearity of OPD -> signal ...
piston_signal_list = []
cmd = np.zeros( dm.surface.reshape(1,-1)[0].shape ) 
dm.update_shape(cmd) #zero dm first
actuation_range = np.linspace(1e-9,1e-6,100)
for i in actuation_range:
    #cmd = i * np.ones( dm.surface.reshape(1,-1)[0].shape ) 
    cmd[len(cmd)//2+20] = i 
    dm.update_shape(cmd)
    
    sig = baldr.detection_chain(calibration_field, dm, FPM, det)
    piston_signal_list.append(sig)

#which indx are we looking at?
#i=10;np.where( np.diff( piston_signal_list[i].signal[mask>0.5] )==np.max(np.diff( piston_signal_list[i].signal[mask>0.5] )) )[0][0]

plt.figure()
plt.plot( 1e9 * actuation_range, [s.signal[mask>0.5] for s in piston_signal_list])
plt.ylabel('detector pixel signal [adu]')
plt.xlabel('actuator amplitude [nm]')
plt.title('pixel response to linear ramp on a singule actuator on the DM')

#plt.plot( [s.signal[s.signal.shape[0]//2,s.signal.shape[0]//2] for s in piston_signal_list])

"""
# poke IM  (zonal)
cmd = np.zeros( dm.surface.reshape(1,-1)[0].shape ) 
dm.update_shape(cmd) #zero dm first

sig_cal = baldr.detection_chain(calibration_field, dm, FPM_cal, det)
poke_signal_list=[]
for i,_ in enumerate(cmd):
    cmd[i]=50e-9
    if i>0:
        cmd[i-1]=0
        
    dm.update_shape(cmd)
    #dm_list.append( dm.surface )      
    
    sig = baldr.detection_chain(calibration_field, dm, FPM, det)
    
    poke_signal_list.append( sig ) 


IM_poke = []
for s in poke_signal_list:
    IM_poke.append( list( (mask * s.signal / sig_cal.signal ).reshape(-1) )  )   # filter out modes that are outside pupil with mask

print('condition of IM_poke=', np.linalg.cond( IM_poke ))
U,S,Vt = np.linalg.svd( IM_poke )
plt.figure()
plt.loglog(S)
plt.ylabel('singular values')

singular_threshold= 1e-4
gains = np.array([1 if s > singular_threshold else 0 for s in S]) 

S_inv_opt = np.array( [g/s if np.isfinite(1/s) else 0 for s,g in zip(S,gains)] )

pinv_IM_poke = Vt.T @ np.diag( S_inv_opt ) @ U.T     


# now look at input field instead of calibration field to see if we can correct it with the DM using IM_poke
sig_cal_turb = baldr.detection_chain(input_field, dm, FPM_cal, det)
sig_turb = baldr.detection_chain(input_field, dm, FPM, det)

plt.figure()
plt.imshow( (mask * sig_turb.signal/sig_cal_turb.signal ) )
cmd = pinv_IM_poke @ (mask * sig_turb.signal/sig_cal_turb.signal ).reshape(-1)
dm.update_shape(cmd)   

plt.figure()
plt.imshow(dm.surface) # ... has some instabilities 
"""

# modal IM  (modal)    
cmd = np.zeros( dm.surface.reshape(-1).shape ) 
dm.update_shape(cmd) #zero dm first

modal_signal_list = []
zernike_control_basis  = [np.nan_to_num(b) for b in zernike.zernike_basis(nterms=20, npix=dm.N_act[0]) ]
for b in zernike_control_basis:
    cmd = 50e-9 * b.reshape(1,-1)[0]
    
    dm.update_shape(cmd)

    sig = baldr.detection_chain(calibration_field, dm, FPM, det)
    #average over a few 
    sig.signal = np.mean( [baldr.detection_chain(calibration_field, dm, FPM, det).signal for _ in range(10)] ,axis=0)
    modal_signal_list.append( sig ) 
    

IM_modal = []
for s in modal_signal_list:
    IM_modal.append( list( (mask * s.signal / sig_cal.signal ).reshape(-1) )  )   # filter out modes that are outside pupil with mask

print('condition of IM_modal=', np.linalg.cond( IM_modal ))


U,S,Vt = np.linalg.svd( IM_modal )
plt.figure()
plt.loglog(S)
plt.ylabel('singular values')

"""
singular_threshold= 1e-4
gains = np.array([1 if s > singular_threshold else 0 for s in S]) 
S_inv_opt = np.array( [g/s if np.isfinite(1/s) else 0 for s,g in zip(S,gains)] )
pinv_IM_poke = Vt.T @ np.diag( S_inv_opt ) @ U.T 
"""
pinv_IM_modal = np.linalg.pinv(IM_modal)
# now look at input field instead of calibration field to see if we can correct it with the DM using IM_poke
sig_cal_turb = baldr.detection_chain(input_field, dm, FPM_cal, det)
sig_turb = baldr.detection_chain(input_field, dm, FPM, det)

plt.figure()
plt.imshow( (mask * sig_turb.signal/sig_cal_turb.signal ) )

modal_reco_list = pinv_IM_modal.T @ (mask * sig_turb.signal/sig_cal_turb.signal ).reshape(-1)
gains =  [0] + list(-25e-9 * np.ones( len(modal_reco_list)-1 )) +[0]
dm_reco = np.sum( np.array([gains[i] * g * m for i,(g,m) in enumerate( zip(modal_reco_list, zernike_control_basis))]) , axis=0)

cmd=dm_reco.reshape(-1)
dm.update_shape(cmd)   

plt.figure()
plt.imshow(dm.surface) 

input_field_dm = input_field.applyDM(dm)


print(f'opd before = {np.std( input_field.phase[wvls[0]] )}nm rms')
print(f'opd after = {np.std( input_field_dm.phase[wvls[0]] )}nm rms')
#sig_after = baldr.detection_chain(input_field, dm, FPM, det)

i=0
plot_ao_correction_process( phase_before= 1e6 *wvls[i]/(2*np.pi)* input_field.phase[wvls[i]] , phase_reco = 1e6 *dm.surface, phase_after = 1e6 *wvls[i]/(2*np.pi)*input_field_dm.phase[wvls[i]] , title_list =None)



#%% using I(theta)-I0
# modal IM  (modal)    
cmd = np.zeros( dm.surface.reshape(-1).shape ) 
dm.update_shape(cmd) #zero dm first

modal_signal_list = []
zernike_control_basis  = [np.nan_to_num(b) for b in zernike.zernike_basis(nterms=20, npix=dm.N_act[0]) ]
for b in zernike_control_basis:
    cmd = 50e-9 * b.reshape(1,-1)[0]
    
    dm.update_shape(cmd)

    sig = baldr.detection_chain(calibration_field, dm, FPM, det)
    #average over a few 
    sig.signal = np.mean( [baldr.detection_chain(calibration_field, dm, FPM, det).signal for _ in range(10)] ,axis=0)
    modal_signal_list.append( sig ) 
    

IM_modal = []
for s in modal_signal_list:
    IM_modal.append( list( (mask * (s.signal - sig_cal.signal) ).reshape(-1) )  )   # filter out modes that are outside pupil with mask

print('condition of IM_modal=', np.linalg.cond( IM_modal ))


U,S,Vt = np.linalg.svd( IM_modal )
plt.figure()
plt.loglog(S)
plt.ylabel('singular values')

"""
singular_threshold= 1e-4
gains = np.array([1 if s > singular_threshold else 0 for s in S]) 
S_inv_opt = np.array( [g/s if np.isfinite(1/s) else 0 for s,g in zip(S,gains)] )
pinv_IM_poke = Vt.T @ np.diag( S_inv_opt ) @ U.T 
"""
pinv_IM_modal = np.linalg.pinv(IM_modal)
# now look at input field instead of calibration field to see if we can correct it with the DM using IM_poke
sig_cal_turb = baldr.detection_chain(input_field, dm, FPM_cal, det)
sig_turb = baldr.detection_chain(input_field, dm, FPM, det)

plt.figure()
plt.imshow( (mask * sig_turb.signal/sig_cal_turb.signal ) )

modal_reco_list = pinv_IM_modal.T @ (mask * (sig_turb.signal-sig_cal_turb.signal) ).reshape(-1)
gains =  [0] + list(-25e-9 * np.ones( len(modal_reco_list)-1 )) +[0]
dm_reco = np.sum( np.array([gains[i] * g * m for i,(g,m) in enumerate( zip(modal_reco_list, zernike_control_basis))]) , axis=0)

cmd=dm_reco.reshape(-1)
dm.update_shape(cmd)   

plt.figure()
plt.imshow(dm.surface) 

input_field_dm = input_field.applyDM(dm)


print(f'opd before = {np.std( input_field.phase[wvls[0]] )}nm rms')
print(f'opd after = {np.std( input_field_dm.phase[wvls[0]] )}nm rms')
#sig_after = baldr.detection_chain(input_field, dm, FPM, det)

i=0
plot_ao_correction_process( phase_before= 1e6 *wvls[i]/(2*np.pi)* input_field.phase[wvls[i]] , phase_reco = 1e6 *dm.surface, phase_after = 1e6 *wvls[i]/(2*np.pi)*input_field_dm.phase[wvls[i]] , title_list =None)




#%% with KL Modes 

# want to get change of basis matrix to go from Zernike to KL modes 
# do this by by diaonalizing covariance matrix of Zernike basis  with SVD , since Hermitian Vt=U^-1 , therefore our change of basis vectors! 
b0 = np.array( [np.nan_to_num(b) for b in zernike_control_basis] )
cov0 = np.cov( b0.reshape(len(b0),-1) )  # have to be careful how nan to zero replacements are made since cov should be calculated only where Zernike basis is valid , ie not nan
KL_B , S,  iKL_B = np.linalg.svd( cov0 )
# take a look plt.figure(): plt.imshow( (b0.T @ KL_B[:,:] ).T [2])



# modal IM  (modal)    
cmd = np.zeros( dm.surface.reshape(-1).shape ) 
dm.update_shape(cmd) #zero dm first

modal_signal_list = []
KL_control_basis  = (b0.T @ KL_B[:,:] ).T  #[b.T @ KL_B[:,:] for b in b0 ]
for b in KL_control_basis:
    cmd = 50e-9 * b.reshape(1,-1)[0]
    
    dm.update_shape(cmd)
    
    sig = baldr.detection_chain(calibration_field, dm, FPM, det)
    # average a few 
    sig.signal = np.mean( [baldr.detection_chain(calibration_field, dm, FPM, det).signal for _ in range(10)] ,axis=0) #  average over a few exposures
        
    modal_signal_list.append( sig ) 
    

IM_modal = []
for s in modal_signal_list:
    IM_modal.append( list( (mask * s.signal / sig_cal.signal ).reshape(-1) )  )   # filter out modes that are outside pupil with mask

print('condition of IM_modal=', np.linalg.cond( IM_modal ))


U,S,Vt = np.linalg.svd( IM_modal )
plt.figure()
plt.loglog(S)
plt.ylabel('singular values')

"""
singular_threshold= 1e-4
gains = np.array([1 if s > singular_threshold else 0 for s in S]) 
S_inv_opt = np.array( [g/s if np.isfinite(1/s) else 0 for s,g in zip(S,gains)] )
pinv_IM_poke = Vt.T @ np.diag( S_inv_opt ) @ U.T 
"""
pinv_IM_modal = np.linalg.pinv(IM_modal)
# now look at input field instead of calibration field to see if we can correct it with the DM using IM_poke
sig_cal_turb = baldr.detection_chain(input_field, dm, FPM_cal, det)
sig_turb = baldr.detection_chain(input_field, dm, FPM, det)

plt.figure()
plt.imshow( (mask * sig_turb.signal/sig_cal_turb.signal ) )

modal_reco_list = pinv_IM_modal.T @ (mask * sig_turb.signal/sig_cal_turb.signal ).reshape(-1)
gains =  [0] + list(-25e-9 * np.ones( len(modal_reco_list)-1 )) + [0]
dm_reco = np.sum( np.array([gains[i] * g * m for i,(g,m) in enumerate( zip(modal_reco_list, KL_control_basis))]) , axis=0)

cmd=dm_reco.reshape(-1)
dm.update_shape(cmd)   

plt.figure()
plt.imshow(dm.surface) 

input_field_dm = input_field.applyDM(dm)


print(f'opd before = {np.std( input_field.phase[wvls[0]] )}nm rms')
print(f'opd after = {np.std( input_field_dm.phase[wvls[0]] )}nm rms')
#sig_after = baldr.detection_chain(input_field, dm, FPM, det)

i=0
plot_ao_correction_process( phase_before= 1e6 *wvls[i]/(2*np.pi)* input_field.phase[wvls[i]] , phase_reco = 1e6 *dm.surface, phase_after = 1e6 *wvls[i]/(2*np.pi)*input_field_dm.phase[wvls[i]] , title_list =None)






#%% 

# define a flat calibration field
wvls = np.linspace(1.4e-6,1.7e-6,10) # input wavelengths 
input_phases = [np.nan_to_num(basis[0]) * (500e-9/w)**(6/5) for w in wvls]
input_fluxes = [ph_flux_H * pup  for _ in wvls] # ph_m2_s_nm

input_field = baldr.field( phases = input_phases  , fluxes = input_fluxes  , wvls=wvls )

input_field.define_pupil_grid(dx=dx, D_pix=D_pix)


FPM = baldr.zernike_phase_mask(A,B,phase_shift_diameter,f_ratio,d_on,d_off,glass_on,glass_off)

FPM.optimise_depths(desired_phase_shift=desired_phase_shift, across_desired_wvls=wvls ,verbose=True)

FPM_cal = baldr.zernike_phase_mask(A,B,phase_shift_diameter,f_ratio,d_on,d_off,glass_on,glass_off)
FPM_cal.d_off = FPM_cal.d_on


# create a DM
N_act=[12,12]
dm = baldr.DM(surface=100e-9 * np.eye(N_act[0]), gain=1, angle=0)

sig_list=[]



cmd = np.zeros( dm.surface.reshape(1,-1)[0].shape ) 
dm.update_shape(cmd)
sig0 = baldr.detection_chain(input_field, dm, FPM_cal, det) # with 0 phase shift in mask 

sig_list=[]

for i,_ in enumerate(cmd):
    cmd[i]=wvls[0]/4
    if i>0:
        cmd[i-1]=0
        
    dm.update_shape(cmd)
    #dm_list.append( dm.surface )      
    
    sig = baldr.detection_chain(input_field, dm, FPM, det)
    
    sig_list.append( sig  ) 
    

IM = []
mask = baldr.pick_pupil(pupil_geometry='disk', dim=N_act[0], diameter=N_act[0]) # baldr.AT_pupil(dim=D_pix, diameter=D_pix) #telescope pupil
for s in sig_list:
    IM.append( list( (mask * s.signal / sig0.signal ).reshape(1,-1)[0] )  )   # filter out modes that are outside pupil with mask

IM=np.array(IM)
plt.figure()
plt.imshow(IM)

sig0 = baldr.detection_chain(input_field, dm, FPM, det)
sig0_cal = baldr.detection_chain(input_field, dm, FPM_cal, det)

fig,ax = plt.subplots(2,1)
ax[0].imshow( sig0.signal ,vmin=0, vmax = np.max(sig0.signal) )
ax[0].set_title('with FPM (90 deg phase shift)')
ax[1].imshow( sig0_cal.signal , vmin=0, vmax = np.max(sig0.signal) )
ax[1].set_title('with FPM_cal (0 deg phase shift)')




# I SHOULD TRY DM CORRECTION WITH LINEAR INTERPOLATION AT FIELD POINTS BETWEEN ACTUATORS (MEM)

#%%



#sig_cal = baldr.detection_chain(input_field, dm, FPM_cal)

#s = (sig.signal/sig_cal.signal).reshape(1,-1)[0]
#s = sig.signal.reshape(1,-1)[0]


#IM = np.linalg.pinv( IM )
#IM = []
#for s in sig_list:
#    IM.append( list( (s.signal ).reshape(1,-1)[0] )  )   

#IM=np.array(IM)
plt.figure()
plt.imshow(IM)

# Matrix is generally near singular! USE singular value decomposition to hep filtering 
U,S,Vt = np.linalg.svd( IM ) # U, S, Vt
    

# lts ee how a poke looks in the modal basis



std_list = []
for singular_threshold in np.logspace(-3,2,100):

    #U@U.T = I
    #Vt@Vt.T = I
    #np.max(abs(U @ np.diag( S )@ Vt- IM))
    #np.max(abs(Vt.T @ np.diag( 1/S )@ U.T - np.linalg.pinv(IM)) )
    #plt.figure()
    #plt.semilogy(S)
    
    #singular_threshold  = 5e3
    #singular_filter = S < singular_threshold 
    #Sfilt = S.copy()
    #Sfilt[singular_filter] = 0 # put (singular) values below threshold to 
    
    gains = np.array([1 if s > singular_threshold else 0 for s in S]) # np.ones( np.sum( ~singular_filter ) )
    
    Sopt = np.array( [g/s if np.isfinite(1/s) else 0 for s,g in zip(S,gains)] )
    IM_svd = U @ np.diag( S ) @ Vt
    pIM_svd = Vt.T @ np.diag( Sopt ) @ U.T     
    
    cmd = pIM_svd @ sig.signal.reshape(1,-1)[0]
    dm.update_shape(cmd)   
    
    input_field_dm = input_field.applyDM(dm)

    std_list.append( np.std( input_field_dm.phase[wvls[0]][pup>0.5] ) )

    


pup_det = baldr.pick_pupil(pupil_geometry='disk', dim=N_act[0], diameter=N_act[0]) # baldr.AT_pupil(dim=D_pix, diameter=D_pix) #telescope pupil

plt.figure()
plt.imshow( pup_det * dm.surface )



#%%
cmd = pIM_n  @ sig.signal.reshape(1,-1)[0]

dm.update_shape(cmd) 


plt.imshow ( (np.linalg.pinv( IM ) @ sig.signal.reshape(1,-1)[0])  )
