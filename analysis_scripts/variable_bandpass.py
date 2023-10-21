#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 20:58:42 2023

@author: bcourtne

how different bandpass's (changing $lambda_0$) effect pupil 

1. compare pupil with / without phase mask on perfect (Strehl~1) source
2. look how these change for changing bandwidth 
"""

import numpy as np
import pylab as plt
import pandas as pd
import os
import pyzelda.utils.zernike as zernike
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy 

#import zelda
os.chdir('/Users/bcourtne/Documents/ANU_PHD2/baldr')
from functions import baldr_functions_2 as baldr


#%% 1. compare pupil with / without phase mask on perfect (Strehl~1) source

dim = 12*20
stellar_dict = { 'Hmag':2,'r0':0.1,'L0':25,'V':50,'airmass':1.3,'extinction': 0.18 }
tel_dict = { 'dim':dim,'D':1.8,'D_pix':12*20,'pup_geometry':'AT' }

FPM_dict = {'A':1, 'B':1, 'f_ratio':21, 'd_on':26.5e-6, 'd_off':26e-6,\
            'glass_on':'sio2', 'glass_off':'sio2','desired_phase_shift':90,\
                'rad_lam_o_D':1.2 ,'N_samples_across_phase_shift_region':10,\
                    'nx_size_focal_plane':dim}
det_dict={'DIT' : 1, 'ron' : 0.5, 'pw' : 20, 'QE' : 1, 'npix_det':12}

baldr_dict={'baldr_lag':0.5e-3,'baldr_lambda0':1.6e-6,'baldr_Ki':0.1, 'baldr_Kp':9}

locals().update(stellar_dict)
locals().update(tel_dict)
locals().update(FPM_dict)
locals().update(det_dict)
locals().update(baldr_dict)

ron = 0
wvls = 1e-6 * np.linspace(1.4,1.8,10)
wvl0 = np.mean(  wvls  )

theta_at_wvl0= np.pi/2

n_on = baldr.nglass(1e6 * wvl0, glass=glass_on)[0]
n_off = baldr.nglass(1e6 * wvl0, glass=glass_off)[0]
n_air = baldr.nglass(1e6 * wvl0, glass='air')[0]

epsilon = 0.4 # mode amplitude
basis = zernike.zernike_basis(nterms=20, npix=D_pix)
pup = baldr.pick_pupil(pupil_geometry='AT', dim=D_pix, diameter=D_pix) 

dx = D/D_pix # grid spatial element (m/pix)
npix_det = int( D_pix//pw ) # number of pixels across detector 
pix_scale_det = dx * pw # m/pix
N_act = [12,12]
m = 4 # aberration index(Zernike)
Nph_wvl_1 = np.logspace(3,4,len(wvls)) #1e3*np.ones(len(wvls)) #np.logspace(3,4,len(wvls))
Nph_wvl_2 = np.logspace(3,4,len(wvls))[::-1] #1e3*np.ones(len(wvls)) #np.logspace(3,4,len(wvls))[::-1]

wvl0_1 = np.sum( Nph_wvl_1 * wvls ) / np.sum( Nph_wvl_1 )
wvl0_2 = np.sum( Nph_wvl_2 * wvls ) / np.sum( Nph_wvl_2 )

# ----- DM
dm = baldr.DM(surface=np.zeros(N_act), gain=1, angle=0,surface_type = 'continuous') 

# ----- PHASE MASKS
N_loD = 1.7
phase_shift_diameter = N_loD  * f_ratio * wvls[0]   ##  f_ratio * wvls[0] = lambda/D  given f_ratio
# calculate on-axis depth for 90deg phase shift
d_on = (  wvl0 * theta_at_wvl0/(np.pi*2) + n_off * d_off -   n_air * d_off ) / (n_on-n_air)
# phase shifting phase mask 
FPM = baldr.zernike_phase_mask(A,B,phase_shift_diameter,f_ratio,d_on,d_off,glass_on,glass_off)
#  nonphase shifting phase mask 
FPM_cal = baldr.zernike_phase_mask(A,B,phase_shift_diameter, f_ratio, d_off, d_off,glass_on,glass_off)
#FPM_cal.d_off = FPM_cal.d_on

# ----- DETECTOR
det = baldr.detector(npix=npix_det, pix_scale = pix_scale_det , DIT= DIT, ron=ron, QE={w:QE for w in wvls})


phase_t = basis[m]  # aberration 
phase_r = basis[0]  # reference 

phase_t = np.array( [ np.nan_to_num(basis[m]) * (500e-9/w)**(6/5) for w in wvls] )
phase_r = np.array( [ np.nan_to_num(basis[0]) * (500e-9/w)**(6/5) for w in wvls] )

input_fluxes_1 = [ p * pup  for p in Nph_wvl_1] # ph_m2_s_nm
input_fluxes_2 = [ p * pup  for p in Nph_wvl_2] # ph_m2_s_nm

input_field_r1 = baldr.field( phases = epsilon * ( phase_r) , fluxes = input_fluxes_1  , wvls=wvls )
input_field_r1.define_pupil_grid(dx=dx, D_pix=D_pix)

input_field_r2 = baldr.field( phases = epsilon * ( phase_r) , fluxes = input_fluxes_2  , wvls=wvls )
input_field_r2.define_pupil_grid(dx=dx, D_pix=D_pix)

#det = baldr.detector(npix=npix_det, pix_scale = pix_scale_det , DIT= DIT, ron=ron, QE={w:QE for w in wvls})
#mask = baldr.pick_pupil(pupil_geometry='disk', dim=det.npix, diameter=det.npix) 

sig_r1_on  = baldr.detection_chain(input_field_r1, dm, FPM, det, include_shotnoise=False)
sig_r2_on  = baldr.detection_chain(input_field_r2, dm, FPM, det, include_shotnoise=False)

sig_r1_off  = baldr.detection_chain(input_field_r1, dm, FPM_cal, det, include_shotnoise=False)
sig_r2_off  = baldr.detection_chain(input_field_r2, dm, FPM_cal, det, include_shotnoise=False)


plt.figure()
plt.semilogy( 1e6 * wvls, Nph_wvl_1,label='red spectrum' ); plt.semilogy( 1e6 * wvls, Nph_wvl_2, label='blue spectrum');
plt.xlabel('wavelength [um]')
plt.ylabel('flux [ph/s/m^2/nm]')


norm_radius = np.linspace(- 0.5, 0.5, N_act[0])    
plt.figure( figsize=(8,5) )
plt.plot(  norm_radius, sig_r1_on.signal[ N_act[0]//2, :] , color='green', label='mask on, red spectrum'); plt.plot( norm_radius, sig_r2_on.signal[ N_act[0]//2, :] , color='lime', label='mask on, blue spectrum' );
plt.plot( norm_radius, sig_r1_off.signal[ N_act[0]//2, :] , color='red', label='mask off, red spectrum'); plt.plot( norm_radius, sig_r2_off.signal[ N_act[0]//2, :] ,color='darkred', label='mask off, blue spectrum');
plt.legend(bbox_to_anchor=(1,1), fontsize=12)
plt.xlabel('normalized radius',fontsize=15)
plt.ylabel('intensity (ADU)',fontsize=15)
plt.title('perfect wavefront ')


# add piston on outer perimeter of DM
dm_basis = zernike.zernike_basis(nterms=20, npix=N_act[0])
dm_surf = 1e-2 * wvl0/4 * np.random.random(N_act) #np.nan_to_num(dm_basis[11])
#dm_surf = 0.5 * wvl0/4 * np.ones( N_act )
#dm_surf[dm_basis[0]>0]=0
dm.update_shape(dm_surf)

strehl =  round( np.exp( -np.var(2*np.pi *dm_surf/wvl0 * 2*np.cos(dm.angle) ) ), 2)

sig_r1_on  = baldr.detection_chain(input_field_r1, dm, FPM, det, include_shotnoise=False)
sig_r2_on  = baldr.detection_chain(input_field_r2, dm, FPM, det, include_shotnoise=False)

sig_r1_off  = baldr.detection_chain(input_field_r1, dm, FPM_cal, det, include_shotnoise=False)
sig_r2_off  = baldr.detection_chain(input_field_r2, dm, FPM_cal, det, include_shotnoise=False)



norm_radius = np.linspace(- 0.5, 0.5, N_act[0])    
plt.figure( figsize=(8,5) )
plt.plot(  norm_radius, sig_r1_on.signal[ N_act[0]//2, :] , color='green', label='mask on, red spectrum'); plt.plot( norm_radius, sig_r2_on.signal[ N_act[0]//2, :] , color='lime', label='mask on, blue spectrum' );
plt.plot( norm_radius, sig_r1_off.signal[ N_act[0]//2, :] , color='red', label='mask off, red spectrum'); plt.plot( norm_radius, sig_r2_off.signal[ N_act[0]//2, :] ,color='darkred', label='mask off, blue spectrum');
plt.legend(bbox_to_anchor=(1,1), fontsize=12)
plt.xlabel('normalized radius',fontsize=15)
plt.ylabel('intensity (ADU)',fontsize=15)
plt.title(f'aberrated wavefront (Strehl={strehl})')


i1,i2=0,5
Nph = np.sum(sig_r1_off.signal[ N_act[0]//2, :])
peak = 1/Nph * np.max(sig_r1_on.signal[ N_act[0]//2, :])
grad = 1/Nph * (sig_r1_on.signal[ N_act[0]//2, i2]-sig_r1_on.signal[ N_act[0]//2, i1]) / (norm_radius[i2]-norm_radius[i1])

#%% Plot peak amp and slope  vs wvl_0 for reference wavefront (no-aberration) with mask on, could also explore this vs design parameters (e.g. depth and diameter!) 

dim = 12*20
stellar_dict = { 'Hmag':2,'r0':0.1,'L0':25,'V':50,'airmass':1.3,'extinction': 0.18 }
tel_dict = { 'dim':dim,'D':1.8,'D_pix':12*20,'pup_geometry':'AT' }

FPM_dict = {'A':1, 'B':1, 'f_ratio':21, 'd_on':26.5e-6, 'd_off':26e-6,\
            'glass_on':'sio2', 'glass_off':'sio2','desired_phase_shift':90,\
                'rad_lam_o_D':1.2 ,'N_samples_across_phase_shift_region':10,\
                    'nx_size_focal_plane':dim}
det_dict={'DIT' : 1, 'ron' : 0.5, 'pw' : 20, 'QE' : 1, 'npix_det':12}

baldr_dict={'baldr_lag':0.5e-3,'baldr_lambda0':1.6e-6,'baldr_Ki':0.1, 'baldr_Kp':9}

locals().update(stellar_dict)
locals().update(tel_dict)
locals().update(FPM_dict)
locals().update(det_dict)
locals().update(baldr_dict)

ron = 0
wvls = 1e-6 * np.linspace(1.4,1.8,10)
wvl0 = np.mean(  wvls  )
dw = 0.4e-6 # bandwidth (i.e: 1.4-1.8um )

theta_at_wvl0= np.pi/2

n_on = baldr.nglass(1e6 * wvl0, glass=glass_on)[0]
n_off = baldr.nglass(1e6 * wvl0, glass=glass_off)[0]
n_air = baldr.nglass(1e6 * wvl0, glass='air')[0]

epsilon = 0.4 # mode amplitude
basis = zernike.zernike_basis(nterms=20, npix=D_pix)
pup = baldr.pick_pupil(pupil_geometry='AT', dim=D_pix, diameter=D_pix) 

dx = D/D_pix # grid spatial element (m/pix)
npix_det = int( D_pix//pw ) # number of pixels across detector 
pix_scale_det = dx * pw # m/pix
N_act = [12,12]
m = 4 # aberration index(Zernike)


# ----- DM
dm = baldr.DM(surface=np.zeros(N_act), gain=1, angle=0,surface_type = 'continuous') 

# ----- PHASE MASKS

# calculate on-axis depth for 90deg phase shift
d_on = (  wvl0 * theta_at_wvl0/(np.pi*2) + n_off * d_off -   n_air * d_off ) / (n_on-n_air)
    
N_loDs = np.linspace(1,2,5)

# ----- DETECTOR
det = baldr.detector(npix=npix_det, pix_scale = pix_scale_det , DIT= DIT, ron=ron, QE={w:QE for w in wvls})

pup_profile_dict={}
for N_loD in N_loDs:
    
    phase_shift_diameter = N_loD  * f_ratio * wvl0   ##  f_ratio * wvls[0] = lambda/D  given f_ratio
    
    # phase shifting phase mask 
    FPM = baldr.zernike_phase_mask(A,B,phase_shift_diameter,f_ratio,d_on,d_off, glass_on, glass_off)
    #  nonphase shifting phase mask 
    FPM_cal = baldr.zernike_phase_mask(A,B,phase_shift_diameter, f_ratio, d_off, d_off, glass_on, glass_off)


    peak=[]
    grad=[]
    central_wvls=[]
    for w0 in 1e-6*np.linspace(0.8, 2, 10):
        wvls = np.linspace(w0, w0+dw, 8)
        det.qe = {w:QE for w in wvls}
        Nph_wvl_1 = 1e4*np.ones(len(wvls)) #np.logspace(3,4,len(wvls))  #np.logspace(3,4,len(wvls))
        #Nph_wvl_2 = np.logspace(3,4,len(wvls))[::-1] #1e3*np.ones(len(wvls)) #np.logspace(3,4,len(wvls))[::-1]
        
        central_wvls.append( np.sum( Nph_wvl_1 * wvls ) / np.sum( Nph_wvl_1 ) )
        #wvl0_2 = np.sum( Nph_wvl_2 * wvls ) / np.sum( Nph_wvl_2 )
        
        
        phase_t = basis[m]  # aberration 
        phase_r = basis[0]  # reference 
        
        phase_t = np.array( [ np.nan_to_num(basis[m]) * (500e-9/w)**(6/5) for w in wvls] )
        phase_r = np.array( [ np.nan_to_num(basis[0]) * (500e-9/w)**(6/5) for w in wvls] )
        
        input_fluxes_1 = [ p * pup  for p in Nph_wvl_1] # ph_m2_s_nm
        #input_fluxes_2 = [ p * pup  for p in Nph_wvl_2] # ph_m2_s_nm
        
        input_field_r1 = baldr.field( phases = epsilon * ( phase_r) , fluxes = input_fluxes_1  , wvls=wvls )
        input_field_r1.define_pupil_grid(dx=dx, D_pix=D_pix)
        
        #input_field_r2 = baldr.field( phases = epsilon * ( phase_r) , fluxes = input_fluxes_2  , wvls=wvls )
        #input_field_r2.define_pupil_grid(dx=dx, D_pix=D_pix)
        
        #det = baldr.detector(npix=npix_det, pix_scale = pix_scale_det , DIT= DIT, ron=ron, QE={w:QE for w in wvls})
        #mask = baldr.pick_pupil(pupil_geometry='disk', dim=det.npix, diameter=det.npix) 
        
        sig_r1_on  = baldr.detection_chain(input_field_r1, dm, FPM, det, include_shotnoise=False)
        #sig_r2_on  = baldr.detection_chain(input_field_r2, dm, FPM, det, include_shotnoise=False)
        
        sig_r1_off  = baldr.detection_chain(input_field_r1, dm, FPM_cal, det, include_shotnoise=False)
        #sig_r2_off  = baldr.detection_chain(input_field_r2, dm, FPM_cal, det, include_shotnoise=False)
        
        i1,i2=0,5
        norm =np.max(sig_r1_off.signal) #np.sum(sig_r1_off.signal)
        peak.append( 1/norm * np.max(sig_r1_on.signal) )
        grad.append( 1/norm * (sig_r1_on.signal[ N_act[0]//2, i2]-sig_r1_on.signal[ N_act[0]//2, i1]) / (norm_radius[i2]-norm_radius[i1]) )
        
    pup_profile_dict[N_loD] = {'peak':peak, 'grad':grad}
        

plt.figure(figsize=(8,5))
for N_loD in pup_profile_dict:
    plt.plot(np.array(central_wvls)/wvl0, pup_profile_dict[N_loD]['peak'], label=f'daim={round(N_loD,2)}'+r'$\lambda_0/D$')
plt.legend(bbox_to_anchor=(1,1), fontsize=12)
plt.xlabel(r'$\lambda_0^{real}/\lambda_0^{design}$',fontsize=15)
plt.ylabel('peak normalized intensity',fontsize=15)
plt.gca().tick_params(labelsize=15)
#plt.title(f'aberrated wavefront (Strehl={strehl})')

plt.figure(figsize=(8,5))
for N_loD in pup_profile_dict:
    plt.plot(np.array(central_wvls)/wvl0, pup_profile_dict[N_loD]['grad'], label=f'daim={round(N_loD,2)}'+r'$\lambda_0/D$')
plt.legend(bbox_to_anchor=(1,1), fontsize=12)
plt.xlabel(r'$\lambda_0^{real}/\lambda_0^{design}$',fontsize=15)
plt.ylabel('edge intensity gradient',fontsize=15)
plt.gca().tick_params(labelsize=15)


#%%

norm_radius = np.linspace(- 0.5, 0.5, N_act[0])    
plt.figure( figsize=(8,5) )
plt.plot(  norm_radius, sig_r1_on.signal[ N_act[0]//2, :] , color='green', label='mask on, red spectrum'); plt.plot( norm_radius, sig_r2_on.signal[ N_act[0]//2, :] , color='lime', label='mask on, blue spectrum' );
plt.plot( norm_radius, sig_r1_off.signal[ N_act[0]//2, :] , color='red', label='mask off, red spectrum'); plt.plot( norm_radius, sig_r2_off.signal[ N_act[0]//2, :] ,color='darkred', label='mask off, blue spectrum');
plt.legend(bbox_to_anchor=(1,1), fontsize=12)
plt.xlabel('normalized radius',fontsize=15)
plt.ylabel('intensity (ADU)',fontsize=15)
plt.title('perfect wavefront ')


# add piston on outer perimeter of DM
dm_basis = zernike.zernike_basis(nterms=20, npix=N_act[0])
dm_surf = 1e-2 * wvl0/4 * np.random.random(N_act) #np.nan_to_num(dm_basis[11])
#dm_surf = 0.5 * wvl0/4 * np.ones( N_act )
#dm_surf[dm_basis[0]>0]=0
dm.update_shape(dm_surf)

strehl =  round( np.exp( -np.var(2*np.pi *dm_surf/wvl0 * 2*np.cos(dm.angle) ) ), 2)

sig_r1_on  = baldr.detection_chain(input_field_r1, dm, FPM, det, include_shotnoise=False)
sig_r2_on  = baldr.detection_chain(input_field_r2, dm, FPM, det, include_shotnoise=False)

sig_r1_off  = baldr.detection_chain(input_field_r1, dm, FPM_cal, det, include_shotnoise=False)
sig_r2_off  = baldr.detection_chain(input_field_r2, dm, FPM_cal, det, include_shotnoise=False)



norm_radius = np.linspace(- 0.5, 0.5, N_act[0])    
plt.figure( figsize=(8,5) )
plt.plot(  norm_radius, sig_r1_on.signal[ N_act[0]//2, :] , color='green', label='mask on, red spectrum'); plt.plot( norm_radius, sig_r2_on.signal[ N_act[0]//2, :] , color='lime', label='mask on, blue spectrum' );
plt.plot( norm_radius, sig_r1_off.signal[ N_act[0]//2, :] , color='red', label='mask off, red spectrum'); plt.plot( norm_radius, sig_r2_off.signal[ N_act[0]//2, :] ,color='darkred', label='mask off, blue spectrum');
plt.legend(bbox_to_anchor=(1,1), fontsize=12)
plt.xlabel('normalized radius',fontsize=15)
plt.ylabel('intensity (ADU)',fontsize=15)
plt.title(f'aberrated wavefront (Strehl={strehl})')


i1,i2=0,5
Nph = np.sum(sig_r1_off.signal[ N_act[0]//2, :])
peak = 1/Nph * np.max(sig_r1_on.signal[ N_act[0]//2, :])
grad = 1/Nph * (sig_r1_on.signal[ N_act[0]//2, i2]-sig_r1_on.signal[ N_act[0]//2, i1]) / (norm_radius[i2]-norm_radius[i1])

