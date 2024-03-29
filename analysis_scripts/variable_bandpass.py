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

#%% Plot peak amp and slope  vs wvl_0 for reference wavefront (no-aberration) with mask on VS dot diameter ,
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



#%% #%% Plot peak amp and slope  vs wvl_0 for reference wavefront (no-aberration) with mask on VS phase shift 

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

#theta_at_wvl0= np.pi/2
thetas = np.linspace(np.pi/6, 2*np.pi/3, 5)
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


N_loDs = 1.06
phase_shift_diameter = N_loD  * f_ratio * wvl0   ##  f_ratio * wvls[0] = lambda/D  given f_ratio
#  nonphase shifting phase mask 
FPM_cal = baldr.zernike_phase_mask(A,B,phase_shift_diameter, f_ratio, d_off, d_off, glass_on, glass_off)



# ----- DETECTOR
det = baldr.detector(npix=npix_det, pix_scale = pix_scale_det , DIT= DIT, ron=ron, QE={w:QE for w in wvls})

pup_profile_theta_dict={}
for theta_at_wvl0 in thetas :
    
    # calculate on-axis depth for 90deg phase shift
    d_on = (  wvl0 * theta_at_wvl0/(np.pi*2) + n_off * d_off -   n_air * d_off ) / (n_on-n_air)
    
    # phase shifting phase mask 
    FPM = baldr.zernike_phase_mask(A,B,phase_shift_diameter,f_ratio,d_on,d_off, glass_on, glass_off)
    

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
        
    pup_profile_theta_dict[theta_at_wvl0] = {'peak':peak, 'grad':grad}
        

plt.figure(figsize=(8,5))
for theta in pup_profile_theta_dict:
    plt.plot(np.array(central_wvls)/wvl0, pup_profile_theta_dict[theta]['peak'], label=f'phase={round(np.rad2deg(theta),2)} [deg]')
plt.legend(bbox_to_anchor=(1,1), fontsize=12)
plt.xlabel(r'$\lambda_0^{real}/\lambda_0^{design}$',fontsize=15)
plt.ylabel('peak normalized intensity',fontsize=15)
plt.gca().tick_params(labelsize=15)
#plt.title(f'aberrated wavefront (Strehl={strehl})')

plt.figure(figsize=(8,5))
for theta in pup_profile_theta_dict:
    plt.plot(np.array(central_wvls)/wvl0, pup_profile_theta_dict[theta]['grad'], label=f'phase={round(np.rad2deg(theta),2)} [deg]')
plt.legend(bbox_to_anchor=(1,1), fontsize=12)
plt.xlabel(r'$\lambda_0^{real}/\lambda_0^{design}$',fontsize=15)
plt.ylabel('edge intensity gradient',fontsize=15)
plt.gca().tick_params(labelsize=15)


# %% Putting plots togeether (I need to run cells to get pupil profile , and pupil profile features vs central wavelength ratio)


# ======== spectral_type_vs_reference_pupil_intensity.png
# Could also use temperature 
fig,ax = plt.subplots(1,2,sharex=False, sharey=False,figsize=(15,7))
plt.subplots_adjust(wspace=0.6)
wvl0_1=np.sum( Nph_wvl_1 * wvls ) /np.sum( Nph_wvl_1 )
wvl0_2=np.sum( Nph_wvl_2 * wvls ) /np.sum( Nph_wvl_2 )
ax[0].semilogy( 1e6 * wvls, Nph_wvl_1,label='red spectrum' , color='r')
ax[0].fill_between( 1e6 * wvls, np.min(Nph_wvl_1)*np.ones(len(wvls)), Nph_wvl_1, alpha=0.4, color='r')
ax[0].semilogy( 1e6 * wvls, Nph_wvl_2, label='blue spectrum',color='b')
ax[0].fill_between( 1e6 * wvls, np.min(Nph_wvl_2)*np.ones(len(wvls)), Nph_wvl_2, alpha=0.4, color='b')
ax[0].set_xlabel('wavelength [um]',fontsize=25)
ax[0].set_ylabel('flux '+r'[ph/s/m$^2$/nm]',fontsize=25)
ax[0].axvline(1e6*wvl0_1,color='r',linestyle=':', label=r'$\lambda_0$ (red)')
ax[0].axvline(1e6*wvl0_2,color='b',linestyle=':', label=r'$\lambda_0$ (blue)')
ax[0].tick_params(labelsize=20)
ax[0].legend(fontsize=15)


norm_radius = np.linspace(- 0.5, 0.5, N_act[0])  
norm = np.max(sig_r1_off.signal)  
ax[1].plot(  norm_radius, 1/np.max(sig_r1_off.signal)  * sig_r1_on.signal[ N_act[0]//2, :] , color='red', label='phase mask in, red spectrum')
ax[1].plot( norm_radius, 1/np.max(sig_r1_off.signal)  * sig_r2_on.signal[ N_act[0]//2, :] , color='blue', label='phase mask in, blue spectrum' )
ax[1].plot( norm_radius, 1/np.max(sig_r1_off.signal)  * sig_r1_off.signal[ N_act[0]//2, :] , color='darkred', label='phase mask out, red spectrum')
ax[1].plot( norm_radius, 1/np.max(sig_r1_off.signal)  * sig_r2_off.signal[  N_act[0]//2, :] ,color='darkblue', label='phase mask out, blue spectrum')
ax[1].legend( fontsize=15 ) #bbox_to_anchor=(1,1),
ax[1].set_xlabel('normalized pupil radius',fontsize=25)
ax[1].set_ylabel('normalized intensity (ADU)',fontsize=25)
#ax[1].text(-0.1, 1/np.max(sig_r1_off.signal) * np.max(sig_r1_on.signal), 'peak',fontsize=20)
ax[1].tick_params(labelsize=20)
#plt.title(f'aberrated wavefront (Strehl={strehl})')

plt.tight_layout()
plt.savefig('/Users/bcourtne/Documents/ANU_PHD2/baldr/figures/spectral_type_vs_reference_pupil_intensity.png',dpi=300)



# ======== pupil_features_vs_phase_shift.png
fig,ax = plt.subplots(1,2,sharex=False, sharey=False,figsize=(15,7))
plt.subplots_adjust(wspace=0.6)

for theta in pup_profile_theta_dict:
    ax[0].plot(np.array(central_wvls)/wvl0, pup_profile_theta_dict[theta]['peak'], label=r'$\Delta \phi(\lambda_0)$'+f'={round(np.rad2deg(theta),2)} [deg]')
ax[0].legend( fontsize=15) #bbox_to_anchor=(1,1),
ax[0].set_xlabel(r'$\lambda_0^{real}/\lambda_0^{expected}$',fontsize=25)
ax[0].set_ylabel('peak normalized \npupil intensity',fontsize=25)
ax[0].tick_params(labelsize=20)
#plt.title(f'aberrated wavefront (Strehl={strehl})')

for theta in pup_profile_theta_dict:
    ax[1].plot(np.array(central_wvls)/wvl0, pup_profile_theta_dict[theta]['grad'], label=r'$\Delta \phi(\lambda_0)$'+f'={round(np.rad2deg(theta),2)} [deg]')
ax[1].legend(fontsize=15) #bbox_to_anchor=(1,1), 
ax[1].set_xlabel(r'$\lambda_0^{real}/\lambda_0^{expected}$',fontsize=25)
ax[1].set_ylabel('normalized pupil (edge) \nintensity gradient',fontsize=25)
ax[1].tick_params(labelsize=20)

plt.tight_layout()
plt.savefig('/Users/bcourtne/Documents/ANU_PHD2/baldr/figures/pupil_features_vs_phase_shift.png',dpi=300)


# ======== pupil_features_vs_dot_diam.png
fig,ax = plt.subplots(1,2,sharex=False, sharey=False,figsize=(15,7))
plt.subplots_adjust(wspace=0.6)

for N_loD in pup_profile_dict:
    ax[0].plot(np.array(central_wvls)/wvl0, pup_profile_dict[N_loD]['peak'], label=f'daim={round(N_loD,2)}'+r'$\lambda_0/D$')
ax[0].legend( fontsize=15 )
ax[0].set_xlabel(r'$\lambda_0^{real}/\lambda_0^{expected}}$',fontsize=25)
ax[0].set_ylabel('peak normalized \npupil intensity',fontsize=25)
ax[0].tick_params(labelsize=20)
#plt.title(f'aberrated wavefront (Strehl={strehl})')

for N_loD in pup_profile_dict:
    ax[1].plot(np.array(central_wvls)/wvl0, pup_profile_dict[N_loD]['grad'], label=f'daim={round(N_loD,2)}'+r'$\lambda_0/D$')
ax[1].legend( fontsize=15 )
ax[1].set_xlabel(r'$\lambda_0^{real}/\lambda_0^{expected}}$',fontsize=25)
ax[1].set_ylabel('normalized pupil (edge) \nintensity gradient',fontsize=25)
ax[1].tick_params(labelsize=20)

plt.tight_layout()
plt.savefig('/Users/bcourtne/Documents/ANU_PHD2/baldr/figures/pupil_features_vs_dot_diam.png',dpi=300)



#%% 
"""
Dear Ben,
 
A key point not completely understood is in the right hand side of “spectral_type_vs_reference_pupil_intensity” plot.
 
If we have a calibration zero point, i.e. pupil target intensity based on a blue spectrum (J-K colour of 0) and then are locking the loop on a red spectrum (J-K colour of 2), then what is the wavefront error?
 
As mentioned, if significant we’ll have to use Heimdallr and Frantz’s asymmetric pupil wavefront sensor as a means to measure remove this effect and add this as a core software mode.
 
Mike.
"""

wvl0_K = 2.2e-6
wvl0_J = 1.2e-6
tmp_wvls = np.linspace(wvl0_J,wvl0_K ,100)

mag = 3
J_flux = baldr.star2photons('J', mag, airmass=1,ph_m2_s_nm = True) 
K_flux_blue = baldr.star2photons('K', mag, airmass=1, ph_m2_s_nm = True) 
K_flux_red = baldr.star2photons('K', mag-2, airmass=1, ph_m2_s_nm = True) 

red_spec = np.linspace(J_flux, K_flux_red, 100)
blue_spec = np.linspace(J_flux, K_flux_blue, 100)

from scipy.interpolate import interp1d 
red_interp_fn = interp1d(tmp_wvls, red_spec)
blue_interp_fn = interp1d(tmp_wvls, blue_spec)

filt = (tmp_wvls >= 1.4e-6) & (tmp_wvls <= 1.6e-6)

red_wvl0 = np.sum(red_spec[filt] * tmp_wvls[filt]) / np.sum(red_spec[filt])
blue_wvl0 = np.sum(blue_spec[filt] * tmp_wvls[filt]) / np.sum(blue_spec[filt])

print(f'red/blue spectrum wvl0= {red_wvl0/blue_wvl0}')


### 
# ============= OK NOW LETS DO THE SAME EXERCISE 

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
Nph_wvl_1 = red_interp_fn(wvls)  # np.logspace(3,4,len(wvls)) #1e3*np.ones(len(wvls)) #np.logspace(3,4,len(wvls))
Nph_wvl_2 = blue_interp_fn(wvls)   #1e3*np.ones(len(wvls)) #np.logspace(3,4,len(wvls))[::-1]
# normalize for same number of total photons 
Nph_wvl_2 *= np.sum(Nph_wvl_1)/np.sum(Nph_wvl_2)

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


# ======== spectral_type_vs_reference_pupil_intensity.png
# Could also use temperature 
fig,ax = plt.subplots(1,2,sharex=False, sharey=False,figsize=(15,7))
plt.subplots_adjust(wspace=0.6)
wvl0_1=np.sum( Nph_wvl_1 * wvls ) /np.sum( Nph_wvl_1 )
wvl0_2=np.sum( Nph_wvl_2 * wvls ) /np.sum( Nph_wvl_2 )
ax[0].plot( 1e6 * wvls, Nph_wvl_1,label='red spectrum (J-K=2)' , color='r')
#ax[0].fill_between( 1e6 * wvls, np.min(Nph_wvl_1)*np.ones(len(wvls)), Nph_wvl_1, alpha=0.4, color='r')
ax[0].plot( 1e6 * wvls, Nph_wvl_2, label='blue spectrum (J-K=0)',color='b')
#ax[0].fill_between( 1e6 * wvls, np.min(Nph_wvl_2)*np.ones(len(wvls)), Nph_wvl_2, alpha=0.4, color='b')
ax[0].set_xlabel('wavelength [um]',fontsize=25)
ax[0].set_ylabel('flux '+r'[ph/s/m$^2$/nm]',fontsize=25)
ax[0].axvline(1e6*wvl0_1,color='r',linestyle=':', label=r'$\lambda_0$ (red)')
ax[0].axvline(1e6*wvl0_2,color='b',linestyle=':', label=r'$\lambda_0$ (blue)')
ax[0].tick_params(labelsize=20)
ax[0].legend(fontsize=15)


norm_radius = np.linspace(- 0.5, 0.5, N_act[0])  
norm = np.max(sig_r1_off.signal)  
ax[1].plot(  norm_radius, 1/np.max(sig_r1_off.signal)  * sig_r1_on.signal[ N_act[0]//2, :] , color='red', label='phase mask in, red spectrum')
ax[1].plot( norm_radius, 1/np.max(sig_r1_off.signal)  * sig_r2_on.signal[ N_act[0]//2, :] , color='blue', label='phase mask in, blue spectrum' )
ax[1].plot( norm_radius, 1/np.max(sig_r1_off.signal)  * sig_r1_off.signal[ N_act[0]//2, :] , color='darkred', label='phase mask out, red spectrum')
ax[1].plot( norm_radius, 1/np.max(sig_r1_off.signal)  * sig_r2_off.signal[  N_act[0]//2, :] ,color='darkblue', label='phase mask out, blue spectrum')
ax[1].legend( fontsize=15 ) #bbox_to_anchor=(1,1),
ax[1].set_xlabel('normalized pupil radius',fontsize=25)
ax[1].set_ylabel('normalized intensity (ADU)',fontsize=25)
#ax[1].text(-0.1, 1/np.max(sig_r1_off.signal) * np.max(sig_r1_on.signal), 'peak',fontsize=20)
ax[1].tick_params(labelsize=20)
#plt.title(f'aberrated wavefront (Strehl={strehl})')

plt.tight_layout()