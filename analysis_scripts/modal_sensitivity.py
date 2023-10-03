#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:34:26 2023

@author: bcourtne

sensitivity again 
definition in Chambouleyron 2021

dI(\phi) = (I(\epsilon \phi) - I(-\epsilon \phi)) / (2\epsilon)

sensitivity 

S(\phi) = ||dI(\phi)||_l2 / ||\phi||

"""
import pyzelda.utils.mft as mft
import numpy as np
import matplotlib.pyplot as plt
import pyzelda.utils.zernike as zernike
import scipy
from scipy.stats import poisson
import os 
os.chdir('/Users/bcourtne/Documents/ANU_PHD2/baldr')
from functions import baldr_functions_2 as baldr
     
def reco(SIG, SIG_CAL, FPM, DET, b):
    
    P = SIG_CAL.signal / FPM.A
    B = DET.detect_field(b).signal
    beta = np.mean([b.phase[w] for w in b.phase],axis=0)
    beta = interpolate_field_onto_det( beta, det )
    
    m = np.mean( [FPM.get_filter_design_parameter(w) for w in wvls] )
    M = abs(m)
    mu = np.angle(m)

    phi_reco = np.arccos( (SIG.signal**2 / FPM.A - P**2 - (B*M)**2) / (2*B*P*M) ) + mu + beta    
    
    #phi_reco = np.arccos( (SIG.signal / FPM.A - P - (B*M**2)) / (2*np.sqrt(B*P)*M) ) + mu + beta    
    
    return(phi_reco)

def reco_linear(SIG, SIG_CAL, FPM, DET, b):
    
    P = SIG_CAL.signal / FPM.A
    B = DET.detect_field(b).signal
    beta = np.mean([b.phase[w] for w in b.phase],axis=0)
    beta = interpolate_field_onto_det( beta, det )
    
    m = np.mean( [FPM.get_filter_design_parameter(w) for w in wvls] )
    M = abs(m)
    mu = np.angle(m)

    #phi_reco = np.arccos( (SIG.signal**2 / FPM.A - P**2 - (B*M)**2) / (2*B*P*M) ) + mu + beta    
    
    phi_reco = np.pi/2 - ( (SIG.signal**2 / FPM.A - P**2 - (B*M)**2) / (2*B*P*M) ) + mu #+ beta    
    
    return(phi_reco)
        
 
def get_out_field(Nph, phase):
    Psi_A = Nph**0.5 * np.exp( 1j * np.nan_to_num( phase ) )
    
    x_focal_plane = np.linspace(-nx_pix  * dx / 2, nx_pix  * dx / 2, nx_pix)
    nx_size_focal_plane = len(x_focal_plane)
    
    m1 = (x_focal_plane[-1] - x_focal_plane[0] ) / (wvl * f_ratio) 
                
    Psi_B = mft.mft(Psi_A, nx_pix, nx_size_focal_plane , m1, cpix=False)
    
    Psi_C = mft.imft( H * Psi_B , nx_size_focal_plane, nx_pix, m1, cpix=False)
    return(Psi_C)
    
    
def interpolate_field_onto_det( my_field, det):
    """
    interpolate any 2d matrix (my_field) onto a given detector grid

    Parameters
    ----------
    my_field : TYPE 2d array 
        DESCRIPTION.
    det : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    X,Y = np.meshgrid(np.linspace( -1,1, my_field.shape[0] ),np.linspace( -1,1, my_field.shape[1] ))
    coords = np.vstack([X.ravel(), Y.ravel()]).T
    nearest_interp_fn = scipy.interpolate.LinearNDInterpolator(coords, my_field.reshape(-1))
    
    X,Y = np.meshgrid(np.linspace( -1,1, det.npix ),np.linspace( -1,1, det.npix ))
    ncoords = np.vstack([X.ravel(), Y.ravel()]).T
    
    b_interp = nearest_interp_fn(ncoords).reshape(det.npix,det.npix)

    return(b_interp)


#%% sensitivity vs dot diameter monochromatic
wvl = 0.6e-6
A,B=1,1
f_ratio=21
theta=np.pi/2

nx_pix = 2**10
#D = 1.8 #m
dx = f_ratio*wvl/20

epsilon=0.4

basis = zernike.zernike_basis(nterms=15,npix=nx_pix)

Nph = 100

N_loDs = np.linspace(0.,5,30)

S_dic = {}
for m in [2,3,5,7]:
    print(f'looking at mode {m}')
    S=[]
    for N_loD  in N_loDs :
        
        phase_shift_diameter = N_loD * f_ratio * wvl 
        
        phase_shift_region = baldr.pick_pupil('disk', dim=nx_pix, diameter=round(phase_shift_diameter/dx) )
        
        H =  A*(1 + (B/A * np.exp(1j * theta) - 1) * phase_shift_region  )
    
        phase_t = basis[m] 
        phase_r = basis[0]
    
        # NOW INCLUDE TURBULENT AND REFERENCE PHASE LIKE
        
        fC1_t = get_out_field(Nph, epsilon * (phase_r+phase_t) )
        fC2_t = get_out_field(Nph, -epsilon *(phase_r+phase_t) )
        
        fC1_r = get_out_field(Nph, epsilon * (phase_r) )
        fC2_r = get_out_field(Nph, -epsilon *(phase_r) )
        
        IC1 = poisson.rvs( abs(fC1_t)**2 ) - poisson.rvs( abs(fC1_r)**2 )
        IC2 = poisson.rvs( abs(fC2_t)**2 ) -  poisson.rvs( abs(fC2_r)**2 )
        # get intesnity with shot noise 
        dI = 1/Nph * ( IC1 - IC2 )/(2*epsilon)
        
        S.append(np.sqrt( np.nansum(basis[0] * dI**2) / np.nansum(phase_t**2) ))
        #print( 'sensitivity for mode = ',S[-1])
        
    S_dic[zernike.zern_name(m+1)] = S

plt.figure(figsize=(8,5))
for s in S_dic:
    plt.plot(N_loDs ,S_dic[s], label=s )
plt.legend(fontsize=15)
plt.gca().tick_params(labelsize=20)
plt.xlabel('dot diameter [$\lambda$/D]',fontsize=20)
plt.ylabel('Sensitivity',fontsize=20)
plt.axvline(1.06,color='k',linestyle=':')
plt.axvline(2,color='k',linestyle=':')
plt.tight_layout()
plt.savefig(f'/Users/bcourtne/Documents/ANU_PHD2/baldr/figures/sensitivity_vs_dot_diam_monochromatic_theta-{int(np.rad2deg(theta))}.png',dpi=300)

#%% sensitivity vs phase shift monochromatic
wvl = 0.6e-6
A,B=1,1
f_ratio=21
#theta=np.pi/2

nx_pix = 2**10
#D = 1.8 #m
dx = f_ratio*wvl/20

epsilon=0.4

basis = zernike.zernike_basis(nterms=15,npix=nx_pix)

Nph = 100

#N_loDs = np.linspace(0.,5,30)
thetas = np.linspace(np.pi/100, np.pi, 20)
N_loD = 1.06
phase_shift_diameter = N_loD * f_ratio * wvl 
phase_shift_region = baldr.pick_pupil('disk', dim=nx_pix, diameter=round(phase_shift_diameter/dx) )
        
S_dic = {}
for m in [2,3,5,7]:
    print(f'looking at mode {m}')
    S=[]
    for theta  in thetas  :

        H =  A*(1 + (B/A * np.exp(1j * theta) - 1) * phase_shift_region  )
    
        phase_t = basis[m] 
        phase_r = basis[0]
    
        # NOW INCLUDE TURBULENT AND REFERENCE PHASE LIKE
        
        fC1_t = get_out_field(Nph, epsilon * (phase_r+phase_t) )
        fC2_t = get_out_field(Nph, -epsilon *(phase_r+phase_t) )
        
        fC1_r = get_out_field(Nph, epsilon * (phase_r) )
        fC2_r = get_out_field(Nph, -epsilon *(phase_r) )
        
        IC1 = poisson.rvs( abs(fC1_t)**2 ) - poisson.rvs( abs(fC1_r)**2 )
        IC2 = poisson.rvs( abs(fC2_t)**2 ) -  poisson.rvs( abs(fC2_r)**2 )
        # get intesnity with shot noise 
        dI = 1/Nph * ( IC1 - IC2 )/(2*epsilon)
        
        S.append(np.sqrt( np.nansum(basis[0] * dI**2) / np.nansum(phase_t**2) ))
        #print( 'sensitivity for mode = ',S[-1])
        
    S_dic[zernike.zern_name(m+1)] = S

plt.figure(figsize=(8,5))
for s in S_dic:
    plt.plot(np.rad2deg( thetas ),S_dic[s], label=s )
plt.legend(fontsize=15)
plt.gca().tick_params(labelsize=20)
plt.xlabel('phase shift [deg]',fontsize=20)
plt.ylabel('Sensitivity',fontsize=20)
plt.axvline(90,color='k',linestyle=':')
#plt.axvline(2,color='k',linestyle=':')
plt.tight_layout()
plt.savefig(f'/Users/bcourtne/Documents/ANU_PHD2/baldr/figures/sensitivity_vs_dot_depth_monochromatic_diam-{N_loD}.png',dpi=300)


#%% sensitivity vs dot diameter chromatic
wvl0 = 1.4e-6
wvls = wvl0 + 1e-6 * np.linspace(-0.3,0.3,4)
A,B=1,1
f_ratio=21

#desired_phase_shift = np.pi/2
#d_on = 26.5e-6
d_off = 26e-6

glass_on = 'sio2'
glass_off = 'sio2'

n_on = baldr.nglass(1e6 * wvl, glass=glass_on)[0]
n_off = baldr.nglass(1e6 * wvl, glass=glass_off)[0]
n_air = baldr.nglass(1e6 * wvl, glass='air')[0]

theta_at_wvl0  = np.pi/2
# calculate d_on such that we get theta_at_wvl0 phase shift at wvl0
d_on = (  wvl0 * theta_at_wvl0/(np.pi*2) + n_off * d_off -   n_air * d_off ) / (n_on-n_air)
# to check 2*np.pi/wvl0 * (d_on * n_on - (d_off * n_off  + (d_on-d_off) * n_air) ) 

nx_pix = 2**10
#D = 1.8 #m
dx = f_ratio*wvl/20

epsilon_0=0.4

basis = zernike.zernike_basis(nterms=15,npix=nx_pix)

Nph = 10000

bandwidths= np.logspace(-1, 0, 5)
dlambda = bandwidths[0]/2

N_loDs = np.linspace(1,2,3)



# calculate d_on,d_off, at wvl0 
FPMs = {}
for N_loD in N_loDs : 
        
    phase_shift_diameter = N_loD * f_ratio * wvl0
    
    FPM = baldr.zernike_phase_mask(A,B,phase_shift_diameter,f_ratio,d_on,d_off,glass_on,glass_off)
    
    FPMs[N_loD] = FPM
        
#check plt.plot(wvls, [FPM.phase_mask_phase_shift(w) for w in wvls])

S_dic = {}
for m in [3]: #[2,3,5,7]:
    S_dic[zernike.zern_name(m+1)] = {}
    print(f'looking at mode {m}')
    for bw in bandwidths:
        print(f'   looking at bw {bw}')
        wvls = wvl0 + 1e-6 * np.arange(-bw/2,bw/2,dlambda)
        
        
        S=[]
        Nph_lambda = Nph / len(wvls)
        
        for N_loD  in N_loDs:
            
            FPM = FPMs[N_loD]
            
            dI_lambda=[]
            
            phase_shift_diameter = N_loD * f_ratio * wvl0  # phase mask diameter defined at wvl0
                
            phase_shift_region = baldr.pick_pupil('disk', dim=nx_pix, diameter=round(FPM.phase_shift_diameter/dx) )
                
            for wvl in wvls :
                
                epsilon = epsilon_0 * (wvl/wvl0)**(-1)

                theta = np.deg2rad( FPM.phase_mask_phase_shift(wvl) )
                
                H =  A*(1 + (B/A * np.exp(1j * theta) - 1) * phase_shift_region  )
            
                phase_t = basis[m] 
                phase_r = basis[0]
            
                # NOW INCLUDE TURBULENT AND REFERENCE PHASE LIKE
                
                fC1_t = get_out_field(Nph_lambda, epsilon * (phase_r+phase_t) )
                fC2_t = get_out_field(Nph_lambda, -epsilon *(phase_r+phase_t) )
                
                fC1_r = get_out_field(Nph_lambda, epsilon * (phase_r) )
                fC2_r = get_out_field(Nph_lambda, -epsilon *(phase_r) )
                
                
                IC1 = poisson.rvs( abs(fC1_t)**2 ) - poisson.rvs( abs(fC1_r)**2 )
                IC2 = poisson.rvs( abs(fC2_t)**2 ) -  poisson.rvs( abs(fC2_r)**2 )
                # get intesnity with shot noise 
                dI_lambda.append(  ( IC1 - IC2 )/(2*epsilon) )
                
                
                #S_lambda.append(np.sqrt( np.nansum(basis[0] * dI**2) / np.nansum(phase_t**2) ))
            #print( 'sensitivity for mode = ',S[-1])
            
            dI = 1/Nph * np.sum( dI_lambda , axis=0)  #* np.median( np.diff(wvls) )# np.trapz(dI_lambda , wvls)
            S.append( np.sqrt( np.nansum(basis[0] * dI**2) / np.nansum(phase_t**2) ) )
        
        S_dic[zernike.zern_name(m+1)][bw] = S
    

plt.figure(figsize=(8,5))
for s in S_dic:
    plt.plot(N_loDs ,S_dic[s][bandwidths[0]], label=s )
plt.legend(fontsize=15)
plt.gca().tick_params(labelsize=20)
plt.xlabel('dot diameter [$\lambda$/D]',fontsize=20)
plt.ylabel('Sensitivity',fontsize=20)
plt.axvline(1.06,color='k',linestyle=':')
plt.axvline(2,color='k',linestyle=':')
plt.tight_layout()
#plt.savefig(f'/Users/bcourtne/Documents/ANU_PHD2/baldr/figures/sensitivity_vs_dot_diam_chromatic_theta-{int(np.rad2deg(theta))}.png',dpi=300)



m = list(S_dic.keys())[0]
plt.figure(figsize=(8,5))
for i, N_loD in enumerate(N_loDs):
    plt.plot( bandwidths, [S_dic[m][bw][i] for bw in bandwidths] ,label=f'dot diameter = {N_loD}'+r'$\lambda/D$')
plt.legend(fontsize=15)
plt.gca().tick_params(labelsize=20)
plt.xlabel(r'Spectral Bandwidth [$\mu m$]',fontsize=20)
plt.ylabel('Modal Sensitivity',fontsize=20)
plt.title(f'mode = {m}\n'+r'$\theta(\lambda_0)$=90deg',fontsize=20)
plt.savefig(f'/Users/bcourtne/Documents/ANU_PHD2/baldr/figures/sensitivity_vs_bandwidth_vs_dot_diam_chromatic_theta-{int(np.rad2deg(theta_at_wvl0 ))}.png',dpi=300)


#%% sensitivity vs phase shift chromatic
wvl0 = 1.4e-6
wvls = wvl0 + 1e-6 * np.linspace(-0.3,0.3,4)
A,B=1,1
f_ratio=21

#desired_phase_shift = np.pi/2
#d_on = 26.5e-6
d_off = 26e-6

glass_on = 'sio2'
glass_off = 'sio2'

n_on = baldr.nglass(1e6 * wvl, glass=glass_on)[0]
n_off = baldr.nglass(1e6 * wvl, glass=glass_off)[0]
n_air = baldr.nglass(1e6 * wvl, glass='air')[0]

#theta_at_wvl0  = np.pi/2
theta_at_wvl0s = np.linspace(np.pi/10, np.pi/1.2, 10)
# calculate d_on such that we get theta_at_wvl0 phase shift at wvl0
d_on = (  wvl0 * theta_at_wvl0/(np.pi*2) + n_off * d_off -   n_air * d_off ) / (n_on-n_air)
# to check 2*np.pi/wvl0 * (d_on * n_on - (d_off * n_off  + (d_on-d_off) * n_air) ) 

nx_pix = 2**10
#D = 1.8 #m
dx = f_ratio*wvl/20

epsilon_0=0.4

basis = zernike.zernike_basis(nterms=15,npix=nx_pix)

Nph = 10000

bandwidths= np.logspace(-1, 0, 5)
dlambda = bandwidths[0]/2

#N_loDs = np.linspace(1,2,3)
N_loD = 1.06


# calculate d_on,d_off, at wvl0 

#for N_loD in N_loDs : 
        

phase_shift_diameter = N_loD * f_ratio * wvl0  # phase mask diameter defined at wvl0                
            
FPM = baldr.zernike_phase_mask(A,B,phase_shift_diameter,f_ratio,d_on,d_off,glass_on,glass_off)
phase_shift_region = baldr.pick_pupil('disk', dim=nx_pix, diameter=round(FPM.phase_shift_diameter/dx) )
        
#check plt.plot(wvls, [FPM.phase_mask_phase_shift(w) for w in wvls])

S_dic = {}
for m in [3]: #[2,3,5,7]:
    S_dic[zernike.zern_name(m+1)] = {}
    print(f'looking at mode {m}')
    for bw in bandwidths:
        print(f'   looking at bw {bw}')
        wvls = wvl0 + 1e-6 * np.arange(-bw/2,bw/2,dlambda)
        
        
        S=[]
        Nph_lambda = Nph / len(wvls)
        
        for theta_at_wvl0  in theta_at_wvl0s:
            
            d_on = (  wvl0 * theta_at_wvl0/(np.pi*2) + n_off * d_off -   n_air * d_off ) / (n_on-n_air)
            
            FPM.d_on = d_on
            
            dI_lambda=[]
            
            for wvl in wvls :
                
                epsilon = epsilon_0 * (wvl/wvl0)**(-1)

                theta = np.deg2rad( FPM.phase_mask_phase_shift(wvl) )
                
                H =  A*(1 + (B/A * np.exp(1j * theta) - 1) * phase_shift_region  )
            
                phase_t = basis[m] 
                phase_r = basis[0]
            
                # NOW INCLUDE TURBULENT AND REFERENCE PHASE LIKE
                
                fC1_t = get_out_field(Nph_lambda, epsilon * (phase_r+phase_t) )
                fC2_t = get_out_field(Nph_lambda, -epsilon *(phase_r+phase_t) )
                
                fC1_r = get_out_field(Nph_lambda, epsilon * (phase_r) )
                fC2_r = get_out_field(Nph_lambda, -epsilon *(phase_r) )
                
                
                IC1 = poisson.rvs( abs(fC1_t)**2 ) - poisson.rvs( abs(fC1_r)**2 )
                IC2 = poisson.rvs( abs(fC2_t)**2 ) -  poisson.rvs( abs(fC2_r)**2 )
                # get intesnity with shot noise 
                dI_lambda.append(  ( IC1 - IC2 )/(2*epsilon) )
                
                
                #S_lambda.append(np.sqrt( np.nansum(basis[0] * dI**2) / np.nansum(phase_t**2) ))
            #print( 'sensitivity for mode = ',S[-1])
            
            dI = 1/Nph * np.sum( dI_lambda , axis=0)  #* np.median( np.diff(wvls) )# np.trapz(dI_lambda , wvls)
            S.append( np.sqrt( np.nansum(basis[0] * dI**2) / np.nansum(phase_t**2) ) )
        
        S_dic[zernike.zern_name(m+1)][bw] = S


m = list(S_dic.keys())[0]
plt.figure(figsize=(8,5))
for i, theta in enumerate(theta_at_wvl0s):
    plt.plot( bandwidths, [S_dic[m][bw][i] for bw in bandwidths] ,label=f'theta = {np.rad2deg(theta)} [deg]')
plt.legend(fontsize=15)
plt.gca().tick_params(labelsize=20)
plt.xlabel(r'Spectral Bandwidth [$\mu m$]',fontsize=20)
plt.ylabel('Modal Sensitivity',fontsize=20)
plt.title(f'mode = {m}\n'+r'$\theta(\lambda_0)$=90deg',fontsize=20)
#%%
# prelims 
dim = 12*20
stellar_dict = { 'Hmag':2,'r0':0.1,'L0':25,'V':50,'airmass':1.3,'extinction': 0.18 }
tel_dict = { 'dim':dim,'D':1.8,'D_pix':12*20,'pup_geometry':'AT' }
naomi_dict = { 'naomi_lag':4.5e-3, 'naomi_n_modes':14, 'naomi_lambda0':0.6e-6,\
              'naomi_Kp':1.1, 'naomi_Ki':0.93 }
FPM_dict = {'A':1, 'B':1, 'f_ratio':21, 'd_on':26.5e-6, 'd_off':26e-6,\
            'glass_on':'sio2', 'glass_off':'sio2','desired_phase_shift':90,\
                'rad_lam_o_D':1.2 ,'N_samples_across_phase_shift_region':10,\
                    'nx_size_focal_plane':dim}
det_dict={'DIT' : 1, 'ron' : 0.5, 'pw' : 20, 'QE' : 1,'npix_det':12}

baldr_dict={'baldr_lag':0.5e-3,'baldr_lambda0':1.6e-6,'baldr_Ki':0.1, 'baldr_Kp':9}

locals().update(stellar_dict)
locals().update(tel_dict)
locals().update(naomi_dict)
locals().update(FPM_dict)
locals().update(det_dict)
locals().update(baldr_dict)

dx = D/D_pix # grid spatial element (m/pix)

npix_det = D_pix//pw # number of pixels across detector 
pix_scale_det = dx * pw # m/pix

det_dict['[calc]npix_det'] = npix_det 
det_dict['[calc]pix_scale_det'] = pix_scale_det

ph_flux_H = baldr.star2photons('H', Hmag, airmass=airmass, k = extinction, ph_m2_s_nm = True) # convert stellar mag to ph/m2/s/nm 

pup = baldr.pick_pupil(pupil_geometry='disk', dim=D_pix, diameter=D_pix) # baldr.AT_pupil(dim=D_pix, diameter=D_pix) #telescope pupil

basis = zernike.zernike_basis(nterms=20, npix=D_pix)


# apply a Zernike mode to the input phase 
wvls = np.array([1.65e-6, 1.66e-6, 1.67e-6]) #np.linspace(1.4e-6,1.41e-6,2) # input wavelengths 

epsilon = 5
input_phases = np.array( [ np.nan_to_num(basis[15]) * (500e-9/w)**(6/5) for w in wvls] )

input_fluxes = [ph_flux_H * pup  for _ in wvls] # ph_m2_s_nm

input_field_1 = baldr.field( phases = epsilon * input_phases  , fluxes = input_fluxes  , wvls=wvls )
input_field_2 = baldr.field( phases = -epsilon * input_phases  , fluxes = input_fluxes  , wvls=wvls )

input_field_1.define_pupil_grid(dx=dx, D_pix=D_pix)
input_field_2.define_pupil_grid(dx=dx, D_pix=D_pix)
#calibration_phases = [np.nan_to_num(basis[0])  for w in wvls]

#calibration_field = baldr.field( phases = calibration_phases, fluxes = input_fluxes  , wvls=wvls )
#calibration_field.define_pupil_grid( dx=dx, D_pix=D_pix )

N_act=[12,12]
dm = baldr.DM(surface=np.zeros(N_act), gain=1, angle=0,surface_type = 'continuous') 

phase_shift_diameter = 1 * f_ratio * wvls[0]   ##  f_ratio * wvls[0] = lambda/D  given f_ratio
FPM = baldr.zernike_phase_mask(A,B,phase_shift_diameter,f_ratio,d_on,d_off,glass_on,glass_off)

FPM.optimise_depths(desired_phase_shift=desired_phase_shift, across_desired_wvls=wvls ,verbose=True)

FPM_cal = baldr.zernike_phase_mask(A,B,phase_shift_diameter,f_ratio,d_on,d_off,glass_on,glass_off)
FPM_cal.d_off = FPM_cal.d_on


# set up detector object
det = baldr.detector(npix=npix_det, pix_scale = pix_scale_det , DIT= DIT, ron=ron, QE={w:QE for w in wvls})
mask = baldr.pick_pupil(pupil_geometry='disk', dim=det.npix, diameter=det.npix) 

sig = baldr.detection_chain(input_field_1, dm, FPM, det)

sig_cal = baldr.detection_chain(input_field_1, dm, FPM_cal, det)

b = input_field_1.calculate_b(FPM)



phi_reco = reco(sig, sig_cal, FPM, det, b) #reco_linear(sig, sig_cal, FPM, det, b) #
#phi_reco = reco_linear(sig, sig_cal, FPM, det, b) #

N_act=[12,12]
dm = baldr.DM(surface=np.zeros(N_act), gain=1, angle=0, surface_type = 'continuous') 
cmd =  wvls[0]/(2*np.pi) * (phi_reco-np.mean(phi_reco)).reshape(-1)

dm.update_shape(cmd = cmd)
input_field_dm = input_field_1.applyDM(dm)

#plt.plot(np.linspace(-1,1,100), np.arccos( np.linspace(-1,1,100) ) ) 
print('Strehl before = ',np.exp(-np.var( input_field_1.phase[wvls[0]][pup>0.5] ) ))
print('Strehl after = ',np.exp(-np.var( input_field_dm.phase[wvls[0]][pup>0.5] ) ))



# can we just do a least squares thing 
# phi = A * M + x, find A, X

# generate x and y
x = np.linspace(0, 1, 101)
y = 1 + x + x * np.random.random(len(x))

# assemble matrix A
A = np.vstack([x, np.ones(len(x))]).T

# turn y into a column vector
y = y[:, np.newaxis]

# Direct least square regression
alpha = np.dot((np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)),y)
print(alpha)


#%%
#NOW TO TRY CONSTRUCT PHASE 

Psi_C = FPM.get_output_field(input_field_1, keep_intermediate_products=True)

# #photons
P = sig_cal.signal / FPM.A

#b parameter - make it a field 
b = baldr.field( wvls=wvls, fluxes=[abs(b) for b in FPM.b] , phases=[np.angle(b) for b in FPM.b] )
b.define_pupil_grid(dx=dx,D_pix=D_pix)

B = det.detect_field(b).signal
beta = np.mean([b.phase[w] for w in b.phase],axis=0)
beta = interpolate_field_onto_det( beta, det)


#b = FPM.b[0]
#B = np.abs(b)
#B = interpolate_field_onto_det( B, det)

#det.DIT * np.mean( list( det.qe.values() ) ) * B * np.diff(wvls)[0] * 1e9


#beta = np.angle(b)
#beta = interpolate_field_onto_det( beta, det)
# detect it ,

#complex filter
m = FPM.get_filter_design_parameter(wvls[0])
M = abs(m)
mu = np.angle(m)

#phi = interpolate_field_onto_det( input_field_1.phase[wvls[0]], det)
Ic_theory = A*(P**2 + (B*M)**2 + 2*B*P*M*np.cos(phi-mu-beta))

phi_theory = np.arccos( (sig.signal**2 / A - P**2 - (B*M)**2) / (2*B*P*M) ) + mu + beta

#Ic_theory = A*(P + (B*M**2) + 2*np.sqrt(B*P)*M*np.cos(phi-mu-beta))

det_pup = baldr.pick_pupil('disk', dim=det.npix,diameter=det.npix)
plt.figure()
plt.loglog(sig.signal[det_pup >0.5],sig.signal[det_pup >0.5],color='r'); 
plt.loglog(Ic_theory[det_pup >0.5]**0.5, sig.signal[det_pup >0.5], 'x',color='k',alpha=0.5)


n1 = len(input_field_1.phase[wvls[0]])
n2 = len(phi_theory)
plt.plot( np.linspace(-1,1,n1), input_field_1.phase[wvls[0]][n1//2,:] ); 
plt.plot( np.linspace(-1,1,n2), np.pi /2 - phi_theory[n2//2,:] )


def reco(SIG, SIG_CAL, FPM, DET, b):
    
    P = SIG_CAL.signal / A
    B = DET.detect_field(b).signal
    beta = np.mean([b.phase[w] for w in b.phase],axis=0)
    beta = interpolate_field_onto_det( beta, det )
    
    m = np.mean( [FPM.get_filter_design_parameter(w) for w in wvls] )
    M = abs(m)
    mu = np.angle(m)

    phi_reco = np.arccos( (SIG.signal**2 / FPM.A - P**2 - (B*M)**2) / (2*B*P*M) ) + mu + beta    
    
    return(phi_reco)
    

#%%
#OK LETS TRY PREDICT THE OUTPUT 

"""
P_det = sig_cal.signal/A  # = P  ph_flux_H 

from PIL import Image
import numpy as np
n = 3 # repeatation
im = Image.fromarray(P)
up_im = im.resize(b.shape,resample=Image.NEAREST)
P = np.array(up_im) 
# I need to scale by pixel size etc
"""

P = input_field_1.flux[wvls[0]]

Psi_C = FPM.get_output_field(input_field_1, keep_intermediate_products=True)
#Psi_C.define_pupil_grid(dx=dx,D_pix=D_pix)
#s = det.detect_field(Psi_C); plt.imshow(s.signal) # - it matches plt.imshow(sig.signal)

#b parameter
b = FPM.b[0]
B = np.abs(b)
beta = np.angle(b)
# detect it ,

#complex filter
m = FPM.get_filter_design_parameter(wvls[0])
M = abs(m)
mu = np.angle(m)



phi=input_field_1.phase[wvls[0]]
Ic_theory = A*(P**2 + (B*M)**2 + 2*B*P*M*np.cos(phi-mu-beta))

#sig_1 = baldr.detection_chain(input_field_1, dm, FPM, det)
#sig_2 = baldr.detection_chain(input_field_2, dm, FPM, det)

#dI = ( sig_1.signal - sig_2.signal ) / (2*epsilon)

#print( np.sqrt( np.sum( dI[mask>0.5]**2 ) / np.sum( np.array([p[pup >0.5] for p in input_phases])**2 ) ) )
#print( np.sqrt( ((D/D_pix)/det.pix_scale)**2 * np.sum( dI[mask>0.5]**2 ) / np.sum( np.array([p[pup >0.5] for p in input_phases])**2 ) ) )