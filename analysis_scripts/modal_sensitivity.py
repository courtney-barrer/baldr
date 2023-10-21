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
import pandas as pd
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
        
 
def get_out_field(Nph, phase, nx_pix, dx):
    
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



print('begin')


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
        
        fC1_t = get_out_field(Nph, epsilon * (phase_r+phase_t) , nx_pix,  dx)
        fC2_t = get_out_field(Nph, -epsilon *(phase_r+phase_t) , nx_pix,  dx)
        
        fC1_r = get_out_field(Nph, epsilon * (phase_r) , nx_pix,  dx)
        fC2_r = get_out_field(Nph, -epsilon *(phase_r) , nx_pix, dx)
        
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
#plt.savefig(f'/Users/bcourtne/Documents/ANU_PHD2/baldr/figures/sensitivity_vs_dot_diam_monochromatic_theta-{int(np.rad2deg(theta))}.png',dpi=300)


#%% optimal dot diameter vs mode monochromatic
wvl = 1.6e-6
A,B=1,1
f_ratio=21
theta=np.pi/2

nx_pix = 2**9
#D = 1.8 #m
dx = f_ratio*wvl/20

epsilon=0.4

basis = zernike.zernike_basis(nterms=200,npix=nx_pix)

Nph = 1000

N_loDs = np.linspace(0.8,5,30)

mode_indxs = [1,3,5,10,30,50,100] # [1, 3, 6, 11, 19, 35, 62, 110, 199]

sens_dic = {}
for m in mode_indxs: #[1,3,5,7]:
    print(f'looking at mode {m}')
    S=[]
    for N_loD  in N_loDs :
        
        phase_shift_diameter = N_loD * f_ratio * wvl 
        
        phase_shift_region = baldr.pick_pupil('disk', dim=nx_pix, diameter=round(phase_shift_diameter/dx) )
        
        H =  A*(1 + (B/A * np.exp(1j * theta) - 1) * phase_shift_region  )
    
        phase_t = basis[m] 
        phase_r = basis[0]
    
        # NOW INCLUDE TURBULENT AND REFERENCE PHASE LIKE
        
        fC1_t = get_out_field(Nph, epsilon * (phase_r+phase_t) ,nx_pix,  dx)
        fC2_t = get_out_field(Nph, -epsilon *(phase_r+phase_t) ,nx_pix,  dx)
        
        fC1_r = get_out_field(Nph, epsilon * (phase_r) ,nx_pix, dx)
        fC2_r = get_out_field(Nph, -epsilon *(phase_r) ,nx_pix, dx)
        
        IC1 = poisson.rvs( abs(fC1_t)**2 ) - poisson.rvs( abs(fC1_r)**2 )
        IC2 = poisson.rvs( abs(fC2_t)**2 ) -  poisson.rvs( abs(fC2_r)**2 )
        # get intesnity with shot noise 
        dI = 1/Nph * ( IC1 - IC2 )/(2*epsilon)
        
        S.append(np.sqrt( np.nansum(basis[0] * dI**2) / np.nansum(phase_t**2) ))
        #print( 'sensitivity for mode = ',S[-1])
        
    sens_dic[m] = S #N_loDs[np.argmax(S)]

plt.figure(figsize=(8,5))
opt_diams = [N_loDs[np.argmax(sens_dic[m])] for m in sens_dic]
for m in sens_dic:
    plt.semilogx(mode_indxs, opt_diams ,color='k')
#plt.legend(fontsize=15)
plt.gca().tick_params(labelsize=20)
plt.ylabel('optimal dot diameter [$\lambda$/D]',fontsize=20)
plt.xlabel('Zernike index',fontsize=20)
plt.axhline(1.06,color='k',linestyle=':')
plt.axvline(3, color='k',linestyle=':')
plt.tight_layout()
plt.savefig(f'/Users/bcourtne/Documents/ANU_PHD2/baldr/figures/optimal_dot_diameter_vs_mode_monochromatic_theta-{int(np.rad2deg(theta))}.png',dpi=300)


plt.figure(figsize=(11,5))
for m in sens_dic: plt.plot(N_loDs, sens_dic[m] , label=f'mode = {m}')
plt.legend(bbox_to_anchor=(1, 1.), fontsize=15)
plt.gca().tick_params(labelsize=20)
plt.xlabel('dot diameter [$\lambda$/D]',fontsize=20)
plt.ylabel('Sensitivity',fontsize=20)
#plt.axvline(1.06,color='k',linestyle=':')
#plt.axvline(2,color='k',linestyle=':')
#plt.bbox_to_anchor=[1, 1]
plt.tight_layout()
plt.savefig(f'/Users/bcourtne/Documents/ANU_PHD2/baldr/figures/sensitivity_vs_dot_diam_monochromatic_100modes_theta-{int(np.rad2deg(theta))}.png',dpi=300)



#%% sensitivity vs phase shift monochromatic
wvl = 1.6e-6
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
        
        fC1_t = get_out_field(Nph, epsilon * (phase_r+phase_t) ,nx_pix, dx)
        fC2_t = get_out_field(Nph, -epsilon *(phase_r+phase_t) ,nx_pix, dx)
        
        fC1_r = get_out_field(Nph, epsilon * (phase_r) ,nx_pix, dx)
        fC2_r = get_out_field(Nph, -epsilon *(phase_r) ,nx_pix, dx)
        
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
#plt.savefig(f'/Users/bcourtne/Documents/ANU_PHD2/baldr/figures/sensitivity_vs_dot_depth_monochromatic_diam-{N_loD}.png',dpi=300)


#%% Sensitivity vs pixels per pupil monochromatic ( not binning detector!!) 
wvl = 1.6e-6
A,B=1,1
f_ratio=21
theta=np.pi/2

nx_pix = 2**9
#D = 1.8 #m
dx = f_ratio*wvl/20

epsilon=0.3


Nph = 1000

N_loDs = np.linspace(0.8,5,30)

mode_indxs = [1,3,5,10,30,50] # [1, 3, 6, 11, 19, 35, 62, 110, 199]

sens_dic = {}
nx_pixs = np.arange(4,120,2) #[2**x for x in range(2,8)]
N_loD = 1.06

for m in mode_indxs: #[1,3,5,7]:
    print(f'looking at mode {m}')
    S=[]
    for nx_pix  in nx_pixs:
        
        basis = zernike.zernike_basis(nterms=m+1, npix=nx_pix)
        phase_shift_diameter = N_loD * f_ratio * wvl 
        
        phase_shift_region = baldr.pick_pupil('disk', dim=nx_pix, diameter=round(phase_shift_diameter/dx) )
        
        H =  A*(1 + (B/A * np.exp(1j * theta) - 1) * phase_shift_region  )
    
        phase_t = basis[m] 
        phase_r = basis[0]
    
        # NOW INCLUDE TURBULENT AND REFERENCE PHASE LIKE
        
        fC1_t = get_out_field(Nph, epsilon * (phase_r+phase_t) ,nx_pix,  dx)
        fC2_t = get_out_field(Nph, -epsilon *(phase_r+phase_t) ,nx_pix,  dx)
        
        fC1_r = get_out_field(Nph, epsilon * (phase_r) ,nx_pix, dx)
        fC2_r = get_out_field(Nph, -epsilon *(phase_r) ,nx_pix, dx)
        
        IC1 = poisson.rvs( abs(fC1_t)**2 ) - poisson.rvs( abs(fC1_r)**2 )
        IC2 = poisson.rvs( abs(fC2_t)**2 ) -  poisson.rvs( abs(fC2_r)**2 )
        # get intesnity with shot noise 
        dI = 1/Nph * ( IC1 - IC2 )/(2*epsilon)
        
        S.append(np.sqrt( np.nansum(basis[0] * dI**2) / np.nansum(phase_t**2) ))
        #print( 'sensitivity for mode = ',S[-1])
        
    sens_dic[m] = S #N_loDs[np.argmax(S)]

plt.figure(figsize=(8,5))
for m in sens_dic: plt.semilogx( nx_pixs[1:], sens_dic[m][1:], label=f'mode:{m}' )
plt.legend(fontsize=15)
plt.xlabel('Pixels per pupil', fontsize=20)
plt.ylabel('Sensitivity', fontsize=20)
plt.gca().tick_params(labelsize=20)
plt.tight_layout()
plt.savefig(f'/Users/bcourtne/Documents/ANU_PHD2/baldr/figures/sensitivity_vs_pixels_per_pupil-loD-{N_loD}.png',dpi=300)

#%% Sensitivity vs pixels per pupil monochromatic 

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


S = []
pix_per_pupil = []

for pw in [2**x for x in range(2,10)]:
    print(np.log2(pw))
    
    D_pix = 2**10
    
    dx = D/D_pix # grid spatial element (m/pix)
    
    npix_det = int( D_pix//pw ) # number of pixels across detector 
    pix_scale_det = dx * pw # m/pix
    
    det_basis = zernike.zernike_basis(nterms=2, npix= npix_det)
    
    det_dict['[calc]npix_det'] = npix_det 
    det_dict['[calc]pix_scale_det'] = pix_scale_det
    
    ph_flux_H = baldr.star2photons('H', Hmag, airmass=airmass, k = extinction, ph_m2_s_nm = True) # convert stellar mag to ph/m2/s/nm 
    
    pup = baldr.pick_pupil(pupil_geometry='disk', dim=D_pix, diameter=D_pix) # baldr.AT_pupil(dim=D_pix, diameter=D_pix) #telescope pupil
    
    basis = zernike.zernike_basis(nterms=20, npix=D_pix)
    
    m = 8 # mode index 
    
    N_loD = 2 #1.06 # # lambda/D
    
    # apply a Zernike mode to the input phase 
    wvls = np.array([1.65e-6, 1.66e-6, 1.67e-6]) #np.linspace(1.4e-6,1.41e-6,2) # input wavelengths 
    wvl0 = np.mean( wvls )
    
    #Nph = ph_flux_H * np.pi*(D/2)**2 * det.DIT * (wvls[-1]-wvls[0])*1e9 
    
    theta_at_wvl0  = np.pi/2
    # calculate d_on such that we get theta_at_wvl0 phase shift at wvl0
    d_on = (  wvl0 * theta_at_wvl0/(np.pi*2) + n_off * d_off -   n_air * d_off ) / (n_on-n_air)
    
    # amplitude of mode to be applied 
    epsilon = 0.4
    
    phase_t = basis[m]  # aberration 
    phase_r = basis[0]  # reference 
    
    phase_t = np.array( [ np.nan_to_num(basis[m]) * (500e-9/w)**(6/5) for w in wvls] )
    phase_r = np.array( [ np.nan_to_num(basis[0]) * (500e-9/w)**(6/5) for w in wvls] )
    
    # IMPORTANT ! DIVIDE ph_flux_H BY NUMBER OF PHOTONS BINS 
    input_fluxes = [ph_flux_H / len(wvls)  * pup  for _ in wvls] # ph_m2_s_nm
    
    # field with applied aberration
    input_field_t1 = baldr.field( phases = epsilon * (phase_t + phase_r) , fluxes = input_fluxes  , wvls=wvls )
    input_field_t2 = baldr.field( phases = -epsilon * (phase_t + phase_r)   , fluxes = input_fluxes  , wvls=wvls )
    
    # reference field
    input_field_r1 = baldr.field( phases = epsilon * phase_r , fluxes = input_fluxes  , wvls=wvls )
    input_field_r2 = baldr.field( phases = epsilon * phase_r , fluxes = input_fluxes  , wvls=wvls )
    
    # define grids
    input_field_t1.define_pupil_grid(dx=dx, D_pix=D_pix)
    input_field_t2.define_pupil_grid(dx=dx, D_pix=D_pix)
    input_field_r1.define_pupil_grid(dx=dx, D_pix=D_pix)
    input_field_r2.define_pupil_grid(dx=dx, D_pix=D_pix)
    #calibration_phases = [np.nan_to_num(basis[0])  for w in wvls]
    
    #calibration_field = baldr.field( phases = calibration_phases, fluxes = input_fluxes  , wvls=wvls )
    #calibration_field.define_pupil_grid( dx=dx, D_pix=D_pix )
    
    N_act=[12,12]
    dm = baldr.DM(surface=np.zeros(N_act), gain=1, angle=0,surface_type = 'continuous') 
    
    phase_shift_diameter = N_loD  * f_ratio * wvls[0]   ##  f_ratio * wvls[0] = lambda/D  given f_ratio
    
    
    FPM = baldr.zernike_phase_mask(A,B,phase_shift_diameter,f_ratio,d_on,d_off,glass_on,glass_off)
    
    #FPM.optimise_depths(desired_phase_shift=desired_phase_shift, across_desired_wvls=wvls ,verbose=True)
    
    FPM_cal = baldr.zernike_phase_mask(A,B,phase_shift_diameter, f_ratio, d_off, d_off,glass_on,glass_off)
    #FPM_cal.d_off = FPM_cal.d_on
    
    
    # set up detector object
    det = baldr.detector(npix=npix_det, pix_scale = pix_scale_det , DIT= DIT, ron=ron, QE={w:QE for w in wvls})
    #mask = baldr.pick_pupil(pupil_geometry='disk', dim=det.npix, diameter=det.npix) 
    
    sig_t1  = baldr.detection_chain(input_field_t1, dm, FPM, det)
    sig_t2  = baldr.detection_chain(input_field_t2, dm, FPM, det)
    
    # it seems we use same phase mask for the reference field when calculating sensitivity 
    sig_r1 = baldr.detection_chain(input_field_r1, dm, FPM, det)
    sig_r2 = baldr.detection_chain(input_field_r1, dm, FPM, det)
    
    IC1 = sig_t1.signal - sig_r1.signal
    IC2 = sig_t2.signal - sig_r2.signal
    
    # NEED TO DEAL WITH PHOTONS
    Nph = np.nansum(sig_r1.signal)**0.5  ### we sqrt !?!?!?!?!?!?!?!?!?!?!?!?!??!?!?!
    dI = 1/Nph * ( IC1 - IC2 )/(2*epsilon)
    
    S.append( np.sqrt( np.nansum( (det_basis[0] * dI) **2) / np.nansum(phase_t**2)) )
    pix_per_pupil.append( npix_det )

print(f'sensitivity = {S}')

plt.semilogx(pix_per_pupil , S)

#%% CHECK HOW PHOTONS SCALE WITH SUM of ADU (energy conservation ) 

pw = 2**5
D_pix = 2**10
    
dx = D/D_pix # grid spatial element (m/pix)

npix_det = int( D_pix//pw ) # number of pixels across detector 
pix_scale_det = dx * pw # m/pix
    
ph_tmp = np.logspace(1,8,10) # ph/s/m^2/nm
XXX=[]
for p in ph_tmp:
    phase_t = basis[m]  # aberration 
    phase_r = basis[0]  # reference 
    
    phase_t = np.array( [ np.nan_to_num(basis[m]) * (500e-9/w)**(6/5) for w in wvls] )
    phase_r = np.array( [ np.nan_to_num(basis[0]) * (500e-9/w)**(6/5) for w in wvls] )
    
    input_fluxes = [p  * pup  for _ in wvls] # ph_m2_s_nm
    
    input_field_t1 = baldr.field( phases = epsilon * ( phase_r) , fluxes = input_fluxes  , wvls=wvls )
    input_field_t1.define_pupil_grid(dx=dx, D_pix=D_pix)
    
    det = baldr.detector(npix=npix_det, pix_scale = pix_scale_det , DIT= DIT, ron=ron, QE={w:QE for w in wvls})
    mask = baldr.pick_pupil(pupil_geometry='disk', dim=det.npix, diameter=det.npix) 
    
    sig_t1  = baldr.detection_chain(input_field_t1, dm, FPM, det)
    
    XXX.append( np.nansum( sig_t1.signal ) )
    
plt.figure()
plt.loglog( ph_tmp * np.pi/4 * D**2 * det.DIT * 1e9 * (wvls[-1]-wvls[0]), np.array(XXX) )
plt.loglog( XXX, XXX, color='r', label='1:1' )
plt.xlabel('# photons')
plt.ylabel(r'$\Sigma$ ADU')
plt.legend()
    
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
                
                fC1_t = get_out_field(Nph_lambda, epsilon * (phase_r+phase_t), nx_pix, dx )
                fC2_t = get_out_field(Nph_lambda, -epsilon *(phase_r+phase_t), nx_pix, dx )
                
                fC1_r = get_out_field(Nph_lambda, epsilon * (phase_r) ,nx_pix, dx)
                fC2_r = get_out_field(Nph_lambda, -epsilon *(phase_r) ,nx_pix, dx)
                
                
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



#%% sensitivity vs phase shift chromatic ( this cell corresponds to  bcourtne@chapman3:/home/bcourtne/baldr/modal_sensitivity_vs_phaseshift.py )
wvl0 = 1.4e-6
A,B=1,1
f_ratio=21

#desired_phase_shift = np.pi/2
#d_on = 26.5e-6
d_off = 26e-6

glass_on = 'sio2'
glass_off = 'sio2'

n_on = baldr.nglass(1e6 * wvl0, glass=glass_on)[0]
n_off = baldr.nglass(1e6 * wvl0, glass=glass_off)[0]
n_air = baldr.nglass(1e6 * wvl0, glass='air')[0]

nx_pix = 2**10
#D = 1.8 #m
dx = f_ratio*wvl0/20

epsilon_0=0.4

basis = zernike.zernike_basis(nterms=15,npix=nx_pix)

Nph = 10000

theta_at_wvl0s =np.linspace(np.pi/15, np.pi/1.2, 40)

bandwidths= np.logspace(-1, 0, 20)
dlambda = bandwidths[0]/2

#N_loDs = np.linspace(1,2,3)
N_loD = 1.06


# calculate d_on,d_off, at wvl0 

#for N_loD in N_loDs : 
        

phase_shift_diameter = N_loD * f_ratio * wvl0  # phase mask diameter defined at wvl0                
# init phase mask - note d_off = d_on in initialization 
FPM = baldr.zernike_phase_mask(A,B,phase_shift_diameter,f_ratio, d_off, d_off, glass_on,glass_off)
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
                
                fC1_t = get_out_field(Nph_lambda, epsilon * (phase_r+phase_t) ,nx_pix, dx)
                fC2_t = get_out_field(Nph_lambda, -epsilon *(phase_r+phase_t) ,nx_pix, dx)
                
                fC1_r = get_out_field(Nph_lambda, epsilon * (phase_r) ,nx_pix, dx)
                fC2_r = get_out_field(Nph_lambda, -epsilon *(phase_r) ,nx_pix, dx)
                
                
                IC1 = poisson.rvs( abs(fC1_t)**2 ) - poisson.rvs( abs(fC1_r)**2 )
                IC2 = poisson.rvs( abs(fC2_t)**2 ) -  poisson.rvs( abs(fC2_r)**2 )
                # get intesnity with shot noise 
                dI_lambda.append(  ( IC1 - IC2 )/(2*epsilon) )
                
                
                #S_lambda.append(np.sqrt( np.nansum(basis[0] * dI**2) / np.nansum(phase_t**2) ))
            #print( 'sensitivity for mode = ',S[-1])
            
            dI = 1/Nph * np.sum( dI_lambda , axis=0)  #* np.median( np.diff(wvls) )# np.trapz(dI_lambda , wvls)
            S.append( np.sqrt( np.nansum(basis[0] * dI**2) / np.nansum(phase_t**2) ) )
        
        S_dic[zernike.zern_name(m+1)][bw] = S

"""
m = list(S_dic.keys())[0]
plt.figure(figsize=(8,5))
for i, theta in enumerate(theta_at_wvl0s):
    plt.plot( bandwidths, [S_dic[m][bw][i] for bw in bandwidths] ,label=f'theta = {np.rad2deg(theta)} [deg]')
plt.legend(fontsize=15)
plt.gca().tick_params(labelsize=20)
plt.xlabel(r'Spectral Bandwidth [$\mu m$]',fontsize=20)
plt.ylabel('Modal Sensitivity',fontsize=20)
plt.title(f'mode = {m}\n'+r'dot diameter =$1.06 \lambda/D$',fontsize=20)


m = list(S_dic.keys())[0]
plt.figure(figsize=(8,5))
bw = bandwidths[-1]
for bw in [bandwidths[-1], bandwidths[2]]:
    plt.plot( np.rad2deg(theta_at_wvl0s), S_dic[m][bw] ,label=r'$\Delta \lambda$='+f'{round(bw,2)}um')
plt.legend(fontsize=15)
plt.gca().tick_params(labelsize=20)
plt.xlabel(r'phase shift at $\lambda_0$ [deg]',fontsize=20)
plt.ylabel('Modal Sensitivity',fontsize=20)
plt.title(f'mode = {m}\n'+r'dot diameter =$1.06 \lambda/D$',fontsize=20)   
#plt.savefig(f'/Users/bcourtne/Documents/ANU_PHD2/baldr/figures/sensitivity_vs_phaseshift_vs_bandwidth_chromatic_diam-{N_loD}.png',dpi=300)
"""
m = list(S_dic.keys())[0]
best_phase_shift = []
for bw in bandwidths:
    best_phase_shift.append( np.rad2deg(theta_at_wvl0s)[np.argmax(S_dic[m][bw])] ) 

plt.figure(figsize=(8,5))
plt.plot( bandwidths/(wvl0*1e6), best_phase_shift)    
plt.legend(fontsize=15)
plt.gca().tick_params(labelsize=20)
plt.xlabel(r'bandwidth [$\Delta \lambda/\lambda_0$]',fontsize=20)
plt.ylabel('optimal phase shift [deg]',fontsize=20)
plt.title(f'mode = {m}\n'+r'dot diameter =$1.06 \lambda/D$',fontsize=20)   
plt.savefig(f'optimal_phaseshift_vs_bandwidth_chromatic_diam-{N_loD}.png',dpi=300)

pc_bandwidth = bandwidths/(1e6 * wvl0)
#pd.DataFrame([pc_bandwidth, best_phase_shift] ).to_csv( 'bandwidth_vs_optimal_phase_shift.csv' )  


# phase shift vs sensitivity 
m = list(S_dic.keys())[0]
plt.figure(figsize=(8,5))
bw = bandwidths[-1]
for bw in [bandwidths[2], bandwidths[-1]]:
    pc_bw = round( bw/(wvl0*1e6) , 2)
    #plt.plot( np.rad2deg(theta_at_wvl0s), S_dic[m][bw] ,label=r'$\Delta \lambda$='+f'{round(bw,2)}um')
    plt.plot( np.rad2deg(theta_at_wvl0s), S_dic[m][bw] ,label=r'$\Delta \lambda/\lambda_0$='+f'{100 * pc_bw}%')
    plt.axvline( np.rad2deg(theta_at_wvl0s)[np.argmax(S_dic[m][bw])] , color='k',linestyle=':')
plt.legend(fontsize=15)
plt.gca().tick_params(labelsize=20)
plt.xlabel(r'phase shift at $\lambda_0$ [deg]',fontsize=20)
plt.ylabel('Modal Sensitivity',fontsize=20)
plt.title(f'mode = {m}\n'+r'dot diameter =$1.06 \lambda/D$',fontsize=20)   
#plt.savefig(f'sensitivity_vs_phaseshift_vs_bandwidth_chromatic_diam-{N_loD}.png',dpi=300)


#for bw in bandwidths:
#    pd.DataFrame([np.rad2deg(theta_at_wvl0s), S_dic[m][bw]] ).to_csv(f'phase_shift_vs_sensitivity_{bw}um.csv' )




#%% Plotting output from chapman Sensitivity vs phase shift for different bandwidths 
import glob
files_tmp = np.sort( glob.glob('/Users/bcourtne/Documents/ANU_PHD2/baldr/figures/misc_data/phase_shift_vs_sensitivity_focus_*csv') )
N_loD=1.06
mode = 'focus'
plt.figure(figsize=(8,5))
for i,col in zip( [2, 12, len(files_tmp)-1], ['g','b','r']):
    # wvl0 = 1.4
    dL_o_L0 = float( files_tmp[i].split('_')[-1].split('um.')[0] ) / 1.4
    sens_v_bandwidth =  pd.read_csv(files_tmp[i],index_col=0)
    
    theta_max = sens_v_bandwidth.loc[0][ np.argmax( sens_v_bandwidth.loc[1] ) ]
    
    plt.plot( sens_v_bandwidth.loc[0], sens_v_bandwidth.loc[1] ,color=col, linestyle='-', label=r'$\Delta \lambda/\lambda_0$='+f'{round(dL_o_L0,1)}')
    plt.axvline( theta_max , color = col, linestyle=':')
#plt.grid()
plt.legend(fontsize=20)
plt.gca().tick_params(labelsize=20)
plt.xlabel(r'Phase shift at '+r'$\lambda_0$ [deg]',fontsize=20)
plt.ylabel('Sensitivity',fontsize=20)
plt.title(f'mode = {m}\n'+r'dot diameter =$1.06\ \lambda_0/D$',fontsize=20)   #r'$\lambda_0=1.4 \mu m$'+
plt.tight_layout()
plt.savefig(f'/Users/bcourtne/Documents/ANU_PHD2/baldr/figures/sensitivity_vs_bandwidth_chromatic_diam-loD{N_loD}_m{mode}.png', dpi=300)



#%% Plotting output from chapman for optimal phase shift vs bandwidth and sensitivity vs phase shift color legendfor different bandwidths 
from scipy.interpolate import interp1d 
import pandas as pd 
bandiwdth_v_phaseshift = pd.read_csv('/Users/bcourtne/Documents/ANU_PHD2/baldr/figures/bandwidth_vs_optimal_phase_shift.csv')

N_loD=1.06
mode = 'focus'
indxs = [1,3,9,15,18,19]
indxs = [4,9,15,18] #[1,5,9,15,18]


x,y = [], [] 
for i in indxs:
    x.append( bandiwdth_v_phaseshift.loc[0][i] )
    y.append( bandiwdth_v_phaseshift.loc[1][i] )
    
ps_fn = interp1d(x,y,kind='quadratic',fill_value='extrapolate')
bw_grid = np.linspace(bandiwdth_v_phaseshift.loc[0][7] , bandiwdth_v_phaseshift.loc[0][-1] , 100 )

plt.figure(figsize=(8,5))
plt.plot( bw_grid, ps_fn(bw_grid) )
plt.grid()
plt.gca().tick_params(labelsize=20)
plt.xlabel(r'bandwidth [$\Delta \lambda/\lambda_0$]',fontsize=20)
plt.ylabel('optimal phase shift\nat '+r'$\lambda_0$'+' [deg]',fontsize=20)
plt.title(f'mode = {m}\n'+r'dot diameter =$1.06\ \lambda_0/D$',fontsize=20)   #r'$\lambda_0=1.4 \mu m$'+
plt.tight_layout()

plt.savefig(f'/Users/bcourtne/Documents/ANU_PHD2/baldr/figures/optimal_phaseshift_vs_bandwidth_chromatic_diam-loD{N_loD}_m{mode}_w{round(wvl0,1)}.png', dpi=300)


d_bw=0.7-0.3
d_ps = 88.2-80

print(f'gradient  = {d_ps/(d_bw*100)} deg/%BW')








#%% lotting output from chapman for sensitivity vs phase shift  legended for different bandwidths 



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

FPM_cal = baldr.zernike_phase_mask(A,B,phase_shift_diameter, f_ratio, d_on,d_off,glass_on,glass_off)
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