#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 15:50:54 2023

@author: bcourtne
"""

import numpy as np
from scipy.special import jv
import matplotlib.pyplot as plt
import aotools
import os
from scipy.interpolate import interp1d
import pyzelda.utils.zernike as zernike
os.chdir('/Users/bcourtne/Documents/ANU_PHD2/heimdallr')

from functions import baldr_functions_2 as baldr

def atm_zernike(j, theta, seeing= 0.7, wvl=2.2e-6, v_mn=20, L_0=np.inf, diam=8):
    """
    Parameters
    ----------
    n : TYPE
        DESCRIPTION. radial degree 
    m : TYPE
        DESCRIPTION. azimuth frequency (m<=n)
    wvl : TYPE, optional
        DESCRIPTION. The default is 2.2e-6.
    seeing : TYPE, optional
        DESCRIPTION. The default is 0.86.
    tau_0 : TYPE, optional
        DESCRIPTION. The default is 4e-3.
    L_0 : TYPE, optional
        DESCRIPTION. The default is np.inf.
    diam : TYPE, optional
        DESCRIPTION. The default is 8.

    Returns
    -------
    None.

    """
    
    # OSA [4] and ANSI single-index Zernike polynomials using
    n, m = noll_indices(j)
    
    #-----Atmospheric Parameters 
    
    r_0 = 0.98*wvl/np.radians(seeing/3600)  #fried parameter at wavelength in m.
    #tau_0 = (1e6*wvl/0.5)**(6/5) * 4e-3        #atm coherence time at wavelength
    #v_mn = 0.314 * r_0 / tau_0               #AO definition for vbar   
    
    atm_params = {'wvl':wvl,'L_0':L_0,'diam':diam,'v_mn':v_mn} #'seeing':seeing,'r_0':r_0,'tau_0':tau_0,
    #-----configuration parameters
    
    #ky needs to have good sampling at small values where VK_model * tel_filter is maximum for the integration (course sampling outside of this since contribution is negligble)
    ky = np.concatenate([-np.logspace(-10,3,2**11)[::-1], np.logspace(-10,3,2**11) ] ) # Spatial frequency  
    nf = int(2**13)    #number of frequency samples
    minf = 1e-2  #min freq
    maxf = 5e2   #max freq
    fs = np.linspace(minf,maxf,nf)
    
    
    #----- PSD (rad^2/Hz)
    
    zernike_psd = [] #to hold diff piston PSD (rad^2/Hz)
    for i, f in enumerate(fs):
        kx = f/v_mn
        k_abs = np.sqrt(kx**2 + ky**2)        
        VK_model = 0.0229 * r_0**(-5/3.) * (L_0**(-2) + k_abs**2)**(-11/6.) #von karman model (Conan 2000)
        if m==0:
            tel_filter = np.sqrt(n+1) * (2*abs( sp.jv( n+1, np.pi*diam*k_abs)/(np.pi*diam*k_abs )) ) * 1  #need to square this 
        else:
            tel_filter = np.sqrt(n+1) * (2*abs( sp.jv( n+1, np.pi*diam*k_abs)/(np.pi*diam*k_abs )) ) #* np.sqrt(2) #* np.cos(theta*m)
        """elif m>0: #even
            tel_filter = np.sqrt(n+1) * (2*abs( sp.jv( n+1, np.pi*diam*k_abs)/(np.pi*diam*k_abs )) ) * np.sqrt(2) * np.cos(theta*m)  #need to square this 
        elif m<0: #odd
            tel_filter = np.sqrt(n+1) * (2*abs( sp.jv( n+1, np.pi*diam*k_abs)/(np.pi*diam*k_abs )) ) * np.sqrt(2) * np.sin(theta*m) 

        """
        Phi = VK_model * abs(tel_filter)**2 
        
        zernike_psd.append( np.trapz(Phi, ky)/v_mn )

    return((fs, zernike_psd), atm_params)



def calibrate_phase_screen2wvl(wvl, screen):
    """
    

    Parameters
    ----------
    wvl : float 
        wavelength (m) to adjust phase using lambda^(5/6) scaling 
    screen : np.array 
        DESCRIPTION. phase screen calibrated with r0(wvl=500nm)
        if from aotools.turbulence.infinitephasescreen.PhaseScreenVonKarman
        then use screen.scrn as input 

    Returns
    -------
    list of adjusted phase screens

    """
    # turbulence gets better for longer wavelengths 
    adjusted_screen = (500e-9/wvl)**(6/5) * screen #avoid long lists for memory 
    
    return(adjusted_screen) 


def DM_mech_stroke_req(D,r0,wvl, include_tip_tilt=True):
    #required mechanical stroke of DM for given telescope diameter (D),turbulence (Fried parameter r0) and wavelength (wvl)
    # taken from Madec 2012 - Overview of Deformable Mirror Technologies for Adaptive Optics and Astronomy
    if include_tip_tilt:
        l=1.03
    else:
        l=0.134
    stroke = 3*wvl / (2*np.pi) * np.sqrt(l) * (D/r0)**(5/3)
    return(stroke)
    

def g(r, eta):
    #synthetic reference wave from Glückstad 2006 (second order expansion) 
    g = 1-\
        jv(0, 1.22*np.pi*eta) -\
        (0.61 * np.pi * eta)**2 * jv(2, 1.22*np.pi*eta) * r**2
        
    return(g)


def eta_2_g(eta):
    g = 1-jv(0, 1.22*np.pi*eta)
    return(g)

def optimal_filter_complex_param(eta, delta_phi  ):
    a = np.sinc(delta_phi)
    g = eta_2_g(eta)
    C = 1/(a*g) * np.exp(1j * (np.pi - delta_phi/2))
    
    return(C)

def optimal_filter_phase_amp(eta, delta_phi ):
    a = np.sinc(delta_phi)
    g = eta_2_g(eta)
    
    BoA = np.sqrt(1 - 2*np.cos(delta_phi/2)/(a*g) + 1/(a*g)**2) 
    theta =  np.arcsin( np.sin(delta_phi/2) / (BoA * a * g) ) 
    return(BoA, theta)
    
r = np.linspace(0,1.2,100) # normalized pupil coordinate 
eta = [0.2,0.4, 0.5, 0.625, 0.7]

plt.figure(figsize=(8,5))
for e in eta:

    plt.plot( r, g(r, e) ,label=r'$\eta$'+f'={e}')

plt.axvline(1,linestyle=':', label='telescope radius')
plt.gca().tick_params(labelsize=14)
plt.legend()
plt.xlabel('normalized radius',fontsize=14)
plt.ylabel('synthetic reference wave (g(r))',fontsize=14)
#plt.savefig('/Users/bcourtne/Documents/ANU_PHD2/heimdallr/synthetic_reference_wave_vs_eta.png')



#%% optimal filter parameters 
lambda_H = 1.65e3 # nm

gpao_dict={'delta_phi_gpao_faint' :2 *  2*np.pi* 150/lambda_H,\
'delta_phi_gpao_bright' : 2 * 2*np.pi* 60/lambda_H,\
'delta_phi_gpao_red' : 2 * 2*np.pi* 150/lambda_H,\
'delta_phi_gpao_blue' : 2 * 2*np.pi* 60/lambda_H}

naomi_dict={'delta_phi_naomi_faint' : 2 * 2*np.pi* 175/lambda_H,\
'delta_phi_naomi_bright' : 2 *  2*np.pi* 100/lambda_H,\
'delta_phi_naomi_red' :2 *  2*np.pi* 175/lambda_H,\
'delta_phi_naomi_blue' : 2 * 2*np.pi* 100/lambda_H}

    
gpao_dict={'delta_phi_gpao_faint' :2 *  2*np.pi* 70/lambda_H,\
'delta_phi_gpao_bright' : 2 * 2*np.pi* 40/lambda_H,\
'delta_phi_gpao_red' : 2 * 2*np.pi* 70/lambda_H,\
'delta_phi_gpao_blue' : 2 * 2*np.pi* 40/lambda_H}

naomi_dict={'delta_phi_naomi_faint' : 2 * 2*np.pi* 100/lambda_H,\
'delta_phi_naomi_bright' : 2 *  2*np.pi* 50/lambda_H,\
'delta_phi_naomi_red' :2 *  2*np.pi* 100/lambda_H,\
'delta_phi_naomi_blue' : 2 * 2*np.pi* 50/lambda_H}
eta=0.625
gpao_Omegas = {}
for k in gpao_dict:
     #gpao_Omegas[k] = optimal_filter_complex_param( eta, gpao_dict[k] )
     gpao_Omegas[k] = optimal_filter_phase_amp( eta, gpao_dict[k] )
naomi_Omegas = {}
for k in naomi_dict:
     #naomi_Omegas[k] = optimal_filter_complex_param( eta, naomi_dict[k] )     
     naomi_Omegas[k] = optimal_filter_phase_amp( eta, naomi_dict[k] )  
     
#np.angle( pd.DataFrame( naomi_Omegas ,index=[0]).T ) * 180/np.pi
#np.abs( pd.DataFrame( naomi_Omegas ,index=[0]).T ) 




#%% limiting magnitude analysis 



tel='AT'
D_dict = {'AT':1.8,'UT':8} # m
Npix = 12**2
D=D_dict[tel] #m
A = (D/2)**2*np.pi #m2

BW_lambda = 200 #nm
DIT=0.5e-3
sigma_ron = 1 #e-

SNR_dict = {}
Hmags = range(18)
BW_lambda = [100,200,300] #nm
N_pixs = [6**2, 12**2]
for BW in BW_lambda:
    SNR_dict[BW] = {}
    for Npix in N_pixs:
        SNR_dict[BW][Npix] = []
        for mag in Hmags:
            f = baldr.star2photons('H', mag, airmass=1, k = 0.18, ph_m2_s_nm = True) #ph/m2/s/nm
            Nph_pix = f * A * DIT * BW/ Npix
            
            SNR_dict[BW][Npix].append(  Nph_pix/(np.sqrt(Nph_pix) + sigma_ron) )


for BW,c_bw in zip(BW_lambda, ['red','blue','green']):
    for N_pix,style_pix in zip(N_pixs,['-',':']):

        plt.semilogy(Hmags, SNR_dict[BW][N_pix], color=c_bw,linestyle=style_pix, label=f'#pixels = {round(np.sqrt(N_pix))}x{round(np.sqrt(N_pix))}, '+r'$\Delta \lambda$'+f'={BW}nm')
     
plt.axhline(1,color='k',lw=3)
plt.ylim([1e-1,1e3])
plt.xlim([0,max(Hmags )])
plt.legend()
plt.xlabel('Hmag',fontsize=14)
plt.ylabel('pixelwise SNR',fontsize=14)
plt.grid()
plt.gca().tick_params(labelsize=14)
plt.title(f'{tel} (D={D}m), Baldr DIT = {DIT*1e3}ms')
plt.tight_layout()
plt.savefig(f'/Users/bcourtne/Documents/ANU_PHD2/heimdallr/PDR_may_2023/SNR_vs_Hmag_{tel}.png')


#%% estimating r0 in partial AO correction 


tel = 'AT'
D_dict = {'AT':10.8,'UT':80} # m
D_pix = 2**9
dx = D_dict[tel] / D_pix


wvl = 1.25e-6 
r0=0.1
L0 = 25

screens = aotools.turbulence.infinitephasescreen.PhaseScreenVonKarman(D_pix, pixel_scale=dx,\
          r0=r0 , L0=L0, n_columns=2,random_seed = 1)


pup_mask = aotools.pupil.circle(D_pix//2, D_pix, circle_centre=(0,0))
pup_mask[pup_mask==0] = np.nan

initial_phase_mask = pup_mask * calibrate_phase_screen2wvl(wvl, screens.add_row())




rho = np.round(np.linspace(1,int( D_pix/D/4 ),5)).astype(int)

r0_dict = {} # to hold r0 as function of number of zernike modes removed 

s_dict={} # to hold r'<|$\phi(r) - \phi(r+\rho)|^2$>' for each removal of a zernike mode 

modes2remove = np.logspace(0,2.5,5).astype(int)
basis = zernike.zernike_basis(nterms=np.max(modes2remove+2), npix=D_pix) 

for noll_index in modes2remove:
    s_dict[noll_index] = {}
    print('calculating for noll index', noll_index)
    # remove Zernike modes up to Noll index
    if  noll_index>1:
        
        Z_coes = zernike.opd_expand( initial_phase_mask , nterms= noll_index, basis=zernike.zernike_basis)
        phi_est = np.sum( basis[:noll_index] * np.array(Z_coes)[:,np.newaxis, np.newaxis] , axis=0) 
        
        s = initial_phase_mask - phi_est 
        
    else:
        s = initial_phase_mask
    
    # calculate <|$\phi(r) - \phi(r+\rho)|^2$> for various values of rho 
    for r in rho:
        print(r)
        delta_s = []
        for i in np.arange(r, s.shape[0]-r,10):
            for j in np.arange(r, s.shape[1]-r,10):
                circ_mask = aotools.pupil.circle(r, D_pix, circle_centre=(i,j))
                # put values outside circle region -> nan 
                circ_mask[circ_mask==0] = np.nan
                
                area=circ_mask * s
                if np.nansum( circ_mask * pup_mask ) > 0.5 * np.pi * r**2: # if not too many nan value
                    delta_s.append( np.nanvar( circ_mask * s) )
                
        s_dict[noll_index][r] = np.nanmean( delta_s )         

    #create interpolation function and calculate from this where <|$\phi(r) - \phi(r+\rho)|^2$>=1rad^2, store it in r0_dict
    fn = interp1d(  np.array(list(s_dict[noll_index].values())  ), dx * np.array(list(s_dict[noll_index].keys()))  )
    
    try:
        r0_dict[noll_index] = fn(1) 
    except:
        print('cannot interpolate for noll index = ',noll_index )
        print('\n radius rho (pixels) vs D(rho) in this case is:\n',s_dict[noll_index])
        print('\nkeeping max  D(rho) ')
        #r0_dict[noll_index] = np.max(list(s_dict[noll_index].values()))


plt.figure(figsize=(8,5))
plt.semilogx( list(r0_dict.keys()), list(r0_dict.values()), label=r'$\lambda$='+f'{wvl*1e6}um, L0={L0}m, r0(500nm) = {r0}m')
plt.legend(fontsize=14)
plt.xlabel('Zernike modes removed',fontsize=14)
plt.ylabel(r'$\rho$ : <|$\phi(r) - \phi(r+\rho)|^2> = 1rad^2$',fontsize=14)
plt.grid()


plt.figure()
plt.plot( dx * np.array( list( s_dict.keys()) ), s_dict.values(), label=r'$\lambda$='+f'{wvl*1e6}um, L0={L0}m, r0(500nm) = {r0}m')
plt.legend()
plt.xlabel(r'$\rho$ (m)')
plt.ylabel(r'<|$\phi(r) - \phi(r+\rho)|^2$>')




#%% transfer function from Woillez 2019

T_wfs = lambda s,Ti : (1-np.exp(-s*Ti))/(s*Ti) 
T_delay = lambda s,Td: np.exp(-s*Td)
T_gain =lambda s,Ki,Tc : Ki/(1-np.exp(-s*Tc))
T_dac =lambda s,Tc : (1-np.exp(-s*Tc))/(s*Tc) 

zernike_psd = {}
for m in np.arange(2,20):
    print(m)
    (fs, zernike_psd_tmp), atm_params = atm_zernike(j=m, theta=0, seeing= 0.7, wvl=1.6e-9, v_mn=50, L_0=np.inf, diam=1.8)

    zernike_psd[m]=np.array( zernike_psd_tmp )

#%%
#T =  T_wfs(fs,1e-1) * T_delay(fs,2e-3)* T_gain(fs,1e-3, 0.5e-3)* T_dac(fs, 0.5e-3)
# using wiki def of closed loop with Woille diagram 
Tint = 25e-3
T =  T_wfs(fs,Tint) / (1 + T_wfs(fs,Tint) * T_delay(fs,1e-3)* T_gain(fs,1/0.4, 0.5e-3) * T_dac(fs, 0.5e-3))

T_baldr =  T_wfs(fs,Tint/2) / (1 + T_wfs(fs,Tint/2) * T_delay(fs,1e-3)* T_gain(fs,0.2, 0.5e-3) * T_dac(fs, 0.5e-3))
#plt.semilogx( fs ,20*np.log10( abs(1/(1+T)**2 * zernike_psd) ) )



import matplotlib as mpl
import matplotlib.cm as cm
   
norm = mpl.colors.Normalize(vmin=-10, vmax=3*len(zernike_psd))
cmap_naomi = cm.Reds
cmap_baldr = cm.Greens


rev_cumsum_naomi_rms = np.cumsum( np.sqrt(np.sum([(abs(T**2) *  np.array( zernike_psd[m] ))**2 for m in zernike_psd],axis=0))[::-1] )[::-1] * np.diff(fs)[0]
rev_cumsum_baldr_rms = np.cumsum( np.sqrt(np.sum([(abs(T_baldr**2) * abs(T**2)  *  np.array( zernike_psd[m] ))**2 for m in zernike_psd],axis=0))[::-1] )[::-1] * np.diff(fs)[0]

plt.figure(figsize=(8,5))
opd_baldr=round( rev_cumsum_baldr_rms[0]**0.5 / (2*np.pi) * 1600 )
opd_naomi=round(rev_cumsum_naomi_rms[0]**0.5 / (2*np.pi) * 1600 )

MMM = len(zernike_psd)
for i,m in enumerate(zernike_psd):
    if m <= 14:
        sig =  abs(T**2) *  np.array( zernike_psd[m] ) 
        opd = round(np.sum(sig  * np.diff(fs)[1] )**0.5 / (2*np.pi) * 1600,0) #nm
        plt.loglog( fs , sig, color=cmap_naomi(norm(MMM-i)),linestyle=':')#label=f'J = {m-1} (opd={opd}nm)')
        
        #plt.loglog( fs , np.cumsum( sig[::-1] )[::-1] * np.diff(fs)[0] )
    else:
        sig = np.array( zernike_psd[m] ) 
        opd = round(np.sum(sig * np.diff(fs)[1] )**0.5 / (2*np.pi) * 1600,0) #nm
        plt.loglog( fs ,sig, color=cmap_naomi(norm(MMM-i)),linestyle=':') #label=f'J = {m-1} (opd={opd}nm)')
plt.loglog(fs, rev_cumsum_naomi_rms, color='r', lw=2,label=f'NAOMI rev. cum. OPD ({opd_naomi}nm rms @ 10s)')
plt.loglog(fs, rev_cumsum_baldr_rms,color='g', lw=2,label=f'Baldr rev. cum. OPD ({opd_baldr}nm rms @ 10s)' )
#print( np.sum(abs(T**2) * zernike_psd  * np.diff(fs)[1] )**0.5 / (2*np.pi) * 1600 )


for i ,m in enumerate(zernike_psd):

    sig = abs(T_baldr**2) * abs(T**2) *  np.array( zernike_psd[m] ) 
    opd = round(np.sum(sig  * np.diff(fs)[1] )**0.5 / (2*np.pi) * 1600,0) #nm
    plt.loglog( fs , sig,color=cmap_baldr(norm(MMM-i)),linestyle='--')#, label=f'J = {m-1} (opd={opd}nm)')
    #plt.loglog( fs , np.cumsum( sig[::-1] )[::-1] * np.diff(fs)[0] )

#print( np.sum(abs(T**2) * zernike_psd  * np.diff(fs)[1] )**0.5 / (2*np.pi) * 1600 )


plt.legend()
plt.xlabel('frequency [Hz]',fontsize=14)
plt.ylabel('PSD '+r'[$rad^2/Hz$]',fontsize=14)
plt.grid()
plt.ylim([1e-13,5e1])
plt.gca().tick_params(labelsize=14)
plt.tight_layout()
plt.savefig('/Users/bcourtne/Documents/ANU_PHD2/heimdallr/PDR_may_2023/naom-baldr_psd_rev-cum.png')
print( rev_cumsum_baldr_rms[0]**0.5 / (2*np.pi) * 1600, rev_cumsum_naomi_rms[0]**0.5 / (2*np.pi) * 1600)

#%%
#T =  T_wfs(fs,1e-1) * T_delay(fs,2e-3)* T_gain(fs,1e-3, 0.5e-3)* T_dac(fs, 0.5e-3)
# using wiki def of closed loop with Woille diagram 
Tint = 20e-3
T =  T_wfs(fs,Tint) / (1 + T_wfs(fs,Tint) * T_delay(fs,4e-3)* T_gain(fs,0.4, 0.5e-3) * T_dac(fs, 0.5e-3))

T_baldr =  T_wfs(fs,Tint/2) / (1 + T_wfs(fs,Tint) * T_delay(fs,1e-3)* T_gain(fs,0.4, 0.5e-3) * T_dac(fs, 0.5e-3))
#plt.semilogx( fs ,20*np.log10( abs(1/(1+T)**2 * zernike_psd) ) )

plt.figure()
for m in zernike_psd:
    if m <= 14:
        sig =  abs(T**2) *  np.array( zernike_psd[m] ) 
        opd = round(np.sum(sig  * np.diff(fs)[1] )**0.5 / (2*np.pi) * 1600,0) #nm
        plt.loglog( fs , sig, label=f'J = {m-1} (opd={opd}nm)')
        #plt.loglog( fs , np.cumsum( sig[::-1] )[::-1] * np.diff(fs)[0] )
    else:
        sig = np.array( zernike_psd[m] ) 
        opd = round(np.sum(sig * np.diff(fs)[1] )**0.5 / (2*np.pi) * 1600,0) #nm
        plt.loglog( fs ,sig, label=f'J = {m-1} (opd={opd}nm)')
#print( np.sum(abs(T**2) * zernike_psd  * np.diff(fs)[1] )**0.5 / (2*np.pi) * 1600 )

plt.legend(bbox_to_anchor=(1,1))
plt.xlabel('frequency [Hz]',fontsize=14)
plt.ylabel('PSD '+r'[$rad^2/Hz$]',fontsize=14)
plt.grid()
plt.gca().tick_params(labelsize=14)


plt.figure()
for m in zernike_psd:
    if m <= 14:
        sig = abs(T_baldr**2) * abs(T**2) *  np.array( zernike_psd[m] ) 
        opd = round(np.sum(sig  * np.diff(fs)[1] )**0.5 / (2*np.pi) * 1600,0) #nm
        plt.loglog( fs ,  label=f'J = {m-1} (opd={opd}nm)')
        #plt.loglog( fs , np.cumsum( sig[::-1] )[::-1] * np.diff(fs)[0] )
    
#print( np.sum(abs(T**2) * zernike_psd  * np.diff(fs)[1] )**0.5 / (2*np.pi) * 1600 )

plt.legend(bbox_to_anchor=(1,1))
plt.xlabel('frequency [Hz]',fontsize=14)
plt.ylabel('PSD '+r'[$rad^2/Hz$]',fontsize=14)
plt.grid()
plt.gca().tick_params(labelsize=14)
#%%
# fitting errors 
sigma_fit = lambda r0, Na, D : 0.257 * (D/r0)**(5/3) * Na**(-5/6)
sigma_servo = lambda tau0, tau_servo:  (tau_servo/tau0)**(5/3) 



#%% 
