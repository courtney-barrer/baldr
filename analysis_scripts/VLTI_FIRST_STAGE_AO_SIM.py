#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 15:38:32 2023

@author: bcourtne

VLTI_FIRST_STAGE_AO_SIM
"""
import numpy as np 
import pandas as pd 
import corner 
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt 


"""
# VEGA zero points from https://www.astronomy.ohio-state.edu/martini.10/usefuldata.html
    # units are :
        #microns
        #microns, UBVRI from Bessell (1990), JHK from AQ
        #1e-20 erg cm-2 s-1 Hz-1, from Bessell et al. (1998)
        #1e-11 erg cm-2 s-1 A-1, from Bessell et al. (1998)
        #photons cm-2 s-1 A-1, calculated from above quantities
        #photons cm-2 s-1 A-1, calculated from above quantities
"""
vega_zero_points = pd.DataFrame({'lambda_eff':[0.36,0.438,0.545,0.641,0.798, 1.22, 1.63, 2.19],\
          'd_lambda':	[0.06, 0.09, 0.085, 0.15, 0.15, 0.26, 0.29, 0.41],\
              'f_v':[1.79, 4.063, 3.636, 3.064, 2.416, 1.589, 1.021, 0.64],\
                  'f_lambda':[417.5, 632, 363.1, 217.7, 112.6, 31.47, 11.38, 3.961],\
                      'ph_lambda':[756.1, 1392.6, 995.5, 702.0, 452.0, 193.1, 93.3, 43.6]},\
                            index = ['U','B','V','R','I','J','H','K'] )
    




def star2photons(band, mag, airmass=1, k = 0.18, ph_m2_s_nm = True):
    """
    # for given photometric band, magnitude, airmass, extinction return Nph/m2/s/wvl     

    Parameters
    ----------
    band : string (e.g. 'R')
        Photometric band. choose from ['U','B','V','R','I','J','H','K']
    mag : float or int
        Vega magnitude in respective band
    airmass : float or int (between 1-5 for physically real telescopes), optional
        DESCRIPTION. the target airmass. default is 1 (i.e. observing at zenith)
    k: float or int, optional
        DESCRIPTION. extinction coefficient. The default is 0.18.
    ph_m2_s_nm: Boolean 
        DESCRIPTION. do we want #photons m-2 s-1 nm-1 (ph_m2_s_nm=True)? OR #photons cm-2 s-1 A-1 (ph_m2_s_nm=False)
    
    Returns
    -------
    ph_flux = #photons cm-2 s-1 A-1 or #photons m-2 s-1 nm-1

    """
        
    
        
    # good examples  http://www.vikdhillon.staff.shef.ac.uk/teaching/phy217/instruments/phy217_inst_phot_problems.html
    ph_flux = vega_zero_points.loc[band]['ph_lambda'] * 10**( -(mag + k * airmass - 0)/2.5 ) #photons cm-2 s-1 A-1
    
    if ph_m2_s_nm: #convert #photons cm-2 s-1 A-1 --> #photons m-2 s-1 nm-1
        ph_flux = ph_flux * 1e4 * 10 #photons m-2 s-1 nm-1
        
    """    examples 
    # sanity check from http://www.vikdhillon.staff.shef.ac.uk/teaching/phy217/instruments/phy217_inst_phot_problems.html
    A star has a measured V-band magnitude of 20.0. How many photons per second 
    are detected from this star by a 4.2 m telescope with an overall 
    telescope/instrument/filter/detector efficiency of 30%? 
    
    351 ~ star2photons('V', 20, airmass=1, k = 0.0) * (4.2/2)**2 * np.pi * 0.3 * (vega_zero_points['d_lambda']['V']*1e3)
    
    for Baldr WFS - how many photons/s for Hmag=10 on ATs assuming 1% throughput, 1.3 airmass with extinction coefficient=0.18?
    
    star2photons('H', 10, airmass=1.3, k = 0.18) * (1.8/2)**2 * np.pi * 0.01 * vega_zero_points['d_lambda']['H']*1e3
    Out[246]: 5550 photons/s
    """
    
    return(ph_flux) 


def get_r0(seeing, wvl):
    r0 = 0.98 * wvl / np.deg2rad( seeing * 1 / 3600 )
    return(r0)

def fit_var(D, r0, J):
    """
    from Glindemann (1999) "Adaptive Optics on Large Telescopes"

    Parameters
    ----------
    D : TYPE
        DESCRIPTION. Telescope diameter 
    r0 : TYPE
        DESCRIPTION. Fried paramaeter (at WFS wvl)
    J : TYPE
        DESCRIPTION. Number of modes corrected by AO system

    Returns
    -------
    phase residual variance (rad)

    """
    
    
    sigma2 = 0.2944 * J**(-np.sqrt(3)/2) * (D/r0)**(5/3)
    return(sigma2)


def aniso_var(theta, theta_0):
    """
    

    Parameters
    ----------
    theta : TYPE
        DESCRIPTION. Angle from off-axis guide star
    theta_0 : TYPE
        DESCRIPTION. isoplanatic angle 

    Returns
    -------
    phase residual variance (rad)

    """
    
    sigma2 = (theta/theta_0)**(5/3)

    return(sigma2)


def servo_var(tau_BW, tau_0):
    """
    from Glindemann (1999) "Adaptive Optics on Large Telescopes"

    Parameters
    ----------
    tau_BW : TYPE
        DESCRIPTION. 1/f_3dB where f_3dB is 3dB freq 
    tau_0 : TYPE
        DESCRIPTION. atmospheric coherence time 

    Returns
    -------
    phase residual variance (rad)

    """
    
    sigma2 = (tau_0/tau_BW)**(5/3)
    
    return(sigma2) 
    

def phot_var(N_ph, r0, d, wvl, diffraction_limited=True):
    """
    from Glindemann (1999) "Adaptive Optics on Large Telescopes"

    Parameters
    ----------
    N_ph : TYPE
        DESCRIPTION. #number of photons per sub-aperture for WFS in integration time 
    r0 : TYPE
        DESCRIPTION. Fried parameter (m)
    d : TYPE
        DESCRIPTION. subpixel size in meters (in pupil plane)
    wvl : TYPE
        DESCRIPTION. wavelength of WFS in meters

    Returns
    -------
    phase residual variance (rad)

    """
    
    if hasattr(r0, "__len__"):
        alpha0 = np.array( [wvl/d if r_0 > d else wvl/r_0 for r_0 in r0 ] )
    else:
        if r0 > d: # diffraction limited in sub-aperture
            alpha0 = wvl/d
        else:
            alpha0 = wvl/r0
    
    sigma2 = np.pi**2/2 * 1/N_ph * (alpha0 * d / wvl )**2

    return(sigma2)



def read_var(  pix_per_subAp, plate_scale, sigma_r, N_ph, d, r0, wvl):
    """
    from Glindemann (1999) "Adaptive Optics on Large Telescopes"

    Parameters
    ----------
    pix_per_subAp : TYPE
        DESCRIPTION. pixels per sub-aperature in the SH-WFS
    
    plate_scale_rad: TYPE
        DESCRIPTION. WFS plate sclae (arcsec/pix)
    
    sigma_r : TYPE
        DESCRIPTION. std of read out noise OR #background photons
    
    N_ph : TYPE
        DESCRIPTION. #number of photons for WFS in integration time
    d : TYPE
        DESCRIPTION. subpixel size in meters (in pupil plane)
    r0 : TYPE
        DESCRIPTION. 

    Returns
    -------
    phase residual variance (rad)

    """
    
    plate_scale_rad = np.deg2rad( plate_scale /3600 )
    
    if hasattr(r0, "__len__"):
        
        
        sigma2= []
        #print(sigma2)
        for ro in r0:
            
            if d > ro:
                
                #qR = 4/np.pi * ((pix_per_subAp * plate_scale_rad) / (wvl/ro) )**2 #the quotient between the area on the detector used for the centroid calculation, and the area of the seeing disk.
            
                N2_alpha0 =  np.pi* (wvl /(2*d * plate_scale_rad))**2  # total number of pixels per Airy disk
                
                #sigma2_i = np.pi**2 / 3 * ( N_alpha0 * qR )**2 * ( sigma_r/N_ph )**2 * ( d/ro)**4
                
                sigma2_i  = np.pi**2 / 3 * ( pix_per_subAp / N2_alpha0 )**2 * ( sigma_r/N_ph )**2  * ( d/ro )**4  # ADAPTIVEOPTICS IN ASTRONOMY 1999 (edited by Roddier) Ch 5 Wave-front sensors (by GEÂRARD ROUSSET)
                #print(aaa, ( d/ro )**4) 
                #print(aaa)
                sigma2.append(  sigma2_i  )
                #print(d,ro, 'd>ro', d>ro, sigma2_i  )
                
            else: #diffraction limited in sub-aperture
                #qR = 4/np.pi * ((pix_per_subAp * plate_scale_rad) / (wvl/d) )**2 #the quotient between the area on the detector used for the centroid calculation, and the area of the seeing disk.
                
                N2_alpha0 =  np.pi* (wvl /(2*d * plate_scale_rad))**2  # total number of pixels per Airy disk
    
                #sigma2_i = np.pi**2 / 3 * ( N_alpha0 * qR )**2 * ( sigma_r/N_ph )**2 
                sigma2_i = np.pi**2 / 3 * ( pix_per_subAp / N2_alpha0 )**2 * ( sigma_r/N_ph )**2 
                #print('hello',sigma2_i)
                sigma2.append(  sigma2_i )
                #print(d,ro, 'd>ro', d>ro, sigma2_i)
            
        sigma2 = np.array( sigma2 )
    """   
    else: # 
    
        if d > r0:
            
            qR = 4/np.pi * ((pix_per_subAp * plate_scale_rad) / (wvl/r0) )**2 #the quotient between the area on the detector used for the centroid calculation, and the area of the seeing disk.
        
            N2_alpha0 =  np.pi* (wvl /(2*r0 * plate_scale_rad))**2  # total number of pixels per Airy disk
            
            #sigma2 = np.pi**2 / 3 * ( N_alpha0 * qR )**2 * ( sigma_r/N_ph )**2 * ( d/r0 )**4
            sigma2 = np.pi**2 / 3 * ( pix_per_subAp / N2_alpha0 )**2 * ( sigma_r/N_ph )**2 * ( d/r0 )**4
        else: #diffraction limited in sub-aperture
            qR = 4/np.pi * ((pix_per_subAp * plate_scale_rad) / (wvl/d) )**2 #the quotient between the area on the detector used for the centroid calculation, and the area of the seeing disk.
        
            N2_alpha0 =  np.pi* (wvl /(2*d * plate_scale_rad))**2  # total number of pixels per Airy disk

            #sigma2 = np.pi**2 / 3 * ( N_alpha0 * qR )**2 * ( sigma_r/N_ph )**2 
            sigma2 = np.pi**2 / 3 * ( pix_per_subAp / N2_alpha0 )**2 * ( sigma_r/N_ph )**2 
    """
    return(sigma2)



def test(a,b):
    dick=[]
    for aa in a:
        if aa>b:
            dick.append(100)
        else:
            dick.append(1)
    return(dick)
    
def noise_var(simga2_r, simga2_bg, sigma2_phot, J):
    """
    from Glindemann (1999) "Adaptive Optics on Large Telescopes"

    Parameters
    ----------
    simga2_r : TYPE
        DESCRIPTION. the read noise variance (rad) calculated from read_var(..) function
    simga2_bg : TYPE
        DESCRIPTION. the photon background noise variance (rad) calculated from read_var(..) function (yes, the same as sigma2_r)
    sigma2_phot : TYPE
        DESCRIPTION. the photon shot noise variance (rad) calculated from phot_var(..) function
    J : TYPE
        DESCRIPTION. Number of modes corrected by AO system

    Returns
    -------
    phase residual variance (rad)

    """

    Pj = 0.34 * np.log(J) + 0.10 
    
    sigma2 = Pj * (simga2_r + simga2_bg + sigma2_phot)

    return(sigma2)



#%% ASM DATA between 2022-01-01 - 2023-06-25 

asm_data = pd.read_csv('/Users/bcourtne/Documents/ANU_PHD2/heimdallr/data/paranal_asm_data_2022-01-01-2023-06-25.csv') 
"""
print( asm_data.columns )

Index(['Date time', 'Free Atmosphere Seeing ["]', 'Free Atmosphere Seeing RMS',
       'MASS Tau0 [s]', 'MASS Tau0 RMS', 'MASS Theta0 ["]', 'MASS Theta0 RMS',
       'MASS Turb Altitude [m]', 'MASS Turb Altitude RMS',
       'MASS-DIMM Cn2 fraction at ground', 'MASS-DIMM Seeing ["]',
       'MASS-DIMM Tau0 [s]', 'MASS-DIMM Theta0 ["]',
       'MASS-DIMM Turb Altitude [m]', 'MASS-DIMM Turb Velocity [m/s]'],
      dtype='object')
"""

asm_filtered = asm_data[['MASS-DIMM Seeing ["]','MASS-DIMM Tau0 [s]', 'MASS-DIMM Theta0 ["]']]
asm_filtered['MASS-DIMM Tau0 [s]'] = 1e3 * asm_filtered['MASS-DIMM Tau0 [s]']
asm_filtered.columns = ['MASS-DIMM Seeing ["]','MASS-DIMM Tau0 [ms]', 'MASS-DIMM Theta0 ["]']

figure = corner.corner(asm_filtered.dropna(axis=0), quantiles=[0.16, 0.5, 0.84],range=[(0, 3), (0, 20), (0, 8)],\
                       labels=['MASS-DIMM Seeing ["]','MASS-DIMM Tau0 [ms]', 'MASS-DIMM Theta0 ["]'])
#figure.savefig("/Users/bcourtne/Documents/ANU_PHD2/heimdallr/data/paranal_atmosphere_corner_plot.jpeg", dpi=300)


kde1 = KernelDensity(bandwidth=0.1).fit( asm_filtered.dropna(axis=0) )
"""
to check it matches use 
corner.corner( kde1.sample(10000) )
# check for example no zero values for sanity check ( np.sum(  kde1.sample(10000) < 0 ) ) 
"""


#%% SIMULATING VLTI FIRST STAGE AO PERFORMANCE 

mode = 'GPAO,NGS,ON-AXIS(HO)' 

AO_setup_dict = { 'GPAO,NGS,ON-AXIS(HO)' : { 'wvl': 0.6e-6, 'd_wvl':0.2e-6, 'D':1.8, 'subAperature':1240, 'pix_per_subAp':6*6 ,'plate_scale':0.42  , 'J':120, 'tau_BW': 2  }}

locals().update(AO_setup_dict[mode] )


wvl_baldr = 1.65e-6 #m

Rmag = 8
ron = 1 #e-
thermal_bg = 1 # need to calculate this properly, what is mag/arcsec^2/s? mult by pixel scale etc then use star2photons
wfs_dit = 1e-3  # how does this reate to tau_BW?

FF = 0.3 # fill factor of SH-WFS lenslet
d = FF * np.sqrt( np.pi*(D/2)**2/subAperature ) # do calculation in telescope pupil 

N_ph = star2photons('R', mag=Rmag, airmass=1, k = 0.18, ph_m2_s_nm = True) * (np.pi*(D/2)**2) * (1e9*d_wvl) * (wfs_dit)

N_ph_per_subAp = star2photons('R', mag=Rmag, airmass=1, k = 0.18, ph_m2_s_nm = True) * (np.pi*(d/2)**2) * (1e9*d_wvl) * (wfs_dit)

atm_samples =  kde1.sample(10000)
seeing = atm_samples[:,0]
tau0 =  atm_samples[:,1]
theta0 = atm_samples[:,2]



r0=get_r0(seeing, wvl)
theta = 1 # off axis distance (arcsec).. we could draw from some distribution

sigma2_contr_dict = {'sigma2_fit':fit_var(D, r0, J) , \
                'sigma2_ansio':aniso_var(theta, theta0 ), \
                'sigma2_bw':servo_var(tau_BW, tau0 ) ,\
                'sigma2_ph':phot_var(N_ph_per_subAp , r0, d, wvl, diffraction_limited=True ),\
                'sigma2_bg':read_var(  pix_per_subAp, plate_scale, ron, N_ph_per_subAp, d, r0 ,wvl),\
                'sigma2_ron':read_var(  pix_per_subAp=pix_per_subAp, plate_scale=plate_scale, \
                                      sigma_r=thermal_bg, N_ph=N_ph_per_subAp, d=d, r0=r0, wvl=wvl ) }
pix_per_subAp, plate_scale, sigma_r, N_ph, d, r0, wvl

sigma2_total = np.zeros( sigma2_contr_dict[list(sigma2_contr_dict.keys())[0]].shape[0] )
plt.figure()
for k,v in sigma2_contr_dict.items():
    print( k, np.mean(v) )
    
    sigma2_total = sigma2_total + v
    
    plt.figure()
    plt.hist(v,label=k)
    plt.legend()
    #plt.plot( v , label=k, alpha=0.1)
    
#plt.plot( np.sqrt(sigma2_total), color='k' )


 