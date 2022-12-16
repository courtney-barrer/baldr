#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 03:30:37 2022

@author: bcourtne



"""



import pyzelda.zelda as zelda
import pyzelda.ztools as ztools
import pyzelda.utils.aperture as aperture
import pyzelda.utils.zernike as zernike
import pyzelda.utils.mft as mft

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


def nglass(l, glass='sio2'):
    """
    (From Mike Irelands opticstools!)
    Refractive index of fused silica and other glasses. Note that C is
    in microns^{-2}
    
    Parameters
    ----------
    l: wavelength 
    """
    try:
        nl = len(l)
    except:
        l = [l]
        nl=1
    l = np.array(l)
    if (glass == 'sio2'):
        B = np.array([0.696166300, 0.407942600, 0.897479400])
        C = np.array([4.67914826e-3,1.35120631e-2,97.9340025])
    elif (glass == 'air'):
        n = 1 + 0.05792105 / (238.0185 - l**-2) + 0.00167917 / (57.362 - l**-2)
        return n
    elif (glass == 'bk7'):
        B = np.array([1.03961212,0.231792344,1.01046945])
        C = np.array([6.00069867e-3,2.00179144e-2,1.03560653e2])
    elif (glass == 'nf2'):
        B = np.array( [1.39757037,1.59201403e-1,1.26865430])
        C = np.array( [9.95906143e-3,5.46931752e-2,1.19248346e2])
    elif (glass == 'nsf11'):
        B = np.array([1.73759695E+00,   3.13747346E-01, 1.89878101E+00])
        C = np.array([1.31887070E-02,   6.23068142E-02, 1.55236290E+02])
    elif (glass == 'ncaf2'):
        B = np.array([0.5675888, 0.4710914, 3.8484723])
        C = np.array([0.050263605,  0.1003909,  34.649040])**2
    elif (glass == 'mgf2'):
        B = np.array([0.48755108,0.39875031,2.3120353])
        C = np.array([0.04338408,0.09461442,23.793604])**2
    elif (glass == 'npk52a'):
        B = np.array([1.02960700E+00,1.88050600E-01,7.36488165E-01])
        C = np.array([5.16800155E-03,1.66658798E-02,1.38964129E+02])
    elif (glass == 'psf67'):
        B = np.array([1.97464225E+00,4.67095921E-01,2.43154209E+00])
        C = np.array([1.45772324E-02,6.69790359E-02,1.57444895E+02])
    elif (glass == 'npk51'):
        B = np.array([1.15610775E+00,1.53229344E-01,7.85618966E-01])
        C = np.array([5.85597402E-03,1.94072416E-02,1.40537046E+02])
    elif (glass == 'nfk51a'):
        B = np.array([9.71247817E-01,2.16901417E-01,9.04651666E-01])
        C = np.array([4.72301995E-03,1.53575612E-02,1.68681330E+02])
    elif (glass == 'si'): #https://refractiveindex.info/?shelf=main&book=Si&page=Salzberg
        B = np.array([10.6684293,0.0030434748,1.54133408])
        C = np.array([0.301516485,1.13475115,1104])**2
    #elif (glass == 'zns'): #https://refractiveindex.info/?shelf=main&book=ZnS&page=Debenham
    #    B = np.array([7.393, 0.14383, 4430.99])
    #    C = np.array([0, 0.2421, 36.71])**2
    elif (glass == 'znse'): #https://refractiveindex.info/?shelf=main&book=ZnSe&page=Connolly
        B = np.array([4.45813734,0.467216334,2.89566290])
        C = np.array([0.200859853,0.391371166,47.1362108])**2
    elif (glass == 'noa61'):
        n = 1.5375 + 8290.45/(l*1000)**2 - 2.11046/(l*1000)**4
        return n
    elif (glass == 'su8'):
        n = 1.5525 + 0.00629/l**2 + 0.0004/l**4
        return n
    elif (glass == 'epocore'):
        n = 1.572 + 0.0076/l**2 + 0.00046/l**4
        return n
    elif (glass == 'epoclad'):
        n = 1.560 + 0.0073/l**2 + 0.00038/l**4
        return n

    else:
        print("ERROR: Unknown glass {0:s}".format(glass))
        raise UserWarning
    n = np.ones(nl)
    for i in range(len(B)):
            n += B[i]*l**2/(l**2 - C[i])
    return np.sqrt(n)


def AT_pupil(dim, diameter, spiders_thickness=0.008, strict=False, cpix=False):
    '''Auxillary Telescope theoretical pupil with central obscuration and spiders
    
    function adapted from pyzelda..
    
    
    Parameters
    ----------
    dim : int
        Size of the output array
    
    diameter : int
        Diameter the disk
    spiders_thickness : float
        Thickness of the spiders, in fraction of the pupil
        diameter. Default is 0.008
    spiders_orientation : float
        Orientation of the spiders. The zero-orientation corresponds
        to the orientation of the spiders when observing in ELEV
        mode. Default is 0
    dead_actuators : array
        Position of dead actuators in the pupil, given in fraction of
        the pupil size. The default values are for SPHERE dead
        actuators but any other values can be provided as a Nx2 array.
    dead_actuator_diameter : float
        Size of the dead actuators mask, in fraction of the pupil
        diameter. This is the dead actuators of SPHERE. Default is
        0.025
    strict : bool optional
        If set to Trye, size must be strictly less than (<), instead of less
        or equal (<=). Default is 'False'
    
    cpix : bool optional
        If set to True, the disc is centered on pixel at position (dim//2, dim//2).
        Default is 'False', i.e. the disc is centered between 4 pixels
    
    Returns
    -------
    pup : array
        An array containing a disc with the specified parameters
    '''

    # central obscuration (in fraction of the pupil)
    obs  = 0.13/1.8
    spiders_orientation = 0

    pp1 = 2.5
    # spiders
    if spiders_thickness > 0:
        # adds some padding on the borders
        tdim = dim+50

        # dimensions
        cc = tdim // 2
        spdr = int(max(1, spiders_thickness*dim))
            
        ref = np.zeros((tdim, tdim))
        ref[cc:, cc:cc+spdr] = 1
        spider1 = aperture._rotate_interp(ref, -pp1 , (cc, cc+diameter/2))

        ref = np.zeros((tdim, tdim))
        ref[:cc, cc-spdr+1:cc+1] = 1
        spider2 = aperture._rotate_interp(ref, -pp1 , (cc, cc-diameter/2))
        
        ref = np.zeros((tdim, tdim))
        ref[cc:cc+spdr, cc:] = 1
        spider3 = aperture._rotate_interp(ref, pp1 , (cc+diameter/2, cc))
        
        ref = np.zeros((tdim, tdim))
        ref[cc-spdr+1:cc+1, :cc] = 1
        spider4 = aperture._rotate_interp(ref, pp1 , (cc-diameter/2, cc))

        spider0 = spider1 + spider2 + spider3 + spider4

        spider0 = aperture._rotate_interp(spider1+spider2+spider3+spider4, 45+spiders_orientation, (cc, cc))
        
        spider0 = 1 - spider0
        spider0 = spider0[25:-25, 25:-25]
    else:
        spider0 = np.ones(dim)

    # main pupil
    pup = aperture.disc_obstructed(dim, diameter, obs, diameter=True, strict=strict, cpix=cpix)

    # add spiders
    pup *= spider0

    return (pup >= 0.5).astype(int)




def calibrate_phase_screen2wvl(wvl, screen):
    """
    

    Parameters
    ----------
    wvl : float 
        wavelength (m) to adjust phase using lambda^(5/6) scaling 
    screen : aotools.turbulence.infinitephasescreen.PhaseScreenVonKarman type calibrated with r0(wvl=500nm)
        DESCRIPTION.

    Returns
    -------
    list of adjusted phase screens

    """
    # turbulence gets better for longer wavelengths 
    adjusted_screen = (500e-9/wvl)**(6/5) * screen.scrn #avoid long lists for memory 
    
    return(adjusted_screen) 
  



def AO_correction( pupil, screen, wvl_0, wvl, n_modes, lag, V_turb, return_in_radians=True, troubleshoot=True):
    """
    

    Parameters
    ----------
    pupil: 2D array 
        describing transmission of telescope pupil (typically real between 0-1).. should be <= size of input phase screen
    screen: aotools.turbulence.infinitephasescreen.PhaseScreenVonKarman type 
        Kolmogorov phase screen (radians) (should be calibrated with r0(wvl=500nm) )
    wvl_0: float 
        central wavelength (m) to calculate AO correction at
    wvl: float 
        wavelength (m) of phase screen to apply AO correction to to 
    n_modes: int
        how many Zernike modes does the Ao system correct?
    lag: float
        what is the latency of the AO system (s)
    V_turb: float
        wind speed of phase screen (m/s)
    

    Returns
    -------
    

    """
    
    if pupil.shape[0] > screen.scrn.shape[0]:
        raise ValueError('pupil is bigger then input phase screen')
    
    
    basis = zernike.zernike_basis(n_modes, npix=pupil.shape[0])

    dx = screen.pixel_scale

    V_pix = V_turb / dx # how many pixels to jump per second 
    
    #AO_lag = 9.5 * 4.6e-3 # full loop delay of AO system between measurement and correction (s) - Woillez AA 629, A41 (2019) 
    
    # how many pixels to jump between AO measurement and correction 
    jumps = int( lag * V_pix )
    
    # screen measured by wfs at central wavelength wvl_0
    then_opd = wvl_0/(2*np.pi) * pupil * crop2center( calibrate_phase_screen2wvl(wvl_0, screen) , pupil ) # crop phase_screen to pupil size (ensuring we can multiply the two)
    
    #forcefully remove piston 
    then_opd[pupil > 0] -= np.nanmean( then_opd[pupil > 0] )
    
    #the measured coeeficients on Zernike basis
    then_coe = zernike.opd_expand(then_opd, nterms=n_modes, basis=zernike.zernike_basis)
    
    #reconstruct DM shape from measured coeeficients 
    DM_opd = np.sum( basis*np.array(then_coe)[:,np.newaxis, np.newaxis] , axis=0) 
            
    #forcefully remove piston 
    DM_opd[basis[0] > 0] -= np.nanmean(DM_opd[basis[0]>0]) #DM_opd[pupil > 0] -= np.nanmean(DM_opd[basis[0]>0])
          
    # propagate phase screen determined by wind velocity and AO latency 
    for j in range(jumps):
        # propagate 
        screen.add_row()
    
    # get the current OPD screen AT wavelength  of interest (wvl) after waiting for AO lag 
    now_opd = wvl/(2*np.pi) * pupil * crop2center( calibrate_phase_screen2wvl(wvl, screen) , pupil )
    
    # again, remove piston 
    now_opd[pupil > 0] -= np.nanmean(now_opd[pupil > 0])
    
    # get the corrected OPD at wavelength = wvl (DM_opd is calculated at wvl_0)
    corrected_opd = pupil * (now_opd - DM_opd) #m
    
    
    if troubleshoot:
        #==== additional calculations 
        
        #the new  coeeficients on Zernike basis
        now_coe = zernike.opd_expand(now_opd, nterms=n_modes, basis=zernike.zernike_basis)
        #the deformable mirrors  coeeficients on Zernike basis
        DM_coe = zernike.opd_expand(DM_opd, nterms=n_modes, basis=zernike.zernike_basis)
        #the corrected phase screen Zernike 
        corrected_coe = zernike.opd_expand(corrected_opd, nterms=n_modes, basis=zernike.zernike_basis)
        # ==== to hold variables for trouble shooting 
        diagnostics_dict={'now_opd':now_opd,'then_opd':then_opd,'DM_opd':DM_opd, \
                         'now_coe':now_coe, 'then_coe':then_coe,'DM_coe':DM_coe, \
                         'corrected_opd':corrected_opd,'corrected_coe':corrected_coe,\
                             'jumps':jumps,'V_pix':V_pix}
            
        return(diagnostics_dict)
    
    else:
        # I can use zernike.opd_expand(opd, nterms=n_modes, basis=zernike.zernike_basis) to get residuals later 
        if return_in_radians:
            
            return( 2*np.pi/wvl * corrected_opd) # , 2*np.pi/wvl * DM_opd, 2*np.pi/wvl * then_opd) )
        
        else:
            
            return( corrected_opd ) #, DM_opd, then_opd) )
        




"""def run_naomi_simulation( parameter_dict = {pupil, screen, wvl_0, wvl, n_modes, lag, V_turb}):
    
     pupil, screen, wvl_0, wvl, n_modes, lag, V_turb
     
     return_in_radians=True,troubleshoot=True"""
     
     


def crop2center(a,b):
    """
    
    crops array 'a' to size of array 'b' in centerput

    Parameters
    ----------
    a : 2d array
    b : 2d array 

    Returns
    -------
    cropped a

    """
    a_cropped = a[a.shape[-1]//2-b.shape[-1]//2 : a.shape[-1]//2+b.shape[-1]//2 , a.shape[-1]//2-b.shape[-1]//2 : a.shape[-1]//2+b.shape[-1]//2 ]
    return( a_cropped )



def putinside_array(a,b):
    """
    overwrite the center of array a with b (but b in a)

    Parameters
    ----------
    a : 2d array
    b : 2d array 

    Returns
    -------
    b centered in a
    """
    #a=a.copy()
    a[a.shape[-1]//2-b.shape[-1]//2 : a.shape[-1]//2+b.shape[-1]//2 , a.shape[-1]//2-b.shape[-1]//2 : a.shape[-1]//2+b.shape[-1]//2 ] = b
    
    return(a)



def naomi_simulation(parameter_dict, sim_time = 0.01, wvl_list= [1.65e-6], save = False, saveAs = None):
    
    """
    simulate AO systemfor sim_time (generally used for simulating naomi) 

    Parameters
    ----------
    parameter_dict : dictionary 
        has to have format like this: (you can replace values as you widsh)
        #-----
        parameter_dict = {'seeing':1 , 'L0':25, 'D':1.8, 'D_pix':2**8, \
                        'wvl_0':0.6e-6, 'wvl':1.65e-6,'n_modes':14,\
                            'lag':3e-6,'V_turb':50}
        #-----
    sim_time  : float
        how long to simulate for (s)
    wvl_list : list / array
        list of wavelengths to calculate the corrected phase screen
    save : boolean 
        do you want to save the output fits files?
        
    where2save : string or None
        if save == True give the path+file_name of where we should save
        e.g. '/Users/bcourtne/Documents/my_AO_sim.fits
        
    Returns
    -------
    fits file with data cube of AO correctied phase screens (in radians) calculated at given wvl
    """
    
    
    #convert key strings to variable names
    #for variable, value in naomi_parameter_dict.items():
    #    exec("{} = {}".format( variable, value ) )
    
    #convert dictionary keys to variable names (explicitly)
    seeing = parameter_dict['seeing']
    L0 = parameter_dict['L0']
    D = parameter_dict['D']
    D_pix = parameter_dict['D_pix']
    wvl_0 = parameter_dict['wvl_0']
    #wvl = parameter_dict['wvl']
    n_modes = parameter_dict['n_modes']
    lag = parameter_dict['lag']
    V_turb = parameter_dict['V_turb']
    pupil_geometry = parameter_dict['pupil_geometry']
    desired_dt = parameter_dict['dt'] # desired simulation sampling rate (should be exact if dt is multiple of lag)
    
    
    #parameters derived from input variables
    dx = D/D_pix  # diff element (pixel size) pupil plane (m)
    r0 = 0.98 * 500e-9 / np.radians(seeing/3600) #Fried parameter defined at 500nm
    
    #how far will it travel in sim_time 
    distance = V_turb * sim_time
    distance_pix = int(distance / dx) # convert to pixels
    
    V_pix = V_turb / dx # how many pixels to jump per second 
    jumps = int( lag * V_pix ) #how many pixels are jumped per AO iteration 
    
    if jumps>0:
        no_iterations = distance_pix // jumps # how many 
    else:
        no_iterations = distance_pix
        
    
    phase_screen = aotools.turbulence.infinitephasescreen.PhaseScreenVonKarman(D_pix, pixel_scale=dx,\
          r0=r0 , L0=L0, n_columns=2,random_seed = 1) # radians
    
    # get the telescope pupil
    pup = pick_pupil(pupil_geometry, dim=D_pix, diameter=D_pix )
       
    # ======
    # apply AO correction and produce phase screen for each AO iteration (lag)
    naomi_screens = [ np.nan * np.zeros((no_iterations, pup.shape[0],pup.shape[1])) for w in wvl_list]
    
    for i in range(no_iterations):
        
        for jjj in range(jumps): # these jumps are don in AO_correction() but only on copy of phase screen (because we do it for each ewvl) .. so we apply them here outside of wvl loop
            phase_screen.add_row()

        for w, wvl in enumerate(wvl_list):
            
            # Very important that we send a copy of phase screen to this function ..and don't roll the master phase screen for every iteration (because then we have different shifts per wvl)
            naomi_screens[w][i] = AO_correction( pupil=pup,  screen= copy.copy( phase_screen ), wvl_0=wvl_0, wvl=wvl, \
                                                n_modes=n_modes, lag=lag, V_turb=V_turb, return_in_radians=True, troubleshoot=False )
            
            #naomi_screens.append( AO_correction( pup, phase_screen, wvl_0, wvl, n_modes, lag, V_turb, return_in_radians=True, troubleshoot=False ) )
        

    # ======
    # To get near the desired sampling we linearly interpolate between pixels on our phase screens  
    screens_between = int( round( lag / desired_dt ) ) # how many frames between lagged steps?
    dt = lag / screens_between 

    
    if screens_between > 1: # if its less then this we don't need to interpolate
        # warning if interpolating will be much longer then tau0
        if lag > 10 * r0 / V_turb :
            print('\n\n=== WARNING:\n AO lag much greeater then tau0, linear interpolation for smaller sampling times may not be valid')
        
        for w, wvl in enumerate(wvl_list):
            
            naomi_screens_interp = []
            
            for i in range( len(naomi_screens[w]) - 1 ):
            
                naomi_screens_interp = naomi_screens_interp + list( np.linspace( naomi_screens[w][i], naomi_screens[w][i+1], screens_between)[:-1] )

            # we update the screen at wvl index to the interpolated version
            naomi_screens[w] = naomi_screens_interp
        
        
        
    #else:
    #    naomi_screens_interp[w] = naomi_screens

    simulation_fits = fits.HDUList() 
    for w, wvl in enumerate(wvl_list):


        hdu = fits.PrimaryHDU( naomi_screens[w] )
        
        hdu.header.set('what is', 'AO corrected phase screens (rad)' , 'at lambda = {}um'.format(round(wvl*1e6,3)))
        hdu.header.set('simulation time', sim_time, 'simulation time (s)')
        hdu.header.set('dt', dt, 'phase screen sampling rate')
        hdu.header.set('wvl', wvl, 'wvl (m) where currrent phase screen is calc. ')
        
        hdu.header.set('seeing', seeing, 'arcsec')
        hdu.header.set('L0', L0, 'outerscale (m)')
        hdu.header.set('V_turb', V_turb, 'velocity of phase screen (m/s)')
        
        hdu.header.set('pupil_geometry', pupil_geometry, 'Valid options are AT, UT, disk')
        hdu.header.set('D', D, 'telescope diameter (m)')
        hdu.header.set('D_pix', D_pix, '#pixels across telescope diameter')
        
        hdu.header.set('NAOMI wvl_0', wvl_0, 'central wavelentgth of WFS (m)')
        hdu.header.set('NAOMI n_modes', n_modes, '#modes sensed & corrected by NAOMI')
        hdu.header.set('NAOMI lag', lag, 'latency of NAOMI (s)')
    
        simulation_fits.append( hdu )
    

    if save:
        simulation_fits.writeto(saveAs,overwrite=True)
        
    return( simulation_fits )




def baldr_simulation( naomi_screens , parameter_dict, save = False, saveAs = None):
    """
    Parameters
    ----------
    naomi_screens : fits file
        DESCRIPTION. fits file with timeseries of NAOMI corrected phase screens at various wavelengths
        (hint: naomi_screens should be the output of naomi_simulation() function )
        
        
    parameter_dict :   dictionary
    
        parameter_dict['baldr_min_wvl'] - float , minimum wavelength (m) that baldr wfs is sensitive to
        parameter_dict['baldr_max_wvl'] - float, maximum wavelength (m) that baldr wfs is sensitive to
        parameter_dict['baldr_wvl_0'] - float, primary wavelength (m) of baldr WFS where correction is optimized. This should be within baldr_min_wvl - baldr_max_wvl
        parameter_dict['f_ratio'] - float indicating the  f ratio of Zernike wfs 
        parameter_dict['DIT'] - float indicating the detector integration time of wfs  
        parameter_dict['processing_latency'] - float indicating the latency of baldr wfs 
        parameter_dict['RON'] - float, read out noise of baldr wfs detector
        
        parameter_dict['input_spectrum_wvls'] - array ike with wavelengths that input spectrum is calculated at 
        parameter_dict['input_spectrum_ph/m2/wvl/s']  - array ike with input spectrum 
        
        parameter_dict['target_phase_shift'] - float, the target phase shift (deg) to optimize phase mask depths for ()
        parameter_dict['T_off'] = float (0-1), phase mask off-axis transmission (outside phase shift region)
        parameter_dict['T_on'] = float (0-1), phase mask on-axis transmission (in phase shift region)
        parameter_dict['d_off'] = float, phase mask off-axis depth (m) 
        parameter_dict['d_on'] = float, phase mask on-axis depth (m) 
        parameter_dict['glass_off'] = string, material of off-axis region 
        parameter_dict['glass_on'] = string, material of on-axis region 
        parameter_dict['phase_mask_rad'] = float, radias of phase shift region at wfs central wvl (units = lambda/D) 
        parameter_dict['achromatic_diffraction'] = boolean, if all wavelengths diffract to the same lambda_0/D  (this is stufy the effect of wvl dependent spatial distribution of phase shift)
        [NOTE I SHOULD CHANGE phase_mask_rad TO PHYSICAL UNITS ]
        
        # parameters calculated based on input spectrum.:
        parameter_dict['redness_ratio']: float, ratio of number of photons above primary wvl (wvl_0) / number of photons below primary wvl (wvl_0) in wfs bandwidth
        parameter_dict['total_flux_in_wfs[ph/s/m2]']: float, total number of photons within 
        parameter_dict['spectral_classification_in_wfs'] : float 'flat'
            

    Returns
    -------
    fits file with timeseries of Baldr corrected phase screens at input phase screen wavelengths (defined in naomi_screens) fits file

    """

    input_screens = copy.deepcopy(naomi_screens) # to get parameters from 
    
    baldr_screens = copy.deepcopy(naomi_screens) # another copy to avoid editing original while assigning 
    
    wvl_0 = parameter_dict['baldr_wvl_0']  # primary wavelength (m) of baldr WFS where correction is optimized. Rhis should be within baldr_min_wvl - baldr_max_wvl
    
    input_wvls = np.array( [it.header['wvl'] for it in input_screens] )
    
    wvl_indx_4_wfs = [i for i,w in enumerate(input_wvls) if (w >= parameter_dict['baldr_min_wvl'] ) & (w <= parameter_dict['baldr_max_wvl'] ) ]
    
    wvl_0_indx = np.argmin(abs(wvl_0 - input_wvls))
    
    #abbreviated phase mask parameters 
    A = parameter_dict['T_off']
    B = parameter_dict['T_on']
    
    # time between sensing and actuation 
    lag = parameter_dict['DIT'] + parameter_dict['processing_latency'] #s
    
    # time steps between input screens (assume that it is sampled uniformly )
    dt = input_screens[0].header['dt']
    
    # number of frames to jump before applying correction 
    jumps = int(round(lag / dt)) 
        
    #init DM_cmd_list to hold previous DM cmds up to the total system lag 
    DM_cmd_list = list( np.zeros( jumps + 1 ) ) #(if jumps=0 list needs to be of len 1 to hold current DM command)
    
    for frame in range(input_screens[0].shape[0]):  # for each timestep (frame)
        #init output intensity to zero
        Ic = np.zeros((baldr_screens[0].header['D_pix'],baldr_screens[0].header['D_pix']))
        N_ph_T = 0
        
        for wvl_indx in wvl_indx_4_wfs:  #  need to filter for valid wvls where baldr sensor operates! 
            
            wvl_in = baldr_screens[wvl_indx].header['wvl']

            
            # input phase screen from *NAOMI* at current wvl
            phi = input_screens[wvl_indx].data[frame]
            phi[~np.isfinite(phi)] = 0 # mft can't deal with nan values  - so convert to zero
            #phi = putinside_array(np.zeros([nx_size, nx_size]), phi.copy()) #put inside new grid 
        
            
            #-------- these should not depend on wvl --------------------
            D = input_screens[wvl_indx].header['D']
            dx = D / input_screens[wvl_indx].header['D_pix'] #m/pixels
            f_ratio = parameter_dict['f_ratio']
                
            pup = pick_pupil(input_screens[wvl_indx].header['pupil_geometry'], dim=input_screens[wvl_indx].header['D_pix'], diameter=input_screens[wvl_indx].header['D_pix'] )
            #------------------------------------------------------------
            
            # create function to interpolate flux to current wavelength 
            flux_interp_fn = interp1d( parameter_dict['input_spectrum_wvls'], parameter_dict['input_spectrum_ph/m2/wvl/s']  )
            
            # interpolate to wvl from input photon flux and multiply by WFS DIT, differential (input) wvl element & telescope area to estimate N_ph in wvl bin
            N_ph_wvl = parameter_dict['throughput'] * flux_interp_fn( wvl_in ) * np.diff( input_wvls )[wvl_indx-1] * parameter_dict['DIT'] * ((D/2)**2 * np.pi)
            
            N_ph_T += N_ph_wvl # add this to total number of photons, this is input to the estimator 
            
            """
            physical radius (m) is conserved for phase mask (unless we have some chromatic lens.. we could keep radius constant lambda/D )
            f_ratio * D * wvl_0 / D 
            therefore scale parameter_dict['phase_mask_rad_at_wvl_0'] by wvl_0/wvl_in
            
            """
            
            if parameter_dict['achromatic_diffraction']:
                # this case is not physically realistic but can be used to study the impact of chromatic spatial distribution of phase shift 
                # can look at it as physical size of phase mask radius changes with wavelength to always be lambda/D
                phase_mask_rad = parameter_dict['phase_mask_rad_at_wvl_0'] 
                 
            elif not parameter_dict['achromatic_diffraction']:
                """
                # this should be default for realistic simulations 
                # sanity check : 
                    phase mask rad = n lambda/D at wvl_0 
                    => physical radius fixed: r = C * n * wvl_0 / D => at wvl : r(wvl_0) = r(wvl) = C * n * (wvl / D) * wvl_0/wvl   
                
                """
                phase_mask_rad =  wvl_0/input_screens[wvl_indx].header['wvl'] * parameter_dict['phase_mask_rad_at_wvl_0']  #TO DO:  make this function of wvl such that = constant in physical radius and parameter_dict['phase_mask_rad_at_wvl_0'] (lambda/D) at wvl_0
                
            else:
                raise TypeError('parameter_dict["achromatic_diffraction"] is not boolean')
            
            #What is phase shift of phase mask at wvl = wvl_in 
            theta = phase_mask_phase_shift( wvl_in , d_on = parameter_dict['d_on'], d_off=parameter_dict['d_off'], glass_on = parameter_dict['glass_on'], glass_off = parameter_dict['glass_off'])
        
            IC_dict = I_C_analytic(A, B, theta, phase_mask_rad, phi, pup, N_ph_wvl, dx, wvl_in, D, f_ratio, troubleshooting=True) # need to changephase_mask_rad to physical radius of scale properly with wvl  if we want chromatic (not lambda /D)
            
            # get b at the central wavlength
            if wvl_0_indx in wvl_indx_4_wfs:
                
                if wvl_indx == wvl_0_indx:
                    
                    b_est =  IC_dict['b'] 
                
            else:
                raise TypeError('central wavelength (wvl_0) not in defined wfs wavelength range (wvl_range_wfs)')
                
                
            # add intensity contribution from spectral bin 
            Ic += IC_dict['Ic'] # + noise 
            
            # Now add shot noise :
                
            # note  that input field calculated in I_C_analytic is normalized so that sum over every pupil pixel intensity = N_phot_wvl , 
            N_pix = np.nansum( pup ) #number of pixels in pupil 
            # average number of photons per pixel = N_ph_wvl/N_pix  (uniform illumination assumption)
            # therefore generate the a possion distribution (shot noise) with zero mean but var =  N_ph_wvl/N_pix  to add to output intesnity 
            # note that this shot noise assume perfect trasmission from input pupil to output detector.. 
            shot_noise = N_ph_wvl/N_pix - poisson.rvs( N_ph_wvl/N_pix, size= Ic.shape)  
            Ic += shot_noise
            # make sure there are no negative values 
            Ic[Ic<0] = 0 
            
            #resample to resize pixels 
            # --- TO DO - need to know how many pixels we reading out on 
            
            # after rebinning (if done) then we convert intensity to adu (integers)
            Ic_adu = Ic.astype(int)
            
        # now that we have our intensity across all wfs spectral channels we can estimate the phase at wvl_0(or wherever we estimated b and theta) - we could also consider estimating b for each spectral chanel and then averaging etc 
        phi_est = zelda_phase_estimator_1(Ic_adu, pup, N_ph_T, dx, A, B,  abs( b_est ), theta, exp_order=1) #radians 
        
        # convert phase estimate to DM OPD at wvl_0
        DM_cmd = wvl_0 / (2*np.pi) * phi_est   # NEED TO HOLD THIS IN A LIST I CAN POP IN
        
        # add new cmd to end of the list 
        DM_cmd_list = DM_cmd_list + [DM_cmd]
        
        # get rid of oldest cmd (first index)
        DM_cmd_list.pop(0) 
        
        #------ now to subtract DM from input phase screen at each wavelength 
        
        
        
        
        # I have to do this loop again bc 1st is to build DM_cmd from all wvl bins.. now to apply it to each wvl!
        for hdu in baldr_screens: # this is for each wavelength again 
            
            hdu.header.set('what is', 'Naomi+Baldr AO corrected phase screens (rad)' )   
            
            # to do - I should also update naomi hrelated headers to reflect clearly they relate to naomi !!
            hdu.header.set( 'BALDR wfs_wvl_min' , parameter_dict['baldr_min_wvl']  , ' minimum wavelength (m) in baldr wfs')
            hdu.header.set( 'BALDR wfs_wvl_max' , parameter_dict['baldr_max_wvl']  , ' maximum wavelength (m) in baldr wfs')
            hdu.header.set( 'BALDR wvl_0', parameter_dict['baldr_wvl_0'], 'primary wavelentgth of Baldr WFS (m) where correction is optimizzed')
            hdu.header.set( 'BALDR wvl_c', (parameter_dict['baldr_max_wvl']  +  parameter_dict['baldr_min_wvl'] )/2, 'central wavelentgth of Baldr WFS (m)')
            hdu.header.set( 'BALDR wfs_bandwidth' , round( 100 * 2*(parameter_dict['baldr_max_wvl']  -  parameter_dict['baldr_min_wvl'] ) / (parameter_dict['baldr_max_wvl']  +  parameter_dict['baldr_min_wvl'] ), 2) , 'baldr wfs badwidth [%]')
            
            hdu.header.set( 'BALDR f_ratio' , parameter_dict['f_ratio'] , ' f ratio of ZWFS')
            hdu.header.set( 'BALDR DIT' , parameter_dict['DIT'] , 'baldr wfs detector integration time (s)')
            hdu.header.set( 'BALDR latency' , parameter_dict['DIT'] , 'baldr latency after detector integration (s) ')
            hdu.header.set( 'BALDR RON' , parameter_dict['RON'] , 'baldr detector read out noise [e-]')
            hdu.header.set( 'BALDR throughput' , parameter_dict['throughput'] , '% of input flux that makes it to baldr wfs (assumes no wvl dependence)')
            
            hdu.header.set( 'T_off', parameter_dict['T_off'] , 'phase mask off-axis transmission (outside phase shift region)')
            hdu.header.set( 'T_on', parameter_dict['T_on'] , 'phase mask on-axis transmission (in phase shift region)')
            hdu.header.set( 'd_off', parameter_dict['d_off'] , 'phase mask off-axis depth (m) ')
            hdu.header.set( 'd_on' , parameter_dict['d_on'] , 'phase mask on-axis depth (m) ')
            hdu.header.set( 'glass_off', parameter_dict['glass_off'] , 'material of off-axis region in phase mask')
            hdu.header.set( 'glass_on', parameter_dict['glass_on'] , 'material of on-axis region in phase mask')
            
            hdu.header.set( 'target_phase_shift', parameter_dict['target_phase_shift'] , 'target phase shift (deg) when optimizing depths across bandpass')
            hdu.header.set( 'mean_phase_shift', parameter_dict['mean_phase_shift'] , 'mean phase mask phase shift (deg) within wfs bandwidth')
            hdu.header.set( 'rmse_phase_shift', parameter_dict['rmse'] , 'rmse of target vs real phase shift (deg) across wfs bandwidth')
            hdu.header.set( 'std_phase_shift', parameter_dict['std_phase_shift'] , 'std of phase mask phase shift (deg) within wfs bandwidth')
            
            hdu.header.set( 'phase_mask_rad', parameter_dict['phase_mask_rad_at_wvl_0'], 'radias of phase shift region at wfs wvl_0 (units = lambda/D) ')
            hdu.header.set( 'achromatic_diffraction', parameter_dict['achromatic_diffraction'] , 'if all wavelengths diffract to the same lambda_0/D (default=False)' )
            
            # parameters calculated based on input spectrum.:
            hdu.header.set( 'redness_ratio' , parameter_dict['redness_ratio'], 'ratio N_ph > wvl_0 / N_ph < wvl_0 in wfs bandwidth')
            hdu.header.set( 'total_flux_in_wfs', parameter_dict['total_flux_in_wfs[ph/s/m2]'], 'total photon flux (ph/s/m2) within wfs bandwidth')
            hdu.header.set( 'spectral_classification_in_wfs', parameter_dict['spectral_classification_in_wfs'] , 'spectral classification (flat, red, blue)')


            wavefront_opd = hdu.header['wvl'] / (2*np.pi) * hdu.data[frame] 
            if frame > jumps:
                hdu.data[ frame ] =  2*np.pi /hdu.header['wvl'] * ( wavefront_opd  -  DM_cmd_list[0] )  # len( DM_cmd_list ) = jumps + 1, with index 0 being the oldest 

    if save:
        baldr_screens.writeto(saveAs,overwrite=True)

    return(baldr_screens)







def pick_pupil(pupil_geometry, dim, diameter ):
        
    if pupil_geometry == 'AT':
        pup = AT_pupil(dim = dim, diameter = diameter) 
    elif pupil_geometry == 'UT':
        pup = aperture.vlt_pupil(dim = dim, diameter =  diameter, dead_actuator_diameter=0) 
    elif pupil_geometry == 'disk':
        pup = aperture.disc( dim = dim, size = diameter//2) 
    else :
        print('no valid geometry defined (try pupil_geometry == disk, or UT, or AT\nassuming disk pupil')
        pup = aperture.disc( dim = dim, size = diameter//2) 

    return(pup)






def get_plane_coord(nx_size, dx, wvl, D, f_ratio):
    
    """
    creates pupil and fourier plane coorindate system from input grid 

    Parameters
    ----------
    nx_size: int - #pixels
    dx: float - pupil plane pixel size (m)
    wvl : wavelength (m)
    D : telescope diamter (m)
    f_ratio : float -  f ratio 

    Returns
    -------
    x- pupil coordinates (m)
    f_x- focal plane coordinates(m)
    df_x - pixel size focal plane (m) 
    f_r- focal plane coordinates (rad)
    df_r - pixel size focal plane (rad) 
    no_lambdaD - # wvl/D in focal plane
    
    
    """
    D_pix = D/dx #(m)
    #focal length 
    foc_len = f_ratio * D 
    
    # pupil plane coordinates (m)
    x = np.linspace(-nx_size//2 * dx, nx_size//2 * dx, nx_size) 
    
    # focal plane coordinates (m)
    f_x = (foc_len * wvl) / (nx_size * dx) * np.arange(-nx_size //2, nx_size //2 ) # grid in focal plane (m)
    df_x = np.diff(f_x)[0] # differential element in focal plane (m)  
    
    # focal plane coordinate (rad) - remember the focal plane coordinates are definied by the telescope diameter (not grid size! )
    f_r = f_x / foc_len  #wvl / (nx_size * dx) * np.arange( -nx_size //2, nx_size //2 )   
    df_r = np.diff(f_x)[0]
    no_lambdaD = 1/2 * nx_size/D_pix * ( np.max( f_r ) - np.min( f_r )) / (wvl/D) # number of resolution elements across focal plane
    #no_lambdaD = (np.max(focal_plane_in_rad ) - np.min(focal_plane_in_rad )) / (wvl/D)
    
    return(x, f_x, df_x, f_r, df_r, no_lambdaD)




def get_phase_shift_region(phase_mask_rad, dx, D, nx_size, no_lambdaD) :
    """
    creates array defining phase shift region 

    Parameters
    ----------
    phase_mask_rad: float- radius of phase shift region in  #lambda/D units (i.e radius = 1 lambda /D => phase_mask_rad = 1 )
    dx: float - pupil plane pixel size (m)
    nx_size: int - #pixels
    D: float, telescope diameter (m)

    Returns
    -------
    phase_shift_region: 2D array defining phaseshift region 
    
    """
    
    D_pix = int( D//dx ) # number of pixels across telescope diameter 
    
    if phase_mask_rad  > 0:
        # phase shift region (number of pixels across radius of phase shifting region)
        f_r_pix = phase_mask_rad  *  1/2 * nx_size/D_pix * nx_size / no_lambdaD #int( np.round( phase_mask_rad * (wvl/D) / df_r  ) )
    
        # phase shift region     
        phase_shift_region = aperture.disc(nx_size, f_r_pix) 
        
    elif phase_mask_rad  == 0:
        phase_shift_region = np.zeros([nx_size,nx_size])
        
    else:
        raise ValueError('non positive phase mask radius')
    
    return(phase_shift_region)


def calc_b(Psi_A, dx, wvl, D, f_ratio, phase_mask_rad , plot=False):
    
    #grid
    nx_size = Psi_A.shape[0]
    
    # get coordinates in pupil and focal plane 
    x, f_x, df_x, f_r, df_r, no_lambdaD = get_plane_coord(nx_size, dx, wvl, D, f_ratio)
    
    #pixels across telescope pupil
    #D_pix = D//dx # 
    # field propagated to focal plane
    Psi_B = mft.mft( Psi_A, Psi_A.shape[-1], Psi_A.shape[-1], no_lambdaD, cpix=False) # (dx / df_x)  *  mft.mft( Psi_A, Psi_A.shape[-1], Psi_A.shape[-1], no_lambdaD, cpix=False)  #mft.mft( Psi_A, Dpup, np.int(2*freq_cutoff*sampling), 2*freq_cutoff ) 
    
    # phase shift region     
    phase_shift_region = get_phase_shift_region(phase_mask_rad, dx, D, nx_size, no_lambdaD) #aperture.disc(nx_size, f_r_pix) 
    
    if plot:
        plt.figure()
        plt.plot( f_r[nx_size//2-100:nx_size//2+100]  / (wvl/D), abs ( Psi_B  )[nx_size//2, nx_size//2-100:nx_size//2+100]/np.max(abs ( Psi_B  ) ) ,label='PSF')
        plt.plot( f_r[nx_size//2-100:nx_size//2+100] / (wvl/D), phase_shift_region[nx_size//2, nx_size//2-100:nx_size//2+100] ,label='phase shift region')
        plt.legend() 
    # b parameter 
    b = mft.imft( Psi_B * phase_shift_region,  Psi_A.shape[-1], Psi_A.shape[-1], no_lambdaD, cpix=False)  # (df_x / dx)  * mft.imft( Psi_B * phase_shift_region,  Psi_A.shape[-1], Psi_A.shape[-1], no_lambdaD, cpix=False) 
    
    return(b) 


def plot_phase_mask_region(pup, dx, wvl, D, f_ratio, phase_mask_rad,zoom=None):
    
    nx_size = pup.shape[0]
    # get coordinates in pupil and focal plane 
    x, f_x, df_x, f_r, df_r, no_lambdaD = get_plane_coord(nx_size, dx, wvl, D, f_ratio)
    
    D_pix = D/dx #(m)
    
    if zoom is None :
        zoom = int( 10 * nx_size//D_pix )
    

    # field propagated to focal plane
    Psi_B =  mft.mft( pup, nx_size, nx_size, no_lambdaD, cpix=False)  #mft.mft( Psi_A, Dpup, np.int(2*freq_cutoff*sampling), 2*freq_cutoff ) 
    
    # phase shift region     
    phase_shift_region = get_phase_shift_region(phase_mask_rad, dx, D, nx_size, no_lambdaD)
    
    plt.figure()
    plt.plot( f_r[nx_size//2-zoom:nx_size//2+zoom]  / (wvl/D), abs ( Psi_B  )[nx_size//2, nx_size//2-zoom:nx_size//2+zoom]/np.max(abs ( Psi_B  ) ) ,label='PSF')
    plt.plot( f_r[nx_size//2-zoom:nx_size//2+zoom] / (wvl/D),  phase_shift_region[nx_size//2, nx_size//2-zoom:nx_size//2+zoom] ,label='phase shift region')
    plt.legend() 
    plt.xlabel(r'radius ($\lambda$/D)')
    plt.xlabel(r'radius ($\lambda$/D)')
    plt.gca().get_yaxis().set_visible(False)
    plt.show()



def I_C_sim(A, B, theta, phase_mask_rad, phi, pup, N_ph, dx, wvl, D, f_ratio):
    """
    returns simulated output intensity from ZWFS 
    
    Parameters
    ----------
    A ~ float in [0,1] for phase mask transmittance in  outside phase shift region 
    B ~ float in [0,1] for phase mask transmittance in  phase shift region 
    theta ~ float in [0,2pi] for phase shift applied in phase shift region
    phase_mask_rad ~ radius of phase shift region ( number of "wvl/D"s )
    
    phi ~ 2d array defining input field phase screen (radians)
    pup ~ 2d array defining transmission in telescope pupil 
    N_ph ~ float defining number of input photons at input wvl
    dx ~ float indicating differential element (pixel size) in pupil plane (m)
    wvl ~ float indicating wavelength (m)
    D ~ telescope diameter (m)
    f_ratio ~ f ratio = foc_len/D 
    
    Returns
    -------
    theoretical output intensity 
    """
    
    # grid size     
    nx_size = pup.shape[0]
    
    # get coordinates in pupil and focal plane 
    x, f_x, df_x, f_r, df_r, no_lambdaD = get_plane_coord(nx_size, dx, wvl, D, f_ratio)
    
    
    
    # input field in pupil
    Psi_A = N_ph * pup * np.exp(1j * phi)
    
    # phase shift region     
    phase_shift_region = get_phase_shift_region(phase_mask_rad, dx, D, nx_size, no_lambdaD)
    
    #phase mask filter 
    H = A*(1 + (B/A * np.exp(1j * theta) - 1) * phase_shift_region  )  

    Psi_B = mft.mft(Psi_A, nx_size, nx_size, no_lambdaD, cpix=False)
    Psi_C = mft.imft( H * Psi_B , nx_size, nx_size, no_lambdaD, cpix=False) 
    
    
    return(Psi_A, Psi_B, Psi_C, H)



def I_C_analytic(A, B, theta, phase_mask_rad, phi, pup, N_ph, dx, wvl, D, f_ratio, troubleshooting=False):
    """
    returns theoretical output intensity from ZWFS 
    
    Parameters
    ----------
    A ~ float in [0,1] for phase mask transmittance in  outside phase shift region 
    B ~ float in [0,1] for phase mask transmittance in  phase shift region 
    theta ~ float in [0,2pi] for phase shift applied in phase shift region
    phase_mask_rad ~ radius of phase shift region ( number of "wvl/D"s )

    phi ~ 2d array defining input field phase screen (radians)
    N_ph ~ float defining number of input photons at wvl 
    pup ~ 2d array defining number of photons in telescope pupil 
    dx ~ float indicating differential element (pixel size) in pupil plane (m)
    wvl ~ float indicating wavelength (m)
    D ~ telescope diameter (m)
    f_ratio ~ f ratio = foc_len/D 
    
    Returns
    -------
    theoretical output intensity 
    """

    # grid size     
    nx_size = pup.shape[0]
    
    # get coordinates in pupil and focal plane 
    x, f_x, df_x, f_r, df_r, no_lambdaD = get_plane_coord(nx_size, dx, wvl, D, f_ratio)
    
    # input field in pupil
    Psi_A = pup * np.exp(1j * phi)
    
    #normalize so that sum over every pupil pixel intensity = N_phot (total number of photons)
    #therefore pupil_field(Nx,Ny) = number of photons within pixel (or equivilantly photons / m^2 considering pixel finite area dfX^2)
    Psi_A *= np.sqrt((N_ph/np.nansum(abs(Psi_A)*dx**2))) #* Psi_A * pup
    
    # b  parameter 
    b = calc_b(Psi_A, dx, wvl, D, f_ratio, phase_mask_rad )

    Psi_C = A * (Psi_A - b) + B * b * np.exp(1j * theta)
    
    I_C = abs(Psi_C)**2
    
    if troubleshooting:
        data_dict = {'Ic':I_C,'Psi_C':Psi_C, 'Psi_A':Psi_A,'b':b,'dx':dx,'df_x':df_x}
        return( data_dict )
    else:
        return(I_C)


def zelda_phase_estimator_1(Ic, pup, N_ph, dx, A, B, b, theta, exp_order=1):
    
    P = pup #  rename to make equations clearer 
    P = np.sqrt((N_ph/np.nansum(abs(pup)*dx**2)))  #normalized so integral P^2 = N_ph
    
    # soln to:  aa * phi**2 + bb * phi + cc == 0
    
    aa = b * (A**2 * P - A * B * P * np.cos(theta) )
    
    bb = 2 * A * b * B * P * np.sin(theta)
    
    cc =  -Ic + (A**2 * P**2 + b**2 * (A**2 + B**2 - 2 * A * B * np.cos(theta) ) +\
        2 * b * ( -A**2 * P  + A * B * P * np.cos(theta) ) ) 

    
    if exp_order == 1:
        phi = - cc / bb
        
    if exp_order == 2:
        phi = ( (-bb + np.sqrt(bb**2 - 4 * aa * cc) ) / (2 * aa) , ( -bb - np.sqrt(bb**2 - 4 * aa * cc) ) / (2 * aa) )
    
    return( phi )  



          
def phase_mask_phase_shift( wvl , d_on, d_off, glass_on, glass_off):
    """
    

    Parameters
    ----------
    wvl : float
        wavelength (m)
    d_on : float
        depth (m) of on-axis (phase shift region) part of mask 
    d_off : TYPE
        depth (m) of off-axis part of mask
    glass_on : string
        name of glass in on-axis (phase shift region) part of mask (see nglass function for options)
    glass_off : string
        name of glass in on-axis (phase shift region) part of mask 

    Returns
    -------
    phase shift (radians) applied to on-axis region in mask at given wavelength
    
           ----------
           |n_in     | n_air
           |         | 
    -------           -----
           |         |    
           |         | n_out 
    -----------------------
    
    
    """
    n_on = nglass(1e6 * wvl, glass=glass_on)[0]
    n_off = nglass(1e6 * wvl, glass=glass_off)[0]
    n_air = nglass(1e6 * wvl, glass='air')[0]
    
    opd = d_on * n_on - ( (d_on-d_off)*n_air + d_off * n_off )
    phase_shift = (2 * np.pi / wvl) * opd  
    
    return(phase_shift)
    
    
def plot_phase_shift_dispersion( phase_mask_dict , wvls ):

    thetas = np.array( [phase_mask_phase_shift( wvl , d_on = phase_mask_dict['d_on'], d_off=phase_mask_dict['d_off'], glass_on = phase_mask_dict['glass_on'], glass_off = phase_mask_dict['glass_off']) for wvl in wvls] )
    
    
    plt.figure()
    plt.plot(1e6 * wvls, 180/np.pi * thetas, color='k')
    plt.ylabel('phase shift [deg] ',fontsize=15)
    plt.xlabel(r'wavelength [$\mu$m]',fontsize=15)
    plt.grid()
    plt.gca().tick_params(labelsize=15)
    plt.show()
    
    
def calculate_don( wvl ,desired_phase_shift,  d_off, glass_on, glass_off):
    """

    Parameters
    ----------
    wvl : float
        wavelength (m)
    desired_phase_shift : float
        desired phase shift (radians)
    d_off : TYPE
        depth (m) of off-axis part of mask
    glass_on : string
        name of glass in on-axis (phase shift region) part of mask (see nglass function for options)
    glass_off : string
        name of glass in on-axis (phase shift region) part of mask 

    Returns
    -------
    on axis depth (d_on) calculated such that we achieve the desired_phase_shift at wvl

    """
    n_on = nglass(1e6 * wvl, glass=glass_on)[0]
    n_off = nglass(1e6 * wvl, glass=glass_off)[0]
    n_air = nglass(1e6 * wvl, glass='air')[0]
    
    opd = desired_phase_shift / (2 * np.pi / wvl)
    
    don = (opd + d_off * ( n_off-n_air )) / ( n_on-n_air )

    return (don)
    


# I need a function that given phase_mask_dict materials, desired_phase_shift , wvl range, optimizes depths to minimize rmse from desired_phase_shift 

def optimize_mask_depths( parameter_dict, wvls,  plot_results=True):
    """
    Parameters
    ----------
    
    parameter_dict - dictionary 
    MUST HOLD THE FOLLOWING KEYS: 
        target_phase_shift : float
            desired phase shift (degrees)
        glass_on : string
            name of glass in on-axis (phase shift region) part of mask (see nglass function for options)
        glass_off : string
            name of glass in on-axis (phase shift region) part of mask 
    
    wvls : array like
        array of wavelengths(m) to optimize (i.e. keep phaseshift as close to target_phase_shift as possible)
        
    for now we just do this the stupid but robust way of a manual grid search over reasonable depth (1um for on axis depth (d_on), wvl_c/10 for d_off) increments 
    
    output
    parameter_dict : dictionary with updated results (optimized depths for target phase shift over wvls )
    """
    
    try:
        if np.isfinite( parameter_dict['target_phase_shift']):
            target_phase_shift_rad = np.pi/180 * parameter_dict['target_phase_shift']
        else:
            raise TypeError('\n\n---->target_phase_shift is not finite!!!')
            
    except:
        raise TypeError('target_phase_shift key probably does not exist in parameter_dict, check this! ')
        
    glass_on = parameter_dict['glass_on']
    glass_off = parameter_dict['glass_off']
    
    
    g1 = np.linspace(20e-6, 30e-6, 10) 
    g2 = np.arange( 0, 5e-6,  (wvls[-1] + wvls[0]) / 2 / 10 )
    
    #init grid and best rmse 
    rmse_grid = np.inf * np.ones([len(g1),len(g2)])
    best_rmse = np.inf

    for i,don in enumerate( g1 ):
        for j, doff in enumerate( don - g2 ):
            phase_shifts = []
            for wvl in wvls:
                n_on = nglass(1e6 * wvl, glass=glass_on)[0]
                n_off = nglass(1e6 * wvl, glass=glass_off)[0]
                n_air = nglass(1e6 * wvl, glass='air')[0]
                
                #opd_desired = target_phase_shift_rad / (2 * np.pi / wvl)
                
                opd = don * n_on  - ( n_air * (don-doff) + n_off * doff ) 
                
                phase_shifts.append( 2*np.pi / wvl * opd ) #radians

            rmse = np.sqrt( np.mean( (target_phase_shift_rad - np.array(phase_shifts))**2 ) )
            
            rmse_grid[i,j] = rmse
            
            if rmse < best_rmse: #then store parameters 
                
                best_rmse = rmse  #radian
                
                mean_shift = np.mean(  np.array(phase_shifts) ) #radian
                std_shift = np.std(  np.array(phase_shifts) ) #radian
                
                don_opt = don
                doff_opt = doff
                
    
    #if phase_mask_dict is None:
    #    results = {'d_on':don_opt , 'd_off':doff_opt,'target_phase_shift': 180/np.pi * target_phase_shift, 'mean_phase_shift':mean_shift, 'std_phase_shift':std_shift, 'rmse':best_rmse, 'rmse_grid':rmse_grid}
    
    if type(parameter_dict) == type(dict()): # in this case return phase_mask_dict with updated items (but no rmse_grid!!)
        
        parameter_dict['d_on'] = don_opt
        parameter_dict['d_off'] = doff_opt
        parameter_dict['mean_phase_shift'] = 180/np.pi *  mean_shift #degree
        parameter_dict['std_phase_shift'] = 180/np.pi *  std_shift   #degree
        parameter_dict['rmse'] = 180/np.pi *  best_rmse   #degree
        parameter_dict['rmse_grid'] = 180/np.pi * rmse_grid   #degree
        
        
        #results = phase_mask_dict.copy() # {'phase_mask_dict':phase_mask_dict, 'don_opt':don_opt , 'doff_opt':doff_opt, 'rmse':best_rmse, 'rmse_grid':rmse_grid}
        
    else:
        raise TypeError('input parameter_dict is not a dictionary')
        
    
    if plot_results:
        
        #phase_mask_dict_tmp = {'d_off':results['d_off'], 'd_on':results['d_on'], 'glass_on':glass_on, 'glass_off':glass_off}

        plot_phase_shift_dispersion( parameter_dict, wvls )
    
    
    return( parameter_dict ) 





def process_input_spectral_features( parameter_dict  ):
    """

    Parameters
    ----------
    
    parameter_dict - dictionary 
    MUST HOLD THE FOLLOWING KEYS: 
        baldr_min_wvl: float
            minimum wavelength (m) for baldr wfs 
        baldr_max_wvl : float
            minimum wavelength (m) for baldr wfs 
        input_spectrum_wvls : array like with floats
            array of the input spectrum wavelengths (m) 
        input_spectrum_ph/m2/wvl/s : array like with floats 
            array holding the number of photons per  at the input wvlm^2 per second per wavelength bin 
    
        
    for now we just do this the stupid but robust way of a manual grid search over reasonable depth (1um for on axis depth (d_on), wvl_c/10 for d_off) increments 
    
    Returns
    --------
    parameter_dict : dictionary with updated features of input spectrum inlcuding total_flux_in_wfs[ph/s/m2], redness_ratio, spectral_classification_in_wfs 

    """
    
    
    
    wvl_c = ( parameter_dict['baldr_min_wvl'] + parameter_dict['baldr_max_wvl'] ) / 2  
    # filter to filter spectrum for the wfs wavelengths 
    wfs_wvl_filter = (parameter_dict['input_spectrum_wvls'] >=  parameter_dict['baldr_min_wvl']) & (parameter_dict['input_spectrum_wvls'] <=  parameter_dict['baldr_max_wvl'])
    #  filter to filter spectrum for the wfs wavelengths longer then the central (wvl_0) wavelength
    wvl_red_filter = (parameter_dict['input_spectrum_wvls'] > wvl_c)
    #  filter to filter spectrum for the wfs wavelengths shorter then the central (wvl_0) wavelength
    wvl_blue_filter = (parameter_dict['input_spectrum_wvls'] < wvl_c)
    
    
    total_flux_in_wfs = np.trapz(parameter_dict['input_spectrum_ph/m2/wvl/s'][wfs_wvl_filter], parameter_dict['input_spectrum_wvls'][wfs_wvl_filter] )  #photons/s/m^2
    
    # number of photons above central wvl of wfs bandpass 
    num = np.trapz( parameter_dict['input_spectrum_ph/m2/wvl/s'][wfs_wvl_filter & wvl_red_filter],parameter_dict['input_spectrum_wvls'][wfs_wvl_filter & wvl_red_filter] )
    # number of photons below central wvl of wfs bandpass 
    den = np.trapz( parameter_dict['input_spectrum_ph/m2/wvl/s'][wfs_wvl_filter & wvl_blue_filter],parameter_dict['input_spectrum_wvls'][wfs_wvl_filter & wvl_blue_filter] )
    

    if den > 0: 
        redness_ratio = num/den
    elif den == 0:
        print( '\n\n ->input spectrum has zero photons below the wavelength bandpass center...\n  setting redness ratio to 999999')
        redness_ratio = 999999
    elif den < 0:
        raise TypeError('\n\n ===== \n number of photons below central wvl of wfs bandpass  evaluated to a negative number.\nThis cannot be physically true!  ')
    else:
        raise TypeError('\n\n ===== \n den != 0 or den == 0 or den < 0:cannot be evaluated, check input to process_input_spectral_features() function ')
        
    # should i output this into input spectrum dict or baldr dict? 
    #maybe I should eventually combine them all into master parameter dictionary !!! 
    parameter_dict['redness_ratio'] =  round(redness_ratio,4)
    parameter_dict['total_flux_in_wfs[ph/s/m2]'] = total_flux_in_wfs
    
    # do some basic classification on spectral type seen by wfs
    if round(redness_ratio,4)==1:
        spectral_classification = 'flat'
    elif round(redness_ratio,4)>1:
        spectral_classification = 'red'
    elif round(redness_ratio,4)<1:
        spectral_classification = 'blue'
    else:
        raise TypeError('\n\n===\n something went wrong when evaluating redness_ratio, non cases met. check process_input_spectral_features() function' )
    
    parameter_dict['spectral_classification_in_wfs'] = spectral_classification
    
    return( parameter_dict )





#%%% Old can Delete stuff 


def baldr_simulation_bk_deleteme_l8er( naomi_screens , input_spectrum, baldr_dict, phase_mask_dict, wvl_0 = 1.65e-6, save = False, saveAs = None):
    """
    Parameters
    ----------
    naomi_screens : fits file
        DESCRIPTION. fits file with timeseries of NAOMI corrected phase screens at various wavelengths
        (hint: naomi_screens should be the output of naomi_simulation() function )
        
        
    master_parameter_dict :   dictionary
    
        baldr_dict['wvl_range_wfs'] - list or tuple with [min wvl, max wvl] of wfs 
        baldr_dict['wvl_0'] - primary wavelength (m) of baldr WFS where correction is optimized. The default is 1.65e-6 m. (this should be within wvl_range_wfs)
        baldr_dict['f_ratio'] - float indicating the  f ratio of Zernike wfs 
        baldr_dict['DIT'] - float indicating the detector integration time of wfs  
        baldr_dict['processing_latency'] - float indicating the latency of baldr wfs 
        baldr_dict['RON'] - read out noise of baldr wfs detector
        
        
        phase_mask_dict['T_off'] = phase mask off-axis transmission (outside phase shift region)
        phase_mask_dict['T_on'] = phase mask on-axis transmission (in phase shift region)
        phase_mask_dict['d_off'] = phase mask off-axis depth (m) 
        phase_mask_dict['d_on'] = phase mask on-axis depth (m) 
        phase_mask_dict['glass_off'] = material of off-axis region 
        phase_mask_dict['glass_on'] = material of on-axis region 
        phase_mask_dict['phase_mask_rad'] = radias of phase shift region at wfs central wvl (units = lambda/D) 
        phase_mask_dict['achromatic_diffraction'] = (boolean) if all wavelengths diffract to the same lambda_0/D  (this is stufy the effect of wvl dependent spatial distribution of phase shift)
        [NOTE I SHOULD CHANGE phase_mask_rad TO PHYSICAL UNITS ]
            
    input_spectrum : dictionary
        DESCRIPTION. input light spectrum (ph/m2/wvl/s) 
            input_spectrum['wvl'] - array like with wavelengths (m)
            input_spectrum['ph/m2/wvl/s'] - array like with flux per wavelength bin 
        
    baldr_dict: dictionary
        DESCRIPTION.  baldr_dict
            baldr_dict['wvl_range_wfs'] - list or tuple with [min wvl, max wvl] of wfs 
            baldr_dict['f_ratio'] - float indicating the  f ratio of Zernike wfs 
            baldr_dict['DIT'] - float indicating the detector integration time of wfs  
            baldr_dict['processing_latency'] - float indicating the latency of baldr wfs 
            baldr_dict['RON'] - read out noise of baldr wfs detector
            
        
    phase_mask_dict : dictionary 
        DESCRIPTION. phase mask parameters :
            phase_mask_dict['T_off'] = phase mask off-axis transmission (outside phase shift region)
            phase_mask_dict['T_on'] = phase mask on-axis transmission (in phase shift region)
            phase_mask_dict['d_off'] = phase mask off-axis depth (m) 
            phase_mask_dict['d_on'] = phase mask on-axis depth (m) 
            phase_mask_dict['glass_off'] = material of off-axis region 
            phase_mask_dict['glass_on'] = material of on-axis region 
            phase_mask_dict['phase_mask_rad'] = radias of phase shift region at wfs central wvl (units = lambda/D) 
            phase_mask_dict['achromatic_diffraction'] = (boolean) if all wavelengths diffract to the same lambda_0/D  (this is stufy the effect of wvl dependent spatial distribution of phase shift)
            [NOTE I SHOULD CHANGE phase_mask_rad TO PHYSICAL UNITS ]
            
            
    wvl_0 : TYPE, float
        DESCRIPTION. Central wavelength (m) of baldr WFS. The default is 1.65e-6 m.

    Returns
    -------
    fits file with timeseries of Baldr corrected phase screens at input phase screen wavelengths (defined in naomi_screens) fits file

    """

    input_screens = copy.deepcopy(naomi_screens) # to get parameters from 
    
    baldr_screens = copy.deepcopy(naomi_screens) # another copy to avoid editing original while assigning 
    
    
    input_wvls = np.array( [it.header['wvl'] for it in input_screens] )
    
    wvl_indx_4_wfs = [i for i,w in enumerate(input_wvls) if (w >= baldr_dict['wvl_range_wfs'][0]) & (w <= baldr_dict['wvl_range_wfs'][1]) ]
    
    wvl_0_indx = np.argmin(abs(wvl_0 - input_wvls))
    
    #abbreviated phase mask parameters 
    A = phase_mask_dict['T_off']
    B = phase_mask_dict['T_on']
    
    
    for frame in range(input_screens[0].shape[0]):
        #init output intensity to zero
        Ic = np.zeros((baldr_screens[0].header['D_pix'],baldr_screens[0].header['D_pix']))
        N_ph_T = 0
        
        for wvl_indx in wvl_indx_4_wfs:  #  need to filter for valid wvls where baldr sensor operates! 
            
            wvl_in = baldr_screens[wvl_indx].header['wvl']
            
            
            
            # input phase screen from *NAOMI* at current wvl
            phi = input_screens[wvl_indx].data[frame]
            phi[~np.isfinite(phi)] = 0 # mft can't deal with nan values  - so convert to zero
            #phi = putinside_array(np.zeros([nx_size, nx_size]), phi.copy()) #put inside new grid 
        
            
            #-------- these should not depend on wvl --------------------
            D = input_screens[wvl_indx].header['D']
            dx = D / input_screens[wvl_indx].header['D_pix'] #m/pixels
            f_ratio = baldr_dict['f_ratio']
                
            pup = pick_pupil(input_screens[wvl_indx].header['pupil_geometry'], dim=input_screens[wvl_indx].header['D_pix'], diameter=input_screens[wvl_indx].header['D_pix'] )
            #------------------------------------------------------------
            
            # create function to interpolate flux to current wavelength 
            flux_interp_fn = interp1d( input_spectrum['wvl'], input_spectrum['ph/m2/wvl/s']  )
            
            # interpolate to wvl from input photon flux and multiply by WFS DIT, differential (input) wvl element & telescope area to estimate N_ph in wvl bin
            N_ph_wvl = flux_interp_fn( wvl_in ) * np.diff( input_wvls )[wvl_indx-1] * baldr_dict['DIT'] * ((D/2)**2 * np.pi)
            N_ph_T += N_ph_wvl # add this to total number of photons
            """
            physical radius (m) is conserved for phase mask (unless we have some chromatic lens.. we could keep radius constant lambda/D )
            f_ratio * D * wvl_0 / D 
            therefore scale phase_mask_dict['phase_mask_rad_at_wvl_0'] by wvl_0/wvl_in
            
            """
            
            if phase_mask_dict['achromatic_diffraction']:
                # this case is not physically realistic but can be used to study the impact of chromatic spatial distribution of phase shift 
                # can look at it as physical size of phase mask radius changes with wavelength to always be lambda/D
                phase_mask_rad = phase_mask_dict['phase_mask_rad_at_wvl_0'] 
                 
            elif not phase_mask_dict['achromatic_diffraction']:
                """
                # this should be default for realistic simulations 
                # sanity check : 
                    phase mask rad = n lambda/D at wvl_0 
                    => physical radius fixed: r = C * n * wvl_0 / D => at wvl : r(wvl_0) = r(wvl) = C * n * (wvl / D) * wvl_0/wvl   
                
                """
                phase_mask_rad =  wvl_0/input_screens[wvl_indx].header['wvl'] * phase_mask_dict['phase_mask_rad_at_wvl_0']  #TO DO:  make this function of wvl such that = constant in physical radius and phase_mask_dict['phase_mask_rad_at_wvl_0'] (lambda/D) at wvl_0
                
            else:
                raise TypeError('phase_mask_dict["achromatic_diffraction"] is not boolean')
            
            #What is phase shift of phase mask at wvl = wvl_in 
            theta = phase_mask_phase_shift( wvl_in , d_on = phase_mask_dict['d_on'], d_off=phase_mask_dict['d_off'], glass_on = phase_mask_dict['glass_on'], glass_off = phase_mask_dict['glass_off'])
        
            IC_dict = I_C_analytic(A, B, theta, phase_mask_rad, phi, pup, N_ph_wvl, dx, wvl_in, D, f_ratio, troubleshooting=True) # need to changephase_mask_rad to physical radius of scale properly with wvl  if we want chromatic (not lambda /D)
            
            # get b at the central wavlength
            if wvl_0_indx in wvl_indx_4_wfs:
                
                if wvl_indx == wvl_0_indx:
                    
                    b_est =  IC_dict['b'] 
                
            else:
                raise TypeError('central wavelength (wvl_0) not in defined wfs wavelength range (wvl_range_wfs)')
                
                
            # add intensity contribution from spectral bin 
            Ic += IC_dict['Ic'] # + noise 
            
            # Now add shot noise :
                
            # note  that input field calculated in I_C_analytic is normalized so that sum over every pupil pixel intensity = N_phot_wvl , 
            N_pix = np.nansum( pup ) #number of pixels in pupil 
            # average number of photons per pixel = N_ph_wvl/N_pix  (uniform illumination assumption)
            # therefore generate the a possion distribution (shot noise) with zero mean but var =  N_ph_wvl/N_pix  to add to output intesnity 
            # note that this shot noise assume perfect trasmission from input pupil to output detector.. 
            shot_noise = N_ph_wvl/N_pix - poisson.rvs( N_ph_wvl/N_pix, size= Ic.shape)  
            Ic += shot_noise
            # make sure there are no negative values 
            Ic[Ic<0] = 0 
            
            #resample to resize pixels 
            # --- TO DO - need to know how many pixels we reading out on 
            
            # after rebinning (if done) then we convert intensity to adu (integers)
            Ic_adu = Ic.astype(int)
            
        # now that we have our intensity across all wfs spectral channels we can estimate the phase at wvl_0(or wherever we estimated b and theta) - we could also consider estimating b for each spectral chanel and then averaging etc 
        phi_est = zelda_phase_estimator_1(Ic_adu, pup, N_ph_T, dx, A, B,  abs( b_est ), theta, exp_order=1) #radians 
        
        # convert phase estimate to DM OPD at wvl_0
        DM_cmd = wvl_0 / (2*np.pi) * phi_est
        
        
        #------ now to subtract DM from input phase screen at each wavelength 
        
        
        lag = baldr_dict['DIT'] + baldr_dict['processing_latency'] #s
        
        dt = input_screens[wvl_indx].header['dt']
        
        jumps = int(round(lag / dt)) # number of frames to jump before applying correction 
        
        for hdu in baldr_screens:
            
            hdu.header.set('what is', 'Naomi+Baldr AO corrected phase screens (rad)' )
            hdu.header.set('wvl_0', wvl_0, 'central wavelentgth of Baldr WFS (m) ')
            
                               
            for  k,v in phase_mask_dict.items() : 
                hdu.header.set(k , v, ' baldr phase_mask parameters ')
                
            
            hdu.header.set('baldr wfs wvl_min' , baldr_dict['wvl_range_wfs'][0] , ' minimum wavelength (m) in baldr wfs')
            hdu.header.set('baldr wfs wvl max' , baldr_dict['wvl_range_wfs'][1] , ' maximum wavelength (m) in baldr wfs')
            hdu.header.set('baldr wfs badwidth ' , round( 100 * (baldr_dict['wvl_range_wfs'][1] -  baldr_dict['wvl_range_wfs'][0]) / wvl_0, 2) , ' maximum wavelength (m) in baldr wfs')
            hdu.header.set('f_ratio' , baldr_dict['f_ratio'] , ' f ratio of ZWFS')
            hdu.header.set('baldr DIT' , baldr_dict['DIT'] , 'baldr wfs detector integration time (s)')
            hdu.header.set('baldr latency' , baldr_dict['DIT'] , 'baldr latency after detector integration (s) ')
            hdu.header.set('baldr RON' , baldr_dict['RON'] , 'baldr detector read out noise [e-]')

        
            if frame + jumps < input_screens[0].shape[0]:
                
                wavefront_opd = hdu.header['wvl'] / (2*np.pi) * hdu.data[frame + jumps]
                hdu.data[ frame + jumps ] =  2*np.pi /hdu.header['wvl'] * ( wavefront_opd  -  DM_cmd )
                
            else:
                print('to do here')
                #hdu.data[ frame +   ] = hdu.header['wvl'] / (2*np.pi) * hdu.data[frame] - DM_cmd 
    
    if save:
        baldr_screens.writeto(saveAs,overwrite=True)

    return(baldr_screens)






def optimize_mask_depths_bk_deletemelater(desired_phase_shift , glass_on, glass_off, wvls, phase_mask_dict = None, plot_results=True):
    """
    Parameters
    ----------
    
    desired_phase_shift : float
        desired phase shift (radians)
    glass_on : string
        name of glass in on-axis (phase shift region) part of mask (see nglass function for options)
    glass_off : string
        name of glass in on-axis (phase shift region) part of mask 
    phase_mask_dict : None or dictionary 
        phase_mask_dict , if provided (not None) then a phase_mask_dict will be returned in results dictionary with the optimized don, doff phase mask depths 
    wvls : array like
        array of wavelengths(m) to optimize (i.e. keep phaseshift as close to desired_phase_shift as possible)
        
    for now we just do this the stupid but robust way of a manual grid search over reasonable depth (1um for on axis depth (d_on), wvl_c/10 for d_off) increments 
    
    output
    results : dictionary with results 
    """
    g1 = np.linspace(20e-6, 30e-6, 10) 
    g2 = np.arange( 0, 5e-6,  (wvls[-1] + wvls[0]) / 2 / 10 )
    
    #init grid and best rmse 
    rmse_grid = np.inf * np.ones([len(g1),len(g2)])
    best_rmse = np.inf

    for i,don in enumerate( g1 ):
        for j, doff in enumerate( don - g2 ):
            phase_shifts = []
            for wvl in wvls:
                n_on = nglass(1e6 * wvl, glass=glass_on)[0]
                n_off = nglass(1e6 * wvl, glass=glass_off)[0]
                n_air = nglass(1e6 * wvl, glass='air')[0]
                
                opd_desired = desired_phase_shift / (2 * np.pi / wvl)
                
                opd = don * n_on  - ( n_air * (don-doff) + n_off * doff ) 
                
                phase_shifts.append( 2*np.pi / wvl * opd )

            rmse = np.sqrt( np.mean( (desired_phase_shift - np.array(phase_shifts))**2 ) )
            
            rmse_grid[i,j] = rmse
            
            if rmse < best_rmse: #then store parameters 
                
                best_rmse = 180/np.pi * rmse  #degree
                
                mean_shift = 180/np.pi * np.mean(  np.array(phase_shifts) ) #degree
                std_shift = 180/np.pi * np.std(  np.array(phase_shifts) ) #degree
                
                don_opt = don
                doff_opt = doff
    
    if phase_mask_dict is None:
        results = {'d_on':don_opt , 'd_off':doff_opt,'target_phase_shift': 180/np.pi * desired_phase_shift, 'mean_phase_shift':mean_shift, 'std_phase_shift':std_shift, 'rmse':best_rmse, 'rmse_grid':rmse_grid}
    
    elif type(phase_mask_dict) == type(dict()): # in this case return phase_mask_dict with updated items (but no rmse_grid!!)
        
        phase_mask_dict['d_on'] = don_opt
        phase_mask_dict['d_off'] = doff_opt
        phase_mask_dict['target_phase_shift'] = 180/np.pi * desired_phase_shift #desired phase shift input is in radians 
        phase_mask_dict['mean_phase_shift'] = mean_shift
        phase_mask_dict['std_phase_shift'] = std_shift
        phase_mask_dict['rmse'] = best_rmse
        
        
        results = phase_mask_dict.copy() # {'phase_mask_dict':phase_mask_dict, 'don_opt':don_opt , 'doff_opt':doff_opt, 'rmse':best_rmse, 'rmse_grid':rmse_grid}
        
    else:
        raise TypeError('input phase_mask_dict is not a dictionary')
        
    
    if plot_results:
        
        phase_mask_dict_tmp = {'d_off':results['d_off'], 'd_on':results['d_on'], 'glass_on':glass_on, 'glass_off':glass_off}

        plot_phase_shift_dispersion( phase_mask_dict_tmp , wvls )
    
    
    return( results ) 



