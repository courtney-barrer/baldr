#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 03:30:37 2022

@author: bcourtne




""""""



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
    

    dx = screen.pixel_scale
    
    basis = zernike.zernike_basis(n_modes, npix=pupil.shape[0])

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



def AO_simulation(parameter_dict, sim_time = 0.01, wvl_list= [1.65e-6], save = False, saveAs = None):
    
    """
    simulate AO systemfor sim_time

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
        hdu.header.set('seeing', seeing, 'arcsec')
        hdu.header.set('L0', L0, 'outerscale (m)')
        hdu.header.set('D', D, 'telescope diameter (m)')
        hdu.header.set('D_pix', D_pix, '#pixels across telescope diameter')
        hdu.header.set('wvl_0', wvl_0, 'central wavelentgth of WFS (m)')
        hdu.header.set('wvl', wvl, 'wvl (m) where corrected phase screen is calc. ')
        hdu.header.set('n_modes', n_modes, '#N modes sensed & corrected by AO system')
        hdu.header.set('lag', lag, 'latency of AO system (s)')
        hdu.header.set('V_turb', V_turb, 'velocity of phase screen (m/s)')
        hdu.header.set('pupil_geometry', pupil_geometry, 'Valid options are AT, UT, disk')
        hdu.header.set('dt', dt, 'phase screen sampling rate')
    
        simulation_fits.append( hdu )
    

    if save:
        simulation_fits.writeto(saveAs,overwrite=True)
        
    return( simulation_fits )




def pick_pupil(pupil_geometry, dim, diameter ):
        
    if pupil_geometry == 'AT':
        pup = AT_pupil(dim = D_pix, diameter = D_pix) 
    elif pupil_geometry == 'UT':
        pup = aperture.vlt_pupil(dim = D_pix, diameter = D_pix, dead_actuator_diameter=0) 
    elif pupil_geometry == 'disk':
        pup = aperture.disc( dim = D_pix, size = D_pix//2) 
    else :
        print('no valid geometry defined (try pupil_geometry == disk, or UT, or AT\nassuming disk pupil')
        pup = aperture.disc( dim = D_pix, size = D_pix//2) 

    return(pup)

#%% Testing Baldr 

"""
initial thoughts 
- set up grid
- set up phase mask materials

""" 





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




def get_phase_shift_region(phase_mask_rad, dx , nx_size, no_lambdaD) :
    """
    creates array defining phase shift region 

    Parameters
    ----------
    phase_mask_rad: float- radius of phase shift region in  #lambda/D units (i.e radius = 1 lambda /D => phase_mask_rad = 1 )
    nx_size: int - #pixels
    dx: float - pupil plane pixel size (m)
    

    Returns
    -------
    phase_shift_region: 2D array defining phaseshift region 
    
    """
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
    phase_shift_region = get_phase_shift_region(phase_mask_rad, dx , nx_size, no_lambdaD) #aperture.disc(nx_size, f_r_pix) 
    
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
    phase_shift_region = get_phase_shift_region(phase_mask_rad, dx , nx_size, no_lambdaD)
    
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
    phase_shift_region = get_phase_shift_region(phase_mask_rad, dx , nx_size, no_lambdaD)
    
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
        data_dict = {'Ic':I_C,'Psi_C':Psi_C,'Psi_B':Psi_B, 'Psi_A':Psi_A,'b':b,'dx':dx,'df_x':df_x}
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
    
    don = (opd + d_off * ( n_off-n_air )) / ( n_off-n_air )

    return (don)
    
    


#%% Testing NAOMI 

#for wvl in np.linspace(1.2e-6 , 1.9e-6, 10):
naomi_parameter_dict_1 = {'seeing':1 , 'L0':25, 'D':1.8, 'D_pix':2**8, \
                        'wvl_0':0.658e-6,'n_modes':14,\
                            'lag':5e-3,'V_turb':40, 'pupil_geometry': 'AT',\
                                'dt': 0.3 * 1e-3}
    
sim_time = 0.1 #s  
wvl_list = np.linspace(0.9e-6,  2.0e-6,  20)

naomi_screens = AO_simulation(naomi_parameter_dict_1, sim_time = sim_time, wvl_list=wvl_list, save = True, saveAs = '/Users/bcourtne/Documents/ANU_PHD2/heimdallr/naomi_screens_sim_1.fits')


"""
# test1. check strehl or rms changes with wvl for same timestamp 
# test2. check for given timestamp that each wvl dependent phase screen has same structure (only scaling difference)

"""

#test1
wvl_indx = 0
print( 'RMSE as function of wvl ', [np.nanstd( naomi_screens[wvl_indx].data[0] - naomi_screens[i].data[0]) for i in range(len(naomi_screens))] )
# 1st index of list should be zero! Passed

#test2
i=0
wvl_indx = 0
plt.figure()
plt.imshow( naomi_screens[wvl_indx].data[i] )
plt.axis('off')
plt.title( 'Example phase screen at {}um\nStrehl = {}'.format(round(1e6 * naomi_screens[wvl_indx].header['wvl'],2), round(np.exp(-np.nanvar(naomi_screens[wvl_indx].data[i])) ,2)) )
   

i=0
wvl_indx = -1
plt.figure()
plt.imshow( naomi_screens[wvl_indx].data[i] )
plt.axis('off')
plt.title( 'Example phase screen at {}um\nStrehl = {}'.format(round(1e6 * naomi_screens[wvl_indx].header['wvl'],2), round(np.exp(-np.nanvar(naomi_screens[wvl_indx].data[i])) ,2)) )


"""strehl_tmp = []
for i in range(200):
    #plt.figure()
    #plt.imshow( naomi_screens[wvl_indx].data[i] )
    strehl_tmp.append(np.exp(-np.nanvar(naomi_screens[wvl_indx].data[i])) )
"""


#%%
# Testing Baldr 

# with NAOMI need to simulate phase screens at Baldr WFS wavelengths and also at Bifrost J-band coupling wavelengths 

# To do: write input parameter dictonary , baldr correction function with lag etc / shot noise http://spiff.rit.edu/classes/phys445/lectures/readout/readout.html
## function to get optical depth. /calculate phase shift as fn of wvl  as function of wavelength for phase_shift region + outer region 
# phi estimator has to become function of wvl!!!

"""
phase_mask_dict 
======
T_off = phase mask off-axis transmission (outside phase shift region)
T_on = phase mask on-axis transmission (in phase shift region)
d_off = phase mask off-axis depth (m) 
d_on = phase mask on-axis depth (m) 
glass_off = material of off-axis region 
glass_on = material of on-axis region 
phase_mask_rad = radias of phase shift region (units = lambda/D)
"""                        
phase_mask_dict = {'T_off':1,'T_on':1,'d_off':2e-6, 'd_on':np.nan, 'glass_off':'sio2', 'glass_on':'sio2','phase_mask_rad_at_wvl_0':1}

"""
Note Chromatic phase mask:
    glass_on = 'sio2'
    d_on = 32um
    glass_off = 'su8'
    d_off=26.27um

"""

input_spectrum = {'wvl':np.linspace(0.9e-6,2e-6,100), 'ph/m2/wvl':1e9 * np.ones(100) } # photons per m2/wvl

baldr_dict = {'wvl_range_wfs':[1.5e-6, 1.8e-6], 'f_ratio':20, 'dt':1e-3, 'DIT':2e-3, 'processing_latency':1e-3, 'RON':2}

input_wvls = [it.header['wvl'] for it in naomi_screens]

wvl_0 = 1.65e-6  # central wvl of baldr wfs 

    
# phase mask 
# =============
#    calculate the phase shift 
phase_mask_dict['d_on'] = calculate_don( wvl = wvl_0 ,desired_phase_shift = np.pi/2,  d_off = phase_mask_dict['d_off'], glass_on = phase_mask_dict['glass_on'], glass_off=phase_mask_dict['glass_off'])

# establish phase mask design parameters
theta = phase_mask_phase_shift( wvl = wvl_0 , d_on=phase_mask_dict['d_on'], d_off=phase_mask_dict['d_off'], glass_on=phase_mask_dict['glass_on'], glass_off=phase_mask_dict['glass_off'])
A = phase_mask_dict['T_off']
B = phase_mask_dict['T_on']

    
#init output intensity to zero
Ic = np.zeros((naomi_screens[0].header['D_pix'],naomi_screens[0].header['D_pix']))
N_ph_T = 0
#wvl_indx=0
for wvl_indx in range(len(naomi_screens)):  # I probably need to filter for valid wvls where baldr sensor operates! 
    
    wvl_in = naomi_screens[wvl_indx].header['wvl']
    phi = naomi_screens[wvl_indx].data[0]
    #-------- these should not depend on wvl --------------------
    D = naomi_screens[wvl_indx].header['D']
    dx = D / naomi_screens[wvl_indx].header['D_pix'] #m/pixels
    f_ratio = baldr_dict['f_ratio']
        
    pup = pick_pupil(naomi_screens[wvl_indx].header['pupil_geometry'], dim=naomi_screens[wvl_indx].header['D_pix'], diameter=naomi_screens[wvl_indx].header['D_pix'] )
    #------------------------------------------------------------
    
    flux_interp_fn = interp1d( input_spectrum['wvl'], input_spectrum['ph/m2/wvl']  )
    
    # interpolate to wvl from input photon flux and multiply by differential (input) wvl element & telescope area to estimate N_ph in wvl bin
    N_ph_wvl = flux_interp_fn( wvl_in ) * np.diff( input_wvls )[wvl_indx-1] * ((D/2)**2 *np.pi)
    N_ph_T += N_ph_wvl # add this to total number of photons

    phase_mask_rad =  phase_mask_dict['phase_mask_rad_at_wvl_0']  #TO DO:  make this function of wvl such that = constant in physical radius and phase_mask_dict['phase_mask_rad_at_wvl_0'] (lambda/D) at wvl_0

    IC_dict = I_C_analytic(A, B, theta, phase_mask_rad, phi, pup, N_ph_wvl, dx, wvl_in, D, f_ratio, troubleshooting=True) # need to changephase_mask_rad to physical radius of scale properly with wvl  if we want chromatic (not lambda /D)
    
    # add noise (to do!)
    Ic += IC_dict['Ic'] # + noise 
    
    
    #resample to resize pixels 
    # --- TO DO - need to know how many pixels we reading out on 
    
#which B do we use?? ) 
phi_est = zelda_phase_estimator_1(Ic, pup, N_ph_T, dx, A, B,  abs( IC_dict['b'] ), theta, exp_order=1)


print( np.nanstd(phi - phi_est ), np.nanstd(phi) )


#%%



#N_ph_wvl_in = np.trapz( flux_interp_fn(input_wvls) * ( (D/2)**2 * np.pi ) , input_wvls)


"""
input_dict 
======
D = telescope diameter  (m)
D_pix = number of pixels across telescope diameter 
pupil_geometry = what pupil do we have? AT, UT, disk?
wvl_in = wavelength (m) of the input phase screen (this will be wvl where baldr calculates correction )
wvl_out = wavelength (m) where the output phase screen is calculated of the input phase screen
N_ph_wvl_in = number of input photons per second at wavelength = wvl_in  
N_ph_wvl_out = number of input photons per second at wavelength = wvl_out 
dt = sampling time of input phase screens
"""

input_dict = {'D':naomi_screens.header['D'], 'D_pix':naomi_screens.header['D_pix'] ,\
                        'wvl_in':naomi_screens.header['wvl'],\
                            'wvl_out':1.35e-6, 'pupil_geometry': naomi_screens.header['pupil_geometry'],\
                                'N_ph_wvl_in':1e6, 'N_ph_wvl_out':1e6,'dt_input':naomi_screens.header['dt'],\
                                    'f_ratio':10}
    
    
                            
"""
phase_mask_dict 
======
T_off = phase mask off-axis transmission (outside phase shift region)
T_on = phase mask on-axis transmission (in phase shift region)
d_off = phase mask off-axis depth (m) 
d_on = phase mask on-axis depth (m) 
glass_off = material of off-axis region 
glass_on = material of on-axis region 
phase_mask_rad = radias of phase shift region (units = lambda/D)
"""                        
phase_mask_dict = {'T_off':1,'T_on':1,'d_off':2e-6, 'd_on':np.nan, 'glass_off':'sio2', 'glass_on':'sio2','phase_mask_rad':1}

"""
Note Chromatic phase mask:
    glass_on = 'sio2'
    d_on = 32um
    glass_off = 'su8'
    d_off=26.27um

"""

baldr_dict = {'wvl_range'[1.5e-6, 1.8e-6], 'dt':1e-3,'DIT':2e-3, 'processing_latency':1e-3, 'RON':2}





# input 
# =============
phi = naomi_screens.data[0]

wvl_in = input_dict['wvl_in'] 
pup = pick_pupil(input_dict['pupil_geometry'], dim=input_dict['D_pix'], diameter=input_dict['D_pix'] )
N_ph_wvl_in = input_dict['N_ph_wvl_in']

D = input_dict['D']
dx = D / input_dict['D_pix'] #m/pixels
f_ratio = input_dict['f_ratio']

# phase mask 
# =============
#    calculate the phase shift 
phase_mask_dict['d_on'] = calculate_don( wvl = input_dict['wvl_in'] ,desired_phase_shift = np.pi/2,  d_off = phase_mask_dict['d_off'], glass_on = phase_mask_dict['glass_on'], glass_off=phase_mask_dict['glass_off'])

theta = phase_mask_phase_shift( wvl =  input_dict['wvl_in'] , d_on=phase_mask_dict['d_on'], d_off=phase_mask_dict['d_off'], glass_on=phase_mask_dict['glass_on'], glass_off=phase_mask_dict['glass_off'])
A = phase_mask_dict['T_off']
B = phase_mask_dict['T_on']
phase_mask_rad =  phase_mask_dict['phase_mask_rad']  # need to change this to physical radius if we want chromatic (not lambda /D)





#for all wavelengths  we create intensity 

# measurement of output intensity 
Nph_T = N_ph_wvl_in * baldr_dict['DIT']



IC_dict = I_C_analytic(A, B, theta, phase_mask_rad, phi, pup, Nph_T, dx, wvl_in, D, f_ratio, troubleshooting=True)

# add noise (to do!)
Ic = IC_dict['Ic'] # + noise 




#get phase estimator 
phi_est = zelda_phase_estimator_1(Ic, pup, N_ph_wvl_in, dx, A, B,  abs( IC_dict['b'] ), theta, exp_order=1)

#subtract of a few phase screens 

    

"""
For each input wavelength across baldr wfs we get one output intensity array. 
These can each be added up to give one OPD correction 
we then apply some lag to the input phase screen 
then apply OPD correction to input phase screen at wvl_out 

"""

    

    
    

    
    

D = naomi_screens.header['D']
dx = D / naomi_screens.header['D_pix'] #m/pixels
wvl = 1.65e-6 
f_ratio = 2 # f/D
N_ph = 1e6 #1e6 
nx_size = naomi_screens.header['D_pix']

pup = pick_pupil(naomi_screens.header['pupil_geometry'], dim=naomi_screens.header['D_pix'], diameter=naomi_screens.header['D_pix'] )
pup = putinside_array(np.zeros([nx_size, nx_size]), pup.copy())

# inputs to ZWFS
phi = naomi_screens.data[0]
phi[~np.isfinite(phi)] = 0 # mft can't deal with nan values  - so convert to zero
phi = putinside_array(np.zeros([nx_size, nx_size]), phi.copy()) #put inside new grid 


# Mask
A=1
B=1
theta= np.pi/3 # function that takes wvl , design (material, depths), and returns phase shift
phase_mask_rad = 1 

# I need a then IC and a now Psi_a, also add shot noise onto it 
I_dict = I_C_analytic(A, B, theta, phase_mask_rad, phi, pup, N_ph, dx, wvl, D, f_ratio, troubleshooting=True)
b = I_dict['b']

phi_est = zelda_phase_estimator_1(I_dict['Ic'], pup, N_ph, dx, A, B,  abs(b), theta, exp_order=1)


plt.figure()
plt.plot(phi[pup>0.5],phi_est[pup>0.5],  '.' ,label='simulation 1st order')
plt.plot(phi_est[pup>0.5],phi_est[pup>0.5], label='1:1')
plt.ylabel('estimate phase ')
plt.xlabel('real phase ')
plt.legend()


print('check conservation \n dx**2 * sum sum Psi_A**2 = {} photons \n df_x**2  * sum sum Psi_B**2 = {} photons'.format(I_dict['dx']**2*sum(sum(abs(I_dict['Psi_A'] )**2)), I_dict['df_x']**2*sum(sum(abs(I_dict['Psi_B'] )**2))) ) 

print('also dx**2*sum(sum(abs(Psi_C)**2)) = {} photons\n this is not necessarily conserved for small phase shift region\n since phase mask diffracts light outside pupil'.format(I_dict['dx']**2*sum(sum(abs(I_dict['Psi_C'] )**2))))
#print('\n1.22 lambda/D = {}arcsec'.format(round(1.22*wvl/D*rad2arcsec,4)))

#%%

Psi_A, Psi_B, Psi_C, H = I_C_sim(A, B, theta, phase_mask_rad, phi, pup, N_ph, dx, wvl, D, f_ratio)







# ------- load NAOMI phase screens 
phi_screens = fits.open('/Users/bcourtne/Documents/ANU_PHD2/heimdallr/naomi_screens_sim_1.fits')
# ------- prelims 
wvl= 1.65e-6  #1.25e-6 #wavelength (m)
D = 1.8 # telescope diameter (m)
f_ratio = 10
nx_size = 2**11 #number of pixels in input pupil grid 
D_pix =  phi_screens[0].data[1].shape[0] #2**8#2**8 #number of pixels across telescope diameter 
dx = D/D_pix  # diff element (pixel size) pupil plane (m)
N_ph = 1e9 # number of photons

pup = aperture.disc(nx_size,  D_pix//2) # AT_pupil(dim = nx_size, diameter =  D_pix)

A = 1 #phase mask non-phase shift region transmission 
B = 0.5 #phase mask phase shift region transmission 
theta = np.pi/3.9  # phase mask phase shift 
phase_mask_rad = 1 # phase mask phase shift region radius (# wvl/D)

basis = zernike.zernike_basis(nterms=10, npix = D_pix) 

#phase screen OPD in units (m). Check <np.exp(-(np.pi*2/1.65e-6)**2 * np.nanvar(phi_screens[0].data[1]))>~0.6 (H-strehl=0.6)
# what wavelength are these simulated in 


phi = 2*np.pi/wvl * phi_screens[0].data[1] 
phi[~np.isfinite(phi)] = 0 # mft can't deal with nan values  - so convert to zero
phi = putinside_array(np.zeros(pup.shape), phi.copy()) #put inside new grid 

# get coordinates in pupil and focal plane 
x, f_x, df_x, f_r, df_r, no_lambdaD = get_plane_coord(nx_size, dx, wvl, D, f_ratio)

#Psi_A = pup * N_ph * np.exp(1j * phi)  #
Psi_A = pup * N_ph * np.exp(1j * phi)

#normalize so that sum over every pupil pixel intensity = N_phot (total number of photons)
#therefore pupil_field(Nx,Ny) = number of photons within pixel (or equivilantly photons / m^2 considering pixel finite area dfX^2)
Psi_A *= np.sqrt((N_ph/np.nansum(abs(Psi_A)*dx**2))) #* Psi_A * pup
# parameter b 

# input field in pupil
#Psi_A = pup * np.exp(1j * phi_tmp)

# b  parameter 
b = calc_b(Psi_A, dx, wvl, D, f_ratio, phase_mask_rad )


I_C, Psi_A, Psi_C = I_C_analytic(A, B, theta, phase_mask_rad, phi, pup, dx, wvl, D, f_ratio)

# estimator assuming abs b
phi_est = pup * zelda_phase_estimator_1(I_C, pup, A, B, 1.5*abs(b)/N_ph, theta, exp_order=1)

plt.figure()
plt.plot(phi_est[pup>0.5], phi[pup>0.5], '.' ,label='simulation 1st order')
plt.plot(phi_est[pup>0.5],phi_est[pup>0.5], label='1:1')
plt.ylabel('estimate phase ')
plt.xlabel('real phase ')
plt.legend()

phi_reco = crop2center( phi_est , phi_screens[0].data[1])
phi_corrected = crop2center( phi_est - phi , phi_screens[0].data[1])


fig,ax = plt.subplots(1,3)
ax[0].imshow( crop2center( phi , phi_screens[0].data[1]) )
ax[0].set_title('phi real')
ax[1].imshow( phi_reco  )
ax[1].set_title('phi reco')
ax[2].imshow( phi_corrected  )
ax[2].set_title('corrected')


ax[0].text(phi_reco.shape[0]//2, phi_reco.shape[0]//2, 'S={}'.format(round( np.exp( - np.nanvar( crop2center( phi, phi_screens[0].data[1]) ) ) , 2)), color='white')
#ax[2].text(250,250,'S={}'.format(round( np.exp( - np.nanvar( crop2center( phi_est, phi_screens[0].data[1]) ) ) , 2)))
ax[2].text(phi_reco.shape[0]//2, phi_reco.shape[0]//2,'S={}'.format(round( np.exp( - np.nanvar( crop2center( phi-phi_est, phi_screens[0].data[1]) ) ) , 2)), color='white')

for axx in ax:
    axx.axis('off')





#%% Testing old 
seeing = 1 # arcsec
L0 = 12 #outerscale (m)
r0 = 0.98 * 500e-9 / np.radians(seeing/3600) #Fried parameter defined at 500nm
D = 1.8 # telescope diameter (m)
nx_size = 2**8 #number of pixels in input pupil grid 
D_pix = 2**8 #2**8#2**8 #number of pixels across telescope diameter 
dx = D/D_pix  # diff element (pixel size) pupil plane (m)
N_ph = 1e9 # number of photons

pup = AT_pupil(dim = D_pix, diameter = D_pix) 


phase_screen = aotools.turbulence.infinitephasescreen.PhaseScreenVonKarman(nx_size, pixel_scale=dx,\
      r0=r0 , L0=L0, n_columns=2,random_seed = 1) # radians
    

pup = AT_pupil(dim = D_pix, diameter = D_pix) 

n_modes_tmp = 14


    



naomi_results=[]
for i in range(20):
    
    AO_correction( pup, phase_screen, wvl_0=0.6e-6, wvl=1.65e-6, n_modes=n_modes_tmp, lag=2.5e-3, V_turb=50, return_in_radians=False, troubleshoot=False)
    
    naomi_results.append(  AO_correction( pup, phase_screen, wvl_0=0.6e-6, wvl=1.65e-6, n_modes=n_modes_tmp, lag=2.5e-3, V_turb=50, return_in_radians=False, troubleshoot=False) )

    #store as fits files

    
    hdu = fits.PrimaryHDU( naomi_results[i] )

    hdulist = fits.HDUList([hdu])
    
    #hdulist.writeto(f'/Users/bcourtne/Documents/ANU_PHD2/heimdallr/naomi_screens_sim_[new]_1.fits')

#%%
for i in range(len(naomi_results)):
    plt.figure()
    plt.imshow(naomi_results[i][0])
    
#%% Checks
"""
1. zero lag => opd_now == opd_then
2. zero lag => DMshape Zernike coefficients = then_opd Zernike coefficients up to n_modes (slight differences could be due to spider)
3. zero lag => variance of corrected_opd should follow (roughly) Noll indicies 

"""    

# TEST  1. zero lag => opd_now == opd_then
pup = AT_pupil(dim = D_pix, diameter = D_pix) 
my_dict = AO_correction( pup, phase_screen, wvl_0=1.6e-6, wvl=1.6e-6, n_modes=14, lag=0, V_turb=50, return_in_radians=False, troubleshoot=True)
fig, ax = plt.subplots(1,3)
#ax.set_title('test 2. zero lag produces perfect correction when center wvl_0=wvl\ncorrected_opd')
ax[0].imshow(my_dict['now_opd'])
ax[0].set_title('now OPD')

ax[1].imshow(my_dict['then_opd'])
ax[1].set_title(r'$\delta$OPD')
ax[1].set_title('then OPD')

ax[2].imshow(my_dict['then_opd']- my_dict['now_opd'])
ax[2].set_title(r'$\delta$OPD')

if np.sum(my_dict['then_opd'] != my_dict['now_opd'])==0:
    print('TEST  1. passed')
else:
    print('TEST  1. failed')

# TEST  2. zero lag => DMshape Zernike coefficients = then_opd Zernike coefficients up to n_modes (slight differences could be due to spider)


n_modes_tmp = 14
basis = zernike.zernike_basis(n_modes_tmp , npix=D_pix)
pup = basis[0]  #AT_pupil(dim = D_pix, diameter = D_pix) 
my_dict = AO_correction( pup, phase_screen, wvl_0=1.6e-6, wvl=1.6e-6, n_modes=n_modes_tmp, lag=0, V_turb=50, return_in_radians=False, troubleshoot=True)

#note by changing the lag we can clearly see divergence from 1:1 line 
plt.figure()
plt.title('test 2. zero lag => DMshape Zernike coefficients = then_opd Zernike coefficients\n up to n_modes (slight differences could be due to spider)')
plt.plot( my_dict['now_coe'], my_dict['DM_coe'] ,'.',label='test 2.',color='b')
plt.plot( my_dict['now_coe'], my_dict['now_coe'] ,'-',label='1:1',color='r',alpha=0.3)
plt.xlabel('now_coe [m]')
plt.ylabel('DM_coe [m]')
plt.legend()


#3. zero lag => variance of corrected_opd should follow (roughly) Noll indicies (for circular poupil! ) <a_i**2> = c_ij * (D/r0)**(5/3), 
# with correction var assymptotically approaching = (D/r0)**(5/3) * 0.2944*J**(np.sqrt(3)/2)

#noll coefficients for first 3 radial orders 
c_ij = np.array( [0.449,0.449,0.0232,0.0232,0.0232, 0.00619, 0.00619, 0.00619,0.00619] )

basis = zernike.zernike_basis(5, npix=D_pix)
pup = basis[0]  #AT_pupil(dim = D_pix, diameter = D_pix) 
wvl_tmp = 1.6e-6
theory_var = []
meas_var = []
for n_modes_tmp in range(10, 15): 
    
    # note each iteration of this continues to move phase screen and does not state
    my_dict = AO_correction( pup, phase_screen, wvl_0=wvl_tmp, wvl=wvl_tmp, n_modes=n_modes_tmp, lag=0, V_turb=50, return_in_radians=False, troubleshoot=True)
    
    #c_ij_meas.append( (2*np.pi / wvl_tmp)**2 * np.nanvar( my_dict['corrected_opd'][pup>0.5] ) )#  / (D/r0)**(5/3) )  
    theory_var.append( (D/r0)**(5/3) * 0.2944 * n_modes_tmp**(-np.sqrt(3)/2) )
    meas_var.append( 0.3 * np.nanvar( (2*np.pi / wvl_tmp) * my_dict['corrected_opd'][pup>0.5] ) )
    
plt.figure()
plt.plot( theory_var , '.')
plt.plot( meas_var )



#4. Issue with units - do we get expected strehl / 

pup = basis[0]  #AT_pupil(dim = D_pix, diameter = D_pix) 
wvl_tmp = 1.6e-6
n_modes_tmp = 3
my_dict = AO_correction( pup, phase_screen, wvl_0=wvl_tmp, wvl=wvl_tmp, n_modes=n_modes_tmp, lag=0, V_turb=50, return_in_radians=False, troubleshoot=True)

print( 'Test X.\nstd (when adjusting for wvl) of raw phase screen =\n {}\nstd after correcting tip/tilt =\n {}'.format(np.nanvar( (2*np.pi / wvl_tmp) * my_dict['corrected_opd'] ) , np.nanvar(pup * crop2center( calibrate_phase_screen2wvl(wvl_tmp, phase_screen) , pup ) ) ) )

#5 . correcting 14 modes with some reasonable lag can we reproduce NAOMI stats 

pup = AT_pupil(dim = D_pix, diameter = D_pix) 

n_modes_tmp = 14
my_dict = AO_correction( pup, phase_screen, wvl_0=0.6e-6, wvl=1.65e-6, n_modes=n_modes_tmp, lag=2.5e-3, V_turb=50, return_in_radians=False, troubleshoot=True)

print('strehl = {}'.format(np.exp(-np.nanvar( (2*np.pi / wvl_tmp) * my_dict['corrected_opd'] ))))



#%%%% OLD FUNCTIONS 


def AO_simulation_old(parameter_dict, sim_time = 0.01, wvl_list= [1.65e-6], save = False, saveAs = None):
    
    """
    simulate AO systemfor sim_time

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
    naomi_screens = []
    for i in range(no_iterations):
        

        
        phase_screen.add_row()
        naomi_screens.append( AO_correction( pup, phase_screen, wvl_0, wvl, n_modes, lag, V_turb, return_in_radians=True, troubleshoot=False ) )
        

    # ======
    # To get near the desired sampling we linearly interpolate between pixels on our phase screens  
    screens_between = int( round( lag / desired_dt ) ) # how many frames between lagged steps?
    dt = lag / screens_between 

    naomi_screens_interp = []
    if screens_between > 1:
        # warning if interpolating will be much longer then tau0
        if lag > 10 * r0 / V_turb :
            print('\n\n=== WARNING:\n AO lag much greeater then tau0, linear interpolation for smaller sampling times may not be valid')
        
        for i in range( len(naomi_screens) - 1 ):
        
            naomi_screens_interp = naomi_screens_interp + list( np.linspace( naomi_screens[i], naomi_screens[i+1], screens_between)[:-1] )
        
    else:
        naomi_screens_interp = naomi_screens
    
    hdu = fits.PrimaryHDU( naomi_screens_interp )
    
    hdu.header.set('what is', 'AO corrected phase screens (rad)' , 'at lambda = {}um'.format(round(wvl*1e6,3)))
    hdu.header.set('simulation time', sim_time, 'simulation time (s)')
    hdu.header.set('seeing', seeing, 'arcsec')
    hdu.header.set('L0', L0, 'outerscale (m)')
    hdu.header.set('D', D, 'telescope diameter (m)')
    hdu.header.set('D_pix', D_pix, '#pixels across telescope diameter')
    hdu.header.set('wvl_0', wvl_0, 'central wavelentgth of WFS (m)')
    hdu.header.set('wvl', wvl, 'wavelentgth where corrected phase screen is calculated (m) ')
    hdu.header.set('n_modes', n_modes, '#N modes sensed & corrected by AO system')
    hdu.header.set('lag', lag, 'latency of AO system (s)')
    hdu.header.set('V_turb', V_turb, 'velocity of phase screen (m/s)')
    hdu.header.set('pupil_geometry', pupil_geometry, 'Valid options are AT, UT, disk')
    hdu.header.set('dt', dt, 'phase screen sampling rate')
    
    
    
    test = fits.PrimaryHDU()
    test.header['simulation time'] = 0.1
    
    
    test_list = fits.HDUList([test, naomi_screens,naomi_screens]) 
    if save:
        hdu.writeto(saveAs,overwrite=True)
        
    return( hdu )