#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 15:44:21 2023

@author: bcourtne

baldr functions 2.0


To Do
======
double check mft is being used properly in getting output field !

- tip/tilt vis 
for i in np.linspace(0,2*np.pi,10):
    plt.figure()
    aaa= basis[1] * 2e2*np.sin(i) ;aaa[np.isnan(aaa)]=0;plt.imshow(np.abs(np.fft.fftshift( np.fft.fft2( np.exp(1j*aaa) ) ) )[len(aaa)//2-10:len(aaa)//2+10,len(aaa)//2-200:len(aaa)//2+200])
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.interpolate import interp2d, interp1d
from scipy.stats import poisson

import pyzelda.utils.aperture as aperture
import pyzelda.utils.zernike as zernike
import pyzelda.utils.mft as mft

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
    

    
    
class field:
  def __init__(self,fluxes, phases, wvls):
    """
    initialize a field flux(wvl,x,y) * exp(1j*phase(wvl,x,y))

    Parameters
    ----------
    fluxes : TYPE dictionary with {wvl_1:phi_1, wvl_2:phi_2 ... } 
        DESCRIPTION.where phi_i is a 2D array of phase (in radians!)
    phases : TYPE TYPE dictionary with {wvl_1:flux_1, wvl_2:flux_2 ... }
        DESCRIPTION.  where flux_i is a 2D array with units like  ph/s/m2/nm
    wvls : TYPE array like 
        DESCRIPTION.array of the wvls 

    Returns
    -------
    None.

    """
    self.phase = {w:p for w,p in zip(wvls,phases)} 
    self.flux = {w:f for w,f in zip(wvls,fluxes)} 
    self.wvl = wvls
  


  def define_pupil_grid(self,dx, D_pix):
      
    if (len( self.flux )>0):
        try: 
            self.nx_size = self.flux[list(input_field.flux.keys())[0]].shape[0]
        except:
            self.nx_size = 0
    else:
        raise TypeError('WARNING: input fluxes are empty. Cannot assign grid size attribute nx_size')
      
    self.dx = dx
    self.D_pix = D_pix
      
          

  def flux_loss(self,losses):
      self.flux = {w:f * losses for w,f in zip(self.wvl, self.flux)}
      
  def phase_shift(self,shifts):
      self.phase = {w:p + shifts for w,p in zip(self.wvl, self.phase)}
      
      
  #def detect(self, detector): # returns a signal 
  


class zernike_phase_mask:
    def __init__(self, A=1, B=1, phase_shift_diameter=1e-6, f_ratio=21, d_on=26.5e-6, d_off=26e-6, glass_on='sio2', glass_off='sio2'):
        """
        

        Parameters
        ----------
        A : TYPE, float between 0,1
            DESCRIPTION. phase mask off-axis transmission (in phase shift region). The default is 1.
        B : TYPE, float between 0,1
            DESCRIPTION.  phase mask on-axis transmission (outside phase shift region). The default is 1.
        phase_shift_diameter : TYPE, float
            DESCRIPTION. diameter (m) where the phase shift gets applied
        f_ratio : float, optional
            DESCRIPTION. f ratio The default is 0.21.
        d_on : TYPE, float
            depth (m) of on-axis (phase shift region) part of mask 
        d_off : TYPE
            depth (m) of off-axis part of mask
        glass_on : TYPE,string
            name of glass in on-axis (phase shift region) part of mask (see nglass function for options)
        glass_off : TYPE, string
            name of glass in on-axis (phase shift region) part of mask 

        Returns
        -------
        a phase mask for ZWFS

        """
        self.A = A 
        self.B = B
        self.phase_shift_diameter = phase_shift_diameter
        self.f_ratio=f_ratio
        self.d_on = d_on
        self.d_off = d_off
        self.glass_on = glass_on
        self.glass_off = glass_off
 
    
    def optimise_depths(self, desired_phase_shift, across_desired_wvls, fine_search = False, verbose=True):
        
        """
        calculate the optimal on & off axis depths in mask to keep phase shifts
        as close as possible to desired_phase_shift across_desired_wvls 
        (minimize rmse)
        
        Parameters
        ----------
        
        desired_phase_shift : float
            desired phase shift (degrees)
       
        
        across_desired_wvls: array like
            array of wavelengths(m) to optimize (i.e. keep phaseshift as close to target_phase_shift as possible)
            
        for now we just do this the stupid but robust way of a manual grid search over reasonable depth (1um for on axis depth (d_on), wvl_c/10 for d_off) increments 
        
        output
        parameter_dict : dictionary with updated results (optimized depths for target phase shift over wvls )
        """
            
        # abbreviations
        glass_on = self.glass_on
        glass_off = self.glass_off 
        wvls = across_desired_wvls
        
        if fine_search :
            g1 = np.linspace(20e-6, 30e-6, 20) 
            g2 = np.arange( -10e-6, 10e-6,  (wvls[-1] + wvls[0]) / 2 / 40 )
        else :
            g1 = np.linspace(20e-6, 50e-6, 20) 
            g2 = np.arange( -10e-6, 10e-6,  (wvls[-1] + wvls[0]) / 2 / 20 )
        
        #init grid and best rmse 
        #rmse_grid = np.inf * np.ones([len(g1),len(g2)])
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
    
                rmse = np.sqrt( np.mean( (np.deg2rad(desired_phase_shift) - np.array(phase_shifts))**2 ) ) #rad

                if rmse < best_rmse: #then store parameters 
                    
                    best_rmse = rmse  #rad (has to be same units as rmse calculation above)
                    #if verbose:
                    #    print(f'best rmse={rmse}')
                    
                    mean_shift = np.rad2deg( np.mean(  np.array(phase_shifts) ) ) #degrees
                    std_shift = np.rad2deg( np.std(  np.array(phase_shifts) ) ) #degrees
                    
                    don_opt = don
                    doff_opt = doff
                    
        self.d_on = don_opt 
        self.d_off = doff_opt
        
        if verbose:
            
            print( f'\n---\noptimal depths [d_on, d_off] = [{don_opt},{doff_opt}] (units should be m)\n\
                  phase rmse (deg) at found optimal depths = {np.rad2deg(best_rmse)}\n\
                  wvl average phase shift (+-std) at optimal depths {mean_shift}+-{std_shift}\n')
              
            #plot results
            thetas = np.array( [self.phase_mask_phase_shift( w ) for w in wvls] ) #degrees

            plt.figure()
            plt.plot(1e6 * wvls,  thetas, color='k')
            plt.ylabel('phase shift [deg] ',fontsize=15)
            plt.xlabel(r'wavelength [$\mu$m]',fontsize=15)
            plt.grid()
            plt.gca().tick_params(labelsize=15)
            plt.show()

        
    def phase_mask_phase_shift(self, wvl): 
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
        phase shift (degrees) applied by on-axis region in mask at given wavelength
        
               ----------
               |n_on     | n_air
               |         | 
        -------           -----
               |         |    
               |         | n_off
        -----------------------
        
        
        """
        n_on = nglass(1e6 * wvl, glass=self.glass_on)[0]
        n_off = nglass(1e6 * wvl, glass=self.glass_off)[0]
        n_air = nglass(1e6 * wvl, glass='air')[0]
        
        opd = self.d_on * n_on - ( (self.d_on-self.d_off)*n_air + self.d_off * n_off )
        phase_shift = (2 * np.pi / wvl) * opd  
        
        return(np.rad2deg(phase_shift))

    
    def sample_phase_shift_region(self, nx_pix, dx, wvl_2_count_res_elements = 1.65e-6, verbose=True):
        """
        create a grid that samples the region where phase mask applies phase shift. 
        1 = phase shift applied, 0 = no phase shift applied

        Parameters
        ----------
        nx_pix : TYPE int
            DESCRIPTION. sample on grid of size nx_pix x nx_pix 
        dx : TYPE float
            DESCRIPTION. the spatial differential spacing in focal plane grid (m)
        wvl_2_count_res_elements : TYPE, optional float
            DESCRIPTION. what wavelength do we count resolution elements to report 
            circular diameter of phase shift in resolution elements (F*wvl).The default is 1.65e-6.

        Returns
        -------
        None.

        """
        # 
        
        if self.phase_shift_diameter/dx < 1:
            print('\n---\nWARNING: in self.sample_phase_shift_region(); the phase_shift_diameter/dx<1\n\
                  this implies phase shift region is less then 1 pixel in focal plane.\n\
                      Consider increasing phase_shift_diameter in the mask or decreasing dx\n')
        
        phase_shift_region = aperture.disc(dim=nx_pix, size= round(self.phase_shift_diameter/dx), diameter=True) 
        
        if verbose:
            print( f'\n---\n discretization error of phase shift diameter = {dx*(self.phase_shift_diameter/dx-round(self.phase_shift_diameter/dx))}m\n\
                   #resolution elements at {np.round(1e6*wvl_2_count_res_elements,2)}um across phase shift diameter \
                       = {np.round(self.phase_shift_diameter/(wvl_2_count_res_elements * self.f_ratio),3)}\n' )
        
        self.phase_shift_region = phase_shift_region 
        self.nx_size_focal_plane = nx_pix 
        self.dx_focal_plane = dx
        self.x_focal_plane = np.linspace(-nx_pix  * dx / 2, nx_pix  * dx / 2, nx_pix)
        
    
    def get_filter_design_parameter(self,wvl):
        """
        return the combined filter parameter defined by 
        Jesper Glückstad & Darwin Palima in Generalized Phase Contrast textbook

        Parameters
        ----------
        wvl : TYPE float 
            DESCRIPTION. wavelengths to calculate the combined filter parameter

        Returns
        -------
        combined filter parameter 

        """
        theta = np.deg2rad( self.phase_mask_phase_shift( wvl ) )
        
        return( self.B/self.A * np.exp(1j * theta) - 1  )

    
    def get_output_field(self, input_field , wvl_lims=[-np.inf, np.inf] , nx_size_focal_plane = None, dx_focal_plane=None,keep_intermediate_products=False):
        """
        get the output field (of class field) from a input field given the 
        current filter 

        Parameters
        ----------
        input_field : TYPE
            DESCRIPTION.
        wvl_lims : TYPE, optional
            DESCRIPTION. The default is [-np.inf, np.inf].
        focal_plane_nx_pix: TYPE, int
            DESCRIPTION. Number of pixels used to sample focal plane (PSF and phase mask) 
            when applyiong Fourier transform (mft). Default is None which sets nx_size_focal_plane=input_field.nx_size 
            (ie.e same grid size as input field)
        dx_focal_plane: TYPE float
            DESCRIPTION. pixel scale in focal plane (m/pix). Default is None which sets dx_focal_plane=self.phase_shift_diameter/20 
            (ie. default is that dx is set so there is 20 pixels over the phase shift region diameter)
        keep_intermediate_products : TYPE, optional
            DESCRIPTION. The default is False.

        Raises
        ------
        TypeError
            DESCRIPTION.

        Returns
        -------
        output_field (class field)
        
                
                     phase mask
            |\          :         /|----- 
            |   \       :      /   |
            |       \  _:  /       |
            |         [_           |    -> ->
            |       /   :  \       |
            |   /       :      \   |  
            |/          :         \|-----
            
         Psi_A        Psi_B      Psi_C (output_field)


        """
        
        """if not hasattr(self,'phase_shift_region'):
            
            raise TypeError('\n---\nphase_shift_region attribute has not been initialized,\n\
                            try method self.sample_phase_shift_region(nx_pix, dx, \
                             wvl_2_count_res_elements = 1.65e-6, verbose=True)\n')
        """

        if nx_size_focal_plane==None:
            nx_size_focal_plane = input_field.nx_size
            
        if dx_focal_plane==None:
            dx_focal_plane = self.phase_shift_diameter/20

        # wavelengths defined in the input field 
        input_wvls = np.array( input_field.wvl )

        #only calculate the output field for the following wvls
        wvl_filt = (input_wvls <= wvl_lims[1]) & (input_wvls >= wvl_lims[0])
        wvls = input_wvls[ wvl_filt ]
        if len(wvls)<1:
            raise TypeError('\n---no wavelengths defined in input_field.wvl are within the wvl limits (wvl_lims)\n')
        
        
        # phase shifts for each wvl
        thetas = np.deg2rad( np.array([self.phase_mask_phase_shift( w ) for w in wvls]) )  #radians

       
        
        # combined filter parameter defined by [Jesper Glückstad & Darwin Palima in Generalized Phase Contrast textbook]
        self.combined_filter_parameter = self.B/self.A * np.exp(1j * thetas) - 1 
        
        
        # init output field 
        output_field = field(fluxes = {}, phases={}, wvls=[])
        
        # now create fields from phase mask filter 
        if keep_intermediate_products:
            self.Psi_A = []
            self.Psi_B = []
            self.Psi_C = []
            self.b = []
            self.diam_mask_lambda_on_D = []
            
       
        # Sample phase mask in focal plane 
        self.sample_phase_shift_region( nx_pix=nx_size_focal_plane, dx=dx_focal_plane, verbose=False) # 
        
        # phase mask filter for each wavelength
        H = {w: self.A*(1 + (self.B/self.A * np.exp(1j * theta) - 1) * self.phase_shift_region  ) for w,theta in zip(wvls,thetas) }
        
        for w in wvls:
            
            # definition of m1 parameter for the Matrix Fourier Transform (MFT)
            # this should be number of resolution elements across the focal plane grid
            m1 = (self.x_focal_plane[-1] - self.x_focal_plane[0] ) / (w * self.f_ratio) 

        
            # --------------------------------
            # plane B (Focal plane)

            
            Psi_A = input_field.flux[w] * np.exp(1j * input_field.phase[w])
            
            #Psi_B = np.fft.fftshift( np.fft.fft2( Psi_A ) )#mft.mft(Psi_A, Na, Nb, n_res_elements, cpix=False)
        
            #Psi_C = np.fft.ifft2( H[w] * Psi_B ) #mft.imft( H[w] * Psi_B , Na, Nb, n_res_elements, cpix=False) 
            
            Psi_B = mft.mft(Psi_A, input_field.nx_size, nx_size_focal_plane , m1, cpix=False)
            
            #print(R_mask , input_field.nx_size, nx_size_focal_plane , H[w].shape, Psi_A.shape, Psi_B.shape)
        
            Psi_C = mft.imft( H[w] * Psi_B , nx_size_focal_plane, input_field.nx_size, m1, cpix=False) 
            
            output_field.flux[w] = abs(Psi_C) #check A=1,B=1,theta=0 that Psi_C preserves photon flux
            output_field.phase[w] = np.angle(Psi_C)
            output_field.wvl.append(w)
    
            
    
            if keep_intermediate_products:
                self.b.append( mft.imft( FPM.phase_shift_region * Psi_B, nx_size_focal_plane, input_field.nx_size, m1, cpix=False) )
                self.Psi_A.append( np.array( Psi_A ) )
                self.Psi_B.append( np.array( Psi_B ) )
                self.Psi_C.append( np.array( Psi_C ) )
                self.diam_mask_lambda_on_D.append( self.phase_shift_diameter / (w * self.f_ratio) )
                
            
        return( output_field )
    

class detector:
    
    def __init__(self, npix, pix_scale, QE={w:1 for w in np.linspace(0.9e-6,2e-6,100)}):
        
        self.npix = npix
        self.pix_scale = pix_scale
        self.det = np.zeros([ self.npix ,  self.npix ] )
        self.qe = QE
        
    def interpolate_QE_to_field_wvls(self, field):
        
        fn = interp1d(self.qe.keys(), self.qe.values() ,bounds_error=False, fill_value = 0)
        
        self.qe = fn(field.wvl)

        
    def detect_field(self, field, DIT=1,  include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True):
        # have to deal with cases that field grid is different to detector grid
        # have to deal with field wavelengths (look at field class structure), detector wavelengths defined in self.qe dict
        # IMPORTANT TO HAVE FAST OPTION WHEN GRIDS ARE SET PROPERLY FROM THE START TO AVOID 2D INTERPOLATION!!
        
        
        # field and detector x coordinates 
        x_field = np.linspace(-field.nx_size * field.dx / 2, field.nx_size * field.dx / 2, field.nx_size)
        x_det =  np.linspace(-self.npix * self.pix_scale / 2, self.npix * self.pix_scale / 2, self.npix)
        
        #wavelengths
        det_wvl = list( self.qe.keys() ) # m
        field_wvls = np.array( field.wvl ) # m
        
        # to deal with case len(wvl)=1
        if not set(field_wvls).issubset(set(det_wvl)): # if wvls defined in field object not defined in detector (quantum efficiency) then we interpolate them
            self.interpolate_QE_to_field_wvls(self, field)

            
        if field.nx_size * field.dx > self.npix * self.pix_scale:
            print('WARNING: some of the input field does not fall on the detector')
        
        if grids_aligned: 
            pw = int(self.pix_scale // field.dx) # how many field grid points fit into a single detector pixel assuming same origin
        
            for n in range(self.det.shape[0]):
                for m in range(self.det.shape[1]):
                    if ph_per_s_per_m2_per_nm:
                        if include_shotnoise:
                        
                            P_wvl = np.array( [DIT * self.qe[w] * np.sum( field.flux[w][pw*n:pw*(n+1), pw*m:pw*(m+1)] * field.dx**2 ) for w in field_wvls] )
                            self.det[n,m] =  poisson.rvs( integrate( P_wvl  , field_wvls * 1e9) ) # draw from poission distribution with mean = np.trapz( P_wvl  , field_wvls)
                            #note 1e9 because field_wvls should be m and flux should be ph_per_s_per_m2_per_nm
                        else:
                            # integrate to get number of photons 
                            P_wvl = np.array( [DIT * self.qe[w] * np.sum( field.flux[w][pw*n:pw*(n+1), pw*m:pw*(m+1)] * field.dx**2 ) for w in field_wvls] )
                            self.det[n,m] =  integrate( P_wvl  , field_wvls * 1e9) # 
                            #DIT * self.qe[wvl] * np.sum( flux[pw*n:pw*(n+1), pw*m:pw*(m+1)] * field.dx**2 ) 
                    else:
                        raise TypeError('make sure flux units are ph_per_s_per_m2_per_nm othersie integrals will be wrong\n\
                                        look at star2photons function')
        
        else: # we try  interpolate to align grids 
            
            print('\n-------\nWARNING detect_field not well tested when grids_aligned=False.. be careful!!\n\n------')
            pixel_window = self.pix_scale / field.dx # how many field grid points fit into a single detector pixel assuming same origin
       
            if pixel_window >= 1:
                # interpolate field such that dx is nearest multiple of pix_scale
                
                                      
                """\
                new_pixel_window =  np.round( pixel_window ).astype(int) = self.pix_scale / new_dx
                self.pix_scale / field.dx * M = new_pixel_window  
                
                => M = (np.round( pixel_window ).astype(int)/ pixel_window )as.type(int) =  ( new_pixel_window  / pixel_window )as.type(int)
                
                therefore
                new_dx =  self.pix_scale/new_pixel_window =  self.pix_scale / (self.pix_scale / field.dx * M) = field.dx * M
                """          
                # field.dx * M = self.pix_scale 
                # np.round( pixel_window ).astype(int) => self.pix_scale / field.dx  = self.pix_scale / (x_field[-1] - x_field[0])/len(x_field) 
                M = (np.round( pixel_window ).astype(int) / pixel_window ).astype(int) 
                new_pixel_window =  np.round( pixel_window ).astype(int) 
                new_dx =  field.dx * M
                
                # interpolate onto detector grid, with interp fn returning zero outside of field grid 
                x_field_new_x = np.linspace( x_det[0], x_det[-1],  new_dx ) 
           
                pw = new_pixel_window
                
                interp_flux = {}
                
                for wvl in field_wvls:
                    flux = field[wvl].flux
                    fn = interp2d(x_field,x_field, flux, bounds_error=False,fill_value=0) #interp(x_field, field)
                    
                    #new flux interpolated onto xgrid such that M * new_dx = self.pix_scale => M = new_pixel_window = self.pix_scale / new_dx  is int 
                    interp_flux[wvl] = fn(x_field_new_x, x_field_new_x) 

                
                for n in range(self.det.shape[0]):
                    for m in range(self.det.shape[0]):
                        # integrate 
                        #self.det[n,m] = DIT * self.qe[wvl] * new_flux[pw*n:pw*(n+1), pw*m:pw*(m+1)] * new_dx**2
                        P_wvl = [DIT * self.qe[w] * np.sum( interp_flux[w][pw*n:pw*(n+1), pw*m:pw*(m+1)] * new_dx**2 ) for w in field_wvls]
                        self.det[n,m] =  np.trapz( P_wvl  , field_wvls * 1e9) # = total #ph
                
                #if include_shotnoise
                #Add Shot noise ! 
                
                
            elif field.dx < self.pix_scale:
                raise TypeError('field.dx > self.pix_scale \ntry increase detector\
                                pixel size to at least correspond to field pixel size (dx) or bigger')
    
            else:
                raise TypeError('not all cases met for checking field.dx > self.pix_scale etc')
                

        
    def add_ron(self, sigma):
        # sigma = standard deviation , ron always 0 mean
        self.det = (self.det + np.random.normal(loc=0, scale=sigma, size=self.det.shape) ).astype(int)


    def ZWFS_phase_estimator_1(self, A, B, b, P, theta, exp_order=1):
        
        aa = b * (A**2 * P - A * B * P * np.cos(theta) )
        bb = 2 * A * b * B * P * np.sin(theta)
        cc =  -self.det + (A**2 * P**2 + b**2 * (A**2 + B**2 - 2 * A * B * np.cos(theta) ) +\
            2 * b * ( -A**2 * P  + A * B * P * np.cos(theta) ) ) 
            
        if exp_order == 1:
            phi = - cc / bb
            
        if exp_order == 2:
            phi = ( (-bb + np.sqrt(bb**2 - 4 * aa * cc) ) / (2 * aa) , ( -bb - np.sqrt(bb**2 - 4 * aa * cc) ) / (2 * aa) )
        
        return( phi )  
    
        """
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
        """
# diffraction limit microscope (m) d = lambda/(2*NA) = lambda * F where F=D/focal length (https://www.telescope-optics.net/telescope_resolution.htm)
#class detector:
#    def __init__(self, N_pixels):
        
        
        
 
#def naomi_correction(fields):    
 
    

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
    
    



def nglass(l, glass='sio2'):
    """
    (From Mike Irelands opticstools!)
    Refractive index of fused silica and other glasses. Note that C is
    in microns^{-2}
    
    Parameters
    ----------
    l: wavelength (um)
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



def aggregate_array(array_A, new_shape, how='mean'):
    pw = int( array_A.shape[0] / new_shape[0] )
    new_array = np.zeros( new_shape )
    
    if how=='mean':
        for n in range(new_array.shape[0]):
            for m in range(new_array.shape[1]):
                new_array[n,m] = np.nanmean( array_A[pw*n:pw*(n+1), pw*m:pw*(m+1)] )
        return( new_array )
    
    elif how=='sum':
        for n in range(new_array.shape[0]):
            for m in range(new_array.shape[1]):
                new_array[n,m] = np.nansum( array_A[pw*n:pw*(n+1), pw*m:pw*(m+1)] )
        return( new_array )
       
    else:
        raise TypeError('how method specified doesn"t exist.\ntry how="mean" or how="sum"')
   
    

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
    a.copy()[a.shape[-1]//2-b.shape[-1]//2 : a.shape[-1]//2+b.shape[-1]//2 , a.shape[-1]//2-b.shape[-1]//2 : a.shape[-1]//2+b.shape[-1]//2 ] = b.copy()
    
    return(a)



def AT_pupil(dim, diameter, spiders_thickness=0.008, strict=False, cpix=False):
    '''Auxillary Telescope theoretical pupil with central obscuration and spiders
    
    function adapted from pyzelda..
    
    
    Parameters
    ----------
    dim : int
        Size of the output array (pixels)
    
    diameter : int
        Diameter the disk (pixels)
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


def integrate(y,x): 
    #on average ~70% quicker than np.trapz and gives same result
    Y = np.sum((y[1:]+y[:-1])/2 * (x[1:] - x[:-1])) 
    return(Y)
#%% testing 


# INPUTFIELD 




# input pupil field grid (WORKS BEST WHEN dim=D_pix )
dim=2**9
D_pix = 2**9

pup = pick_pupil('AT', dim=dim, diameter=D_pix ) #aperture.disc(dim=dim, size=D_pix,diameter=True) #

# pupil basis
basis = zernike.zernike_basis(nterms=10, npix=D_pix)

# input field 
wvls = np.linspace( 1.5e-6, 1.7e-6, 10)

ph_flux_H = star2photons('H', 8, airmass=1.3, k = 0.18, ph_m2_s_nm = True)
fluxes = [pup * (ph_flux_H + noise) for noise in np.random.normal(0, 1e-5*ph_flux_H, len(wvls))]

# NOTE THINGS GET VERY NON-LINEAR IN HIGHER ORDER MODES IF a > 1 ( i.e. 5e-1*basis[5])
phase_tmp = 5e-1 * basis[5]
phase_tmp[np.isnan(phase_tmp)] = 0

phase_tmp2 = np.pad(phase_tmp, [(int((dim-D_pix)/2), int((dim-D_pix)/2)), (int((dim-D_pix)/2), int((dim-D_pix)/2))], mode='constant')#putinside_array( pup.copy(), phase_tmp.copy())
phases = [phase_tmp2 for i in range(len(wvls))]


# focal plane Filter
A=1
B=1
f_ratio=21 
d_on=26.5e-6
d_off=26e-6
glass_on='sio2'
glass_off='sio2'

# focal plane grid
phase_shift_diameter = 1 * f_ratio * wvls[0]   ##  f_ratio * wvls[0] = lambda/D  given f_ratio

nx_size_focal_plane = dim #grid size in focal plane 
N_samples_across_phase_shift_region = 10 # number of pixels across pghase shift region 
dx_focal_plane = phase_shift_diameter / N_samples_across_phase_shift_region  # 



# generate field class for input field 
input_field = field(fluxes , phases, wvls )
# define the grid 
input_field.define_pupil_grid( dx = 1.8 / D_pix, D_pix = D_pix )


# init filter 
FPM = zernike_phase_mask(A,B,phase_shift_diameter,f_ratio,d_on,d_off,glass_on,glass_off)

# optimize depths 
FPM.optimise_depths(desired_phase_shift=90, across_desired_wvls=wvls ,verbose=True)


start_time = time.time()
# get output field 
output_field = FPM.get_output_field( input_field, wvl_lims=[0,100], \
                                    nx_size_focal_plane = nx_size_focal_plane , dx_focal_plane = dx_focal_plane, keep_intermediate_products=True )

output_field.define_pupil_grid(dx=input_field.dx, D_pix=input_field.D_pix)
print("once field, masks are intialized time for getting output field:--- %s seconds ---" % (time.time() - start_time))



"""# init filter 
FPM2 = zernike_phase_mask(A,B,phase_shift_diameter,f_ratio,d_on,d_off,glass_on,glass_off)
# optimize depths 
FPM2.optimise_depths(desired_phase_shift=0, across_desired_wvls=wvls ,verbose=True)
# sample our focal plane mask (FPM). 
FPM2.sample_phase_shift_region(nx_pix=2**9, dx = FPM2.f_ratio * np.min(wvls) / 2 , wvl_2_count_res_elements = 1.65e-6, verbose=True)
# get output field 
output_field2 = FPM2.get_output_field( input_field, wvl_lims=[0,100], keep_intermediate_products=True )
"""

# sanity check - phase_shift_diameter = 1 * f_ratio * wvls[0]  should always be dcomparable to PSF size independent of nx_size_focal_plane, dx_focal_plane
fig, ax = plt.subplots(4,1,figsize=(10,20))
ax[0].set_title('input phase')
ax[0].imshow( input_field.phase[wvls[0]] )

ax[1].set_title('focal plane mask (cropped')
ax[1].imshow( FPM.phase_shift_region)

ax[2].set_title('focal plane PSF')
ax[2].imshow(  abs( FPM.Psi_B[0] )**2 )

ax[3].set_title('output intensity ')
ax[3].imshow( abs(output_field.flux[wvls[0]])**2 )


# ++++++++++++++++++++++
# DETECT 
pw = 2**4
DIT=1
npix = input_field.flux[wvls[0]].shape[0]//pw
pix_scale = input_field.dx * pw
det = detector(npix=npix, pix_scale = pix_scale, QE={w:1 for w in input_field.wvl})

start_detect = time.perf_counter()
det.detect_field(output_field , DIT=1, include_shotnoise=True,ph_per_s_per_m2_per_nm=True,grids_aligned=True)
end_detect = time.perf_counter()
print(f'\n{end_detect-start_detect}s')

# need to fix include_shotnoise=True,

# check sum(det) == np.trapz(DIT * sum(field[w] * dx**2), wvl)
print('check sum(det) ~ np.trapz(DIT * sum(field[w] * dx**2), wvl)\n\nnp.trapz(DIT * sum(field[w] * dx**2), wvl)=')
print('  ', np.trapz([DIT * np.sum(input_field.flux[w] * input_field.dx**2) for w in  input_field.flux.keys()], 1e9*np.array(list(input_field.flux.keys()))) )
print( '\nsum(det)=' )
print( '  ', np.sum(det.det))


fig,ax = plt.subplots(1,2)
ax[1].imshow(det.det)
ax[1].set_title('detector')
ax[0].imshow(phase_tmp)
ax[0].set_title('input phase')
ax[1].axis('off')
ax[0].axis('off')
plt.tight_layout()

#%% calibration phase maskPhase estimate 

FPM_cal = zernike_phase_mask(A,B,phase_shift_diameter,f_ratio,d_on,d_off,glass_on,glass_off)
# optimize depths 
FPM_cal.optimise_depths(desired_phase_shift=0, across_desired_wvls=wvls ,verbose=True)

output_field_cal = FPM_cal.get_output_field( input_field, wvl_lims=[0,100], \
                                    nx_size_focal_plane = nx_size_focal_plane , dx_focal_plane = dx_focal_plane, keep_intermediate_products=False )

output_field_cal.define_pupil_grid(dx=input_field.dx, D_pix=input_field.D_pix)

det_cal = detector(npix=npix, pix_scale = pix_scale, QE={w:1 for w in input_field.wvl})

det_cal.detect_field(output_field_cal , DIT=1, include_shotnoise=True, ph_per_s_per_m2_per_nm=True,grids_aligned=True)

#plt.imshow(output_field_cal.flux[wvls[0]])

#  TRY JUST TAKE MEAN OF B ON DETECTOR PIXELS AT CENTRAL WAVELENGTH 
#plt.imshow( aggregate_array(abs(FPM.b[int(len(wvls)//2)]), det.det.shape, how='mean') )
b_est = aggregate_array(abs(FPM.b[int(len(wvls)//2)]), det.det.shape, how='mean')
#

# OK NOW try detect the field.. because at single wvl isnt working 
det_b = detector(npix=npix, pix_scale = pix_scale, QE={w:1 for w in input_field.wvl})

b_field = field(fluxes=[abs(FPM.b[i]) for i,_ in enumerate(wvls) ],\
                phases=[np.angle(FPM.b[i]) for i,_ in enumerate(wvls) ], wvls=wvls)
    
b_field.define_pupil_grid(dx=input_field.dx, D_pix=input_field.D_pix)
det_b.detect_field(b_field, DIT=1, include_shotnoise=False, ph_per_s_per_m2_per_nm=True,grids_aligned=True)

b_est2 = det_b.det


# does one of these work? 
phi_est = det.ZWFS_phase_estimator_1(A, B, b_est, det_cal.det, np.deg2rad( FPM.phase_mask_phase_shift(np.mean(wvls))) , exp_order=1)
phi_est2 = det.ZWFS_phase_estimator_1(A, B,  1e-3*b_est2, det_cal.det, np.deg2rad( FPM.phase_mask_phase_shift(np.mean(wvls))) , exp_order=1)


app_tmp =  zernike.zernike_basis(nterms=1, npix=det.det.shape[0])[0]
fig,ax = plt.subplots(1,4,figsize=(10,5))
ax[0].imshow(det.det)
ax[1].imshow(det_cal.det)
ax[2].imshow(b_est)
ax[3].imshow(app_tmp * phi_est2)

ax[0].set_title('detector (theta = {})'.format(round( FPM.phase_mask_phase_shift(np.mean(wvls)),1)) )
ax[1].set_title('detector (theta = 0)')
ax[2].set_title('b estimate')
ax[3].set_title('phase estimate')

for i in range(len(ax)):
    ax[i].axis('off')

plt.tight_layout()


np.repeat(det.det, 5, axis=1).repeat(5, axis=0) 

#plt.plot( phi_est[:, det.det.shape[0]//2] )
#X = (det.det - det_cal.det)/b_est
#X * iteraction_matrix = phi => iteraction_matrix = X^-1 * phi 
#np.linalg.inv((det.det - det_cal.det)/b_est) @ 
# (np.ones( [phases[0].shape[0],det.det.shape[0]]) @ det.det @ np.ones( [det.det.shape[0], phases[0].shape[0]]))


#%% speed tests

"""def integrate(y,x): 
    Y = np.sum((y[1:]+y[:-1])/2 * (x[1:] - x[:-1])) 
    return(Y)"""

aaa=np.linspace(0,10,100); bbb=np.linspace(0,1,100)
t_trapz=[]
t_sum = []
residuals = []
for i in range(1000):
    start_trapz = time.perf_counter()
    t = np.trapz(aaa,bbb)
    end_trapz = time.perf_counter()
    
    start_sum = time.perf_counter()
    s = integrate(np.array(aaa),np.array(bbb)) #np.sum((aaa[1:]+aaa[:-1])/2 * (bbb[1:] - bbb[:-1])) #np.sum(aaa[1:] * np.diff(bbb))
    end_sum = time.perf_counter()
    
    t_trapz.append( end_trapz- start_trapz )
    t_sum.append( end_sum- start_sum )
    residuals.append( s-t)
plt.hist( np.log10(t_trapz),bins=40, label='trapz' );plt.hist( np.log10(t_sum),bins=40,  label='sum' ); plt.legend(); plt.xlabel('log10(delta t)')
print(f'max residuals = {np.max( residuals )}, \nintegrate fn = {np.mean(np.array( t_sum )/np.array( t_trapz) )} quicker')