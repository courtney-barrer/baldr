#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 15:44:21 2023

@author: bcourtne

baldr functions 2.0


To Do
======
there is some factor 10 error in  b or P when A=B=1 for phase estimate


- tip/tilt vis 
for i in np.linspace(0,2*np.pi,10):
    plt.figure()
    aaa= basis[1] * 2e2*np.sin(i) ;aaa[np.isnan(aaa)]=0;plt.imshow(np.abs(np.fft.fftshift( np.fft.fft2( np.exp(1j*aaa) ) ) )[len(aaa)//2-10:len(aaa)//2+10,len(aaa)//2-200:len(aaa)//2+200])
"""
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import time
from scipy.interpolate import interp2d, interp1d
from scipy.stats import poisson
from astropy.io import fits

import aotools
import pyzelda.utils.aperture as aperture
import pyzelda.utils.zernike as zernike
import pyzelda.utils.mft as mft


from scipy import linalg
import scipy

from numpy import fft
import numba
from scipy.special import gamma, kv
from scipy.stats import multivariate_normal

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
  


  def define_pupil_grid(self, dx, D_pix=None, center=(0,0)):
    """
      

      Parameters
      ----------
      dx : TYPE
          DESCRIPTION.
      D_pix : TYPE, optional
          DESCRIPTION. number of pixels across telescope pupil The default is None.
      center : TYPE, optional
          DESCRIPTION. center of x,y coordinates The default is (0,0)


      Raises
      ------
      TypeError
          DESCRIPTION.

      Returns
      -------
      None.

      """
    if (len( self.flux )>0):
        try: 
            self.nx_size = self.flux[list(self.flux.keys())[0]].shape[0]
        except:
            print(' CANNOT READ SIZE OF INPUT FLUX FRAMES\nSETTING nx_size=0')
            self.nx_size = 0
    else:
        raise TypeError('WARNING: input fluxes are empty. Cannot assign grid size attribute nx_size')
      
    self.dx = dx
    self.D_pix = D_pix # pixels across telescope diameter (may be smaller then nx_size)

    self.x = center[0] + np.linspace(-self.dx * (self.nx_size //2) , self.dx * (self.nx_size //2) , self.nx_size) #position of the field
    self.y = center[1] + self.x.copy()

    self.X,self.Y = np.meshgrid(self.x, self.y)
    self.coordinates = np.vstack([self.X.ravel(), self.Y.ravel()]).T
    

  def applyDM(self, DM):
      
      if (not hasattr(DM, 'x') ) or (not hasattr(DM, 'y') ): # add coordinate system
          DM.x = np.linspace(-self.dx * (self.nx_size //2) , self.dx * (self.nx_size //2) , DM.N_act[0])
          DM.y =  np.linspace(-self.dx * (self.nx_size //2) , self.dx * (self.nx_size //2) , DM.N_act[1])
          DM.X, DM.Y = np.meshgrid(DM.x,DM.y)  
          DM.coordinates = np.vstack([DM.X.ravel(), DM.Y.ravel()]).T

      if DM.surface_type == 'continuous' :
          DM.nearest_interp_fn = scipy.interpolate.LinearNDInterpolator(DM.coordinates, DM.surface.reshape(1,-1)[0])
      elif DM.surface_type == 'segmented':
          DM.nearest_interp_fn = scipy.interpolate.NearestNDInterpolator(DM.coordinates, DM.surface.reshape(1,-1)[0] )
      else:
          raise TypeError('\nDM object does not have valid surface_type\nmake sure DM.surface_type = "continuous" or "segmented" ')
          
      dm_at_field_pt = DM.nearest_interp_fn( self.coordinates ) # these x, y, points may need to be meshed...and flattened
      
      
      dm_at_field_pt = dm_at_field_pt.reshape( self.nx_size, self.nx_size )

      phase_shifts = {w:2*np.pi/w * (2*np.cos(DM.angle)) * dm_at_field_pt for w in self.wvl} # (2*np.cos(DM.angle)) because DM is double passed
      
      field_despues = copy.copy(self)
      
      field_despues.phase = {w: field_despues.phase[w] + phase_shifts[w] for w in field_despues.wvl}
      
      return(field_despues)

  def calculate_b(self, FPM):
      
      nx_size_focal_plane = self.nx_size

      dx_focal_plane = FPM.phase_shift_diameter/10
      
      
      FPM.sample_phase_shift_region( nx_pix=nx_size_focal_plane, dx=dx_focal_plane, verbose=False) # 
            
      # phase mask filter for each wavelength
      #thetas = np.deg2rad( np.array([self.phase_mask_phase_shift( w ) for w in self.wvl]) )  #radians
      #H = {w: FPM.get_filter_design_parameter(w) for w,theta in zip(self.wvl,thetas) }
            
      bs=[]
      for w in self.wvl:
          
          m1 = (FPM.x_focal_plane[-1] - FPM.x_focal_plane[0] ) / (w * FPM.f_ratio) 
          
          Psi_A = self.flux[w] * np.exp(1j * self.phase[w])
                
          Psi_B = mft.mft(Psi_A, self.nx_size, nx_size_focal_plane , m1, cpix=False)
          
          bs.append( mft.imft( FPM.phase_shift_region * Psi_B, nx_size_focal_plane, self.nx_size, m1, cpix=False) )
      
      b_field = field(fluxes=[abs(b) for b in bs], phases=[np.angle(b) for b in bs], wvls=self.wvl)
      
      if hasattr(self, "dx") :
          b_field.define_pupil_grid(dx= self.dx, D_pix=self.D_pix)
      
      return(b_field)
  
  def flux_loss(self,losses):
      self.flux = {w:f * losses for w,f in zip(self.wvl, self.flux)}
      
  def phase_shift(self,shifts):
      self.phase = {w:p + shifts for w,p in zip(self.wvl, self.phase)}


class DM:
  def __init__(self, surface, gain=1, angle=0, surface_type='continuous' ):
      self.surface = surface
      self.N_act = self.surface.shape
      self.gain = gain #um/V
      self.angle = angle # angle between DM surface normal & input beam (rad)
      self.surface_type = surface_type # 'continuous' or 'segmented' (continuous uses linear interperlation when applying correction to a field, segmented uses nearest neighbour interpolation)
      #self.pitch = pitch # seperation between actuators 
      #self.coords = coords #meshgrid of x,y,z coordinates 
     
      
  def update_shape(self, cmd):
      self.surface = (self.gain  * cmd).reshape(self.N_act)
      
  def create_IM(self, field, det):
      print('to do')

      


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
        self.f_ratio = f_ratio
        self.d_on = d_on
        self.d_off = d_off
        self.glass_on = glass_on
        self.glass_off = glass_off
        
        #nx_size_focal_plane = nx_size_focal_plane , dx_focal_plane = dx_focal_plane,
        #phase_shift_diameter = rad_lam_o_D * f_ratio * wvls[0]   ##  f_ratio * wvls[0] = lambda/D  given f_ratio
        #dx_focal_plane = phase_shift_diameter / N_samples_across_phase_shift_region  # dif elemetn in focal plane (m)
 
    
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

    
    def get_output_field(self, input_field , wvl_lims=[-np.inf, np.inf] , nx_size_focal_plane = None, dx_focal_plane=None, cold_stop_diam = None, keep_intermediate_products=False):
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
        cold_stop_diam: float or array like
            if scalar input is considered the cold stop radius (m). if array like it is considered to be the cold stop mask to multiply 
            by in the focal plane. Note sometimes cold stop given in f-ratio (N=f/D~1/2NA, where NA = nsin(alpha), alpha being acceptance angle,\
                                                                              just need to know pixel scale in mas to get mask radius)
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
            dx_focal_plane = self.phase_shift_diameter/10

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
            
       
        if cold_stop_diam == None:
            # Sample phase mask in focal plane 
            self.sample_phase_shift_region( nx_pix=nx_size_focal_plane, dx=dx_focal_plane, verbose=False) # 
            
            # phase mask filter for each wavelength
            H = {w: self.A*(1 + (self.B/self.A * np.exp(1j * theta) - 1) * self.phase_shift_region  ) for w,theta in zip(wvls,thetas) }
 
        else:
            # if cold stop given in f-ratio (N=f/D~1/2NA, where NA = sin(alpha), alpha being acceptance angle, just need to know pixel scale in mas)
            # create cold stop
            if hasattr(cold_stop_diam, "__len__") : # if has length attribute we consider it a diameter 
                cold_stop = aperture.disc(dim=nx_size_focal_plane, size = cold_stop_diam/dx_focal_plane, diameter=True)
            elif not hasattr(cold_stop_diam, "__len__") : # we assume input is actual cold stop mask
                
                if cold_stop_diam.shape == self.phase_shift_region.shape: #needs to have same shape for multiplying 
                    cold_stop = cold_stop_diam
                else:
                    raise TypeError('cold_stop_diam.shape != self.phase_shift_region.shape\n\
                                    make sure they have same shape if you want to input the cold stop mask yourself\n\
                                        alternatively you can just input a scalar diameter and the code will create (this will be slower) the appropiate mask')
            else:
                raise TypeError('cold_stop_diam fails check hasattr(cold_stop_diam, "__len__").\n\
                                check the input cold_stop_diam ')
            
            # phase mask filter for each wavelength
            H = {w: cold_stop * self.A*(1 + (self.B/self.A * np.exp(1j * theta) - 1) * self.phase_shift_region  ) for w,theta in zip(wvls,thetas) }
            
        
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
                self.b.append( mft.imft( self.phase_shift_region * Psi_B, nx_size_focal_plane, input_field.nx_size, m1, cpix=False) )
                self.Psi_A.append( np.array( Psi_A ) )
                self.Psi_B.append( np.array( Psi_B ) )
                self.Psi_C.append( np.array( Psi_C ) )
                self.diam_mask_lambda_on_D.append( self.phase_shift_diameter / (w * self.f_ratio) )
                
            
        return( output_field )
    

class detector:
    
    def __init__(self, npix, pix_scale, DIT = 1, ron=1, QE={w:1 for w in np.linspace(0.9e-6,2e-6,100)}):
        
        self.npix = npix
        self.pix_scale = pix_scale
        self.det = np.zeros([ self.npix ,  self.npix ] )
        self.qe = QE
        self.DIT = DIT 
        self.ron = ron #e-
        
        
    def interpolate_QE_to_field_wvls(self, field):
        
        fn = interp1d(list(self.qe.keys()), list(self.qe.values()) ,bounds_error=False, fill_value = 0)
        
        new_qe_values = fn(field.wvl)
        
        self.qe = {k:v for k,v in zip(field.wvl, new_qe_values)}

        
    def detect_field(self, field, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True):
        # have to deal with cases that field grid is different to detector grid
        # have to deal with field wavelengths (look at field class structure), detector wavelengths defined in self.qe dict
        # IMPORTANT TO HAVE FAST OPTION WHEN GRIDS ARE SET PROPERLY FROM THE START TO AVOID 2D INTERPOLATION!!
        
        
        self.det = np.zeros([ self.npix ,  self.npix ] )
        
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
                            P_wvl = np.array( [self.DIT * self.qe[w] * np.sum( field.flux[w][pw*n:pw*(n+1), pw*m:pw*(m+1)] * field.dx**2 ) for w in field_wvls] )
                            self.det[n,m] =  poisson.rvs( integrate( P_wvl  , field_wvls * 1e9) ) # draw from poission distribution with mean = np.trapz( P_wvl  , field_wvls)
                            #note 1e9 because field_wvls should be m and flux should be ph_per_s_per_m2_per_nm
                        else:
                            # integrate to get number of photons 
                            P_wvl = np.array( [self.DIT * self.qe[w] * np.sum( field.flux[w][pw*n:pw*(n+1), pw*m:pw*(m+1)] * field.dx**2 ) for w in field_wvls] )
                            self.det[n,m] =  integrate( P_wvl  , field_wvls * 1e9) # 
                            #DIT * self.qe[wvl] * np.sum( flux[pw*n:pw*(n+1), pw*m:pw*(m+1)] * field.dx**2 ) 
                    else:
                        raise TypeError('make sure flux units are ph_per_s_per_m2_per_nm othersie integrals will be wrong\n\
                                        look at star2photons function')
        
            # add the read noise 
            self.add_ron( self.ron ) 
            
            # convert detected signal to signal class
            det_sig = signal( self.det )
            
            return( det_sig )
        
        
        # ------ caution here, not well tested 
        else: # we try  interpolate to align grids 
            
            print('\n-------\nWARNING detect_field not well tested when grids_aligned=False.. be careful!!\n\n------')
            
            # field and detector x coordinates 
            x_field = np.linspace(-field.nx_size * field.dx / 2, field.nx_size * field.dx / 2, field.nx_size)
            x_det =  np.linspace(-self.npix * self.pix_scale / 2, self.npix * self.pix_scale / 2, self.npix)
        
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
                        
                        if ph_per_s_per_m2_per_nm:
                            if include_shotnoise:
                            
                                P_wvl = np.array( [self.DIT * self.qe[w] * np.sum( field.flux[w][pw*n:pw*(n+1), pw*m:pw*(m+1)] * field.dx**2 ) for w in field_wvls] )
                                self.det[n,m] =  poisson.rvs( integrate( P_wvl  , field_wvls * 1e9) ) # draw from poission distribution with mean = np.trapz( P_wvl  , field_wvls)
                                #note 1e9 because field_wvls should be m and flux should be ph_per_s_per_m2_per_nm
                            else:
                                # integrate to get number of photons 
                                P_wvl = np.array( [self.DIT * self.qe[w] * np.sum( field.flux[w][pw*n:pw*(n+1), pw*m:pw*(m+1)] * field.dx**2 ) for w in field_wvls] )
                                self.det[n,m] =  integrate( P_wvl  , field_wvls * 1e9) # 
                                #DIT * self.qe[wvl] * np.sum( flux[pw*n:pw*(n+1), pw*m:pw*(m+1)] * field.dx**2 ) 
                        else:
                            raise TypeError('make sure flux units are ph_per_s_per_m2_per_nm othersie integrals will be wrong\n\
                                            look at star2photons function')
                #if include_shotnoise
                #Add Shot noise ! 
                
                # add the read noise 
                self.add_ron( self.ron ) 
                
                # convert detected signal to signal class
                det_sig = signal( self.det )
                
                return( det_sig )
                
            elif field.dx < self.pix_scale:
                raise TypeError('field.dx > self.pix_scale \ntry increase detector\
                                pixel size to at least correspond to field pixel size (dx) or bigger')
    
            else:
                raise TypeError('not all cases met for checking field.dx > self.pix_scale etc')
                

        
    def add_ron(self, sigma):
        # sigma = standard deviation , ron always 0 mean
        self.det = (self.det + np.random.normal(loc=0, scale=sigma, size=self.det.shape) ).astype(int)


    
    
    
# diffraction limit microscope (m) d = lambda/(2*NA) = lambda * F where F=D/focal length (https://www.telescope-optics.net/telescope_resolution.htm)
#class detector:
#    def __init__(self, N_pixels):
        
        
        
 
#def naomi_correction(fields):    
class signal():
     def __init__(self, signal):
         self.signal = signal
         
    
     def ZWFS_phase_estimator_1(self, A, B, b, P, theta, exp_order=1):
        # note b needs to be detected with the same detector DIT as signal!
         
        aa = b * (A**2 * P - A * B * P * np.cos(theta) )
        bb = 2 * A * b * B * P * np.sin(theta)
        cc =  -self.signal + (A**2 * P**2 + b**2 * (A**2 + B**2 - 2 * A * B * np.cos(theta) ) +\
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
        
     
def detection_chain(input_field, dm, FPM, det, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True):
    # apply DM correction 
    # Pass through ZWFS (phase mask) onto back pupil
    # detect it on a detector to get a signal 
    input_field_dm = input_field.applyDM(dm) # apply DM phase shifts
    output_field = FPM.get_output_field( input_field_dm, wvl_lims=[0,100], nx_size_focal_plane = None, dx_focal_plane = None, keep_intermediate_products=False )
    output_field.define_pupil_grid(dx=input_field.dx, D_pix=input_field.D_pix)
    
    sig = det.detect_field( output_field, include_shotnoise=include_shotnoise, ph_per_s_per_m2_per_nm=ph_per_s_per_m2_per_nm, grids_aligned=grids_aligned)
    
    return( sig )



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
   
    
def plot_cross_section( array_2d, x = None, xlabel='x',ylabel='y' ):
    
    plt.figure(figsize=(8,5))
    if x==None:
        
        plt.plot( array_2d[len(array_2d)//2,:] )
    
    else:
        plt.plot(x, array_2d[len(array_2d)//2,:] )
    
    plt.xlabel(xlabel,fontsize=15)
    plt.ylabel(ylabel,fontsize=15)
    plt.gca().tick_params(labelsize=15)
    
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



def modal_AO_correction( opd_screens, n_modes, lag, Ki, Kp, V, dx , wvls , wfs_wvl, it=10, pup= None):
    """
    use wvl variable and test if single 

    Parameters
    ----------
    opd_screens : TYPE aotools phase screen type or dictionary (opd_screens[wvl] = array([time,x,y]))
        DESCRIPTION. phase screen object returned from aotools
    n_modes : TYPE. integer
        DESCRIPTION. number of modes that AO system sensors and corrects
    lag : TYPE. scalar
        DESCRIPTION. lag of AO system from sensing to corerection
    Ki : TYPE. scalar
        DESCRIPTION. integral gain
    Kp : TYPE. scalar
        DESCRIPTION. proportional gain
    V : TYPE. scalar
        DESCRIPTION. turbulence speed (e.g. m/s)
    dx : TYPE. scalar
        DESCRIPTION. pixel scale of phase screen (e.g. m)
    wvls : TYPE. list
        DESCRIPTION. wvls to of phase screens to apply wvls correction 
    wfs_wvl : TYPE scalar
        DESCRIPTION. central wavelength of wavefront sensor to calculate the correctionto calculate ao corrected phase screens
    it :  TYPE. integer (default = 10 )
        DESCRIPTION. number of iterations to simulate if type(opd_screens) == aotools.turbulence.infinitephasescreen.PhaseScreenVonKarman
        otherwise we just iterate over all input opd_screens (in the temporal dimension )
    pup : TYPE. 2D numpy array 
        DESCRIPTION. if type(opd_screens) == aotools.turbulence.infinitephasescreen.PhaseScreenVonKarman then you should 
        define a telescope pupil that the screen passes through (array with values between 0-1)
        the default is None in which case we just consider a uniform disk the same size as opd_screens.scrn array
        
        maiva and lara <3
    Returns
    -------
     sci_frames - numpy array with AO corrected opd screen (typically meters). sci_frames[time, wvl, x, y]

    """
    
    dt = dx / V # sample time

    wfs_frames=[] # AO residuals calcualted at wfs_wvl
    sci_frames = [] # AO residuals calcualted at wvls
    
    if not type(opd_screens) == aotools.turbulence.infinitephasescreen.PhaseScreenVonKarman:
        # then we assume screens are pre-processed dictionary opd_screens[wvl] = array([time,x,y]), 
        
        wvl_keys = list(opd_screens.keys())  # dictionary keys must contain wvls and wfs_wvl
        
        npix = np.array(opd_screens[wvl_keys[0]]).shape[-1]
        frame_lag = round(  V * lag / dx ) 
        
        basis = zernike.zernike_basis(nterms=n_modes, npix=npix) # init Zernike basis
        dm_cmd = np.zeros([npix,npix]) # init DM comand 
        
        for i,s in enumerate(  opd_screens[wvl_keys[0]] ):

            #Science frames
            s_sci = np.array([opd_screens[w][i] for w in wvls])
            opd_sci = s_sci - dm_cmd    
            sci_frames.append( opd_sci )
            
            #WFS frames
            s_wfs = np.array( opd_screens[wfs_wvl][i] ) #calibrate_phase_screen2wvl(wfs_wvl, pup * screen.scrn)
            opd_wfs = s_wfs - dm_cmd  # have to do this for each wavelength
            wfs_frames.append( opd_wfs ) #add this to the wfs frame (to be averaged when its time to update DM cmd )
            
            # check if its time to update DM command
            if len( wfs_frames ) >= frame_lag:
                wfs_signal = np.nanmean( wfs_frames, axis=0 )
                wfs_frames = []
                Z_coes = zernike.opd_expand(wfs_signal, nterms=n_modes, basis=zernike.zernike_basis)
                phi_est = np.sum( basis*np.array(Z_coes)[:,np.newaxis, np.newaxis] , axis=0) 
                dm_cmd = Ki * (dm_cmd + Kp * phi_est)
              
    elif type(opd_screens) == aotools.turbulence.infinitephasescreen.PhaseScreenVonKarman:
        
        npix = opd_screens.scrn.shape[0]
        frame_lag = round( lag / dt ) 
        
        basis = zernike.zernike_basis(nterms=n_modes, npix=npix) # init Zernike basis
        dm_cmd = np.zeros([npix,npix]) # init DM comand 
        
        if pup==None:
            pup = basis[0]
    
        for i in range( it ):
            
            # propagate screen
            opd_screens.add_row() #radian
            
            #Science frames (make sure to convert radian to opd! )
            s_sci = np.array( [calibrate_phase_screen2wvl(w, w/(2*np.pi) * pup * opd_screens.scrn) for w in wvls] )
            opd_sci = s_sci - dm_cmd    
            sci_frames.append( opd_sci )
            
            
            #WFS frames
            s_wfs = calibrate_phase_screen2wvl(wfs_wvl, wfs_wvl/(2*np.pi) * pup * opd_screens.scrn) 
            opd_wfs = s_wfs - dm_cmd  # have to do this for each wavelength
            wfs_frames.append( opd_wfs ) #add this to the wfs frame (to be averaged when its time to update DM cmd )
            
            # check if its time to update DM command
            if len( wfs_frames ) >= frame_lag:
                wfs_signal = np.nanmean( wfs_frames, axis=0 )
                wfs_frames = []
                Z_coes = zernike.opd_expand(wfs_signal, nterms=n_modes, basis=zernike.zernike_basis)
                phi_est = np.sum( basis*np.array(Z_coes)[:,np.newaxis, np.newaxis] , axis=0) 
                dm_cmd = Ki * (dm_cmd + Kp * phi_est)
                
    return( sci_frames ) #  sci_frames[time][wvl][x,y]
        

            
    """
    #example : 
        
        
    # =============================================================================
    # D = 1.8
    # D_pix = 2**8
    # dx = D/D_pix
    # wvls =  np.linspace(1.1e-6 ,1.7e-6, 5) 
    # wfs_wvl = 0.6e-6
    # basis = zernike.zernike_basis(nterms=10, npix=D_pix)
    # pup = basis[0]
    # 
    # screens = aotools.turbulence.infinitephasescreen.PhaseScreenVonKarman(nx_size=D_pix,pixel_scale=dx,r0=0.1,L0=25)
    # 
    # 
    # V=50
    # dx=D/pup.shape[0]
    # lag=1e-3
    # 
    # #%% pre processed screens 
    # phase_screens = np.array( [pup * screens.add_row() for i in range(100)] )   
    # 
    # opd_screens={}
    # for w in [wfs_wvl] + list(wvls) :
    #     
    #     opd_screens[w] = np.array([calibrate_phase_screen2wvl(w, w/(2*np.pi) * pup * phi)  for phi in phase_screens])
    # 
    # ao_screens = modal_AO_correction( opd_screens, n_modes=14, lag=lag, Ki=.95, Kp=1.1, V=V, dx=dx , wvls=wvls , wfs_wvl=wfs_wvl, it=1000, pup= None)
    # 
    # w = wvls[1]
    # before_strehl = np.array( [np.exp( -np.nanvar( 2*np.pi/w * p ) ) for p in opd_screens[w]] )
    # after_strehl =  np.array( [np.exp( -np.nanvar( 2*np.pi/w * p ) ) for p in np.array( ao_screens )[:,2,:,:]] )
    # plt.plot(before_strehl); plt.plot( after_strehl, label='after'), plt.legend()
    # 
    # #aa =modal_AO_correction( phase_screen, n_modes=14, lag=lag, Ki=.95, Kp=1.1, V=V, dx=phase_screen.pixel_scale , wvls=[1.1e-6, 1.65e-6] , wfs_wvl=0.6e-6, it=1000, pup= None)
    # 
    # #%% or with the original ao tools screens (no pre processing). This is slower but avoids storing large arrays in memory
    # 
    # ao_screens = modal_AO_correction( screens, n_modes=14, lag=lag, Ki=.95, Kp=1.1, V=V, dx=dx , wvls=wvls , wfs_wvl=wfs_wvl, it=1000, pup= None)
    # 
    # w = wvls[1]
    # #before_strehl = np.array( [np.exp( -np.nanvar( 2*np.pi/w * p ) ) for p in opd_screens[w]] )
    # after_strehl =  np.array( [np.exp( -np.nanvar( 2*np.pi/w * p ) ) for p in np.array( ao_screens )[:,2,:,:]] )
    # plt.plot(before_strehl); plt.plot( after_strehl, label='after'), plt.legend()
    # 
    # =============================================================================

    
    """

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
     
def reco(FIELD, SIG, SIG_CAL, FPM, DET, b, order='first'):
    """
    
    phase reconstruction from a given focal plane mask  (FPM) , detector (DET) and mask in/out SIGNALS

    Parameters
    ----------
    FIELD : TYPE
        DESCRIPTION.
    SIG : TYPE
        DESCRIPTION.
    SIG_CAL : TYPE
        DESCRIPTION.
    FPM : TYPE
        DESCRIPTION.
    DET : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    wvls = FIELD.wvl
    P = SIG_CAL.signal / FPM.A
    B = DET.detect_field(b).signal
    beta = np.mean([b.phase[w] for w in b.phase],axis=0)
    beta = interpolate_field_onto_det( beta, DET )
    
    m = np.mean( [FPM.get_filter_design_parameter(w) for w in wvls] )
    M = abs(m)
    mu = np.angle(m)

    if order == 'full':
        phi_reco = np.arccos( (SIG.signal**2 / FPM.A - P**2 - (B*M)**2) / (2*B*P*M) ) + mu + beta 
    elif order == 'first':
        phi_reco = np.pi/2 - ( (SIG.signal**2 / FPM.A - P**2 - (B*M)**2) / (2*B*P*M) ) + mu #+ beta  
    
    #phi_reco = np.arccos( (SIG.signal / FPM.A - P - (B*M**2)) / (2*np.sqrt(B*P)*M) ) + mu + beta    
    
    return(phi_reco)


def write2fits(screens,wvls, dict4headers, saveAs=None ) :
    """
    

    Parameters
    ----------
    screens : TYPE. 4D array like (screens[time,wvl,x,y])
        DESCRIPTION. phase screens 
    wvls : TYPE. 1D array like
        DESCRIPTION. wavelengths where phase screens are calculated , must match len(wvls)==screens.shape[1]
    dict4headers : TYPE. dictionary 
        DESCRIPTION. dictionary items are written to headers 
    saveAs : TYPE. None or string
        DESCRIPTION. path to save fits file as. Default is None in which case no file is saved 
    Returns
    -------
    fits_file.

    """
    fits_file = fits.HDUList() 
    for w, wvl in enumerate(wvls):

        hdu = fits.PrimaryHDU( np.array( screens )[:,w,:,:] )
        
        hdu.header.set('what is', 'AO corrected opd screens (m)' , 'at lambda = {0:.4g}um'.format(wvl))
        for k,v in dict4headers.items():
            hdu.header.set(k, v)
    
        fits_file.append( hdu )
        
        if saveAs!=None:
            fits_file.writeto(saveAs,overwrite=True)
        
    return( fits_file )


#%% Phase screens of first AO stage 


"""
Also look at post_ao_phasescreen_sim.py script .. inparticular single 

AOtools 
Infinite Phase Screens
----------------------

An implementation of the "infinite phase screen", as deduced by Francois Assemat and Richard W. Wilson, 2006.

I added capacity to calculate covariance of AO residuals in a generic way and generate their phase screens!  

- added self.post_ao attribute on init of PhaseScreenVonKarman, PhaseScreenKolmogorov class, set by default to False
- added PhaseScreen_PostAO class with self.post_ao=True, and same inheritence as PhaseScreenVonKarman
    - if self.post_ao=True (should only be true for PhaseScreen_PostAO class)
        we need to add attributes to feed ao_phase_covariance():
            self.sigma2_ao
            self.D
            self.N_act
                -these should be user inputs to init PhaseScreen_PostAO class
                
- put conditional check on self.post_ao in make_covmats(self) method for PhaseScreen Object when calculating phase_covariance
    if not self.post_ao:
        self.cov_mat = phase_covariance(self.seperations, self.r0, self.L0)
    else:
        self.cov_mat = ao_phase_covariance(r, sigma2_ao, D, N_act, L0, r0, kolmogorov = False, finite_aperture=None,filter_order=None)


Works well.. However with fast AO correction we probably don't expect so much temporal correlation in the phase screens
another option would be to keep original implementation and add option to apply dm command to subtract off average of last N screens

Also we implement Gaussian sampling of phase screens from covariance matrix s

"""




class PhaseScreen(object):
    """
    A "Phase Screen" for use in AO simulation.  Can be extruded infinitely.

    This represents the phase addition light experiences when passing through atmospheric 
    turbulence. Unlike other phase screen generation techniques that translate a large static 
    screen, this method keeps a small section of phase, and extends it as necessary for as many
    steps as required. This can significantly reduce memory consumption at the expense of more
    processing power required.

    The technique is described in a paper by Assemat and Wilson, 2006 and expanded upon by Fried, 2008.
    It essentially assumes that there are two matrices, "A" and "B",
    that can be used to extend an existing phase screen.
    A single row or column of new phase can be represented by 

        X = A.Z + B.b

    where X is the new phase vector, Z is some data from the existing screen,
    and b is a random vector with gaussian statistics.

    This object calculates the A and B matrices using an expression of the phase covariance when it
    is initialised. Calculating A is straightforward through the relationship:

        A =  Cov_xz . (Cov_zz)^(-1).

    B is less trivial.

        BB^t = Cov_xx - A.Cov_zx

    (where B^t is the transpose of B) is a symmetric matrix, hence B can be expressed as 

        B = UL, 

    where U and L are obtained from the svd for BB^t

        U, w, U^t = svd(BB^t)

    L is a diagonal matrix where the diagonal elements are w^(1/2).    

    On initialisation an initial phase screen is calculated using an FFT based method.
    When 'add_row' is called, a new vector of phase is added to the phase screen.

    Existing points to use are defined by a "stencil", than is set to 0 for points not to use
    and 1 for points to use. This makes this a generalised base class that can be used by 
    other infinite phase screen creation schemes, such as for Von Karmon turbulence or 
    Kolmogorov turbulence.

    .. note::
        The phase screen is returned on each iteration as a 2d array, with each element representing the phase 
        change in **radians**. This means that to obtain the physical phase distortion in nanometres, 
        it must be multiplied by (wavelength / (2*pi)), (where `wavellength` here is the same wavelength
        in which r0 is given in the function arguments)
    """
    def set_X_coords(self):
        """
        Sets the coords of X, the new phase vector.
        """
        self.X_coords = np.zeros((self.nx_size, 2))
        self.X_coords[:, 0] = -1
        self.X_coords[:, 1] = np.arange(self.nx_size)
        self.X_positions = self.X_coords * self.pixel_scale

    def set_stencil_coords(self):
        """
        Sets the Z coordinates, sections of the phase screen that will be used to create new phase
        """
        self.stencil = np.zeros((self.stencil_length, self.nx_size))

        max_n = 1
        while True:
            if 2 ** (max_n - 1) + 1 >= self.nx_size:
                max_n -= 1
                break
            max_n += 1

        for n in range(0, max_n + 1):
            col = int((2 ** (n - 1)) + 1)
            n_points = (2 ** (max_n - n)) + 1

            coords = np.round(np.linspace(0, self.nx_size - 1, n_points)).astype('int32')
            self.stencil[col - 1][coords] = 1

        # Now fill in tail of stencil
        for n in range(1, self.stencil_length_factor + 1):
            col = n * self.nx_size - 1
            self.stencil[col, self.nx_size // 2] = 1

        self.stencil_coords = np.array(np.where(self.stencil == 1)).T
        self.stencil_positions = self.stencil_coords * self.pixel_scale

        self.n_stencils = len(self.stencil_coords)

    def calc_seperations(self):
        """
        Calculates the seperations between the phase points in the stencil and the new phase vector
        """
        positions = np.append(self.stencil_positions, self.X_positions, axis=0)
        self.seperations = np.zeros((len(positions), len(positions)))

        if numba:
            calc_seperations_fast(positions, self.seperations)
        else:
            for i, (x1, y1) in enumerate(positions):
                for j, (x2, y2) in enumerate(positions):
                    delta_x = x2 - x1
                    delta_y = y2 - y1

                    delta_r = np.sqrt(delta_x ** 2 + delta_y ** 2)

                    self.seperations[i, j] = delta_r



    def make_covmats(self):
        """
        Makes the covariance matrices required for adding new phase
        """
        if not self.post_ao:
            self.cov_mat = phase_covariance(self.seperations, self.r0, self.L0)
        else:
            self.cov_mat = ao_phase_covariance(self.seperations, self.r0, self.L0, self.D, self.sigma2_ao, self.N_act)

        self.cov_mat_zz = self.cov_mat[:self.n_stencils, :self.n_stencils]
        self.cov_mat_xx = self.cov_mat[self.n_stencils:, self.n_stencils:]
        self.cov_mat_zx = self.cov_mat[:self.n_stencils, self.n_stencils:]
        self.cov_mat_xz = self.cov_mat[self.n_stencils:, :self.n_stencils]

    def makeAMatrix(self):
        """
        Calculates the "A" matrix, that uses the existing data to find a new 
        component of the new phase vector.
        """
        # Cholsky solve can fail - if so do brute force inversion
        try:
            cf = linalg.cho_factor(self.cov_mat_zz)
            inv_cov_zz = linalg.cho_solve(cf, np.identity(self.cov_mat_zz.shape[0]))
        except linalg.LinAlgError:
            # print("Cholesky solve failed. Performing SVD inversion...")
            # inv_cov_zz = np.linalg.pinv(self.cov_mat_zz)
            raise linalg.LinAlgError("Could not invert Covariance Matrix to for A and B Matrices. Try with a larger pixel scale or smaller L0")

        self.A_mat = self.cov_mat_xz.dot(inv_cov_zz)

    def makeBMatrix(self):
        """
        Calculates the "B" matrix, that turns a random vector into a component of the new phase.
        """
        # Can make initial BBt matrix first
        BBt = self.cov_mat_xx - self.A_mat.dot(self.cov_mat_zx)

        # Then do SVD to get B matrix
        u, W, ut = np.linalg.svd(BBt)

        L_mat = np.zeros((self.nx_size, self.nx_size))
        np.fill_diagonal(L_mat, np.sqrt(W))

        # Now use sqrt(eigenvalues) to get B matrix
        self.B_mat = u.dot(L_mat)

    def make_initial_screen(self):
        """
        Makes the initial screen usign FFT method that can be extended 
        """

        # phase screen will make it *really* random if no seed at all given.
        # If a seed is here, screen must be repeatable wiht same seed
        self._R = np.random.default_rng(self.random_seed)

        self._scrn = ft_phase_screen(
            self.r0, self.stencil_length, self.pixel_scale, self.L0, 1e-10, seed=self._R
        )

        self._scrn = self._scrn[:, :self.nx_size]

    def get_new_row(self):
        random_data = self._R.normal(0, 1, size=self.nx_size)

        stencil_data = self._scrn[(self.stencil_coords[:, 0], self.stencil_coords[:, 1])]
        new_row = self.A_mat.dot(stencil_data) + self.B_mat.dot(random_data)

        new_row.shape = (1, self.nx_size)
        return new_row

    def add_row(self):
        """
        Adds a new row to the phase screen and removes old ones.
        """

        new_row = self.get_new_row()

        self._scrn = np.append(new_row, self._scrn, axis=0)[:self.stencil_length, :self.nx_size]

        return self.scrn

    @property
    def scrn(self):
        """
        The current phase map held in the PhaseScreen object in radians.
        """
        return self._scrn[:self.requested_nx_size, :self.requested_nx_size]


class PhaseScreenVonKarman(PhaseScreen):
    """
    A "Phase Screen" for use in AO simulation with Von Karmon statistics.

    This represents the phase addition light experiences when passing through atmospheric
    turbulence. Unlike other phase screen generation techniques that translate a large static
    screen, this method keeps a small section of phase, and extends it as necessary for as many
    steps as required. This can significantly reduce memory consumption at the expense of more
    processing power required.

    The technique is described in a paper by Assemat and Wilson, 2006. It essentially assumes that
    there are two matrices, "A" and "B", that can be used to extend an existing phase screen.
    A single row or column of new phase can be represented by

        X = A.Z + B.b

    where X is the new phase vector, Z is some number of columns of the existing screen,
    and b is a random vector with gaussian statistics.

    This object calculates the A and B matrices using an expression of the phase covariance when it
    is initialised. Calculating A is straightforward through the relationship:

        A =  Cov_xz . (Cov_zz)^(-1).

    B is less trivial.

        BB^t = Cov_xx - A.Cov_zx

    (where B^t is the transpose of B) is a symmetric matrix, hence B can be expressed as

        B = UL,

    where U and L are obtained from the svd for BB^t

        U, w, U^t = svd(BB^t)

    L is a diagonal matrix where the diagonal elements are w^(1/2).

    On initialisation an initial phase screen is calculated using an FFT based method.
    When ``add_row`` is called, a new vector of phase is added to the phase screen using `nCols`
    columns of previous phase. Assemat & Wilson claim that two columns are adequate for good
    atmospheric statistics. The phase in the screen data is always accessed as ``<phasescreen>.scrn`` and is in radians.

        .. note::
        The phase screen is returned on each iteration as a 2d array, with each element representing the phase 
        change in **radians**. This means that to obtain the physical phase distortion in nanometres, 
        it must be multiplied by (wavelength / (2*pi)), (where `wavellength` here is the same wavelength
        in which r0 is given in the function arguments)

    Parameters:
        nx_size (int): Size of phase screen (NxN)
        pixel_scale(float): Size of each phase pixel in metres
        r0 (float): fried parameter (metres)
        L0 (float): Outer scale (metres)
        random_seed (int, optional): seed for the random number generator
        n_columns (int, optional): Number of columns to use to continue screen, default is 2
    """
    def __init__(self, nx_size, pixel_scale, r0, L0, random_seed=None, n_columns=2):

        self.n_columns = n_columns

        self.requested_nx_size = nx_size
        self.nx_size = nx_size
        self.pixel_scale = pixel_scale
        self.r0 = r0
        self.L0 = L0
        self.stencil_length_factor = 1
        self.stencil_length = self.nx_size

        self.random_seed = random_seed

        self.set_X_coords()
        self.set_stencil_coords()

        self.calc_seperations()
        
        self.post_ao = False
        self.make_covmats()

        self.makeAMatrix()
        self.makeBMatrix()
        self.make_initial_screen()


    def set_stencil_coords(self):
        self.stencil = np.zeros((self.stencil_length, self.nx_size))
        self.stencil[:self.n_columns] = 1

        self.stencil_coords = np.array(np.where(self.stencil==1)).T
        self.stencil_positions = self.stencil_coords * self.pixel_scale

        self.n_stencils = len(self.stencil_coords)


class PhaseScreen_PostAO(PhaseScreen):
    """
    A "Phase Screen" for use in AO simulation with Von Karmon statistics.

    This represents the phase addition light experiences when passing through atmospheric
    turbulence. Unlike other phase screen generation techniques that translate a large static
    screen, this method keeps a small section of phase, and extends it as necessary for as many
    steps as required. This can significantly reduce memory consumption at the expense of more
    processing power required.

    The technique is described in a paper by Assemat and Wilson, 2006. It essentially assumes that
    there are two matrices, "A" and "B", that can be used to extend an existing phase screen.
    A single row or column of new phase can be represented by

        X = A.Z + B.b

    where X is the new phase vector, Z is some number of columns of the existing screen,
    and b is a random vector with gaussian statistics.

    This object calculates the A and B matrices using an expression of the phase covariance when it
    is initialised. Calculating A is straightforward through the relationship:

        A =  Cov_xz . (Cov_zz)^(-1).

    B is less trivial.

        BB^t = Cov_xx - A.Cov_zx

    (where B^t is the transpose of B) is a symmetric matrix, hence B can be expressed as

        B = UL,

    where U and L are obtained from the svd for BB^t

        U, w, U^t = svd(BB^t)

    L is a diagonal matrix where the diagonal elements are w^(1/2).

    On initialisation an initial phase screen is calculated using an FFT based method.
    When ``add_row`` is called, a new vector of phase is added to the phase screen using `nCols`
    columns of previous phase. Assemat & Wilson claim that two columns are adequate for good
    atmospheric statistics. The phase in the screen data is always accessed as ``<phasescreen>.scrn`` and is in radians.

        .. note::
        The phase screen is returned on each iteration as a 2d array, with each element representing the phase 
        change in **radians**. This means that to obtain the physical phase distortion in nanometres, 
        it must be multiplied by (wavelength / (2*pi)), (where `wavellength` here is the same wavelength
        in which r0 is given in the function arguments)

    Parameters:
        nx_size (int): Size of phase screen (NxN)
        pixel_scale(float): Size of each phase pixel in metres
        r0 (float): fried parameter (metres)
        L0 (float): Outer scale (metres)
        random_seed (int, optional): seed for the random number generator
        n_columns (int, optional): Number of columns to use to continue screen, default is 2
    """
    def __init__(self, nx_size, pixel_scale, r0, L0, D, sigma2_ao, N_act, random_seed=None, n_columns=2):

        self.n_columns = n_columns

        self.D = D# telescope diameter 
        self.sigma2_ao = sigma2_ao # AO residual variance (calculated Ao error budgets)
        self.N_act = N_act # number of total actuators on DM (e.g. N_act = Nx_act x Ny_act for square grid)

        self.requested_nx_size = nx_size
        self.nx_size = nx_size
        self.pixel_scale = pixel_scale
        self.r0 = r0
        self.L0 = L0
        self.stencil_length_factor = 1
        self.stencil_length = self.nx_size

        self.random_seed = random_seed

        self.set_X_coords()
        self.set_stencil_coords()

        self.calc_seperations()
        
        self.post_ao = True
        self.make_covmats()

        self.makeAMatrix()
        self.makeBMatrix()
        self.make_initial_screen()


    def set_stencil_coords(self):
        self.stencil = np.zeros((self.stencil_length, self.nx_size))
        self.stencil[:self.n_columns] = 1

        self.stencil_coords = np.array(np.where(self.stencil==1)).T
        self.stencil_positions = self.stencil_coords * self.pixel_scale

        self.n_stencils = len(self.stencil_coords)


def find_allowed_size(nx_size):
    """
    Finds the next largest "allowed size" for the Fried Phase Screen method
    
    Parameters:
        nx_size (int): Requested size
    
    Returns:
        int: Next allowed size
    """
    n = 0
    while (2 ** n + 1) < nx_size:
        n += 1

    nx_size = 2 ** n + 1
    return nx_size


class PhaseScreenKolmogorov(PhaseScreen):
    """
    A "Phase Screen" for use in AO simulation using the Fried method for Kolmogorov turbulence.

    This represents the phase addition light experiences when passing through atmospheric 
    turbulence. Unlike other phase screen generation techniques that translate a large static 
    screen, this method keeps a small section of phase, and extends it as neccessary for as many 
    steps as required. This can significantly reduce memory consuption at the expense of more 
    processing power required.

    The technique is described in a paper by Assemat and Wilson, 2006 and expanded upon by Fried, 2008.
    It essentially assumes that there are two matrices, "A" and "B",
    that can be used to extend an existing phase screen.
    A single row or column of new phase can be represented by 

        X = A.Z + B.b

    where X is the new phase vector, Z is some data from the existing screen,
    and b is a random vector with gaussian statistics.

    This object calculates the A and B matrices using an expression of the phase covariance when it
    is initialised. Calculating A is straightforward through the relationship:

        A =  Cov_xz . (Cov_zz)^(-1).

    B is less trivial.

        BB^t = Cov_xx - A.Cov_zx

    (where B^t is the transpose of B) is a symmetric matrix, hence B can be expressed as 

        B = UL, 

    where U and L are obtained from the svd for BB^t

        U, w, U^t = svd(BB^t)

    L is a diagonal matrix where the diagonal elements are w^(1/2).    

    The Z data is taken from points in a "stencil" defined by Fried that samples the entire screen.
    An additional "reference point" is also considered, that is picked from a point separate from teh stencil 
    and applied on each iteration such that the new phase equation becomes:
    
    On initialisation an initial phase screen is calculated using an FFT based method.
    When ``add_row`` is called, a new vector of phase is added to the phase screen. The phase in the screen data
    is always accessed as ``<phasescreen>.scrn`` and is in radians.

    .. note::
        The phase screen is returned on each iteration as a 2d array, with each element representing the phase 
        change in **radians**. This means that to obtain the physical phase distortion in nanometres, 
        it must be multiplied by (wavelength / (2*pi)), (where `wavellength` here is the same wavelength
        in which r0 is given in the function arguments)

    Parameters:
        nx_size (int): Size of phase screen (NxN)
        pixel_scale(float): Size of each phase pixel in metres
        r0 (float): fried parameter (metres)
        L0 (float): Outer scale (metres)
        random_seed (int, optional): seed for the random number generator
        stencil_length_factor (int, optional): How much longer is the stencil than the desired phase? default is 4
    """
    def __init__(self, nx_size, pixel_scale, r0, L0, random_seed=None, stencil_length_factor=4):

        self.requested_nx_size = nx_size
        self.nx_size = find_allowed_size(nx_size)
        self.pixel_scale = pixel_scale
        self.r0 = r0
        self.L0 = L0
        self.stencil_length_factor = stencil_length_factor
        self.stencil_length = stencil_length_factor * self.nx_size
        self.random_seed = random_seed

        # Coordinate of Fried's "reference point" that stops the screen diverging
        self.reference_coord = (1, 1)

        self.set_X_coords()
        self.set_stencil_coords()

        self.calc_seperations()
        
        self.post_ao = False
        self.make_covmats()

        self.makeAMatrix()
        self.makeBMatrix()
        self.make_initial_screen()

    def get_new_row(self):
        random_data = self._R.normal(0, 1, size=self.nx_size)

        stencil_data = self._scrn[(self.stencil_coords[:, 0], self.stencil_coords[:, 1])]

        reference_value = self._scrn[self.reference_coord]

        new_row = self.A_mat.dot(stencil_data - reference_value) + self.B_mat.dot(random_data) + reference_value

        new_row.shape = (1, self.nx_size)
        return new_row

    def __repr__(self):
        return str(self.scrn)

    
def ft_phase_screen(r0, N, delta, L0, l0, FFT=None, seed=None):
    """
    Creates a random phase screen with Von Karmen statistics.
    (Schmidt 2010)
    
    Parameters:
        r0 (float): r0 parameter of scrn in metres
        N (int): Size of phase scrn in pxls
        delta (float): size in Metres of each pxl
        L0 (float): Size of outer-scale in metres
        l0 (float): inner scale in metres
        seed (int, optional): seed for random number generator. If provided, 
            allows for deterministic screens  

    .. note::
        The phase screen is returned as a 2d array, with each element representing the phase 
        change in **radians**. This means that to obtain the physical phase distortion in nanometres, 
        it must be multiplied by (wavelength / (2*pi)), (where `wavellength` here is the same wavelength
        in which r0 is given in the function arguments)

    Returns:
        ndarray: numpy array representing phase screen in radians
    """
    delta = float(delta)
    r0 = float(r0)
    L0 = float(L0)
    l0 = float(l0)

    R = np.random.default_rng(seed)

    del_f = 1./(N*delta)

    fx = np.arange(-N/2., N/2.) * del_f

    (fx, fy) = np.meshgrid(fx,fx)
    f = np.sqrt(fx**2. + fy**2.)

    fm = 5.92/l0/(2*np.pi)
    f0 = 1./L0

    PSD_phi = (0.023*r0**(-5./3.) * np.exp(-1*((f/fm)**2)) / (((f**2) + (f0**2))**(11./6)))

    PSD_phi[int(N/2), int(N/2)] = 0

    cn = ((R.normal(size=(N, N))+1j * R.normal(size=(N, N))) * np.sqrt(PSD_phi)*del_f)

    phs = ift2(cn, 1, FFT).real

    return phs


def ift2(G, delta_f, FFT=None):
    """
    Wrapper for inverse fourier transform

    Parameters:
        G: data to transform
        delta_f: pixel seperation
        FFT (FFT object, optional): An accelerated FFT object
    """

    N = G.shape[0]

    if FFT:
        g = np.fft.fftshift(FFT(np.fft.fftshift(G))) * (N * delta_f) ** 2
    else:
        g = fft.ifftshift(fft.ifft2(fft.fftshift(G))) * (N * delta_f) ** 2

    return g


def D_phi(r,L0,r0, kolmogorov = False, finite_aperture=None,filter_order=None ): # atmospheric phase structure function 
    
    if r==0:
        Dphi=0
    else:
        if  kolmogorov :
            Dphi = 6.88 * (r/r0)**(5/3)
        else:
            if finite_aperture is not None: #we apply telescope filtering 
                if r < finite_aperture:
                    Dphi = (L0/r0)**(5/3) * 2**(1/6) * gamma(11/6)/np.pi**(8/3) * ((24/5)*gamma(6/5))**(5/6) * (gamma(5/6)/2**(1/6)-(2*np.pi*r/L0)**(5/6)*kv(5/6,2*np.pi*r/L0))
                    # saturates at (L0/r0)**(5/3) * 2**(1/6) * gamma(11/6)/np.pi**(8/3) * ((24/5)*gamma(6/5))**(5/6) * gamma(5/6)/2**(1/6)
                else:
                    if filter_order is not None:
                        #power law decay (power law index = filter_order )
                        Dphi = (L0/r0)**(5/3) * 2**(1/6) * gamma(11/6)/np.pi**(8/3) * ((24/5)*gamma(6/5))**(5/6) * (gamma(5/6)/2**(1/6)-(2*np.pi*finite_aperture/L0)**(5/6)*kv(5/6,2*np.pi*finite_aperture/L0)) / (1+r-finite_aperture)**filter_order
                        #exponetial decay 
                        #(L0/r0)**(5/3) * 2**(1/6) * gamma(11/6)/np.pi**(8/3) * ((24/5)*gamma(6/5))**(5/6) * (gamma(5/6)/2**(1/6)-(2*np.pi*finite_aperture/L0)**(5/6)*kv(5/6,2*np.pi*finite_aperture/L0)) * np.exp( -filter_order * (r-finite_aperture) )
                    else:
                        Dphi = 0
            else:
                Dphi = (L0/r0)**(5/3) * 2**(1/6) * gamma(11/6)/np.pi**(8/3) * ((24/5)*gamma(6/5))**(5/6) * (gamma(5/6)/2**(1/6)-(2*np.pi*r/L0)**(5/6)*kv(5/6,2*np.pi*r/L0))
                
    return(Dphi)




def Dr_phi(r,sigma2_ao, D, N_act, L0, r0, kolmogorov = False, finite_aperture=None,filter_order=None ): # ao residual phase structure function 


    #if r==0:
    #    Dr=0
    if 1: #important that r!=0 otherwise will crash 
        #interactuator distance in telescope pupil space 
        d_act = np.sqrt( (D/2)**2 /  N_act ) # pi*r_act^2 = np.pi(D/2)^2 / N_act
    
        if finite_aperture is not None: #we apply telescope filtering 
            if r<=finite_aperture: # if the distance is less then the DM inter-actuator spacing then we have no AO correction and turbulence structure is from atmosphere
                Dr =  sigma2_ao * (1 - (2*np.pi*r/d_act)**(5/6)*kv(5/6,2*np.pi*r/d_act) )   # saturates to sigma2_ao at r=d_act    
            else: # outside aperture, so structure function fgoes to zero
                if filter_order is not None:
                    #power law decay (power law index = filter_order )
                    Dr = sigma2_ao * (1 - (2*np.pi*finite_aperture/d_act)**(5/6)*kv(5/6,2*np.pi*finite_aperture/d_act) ) / (1+r-finite_aperture)**filter_order #0 #sigma2_ao + D_phi(r-D/2, L0, r0, kolmogorov) 
                    #exponetial decay 
                    #Dr = sigma2_ao * (1 - (2*np.pi*finite_aperture/d_act)**(5/6)*kv(5/6,2*np.pi*finite_aperture/d_act) ) * np.exp( -filter_order * (r-finite_aperture) )
                else:
                    Dr=0
        else: 
            Dr =  sigma2_ao * (1 - (2*np.pi*r/d_act)**(5/6)*kv(5/6,2*np.pi*r/d_act) ) 
            
    return(Dr)




def ao_phase_covariance(r, sigma2_ao, D, N_act, L0, r0, kolmogorov = False, finite_aperture=None,filter_order=None):
    """
    Calculate the phase covariance between two points seperated by `r`, 
    after AO correction
   
    """
    
    # Make sure everything is a float to avoid nasty surprises in division!
    r = np.float32(r)
    r0 = float(r0)
    L0 = float(L0)

    
    # Get rid of any zeros
    r += 1e-40

    cov = np.exp( -Dr_phi(r,sigma2_ao, D, N_act, L0, r0, kolmogorov = False, finite_aperture=None,filter_order=None ) ) 

    return cov


def phase_covariance(r, r0, L0 ):
    """
    Calculate the phase covariance between two points seperated by `r`, 
    in turbulence with a given `r0 and `L0`.
    Uses equation 5 from Assemat and Wilson, 2006.

    Parameters:
        r (float, ndarray): Seperation between points in metres (can be ndarray)
        r0 (float): Fried parameter of turbulence in metres
        L0 (float): Outer scale of turbulence in metres
    """
    
    # Make sure everything is a float to avoid nasty surprises in division!
    r = np.float32(r)
    r0 = float(r0)
    L0 = float(L0)

    
    # Get rid of any zeros
    r += 1e-40

    A = (L0 / r0) ** (5. / 3)

    B1 = (2 ** (-5. / 6)) * gamma(11. / 6) / (np.pi ** (8. / 3))
    B2 = ((24. / 5) * gamma(6. / 5)) ** (5. / 6)

    C = (((2 * np.pi * r) / L0) ** (5. / 6)) * kv(5. / 6, (2 * np.pi * r) / L0)

    cov = A * B1 * B2 * C

    return cov



def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = linalg.cholesky(B)
        return True
    except linalg.LinAlgError:
        return False
    

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
    _, s, V = linalg.svd(B)

    H =np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(linalg.norm(A))
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
        mineig = np.min(np.real(linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3
    



def create_atm_phase_screen_Guassian_sampling(Npix=2**5, sigma2=0.1, r0=0.1, L0=25, D=1.8,singular_threshold=1e-6,kolmogorov = False, finite_aperture = None,filter_order = None,verbose = True):
    """
    my own attempt at creating phase screens from gaussian sampling of covariance matrix.. works but is much slower than fft methods for init , but faster for sampling once created 

    Parameters
    ----------
    Npix : TYPE
        DESCRIPTION.
    sigma2: TYPE float
        DESCRIPTION. target variance of phase screen to be sampled 
    r0 : TYPE
        DESCRIPTION.
    L0 : TYPE
        DESCRIPTION.
    D : TYPE
        DESCRIPTION.
    singular_threshold : TYPE, optional
        DESCRIPTION. The default is 1e-6.
    kolmogorov : TYPE, optional
        DESCRIPTION. The default is False.
    finite_aperture : TYPE, optional
        DESCRIPTION. The default is None.
    filter_order : TYPE, optional
        DESCRIPTION. The default is None.
    verbose : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    rv = scipy.stats.multivariate_normal() object
    N random (flattened 1D)  phase screen samples can be generated from this object via:
        screen_list = rv.rvs(N)
    
    the final phase screen at a given index i must be reshaped to a 2D array
        screen = screen_list[i].reshape(Npix,Npix)
        
    e.g.
    Npix = 2**5
    rv = create_atm_phase_screen_Guassian_sampling(Npix)
    screens_1D = rv.rvs(2) # sample 2 screens  
    
    final_screen0 = screens_1D[0].reshape(Npix,Npix)
    #(note if only sampling 1 screen we should reshape without index e.g. final_screen0 = screens_1D.reshape(Npix,Npix))
    
    plt.figure()
    plt.imshow( final_screen0  )

    """

    
    
    dx = D/Npix #m
    #P = np.zeros([Npix,Npix]) # init pupil 
    rho = {} # Npix x Npix x Npix**2
    
    
    for i0 in range(Npix): # x pos in screen that we calculate distance from 
        for j0 in range( Npix): # y pos in screen that we calculate distance from 
            #rho[(i0,j0)] = np.zeros([Npix,Npix])
            tmp = np.zeros([Npix,Npix]) # we just fill the upper triangle of matrix then reflect on diagonal (matrix is symmetric by construction!) 
            for i in range(Npix):
                for j in range( Npix):
                    r = dx * np.sqrt( (i0-i)**2 + (j0-j)**2 ) 
                    tmp[i,j] = cov_function(r , sigma2 ,L0, r0, kolmogorov,  finite_aperture, filter_order) 
            rho[(i0,j0)] = tmp
            #if i0!=j0:
            #    rho[(j0,i0)] = tmp
            #lower_diag = np.rot90( np.fliplr(upper_diag) )
            #rho[(i0,j0)] = upper_diag + lower_diag - np.diag( np.diag( upper_diag ) ) # add reflect and subtract half diagonal (since this gets added twice) 
            
    """
    for i0 in range(Npix): # x pos in screen that we calculate distance from 
        for j0 in range( Npix): # y pos in screen that we calculate distance from 
            #rho[(i0,j0)] = np.zeros([Npix,Npix])
            upper_diag = np.zeros([Npix,Npix]) # we just fill the upper triangle of matrix then reflect on diagonal (matrix is symmetric by construction!) 
            for i in range(Npix):
                for j in range(i , Npix):
                    r = dx * np.sqrt( (i0-i)**2 + (j0-j)**2 ) 
                    upper_diag[i,j] = cov_function(r , sigma2 ,L0, r0, kolmogorov,  finite_aperture, filter_order) 
            lower_diag = np.rot90( np.fliplr(upper_diag) )
            rho[(i0,j0)] = upper_diag + lower_diag - np.diag( np.diag( upper_diag ) ) # add reflect and subtract half diagonal (since this gets added twice) 
    """

    # Create our 3D covariance matrix        
    Sigma = np.array( [rho[x] for x in rho] ) # cov_matrix
    
    
    # reshape to 2D to make compatible with MVG 
    cov_matrix = Sigma.reshape(Npix**2,Npix**2) # have to make #D covariance matrix 2D 
    
    # cov_matrix as a few (small) negative or non-zero  imaginary eigenvalues meaning not semi positive definite.. 
    # so we find the nearest positive semi-definite matrix 
    cov_matrix_nearestPD = nearestPD( cov_matrix ) 
    
    # Matrix is generally near singular! USE singular value decomposition to hep filtering 
    U,S,Vt = la.svd(cov_matrix_nearestPD) # U, S, V
    
    # filter near singular values 
    singular_filter = S < singular_threshold 
    Sfilt = S.copy()
    Sfilt[singular_filter] = np.min(S[~singular_filter]) # put (singular) values below threshold to 
    
    # now create our new (non-singular) matrix
    cov_mat_n  = U @ np.diag(Sfilt) @ Vt
    
    # make zero mean 
    mu = np.zeros(cov_mat_n.shape[0]) 
    
    # create our multivariant Guassian to sample screens from 
    rv = multivariate_normal( mean=mu, cov=cov_mat_n   )
    
    # generate a 1D phase screen 
    #screen0 = rv.rvs(1)
    
    # reshape it
    #screen = screen0.reshape(Npix,Npix)
    
    """# lets look
    plt.figure()
    plt.title('simulated phase screen')
    plt.imshow(screen)"""
    
    if verbose:
        # lets look at differences 
        fig,ax = plt.subplots(1,2,figsize=(10,5))
        ax[0].imshow(cov_matrix ) 
        ax[0].set_title(r'$\Sigma$ ',fontsize=20)
        ax[1].imshow( cov_matrix_nearestPD  )
        ax[1].set_title('nearest semi-definite \n'+r'matrix to $\Sigma$',fontsize=20)
        
        plt.figure() # eigenvalues 
        plt.plot( np.real( la.eigvals(cov_matrix)) ,color='b',linestyle='-',label=r'real eigenvals of $\Sigma$')
        plt.plot( np.imag( la.eigvals(cov_matrix)) ,color='b',linestyle=':',label=r'imaginary eigenvals of $\Sigma$')
        
        plt.plot( np.real( la.eigvals(cov_matrix_nearestPD)) ,color='g',linestyle='-',label=r'real eigenvals of nearest PD to $\Sigma$')
        plt.plot( np.imag( la.eigvals(cov_matrix_nearestPD)) ,color='g',linestyle=':',label=r'imaginary eigenvals of nearest PD to $\Sigma$')
        plt.xlabel('eigenvalue index ',fontsize=14)
        plt.ylabel('eigenvalue',fontsize=14)
        plt.yscale('symlog')
        plt.xscale('log')
        plt.gca().tick_params(labelsize=14)
        plt.legend()
        
        
        # check for zero (or very small values) in S
        plt.figure()
        plt.loglog( S, label='without filter' )
        plt.loglog( Sfilt, label='with filter' )
        plt.ylabel('S in SVD',fontsize=14)
        plt.xlabel('index',fontsize=14)
        plt.gca().tick_params(labelsize=14)
        plt.legend()
    
        # how does covariance look after removing singular values 
        fig,ax = plt.subplots(1,2,figsize=(10,5))
        
        ax[0].imshow( cov_matrix_nearestPD ) 
        ax[0].set_title('covariance matrix \nBEFORE removing singular values')
        ax[1].imshow( cov_mat_n ) 
        ax[1].set_title('covariance matrix \nAFTER removing singular values')
    
    return( rv )

@numba.jit(nopython=True, parallel=True)
def calc_seperations_fast(positions, seperations):

    for i in numba.prange(len(positions)):
        x1, y1 = positions[i]
        for j in range(len(positions)):
            x2, y2 = positions[j]
            delta_x = x2 - x1
            delta_y = y2 - y1

            delta_r = np.sqrt(delta_x ** 2 + delta_y ** 2)

            seperations[i, j] = delta_r
            







# less likely to use 
# generate Non object oriented list of phase screens that can't be rolled '

def cov_function(r , sigma2 ,L0, r0, kolmogorov = False,  finite_aperture=None, filter_order=None):
    
    
    if finite_aperture is not None: 
        if r<=finite_aperture:
            B = sigma2 - D_phi(r,L0,r0, kolmogorov, finite_aperture,filter_order) / 2 #covariance 
        else:
            if filter_order is not None:
                #power law decay (power law index = filter_order )
                B = (sigma2 - D_phi(finite_aperture,L0,r0, kolmogorov, finite_aperture,filter_order) / 2) / (1+r-finite_aperture)**filter_order #0 #sigma2_ao + D_phi(r-D/2, L0, r0, kolmogorov) 
                #exponetial decay 
                #B = (sigma2 - D_phi(finite_aperture,L0,r0, kolmogorov, finite_aperture,filter_order) / 2) * np.exp( -filter_order * (r-finite_aperture) )#0 #sigma2_ao + D_phi(r-D/2, L0, r0, kolmogorov) 
            else:
                B=0
    else:
        B = sigma2 - D_phi(r,L0,r0, kolmogorov, finite_aperture,filter_order) / 2 #covariance 
        
    #BB=[b if b>0 else 0 for b in B]
    return(B) 


def seeing2r0(seeing,wvl):
    r0 = 0.98 * wvl / np.deg2rad(seeing / 3600)
    return(r0)


def r02seeing(r0,wvl):
    seeing = np.rad2deg( 1/( 1/0.98 * r0/wvl ) ) * 3600 
    return(seeing)

def create_atm_phase_screen(N_screens, x, sigma2 ,D, L0, seeing, img_wave, kolmogorov = False, finite_aperture=None,filter_order=None, verbose=False, rad_units=True):
    """
    Non object oriented list of phase screens that can't be rolled '
    
    Parameters
    ----------
    N_screens : int
        DESCRIPTION. number of screens to generate 
    x : TYPE : array like 
        DESCRIPTION. x coorindates (m) that the phase grid will be created on 
    sigma2 : TYPE float
        DESCRIPTION. expected spatial phase variance (rms)of the screen. From kolmogorov theory this should be 1.0299 * (D/r0)**(5/3) rad for the atmosphere. But is left to the user as a free input 
    D : TYPE float
        DESCRIPTION. telescope diameter 
    L0 : TYPE float
        DESCRIPTION. outerscale (m)
    seeing : TYPE float
        DESCRIPTION. atmospheric seeing (arcsec) @ 500nm
    img_wave : TYPE float
        DESCRIPTION. wavelength that the phase screen is calcualted at 
    kolmogorov : TYPE, Bolean (optional)
        DESCRIPTION. Von Karman (finite outer scale) or Kolmogorov model?The default is False.
    finite_aperture : TYPE, optional, None or float  g
        DESCRIPTION. ive finite value to define a cut off distance in the structure function used to calculate the phase screen. The default is None.
    filter_order : TYPE, optional None or float 
        DESCRIPTION. if finite_aperture is defined (!=None) do we apply a hard cut at r=finite_aperture (filter_order=None) or a filtered cut of some order (filter_order=float) The default is None.
    verbose : TYPE, Bolean optional 
        DESCRIPTION. Do we plot intermediate stuff? The default is False.
    rad_units : TYPE, Bolean optional
        DESCRIPTION. are the output phase screens in units = radians? The default is True. Otherwise returned in units = meters

    Returns
    -------
    screen_list - a list of phase screens

    """
    
    X,Y =np.meshgrid(x,x)
    R = (X**2+Y**2)**0.5
    dr = abs( np.diff(R)[-1,-1] )
    
    r0 = seeing2r0(seeing, img_wave) # Fried parameter @ img_wave
    
    B_atm =  np.array( [cov_function(rtmp , sigma2 , L0, r0, kolmogorov = False, finite_aperture=finite_aperture,filter_order=filter_order) for rtmp in R.reshape(-1)] )
    #B_atm = np.array([ao_cov_function(rtmp,sigma2_ao, D, N_act, L0, r0, kolmogorov = False, finite_aperture=finite_aperture,filter_order=filter_order) for rtmp in R.reshape(-1)] )
    B_atm = B_atm.reshape(X.shape) # we flattened it in cov calculation so we put it back to original shape 
    # plt.imshow( np.log10( abs(B_atm ) ) )
    
    # PSD (spatial)
    k_atm =  np.fft.fftfreq(  len(B_atm), d=dr ) #[:len(B_atm)//2]
    #Kx, Ky = np.meshgrid(k_atm,k_atm)
    #Kr =  (Kx**2+Ky**2)**0.5
    
    psd_atm = dr**2 * np.fft.fftshift( np.fft.fft2( B_atm ) ) #/ Kr #[:len(B_atm)//2]  
    psd_atm[~np.isfinite( psd_atm )] = 0 
    #pd.DataFrame( abs(psd_atm) ).interpolate().values
    
    # integrate along 1 axis 
    if verbose:
        psd_atm_1D =  np.sum(abs(psd_atm), axis=0) * np.diff(k_atm)[0]
        psd_theory = 7.2e-3 * (D/r0)**(5/3) * abs(k_atm)**(-11/3) # Tatarski 1961
        
        plt.figure()
        plt.loglog( k_atm[:len(B_atm)//2],  psd_atm_1D[len(B_atm)//2:] ) #np.pi/k_atm[:len(B_atm)//2] * psd_atm_1D[len(B_atm)//2:] 
        plt.loglog(k_atm[:len(B_atm)//2], psd_theory[:len(B_atm)//2])
    
    screen_list = []
    for it in  range(N_screens):
        
        
        #phs_np = array_to_numpy(phs_data, phs_shape, dtype=np.float32)
        
        local_dim = psd_atm.shape[0]
        local_L = np.max(x) - np.min(x)
        img_wave = 1e-6 #
        
        # random draw of Gaussian noise
        tab = np.random.normal(loc=0, scale=1, size=(psd_atm.shape[0], psd_atm.shape[1]))
        
        # switch to Fourier space
        tab = np.fft.ifft2(tab)
        
        # normalize
        tab *= local_dim*local_L
        
        # multiply with PSD
        tab = tab * np.sqrt(abs(psd_atm))
        
        # switch back to direct space
        tab = np.fft.fft2(tab).real
        
        # normalize
        if rad_units:
            tab *=  1 / local_L**2 # rad
        else: # return screen in m
            tab *=  img_wave / (2*np.pi) / local_L**2 # m 
        
        screen_list.append(abs(tab))
    
    return(screen_list)


#%% pre processed screens 
# =============================================================================
# phase_screens = np.array( [pup * screens.add_row() for i in range(100)] )   
# 
# opd_screens={}
# for w in [wfs_wvl] + list(wvls) :
#     
#     opd_screens[w] = np.array([calibrate_phase_screen2wvl(w, w/(2*np.pi) * pup * phi)  for phi in phase_screens])
# 
# ao_screens = modal_AO_correction( opd_screens, n_modes=14, lag=lag, Ki=.95, Kp=1.1, V=V, dx=dx , wvls=wvls , wfs_wvl=wfs_wvl, it=1000, pup= None)
# 
# w = wvls[1]
# before_strehl = np.array( [np.exp( -np.nanvar( 2*np.pi/w * p ) ) for p in opd_screens[w]] )
# after_strehl =  np.array( [np.exp( -np.nanvar( 2*np.pi/w * p ) ) for p in np.array( ao_screens )[:,2,:,:]] )
# plt.plot(before_strehl); plt.plot( after_strehl, label='after'), plt.legend()
# 
# =============================================================================
#aa =modal_AO_correction( phase_screen, n_modes=14, lag=lag, Ki=.95, Kp=1.1, V=V, dx=phase_screen.pixel_scale , wvls=[1.1e-6, 1.65e-6] , wfs_wvl=0.6e-6, it=1000, pup= None)

#%% or with the original ao tools screens 
# =============================================================================
# 
# ao_screens = modal_AO_correction( screens, n_modes=14, lag=lag, Ki=.95, Kp=1.1, V=V, dx=dx , wvls=wvls , wfs_wvl=wfs_wvl, it=1000, pup= None)
# 
# w = wvls[1]
# #before_strehl = np.array( [np.exp( -np.nanvar( 2*np.pi/w * p ) ) for p in opd_screens[w]] )
# after_strehl =  np.array( [np.exp( -np.nanvar( 2*np.pi/w * p ) ) for p in np.array( ao_screens )[:,2,:,:]] )
# plt.plot(before_strehl); plt.plot( after_strehl, label='after'), plt.legend()
# 
# 
# =============================================================================
#%% testing 



# # input pupil field grid (WORKS BEST WHEN dim=D_pix )
# dim=2**9
# D_pix = 2**9

# pup = pick_pupil('AT', dim=dim, diameter=D_pix ) #aperture.disc(dim=dim, size=D_pix,diameter=True) #

# # pupil basis
# basis = zernike.zernike_basis(nterms=10, npix=D_pix)

# # input field 
# wvls = np.linspace( 1.5e-6, 1.7e-6, 5)

# ph_flux_H = star2photons('H', 11, airmass=1.3, k = 0.18, ph_m2_s_nm = True)
# fluxes = [pup * (ph_flux_H + noise) for noise in np.random.normal(0, 1e-5*ph_flux_H, len(wvls))]

# # NOTE THINGS GET VERY NON-LINEAR IN HIGHER ORDER MODES IF a > 1 ( i.e. 5e-1*basis[5])
# phase_tmp = 5e-1 * basis[9]
# phase_tmp[np.isnan(phase_tmp)] = 0

# phase_tmp2 = np.pad(phase_tmp, [(int((dim-D_pix)/2), int((dim-D_pix)/2)), (int((dim-D_pix)/2), int((dim-D_pix)/2))], mode='constant')#putinside_array( pup.copy(), phase_tmp.copy())
# phases = [phase_tmp2 for i in range(len(wvls))]


# # focal plane Filter
# A=1
# B=1
# f_ratio=21 
# d_on=26.5e-6
# d_off=26e-6
# glass_on='sio2'
# glass_off='sio2'

# desired_phase_shift = 60 # deg

# # focal plane grid
# phase_shift_diameter = 1 * f_ratio * wvls[0]   ##  f_ratio * wvls[0] = lambda/D  given f_ratio

# nx_size_focal_plane = dim #grid size in focal plane 
# N_samples_across_phase_shift_region = 10 # number of pixels across pghase shift region 
# dx_focal_plane = phase_shift_diameter / N_samples_across_phase_shift_region  # 


# # init filter 
# FPM = zernike_phase_mask(A,B,phase_shift_diameter,f_ratio,d_on,d_off,glass_on,glass_off)

# # optimize depths 
# FPM.optimise_depths(desired_phase_shift=desired_phase_shift, across_desired_wvls=wvls ,verbose=True)


# # generate field class for input field 
# input_field = field(fluxes , phases, wvls )

# # define the grid 
# input_field.define_pupil_grid( dx = 1.8 / D_pix, D_pix = D_pix )

# # get output field after phase mask
# output_field = FPM.get_output_field( input_field, wvl_lims=[0,100], \
#                                     nx_size_focal_plane = nx_size_focal_plane , dx_focal_plane = dx_focal_plane, keep_intermediate_products=True )

# output_field.define_pupil_grid(dx=input_field.dx, D_pix=input_field.D_pix)
# #print("once field, masks are intialized time for getting output field:--- %s seconds ---" % (time.time() - start_time))

# # detector 
# DIT = 1 # integration time (s)
# ron = 1 #read out noise (e-) 
# pw = 2**4 # windowing (pixel_size = field_dx * pw)
# npix_det = input_field.flux[wvls[0]].shape[0]//pw # number of pixels across detector 
# pix_scale = input_field.dx * pw # m/pix

# # ++++++++++++++++++++++
# # DETECT 

# det = detector(npix=npix_det, pix_scale = pix_scale, DIT= DIT, ron=ron, QE={w:1 for w in input_field.wvl})

# sig1 = det.detect_field( output_field, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True)


# # need to fix include_shotnoise=True,

# # check sum(det) == np.trapz(DIT * sum(field[w] * dx**2), wvl)
# print('check sum(det) ~ np.trapz(DIT * sum(field[w] * dx**2), wvl)\n\nnp.trapz(DIT * sum(field[w] * dx**2), wvl)=')
# print('  ', np.trapz([DIT * np.sum(input_field.flux[w] * input_field.dx**2) for w in  input_field.flux.keys()], 1e9*np.array(list(input_field.flux.keys()))) )
# print( '\nsum(det)=' )
# print( '  ', np.sum(det.det))


# fig,ax = plt.subplots(1,2)
# ax[1].imshow(sig1.signal)
# ax[1].set_title('detector')
# ax[0].imshow(phase_tmp)
# ax[0].set_title('input phase')
# ax[1].axis('off')
# ax[0].axis('off')
# plt.tight_layout()


# # calibration phase maskPhase estimate 

# FPM_cal = zernike_phase_mask(A,B,phase_shift_diameter,f_ratio,d_on,d_off,glass_on,glass_off)

# # optimize depths 
# FPM_cal.d_off = FPM_cal.d_on
# #FPM_cal.optimise_depths(desired_phase_shift=0, across_desired_wvls=wvls ,verbose=True)

# output_field_cal = FPM_cal.get_output_field( input_field, wvl_lims=[0,100], \
#                                     nx_size_focal_plane = nx_size_focal_plane , dx_focal_plane = dx_focal_plane, keep_intermediate_products=False )

# output_field_cal.define_pupil_grid(dx=input_field.dx, D_pix=input_field.D_pix)


# sig_cal = det.detect_field( output_field_cal , include_shotnoise=True, ph_per_s_per_m2_per_nm=True,grids_aligned=True)

# #plt.imshow(output_field_cal.flux[wvls[0]])

# #  TRY JUST TAKE MEAN OF B ON DETECTOR PIXELS AT CENTRAL WAVELENGTH 
# #plt.imshow( aggregate_array(abs(FPM.b[int(len(wvls)//2)]), det.det.shape, how='mean') )
# #b_est = aggregate_array(abs(FPM.b[int(len(wvls)//2)]), det.det.shape, how='mean')
# #


# b_field = field(fluxes=[abs(FPM.b[i]) for i,_ in enumerate(wvls) ],\
#                 phases=[np.angle(FPM.b[i]) for i,_ in enumerate(wvls) ], wvls=wvls)
    
# b_field.define_pupil_grid(dx=input_field.dx, D_pix=input_field.D_pix)

# sig_b = det.detect_field(b_field, include_shotnoise=False, ph_per_s_per_m2_per_nm=True, grids_aligned=True)

# # WHY factor of 10 error in b when A=B=1??????
# phi_est = sig1.ZWFS_phase_estimator_1(A=A, B=B, b = (0.1 * sig_b.signal)**0.5, P = sig_cal.signal**0.5, \
#                                       theta = np.deg2rad( FPM.phase_mask_phase_shift(np.mean(wvls))) , exp_order=1)


# # residual 
# phi_est_in_pup = np.repeat(phi_est , pup.shape[0]/det.det.shape[0], axis=1).repeat(pup.shape[0]/det.det.shape[0], axis=0)
# #phi_est_in_pup_2 = np.repeat(phi_est2 , pup.shape[0]/det.det.shape[0], axis=1).repeat(pup.shape[0]/det.det.shape[0], axis=0)


# plt.figure()
# plt.plot( input_field.phase[wvls[0]][pup>0.5], phi_est_in_pup[pup>0.5],'.',alpha=0.01)
# plt.plot( input_field.phase[wvls[0]][pup>0.5], input_field.phase[wvls[0]][pup>0.5],'-',label='1:1')
# plt.xlabel('input phase (rad)')
# plt.ylabel('phase estimate (rad)')
# plt.legend()  




# def subplot_additions(ax,im , title,cbar_label, axis_off=False, fontsize=18):
#     ax[0].set_title( title, fontsize=fontsize )
#     divider = make_axes_locatable(ax[0])
#     cax = divider.append_axes('bottom', size='5%', pad=0.05)
#     cbar = fig.colorbar( im, cax=cax, orientation='horizontal')
#     cbar.set_label(cbar_label, rotation=0, fontsize=fontsize)
#     #cbar.tick_params(labelsize=fontsize)
#     if axis_off:
#         ax[0].axis('off')

# wvl_indx = wvls[0]   
# residual_wvl = ( phi_est_in_pup  - input_field.phase[wvl_indx] )

# from mpl_toolkits.axes_grid1 import make_axes_locatable
# app_tmp =  zernike.zernike_basis(nterms=1, npix=det.det.shape[0])[0]
# fig = plt.figure(figsize=(20, 20))

# ax1 = fig.add_subplot(231) # no rows, no cols, ax number 
# im1 = ax1.imshow( input_field.phase[wvl_indx] )
# subplot_additions([ax1], im1 , title='input phase'.format(round( FPM.phase_mask_phase_shift(np.mean(wvls)),1)),\
#                   cbar_label= 'phase (rad)', axis_off=True)
# ax1.text(pup.shape[0]//4, pup.shape[0]//4,'{}um strehl = {}'.format(round(wvl_indx*1e6,2), round( np.exp(-np.var(input_field.phase[wvl_indx][pup>0.5])),2) ),\
#           fontsize=18,color='w')
    
# ax2 = fig.add_subplot(232) # no rows, no cols, ax number 
# im2 = ax2.imshow( sig1.signal )
# subplot_additions([ax2], im2 , title='detector (theta = {})'.format(round( FPM.phase_mask_phase_shift(np.mean(wvls)),1)),\
#                   cbar_label= r'Intensity (adu)', axis_off=True)    

# ax3 = fig.add_subplot(233) # no rows, no cols, ax number 
# im3 = ax3.imshow( sig_cal.signal )
# subplot_additions([ax3], im3 , title='detector (theta = {})'.format(round(0,1)),\
#                   cbar_label= r'Intensity (adu)', axis_off=True)   
    
    
# ax4 = fig.add_subplot(234) # no rows, no cols, ax number 
# im4 = ax4.imshow( phi_est * app_tmp )
# subplot_additions([ax4], im4 , title='phase estimate',\
#                   cbar_label= r'phase (rad)', axis_off=True)   

# ax5 = fig.add_subplot(235) # no rows, no cols, ax number 
# im5 = ax5.imshow( pup * ( residual_wvl ) )
# ax5.text(pup.shape[0]//4, pup.shape[0]//4,'{}um strehl = {}'.format(round(wvl_indx*1e6,2), round( np.exp(-np.var(residual_wvl[pup>0.5])),2) ),\
#           fontsize=18,color='w')
# subplot_additions([ax5], im5 , title='phase residual',\
#                   cbar_label= r'phase (rad)', axis_off=True)   

    
# ax6 = fig.add_subplot(236) # no rows, no cols, ax number 
# ax6.hist( (input_field.phase[wvl_indx] - phi_est_in_pup)[pup>0] , bins= np.linspace(-np.pi,np.pi,30), alpha=0.4, label='residual')
# ax6.hist( input_field.phase[wvls[0]][pup>0] , bins= np.linspace(-2*np.pi,2*np.pi,30), label='prior correction',alpha=0.4)
# ax6.legend(fontsize=15)
# ax6.set_xlabel('phase (rad)',fontsize=18)

# #%%
# # does one of these work? 
# phi_est = det.ZWFS_phase_estimator_1(A=A, B=B, b = (1 * b_est)**0.5, P = det_cal.det**0.5, \
#                                       theta = np.deg2rad( FPM.phase_mask_phase_shift(np.mean(wvls))) , exp_order=2)
# #phi_est2 = det.ZWFS_phase_estimator_1(A=A, B=B,  b = 2*b_est2**0.5, P=det_cal.det**0.5, theta=np.deg2rad( FPM.phase_mask_phase_shift(np.mean(wvls))) , exp_order=1)


# # residual 
# phi_est_in_pup = np.repeat(phi_est[0] , pup.shape[0]/det.det.shape[0], axis=1).repeat(pup.shape[0]/det.det.shape[0], axis=0)
# #phi_est_in_pup_2 = np.repeat(phi_est2 , pup.shape[0]/det.det.shape[0], axis=1).repeat(pup.shape[0]/det.det.shape[0], axis=0)


# plt.figure()
# plt.plot( input_field.phase[wvls[0]][pup>0.5], phi_est_in_pup[pup>0.5],'.',alpha=0.01)
# plt.plot( input_field.phase[wvls[0]][pup>0.5], input_field.phase[wvls[0]][pup>0.5],'-',label='1:1')
# plt.xlabel('input phase (rad)')
# plt.ylabel('phase estimate (rad)')
# plt.legend()





# #%%
# def subplot_additions(ax,im , title,cbar_label, axis_off=False):
#     ax[0].set_title( title )
#     divider = make_axes_locatable(ax[0])
#     cax = divider.append_axes('bottom', size='5%', pad=0.05)
#     cbar = fig.colorbar( im, cax=cax, orientation='horizontal')
#     cbar.set_label(cbar_label, rotation=0)
#     if axis_off:
#         ax[0].axis('off')

# wvl_indx = wvls[0]   


# from mpl_toolkits.axes_grid1 import make_axes_locatable
# app_tmp =  zernike.zernike_basis(nterms=1, npix=det.det.shape[0])[0]
# fig = plt.figure(figsize=(20, 20))

# ax1 = fig.add_subplot(831) # no rows, no cols, ax number 
# im1 = ax1.imshow( det.det )
# subplot_additions([ax1],im1 , title='detector (theta = {})'.format(round( FPM.phase_mask_phase_shift(np.mean(wvls)),1)),\
#                   cbar_label= r'Intensity $(ph)$', axis_off=True)
    
# #ax2 = fig.add_subplot(832)
# #im2 = ax2.imshow( det_cal.det  )
# #subplot_additions([ax2],im2 , title='detector (theta = {})'.format(round( FPM_cal.phase_mask_phase_shift(np.mean(wvls)),1)),\
# #                  cbar_label= r'Intensity $(ph)$', axis_off=True)
    
    
# ax3 = fig.add_subplot(833)
# im3 = ax3.imshow( 1e6 * wvl_indx/(2*np.pi) * input_field.phase[wvl_indx] )
# subplot_additions([ax3],im3 , title='input phase',\
#                   cbar_label= r'opd ($\mu$m)', axis_off=True)
    

# """ax5 = fig.add_subplot(834)
# im5 = ax5.imshow(  b_est  )
# subplot_additions([ax5],im5, title='b estimate 1',\
#                   cbar_label= r'Intensity $(ph)$', axis_off=True)
    
# ax6 = fig.add_subplot(835)
# im6 = ax6.imshow(  b_est2 )
# subplot_additions([ax6],im6 , title='b estimate 2',\
#                   cbar_label= r'Intensity $(ph)$', axis_off=True)"""
    

# ax7 = fig.add_subplot(834)
# im7 = ax7.imshow(  1e6 * wvl_indx/(2*np.pi) * app_tmp * phi_est  )
# subplot_additions([ax7],im7 , title='phi est 1 ',\
#                   cbar_label= r'Intensity $(ph)$', axis_off=True)
    
# #ax8 = fig.add_subplot(835)
# #im8 = ax8.imshow( 1e6 * wvl_indx/(2*np.pi) * app_tmp * phi_est2  )
# #subplot_additions([ax8],im8 , title='phi est 2 ',\
# #                  cbar_label= r'Intensity $(ph)$', axis_off=True)
    

# ax9 = fig.add_subplot(836)
# im9 = ax9.imshow(  1e6 * wvl_indx/(2*np.pi) * pup * ( input_field.phase[wvl_indx] - phi_est_in_pup ) )
# subplot_additions([ax9],im9 , title='residual 1 ',\
#                   cbar_label= r'residual (um)', axis_off=True)
    
# #ax10 = fig.add_subplot(837)
# #im10 = ax10.imshow(  1e6 * wvl_indx/(2*np.pi) *  pup * ( input_field.phase[wvl_indx] - phi_est_in_pup_2 ) )
# #subplot_additions([ax10],im10 , title='phi est 1 ',\
# #                  cbar_label= r'residual2 ($\mu$m)', axis_off=True)

# ax11 = fig.add_subplot(838)
# ax11.hist( (input_field.phase[wvl_indx] - phi_est_in_pup)[pup>0] ,alpha=0.4, label='residual');
# ax11.hist( input_field.phase[wvls[0]][pup>0] , label='prior correction',alpha=0.4)
# ax11.legend()

# #ax12 = fig.add_subplot(839)
# #ax12.hist( (input_field.phase[wvl_indx] - phi_est_in_pup_2)[pup>0] ,alpha=0.4, label='residual');
# #ax12.hist( input_field.phase[wvls[0]][pup>0] , label='prior correction',alpha=0.4)
# #ax12.legend()


# #%%






# ax1.set_title( )
# ax1.axis('off')
# im1 = ax1.imshow( det.det )

# divider = make_axes_locatable(ax1)
# cax = divider.append_axes('bottom', size='5%', pad=0.05)
# cbar = fig.colorbar( im1, cax=cax, orientation='horizontal')
# cbar.set_label( r'Intensity $(ph)$', rotation=0)






# fig,ax = plt.subplots(1,6,figsize=(10,5))
# ax[0].imshow( det.det )
# ax[1].imshow( det_cal.det )
# ax[2].imshow( b_est )
# ax[3].imshow( app_tmp * phi_est )
# ax[4].imshow( app_tmp * phi_est2 )
# ax[5].imshow (pup*(input_field.phase[wvls[0]] - phi_est_in_pup_2 ) )
# ax[0].set_title('detector (theta = {})'.format(round( FPM.phase_mask_phase_shift(np.mean(wvls)),1)) )
# ax[1].set_title('detector (theta = 0)')
# ax[2].set_title('b estimate')
# ax[3].set_title('phase estimate 1')
# ax[4].set_title('phase estimate 2')
# ax[4].set_title('residual')

# for i in range(len(ax)):
#     ax[i].axis('off')

# plt.tight_layout()

# #note DM correction should be an opd not phase!!!
# strehl_before = np.exp(-np.var( (input_field.phase[wvls[0]])[pup>0.5] ) )
# strehl_after = np.exp(-np.var( (input_field.phase[wvls[0]] - phi_est_in_pup_2)[pup>0.5] ) )

# print('strehl before = {}, strehl after = {}'.format( strehl_before , strehl_after) )

# plt.figure()
# plt.hist( (input_field.phase[wvls[0]] - phi_est_in_pup_2)[pup>0] ,alpha=0.4, label='residual');
# plt.hist( input_field.phase[wvls[0]][pup>0] , label='prior correction',alpha=0.4)
# plt.legend()

# #plt.plot( phi_est[:, det.det.shape[0]//2] )
# #X = (det.det - det_cal.det)/b_est
# #X * iteraction_matrix = phi => iteraction_matrix = X^-1 * phi 
# #np.linalg.inv((det.det - det_cal.det)/b_est) @ 
# # (np.ones( [phases[0].shape[0],det.det.shape[0]]) @ det.det @ np.ones( [det.det.shape[0], phases[0].shape[0]]))


# #%% speed tests

# """def integrate(y,x): 
#     Y = np.sum((y[1:]+y[:-1])/2 * (x[1:] - x[:-1])) 
#     return(Y)"""

# aaa=np.linspace(0,10,100); bbb=np.linspace(0,1,100)
# t_trapz=[]
# t_sum = []
# residuals = []
# for i in range(1000):
#     start_trapz = time.perf_counter()
#     t = np.trapz(aaa,bbb)
#     end_trapz = time.perf_counter()
    
#     start_sum = time.perf_counter()
#     s = integrate(np.array(aaa),np.array(bbb)) #np.sum((aaa[1:]+aaa[:-1])/2 * (bbb[1:] - bbb[:-1])) #np.sum(aaa[1:] * np.diff(bbb))
#     end_sum = time.perf_counter()
    
#     t_trapz.append( end_trapz- start_trapz )
#     t_sum.append( end_sum- start_sum )
#     residuals.append( s-t)
# plt.hist( np.log10(t_trapz),bins=40, label='trapz' );plt.hist( np.log10(t_sum),bins=40,  label='sum' ); plt.legend(); plt.xlabel('log10(delta t)')
# print(f'max residuals = {np.max( residuals )}, \nintegrate fn = {np.mean(np.array( t_sum )/np.array( t_trapz) )} quicker')