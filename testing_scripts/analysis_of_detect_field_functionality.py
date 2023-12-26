#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 11:27:27 2023

@author: bcourtne

field coordinates set with define_grid method in field object. You can shift the field center with the center tuple variable. 
applyDM method in field object makes DM inherit (if it doesn't already have ') field coordinates with center at origin. 
Therefore should set x, y coorindates of DM prioer 


Q1: What happens if DM outside of field with interpolation when applying DM?
    phase -> np.nan, flux goes to zero,
Q2: how to best detect a field in generic way given detector ( interpolate and bin)? requirements is speed


"""

import numpy as np
import pylab as plt
import pandas as pd
import os
from scipy.interpolate import interp1d
import pyzelda.utils.zernike as zernike
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy 
import copy
import aotools
from astropy.io import fits
import json
import time
#import zelda
os.chdir('/Users/bcourtne/Documents/ANU_PHD2/baldr')

from functions import baldr_functions_2 as baldr
from functions import data_structure_functions as config

tel_config =  config.init_telescope_config_dict(use_default_values = True)
phasemask_config = config.init_phasemask_config_dict(use_default_values = True) 
DM_config = config.init_DM_config_dict(use_default_values = True) 
detector_config = config.init_detector_config_dict(use_default_values = True)
# define a hardware mode for the ZWFS 
mode_dict = config.create_mode_config_dict( tel_config, phasemask_config, DM_config, detector_config)

# init out Baldr ZWFS object with the desired mode 
zwfs = baldr.ZWFS(mode_dict) 


#%% Testing offsets with DM and ZWFS on detector 


input_field_0 =  baldr.init_a_field( Hmag=4, mode='Kolmogorov', wvls=zwfs.wvls, pup_geometry=zwfs.mode['telescope']['pup_geometry'],\
             D_pix=zwfs.mode['telescope']['telescope_diameter_pixels'], \
                 dx=zwfs.mode['telescope']['telescope_diameter']/zwfs.mode['telescope']['telescope_diameter_pixels'], \
                     r0=0.1, L0=25, phase_scale_factor = 1)    

dm_offset = [0, 0]
zwfs_offset = [.8,0]
xDM = np.linspace(input_field_0.x.min(),input_field_0.x.max(),zwfs.dm.N_act[0])  + dm_offset[0]
yDM = np.linspace(input_field_0.y.min(),input_field_0.y.max(),zwfs.dm.N_act[0])  + dm_offset[1]

zwfs.dm.define_coordinates( xDM, yDM )

postDM_field = input_field_0.applyDM( zwfs.dm )


out_field = zwfs.FPM.get_output_field(postDM_field, wvl_lims=[-np.inf, np.inf] , keep_intermediate_products=False, replace_nan_with=0)

    # !!! isue with center coordinates -- does not correspond to center 
out_field.define_pupil_grid(dx=input_field_0.dx, D_pix=input_field_0.D_pix, center=(zwfs_offset[0], zwfs_offset[1]))



t0 = time.time()
sig_nal = zwfs.det.detect_field( out_field, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=False)
t1 = time.time()
print('not aligned time = ', t1-t0)

plt.figure()
plt.scatter(input_field_0.X,input_field_0.Y, c = input_field_0.phase[zwfs.wvls[0]] ) 
plt.scatter( zwfs.dm.X, zwfs.dm.Y, c=zwfs.dm.surface)
plt.title('dots show DM actuators vs input field phase ')

fig,ax = plt.subplots(1,2)
ax[1].pcolormesh( postDM_field.x, postDM_field.y, postDM_field.phase[zwfs.wvls[0]] )
ax[1].set_title('field phase post DM')
ax[0].pcolormesh( postDM_field.x, postDM_field.y, postDM_field.flux[zwfs.wvls[0]] )
ax[0].set_title('field flux post DM')

plt.figure()
plt.pcolormesh( sig_nal.signal ) 


#%% Testing and understanding functionality of applying DM with offsets between field and DM 

input_field_0 =  baldr.init_a_field( Hmag=4, mode='Kolmogorov', wvls=zwfs.wvls, pup_geometry=zwfs.mode['telescope']['pup_geometry'],\
             D_pix=zwfs.mode['telescope']['telescope_diameter_pixels'], \
                 dx=zwfs.mode['telescope']['telescope_diameter']/zwfs.mode['telescope']['telescope_diameter_pixels'], \
                     r0=0.1, L0=25, phase_scale_factor = 1)    

xDM = np.linspace(input_field_0.x.min(),input_field_0.x.max(),zwfs.dm.N_act[0]) + 0.1
yDM = np.linspace(input_field_0.y.min(),input_field_0.y.max(),zwfs.dm.N_act[0]) + 0.1

zwfs.dm.define_coordinates(xDM,yDM)

post_field = input_field_0.applyDM( zwfs.dm )

plt.figure()
plt.scatter(input_field_0.X,input_field_0.Y, c = input_field_0.phase[zwfs.wvls[0]] ) 
plt.scatter( zwfs.dm.X, zwfs.dm.Y, c=zwfs.dm.surface)
plt.title('dots show DM actuators vs input field phase ')

fig,ax = plt.subplots(1,2)
ax[1].pcolormesh( post_field.x, post_field.y, post_field.phase[zwfs.wvls[0]] )
ax[1].set_title('field phase post DM')
ax[0].pcolormesh( post_field.x, post_field.y, post_field.flux[zwfs.wvls[0]] )
ax[0].set_title('field flux post DM')

#%% Testing outputfield of FPM with nan values caused from DM offsets 




out_field = zwfs.FPM.get_output_field(post_field,wvl_lims=[-np.inf, np.inf] ,\
                                          keep_intermediate_products=False, replace_nan_with=0)

out_field.define_pupil_grid(dx=input_field_0.dx, D_pix=input_field_0.D_pix, center=(0.,0))

fig,ax = plt.subplots(1,2)
ax[1].pcolormesh( out_field.x, out_field.y, out_field.phase[zwfs.wvls[0]] )
ax[1].set_title('field phase after focal plane phase mask')
ax[0].pcolormesh( out_field.x, out_field.y, out_field.flux[zwfs.wvls[0]] )
ax[0].set_title('field flux after focal plane phase mask')


#%% use field.define_pupil_grid(self, dx, D_pix=None, center=(0,0)) to control offsets between output field of FPM and detector

"""
work out nearest value to field dx that is devisor of det dex (e.g: dx_det = N*dx)
if grids not aligned interpolate field onto new grid 
detect it
 
"""

dm_offset = [0, 0]
zwfs_offset = [0, 0]

input_field_0 =  baldr.init_a_field( Hmag=4, mode='Kolmogorov', wvls=zwfs.wvls, pup_geometry=zwfs.mode['telescope']['pup_geometry'],\
             D_pix=zwfs.mode['telescope']['telescope_diameter_pixels'], \
                 dx=zwfs.mode['telescope']['telescope_diameter']/zwfs.mode['telescope']['telescope_diameter_pixels'], \
                     r0=0.1, L0=25, phase_scale_factor = 1)    
    

tel_config =  config.init_telescope_config_dict(use_default_values = True)
phasemask_config = config.init_phasemask_config_dict(use_default_values = True) 
DM_config = config.init_DM_config_dict(use_default_values = True) 
detector_config = config.init_detector_config_dict(use_default_values = True)
# define a hardware mode for the ZWFS 
mode_dict = config.create_mode_config_dict( tel_config, phasemask_config, DM_config, detector_config)

# init out Baldr ZWFS object with the desired mode 
zwfs = baldr.ZWFS(mode_dict) 

xDM = np.linspace(input_field_0.x.min(),input_field_0.x.max(),zwfs.dm.N_act[0])  + dm_offset[0]
yDM = np.linspace(input_field_0.y.min(),input_field_0.y.max(),zwfs.dm.N_act[0])  + dm_offset[1]

zwfs.dm.define_coordinates( xDM, yDM )

postDM_field = input_field_0.applyDM( zwfs.dm )


out_field = zwfs.FPM.get_output_field(postDM_field, wvl_lims=[-np.inf, np.inf] ,\
                                          keep_intermediate_products=False, replace_nan_with=0)

# !!! isue with center coordinates -- does not correspond to center 
out_field.define_pupil_grid(dx=input_field_0.dx, D_pix=input_field_0.D_pix, center=(zwfs_offset[0], zwfs_offset[1]))




t0 = time.time()
sig_nal = zwfs.det.detect_field( out_field, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=False)
t1 = time.time()
print('not aligned time = ', t1-t0)

t0 = time.time()
sig_al = zwfs.det.detect_field( out_field, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True)
t1 = time.time()
print('aligned time = ', t1-t0)

plt.figure()
plt.imshow( sig_al.signal )

plt.figure()
plt.imshow( sig_nal.signal )

plt.figure()
plt.title('input phase')
plt.imshow( input_field_0.phase[zwfs.wvls[0]] )

# there seems to be a transpose in the interpoaltion when grids not alligned!! need to check this!!  

"""
not aligned time =  0.13164377212524414
aligned time =  0.030017852783203125
"""



#%% prior testing [old]
zwfs.dm.nearest_interp_fn = scipy.interpolate.LinearNDInterpolator(zwfs.dm.coordinates, zwfs.dm.surface.reshape(1,-1)[0])

#input_field_0.applyDM(zwfs.dm)

    
plt.figure()
plt.scatter(input_field_0.X,input_field_0.Y, c = input_field_0.phase[zwfs.wvls[0]] ); plt.scatter( zwfs.dm.X, zwfs.dm.Y, c=zwfs.dm.surface)


dm_at_field_pt = zwfs.dm.nearest_interp_fn( input_field_0.coordinates ).reshape(input_field_0.nx_size, input_field_0.nx_size)


plt.figure()
plt.scatter(input_field_0.X,input_field_0.Y, c = input_field_0.phase[zwfs.wvls[0]] ); plt.scatter( input_field_0.X,input_field_0.Y, c=dm_at_field_pt)

zwfs.dm.update_shape( np.ones(zwfs.dm.N_act).reshape(-1) )
field_dm = input_field_0.applyDM( zwfs.dm )
    
plt.figure()
plt.scatter(input_field_0.X,input_field_0.Y, c = field_dm.phase[zwfs.wvls[0]] ); plt.scatter( zwfs.dm.X, zwfs.dm.Y, c=zwfs.dm.surface)


# check only first wavelength (nan values should be same across all wavelengths)
nan_fliter = np.isnan( field_dm.phase[zwfs.wvls[0]] )
if np.any( nan_fliter ):
    for w in zwfs.wvls:
        # if nan values (outside interp range of DM ) flux goes to zero and
        #nan_fliter = np.isnan( field_dm.phase[w] )
        field_dm.flux[w][nan_fliter] = 0
        field_dm.phase[w][nan_fliter] = 0
        
        
#field_dm.flux[zwfs.wvls[0]][np.isfinite( field_dm.phase[zwfs.wvls[0]] )] =0 

#%%
scipy.interpolate.LinearNDInterpolator(DM.coordinates, DM.surface.reshape(1,-1)[0])



dm_at_field_pt = DM.nearest_interp_fn( self.coordinates ) # these x, y, points may need to be meshed...and flattened
      
dm_at_field_pt = dm_at_field_pt.reshape( self.nx_size, self.nx_size )
  
phase_shifts = {w:2*np.pi/w * (2*np.cos(DM.angle)) * dm_at_field_pt for w in self.wvl} # (2*np.cos(DM.angle)) because DM is double passed

field_despues = copy.copy(self)

field_despues.phase = {w: field_despues.phase[w] + phase_shifts[w] for w in field_despues.wvl}