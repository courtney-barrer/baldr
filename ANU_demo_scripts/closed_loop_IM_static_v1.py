#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 21:01:02 2024

@author: bencb
"""

import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import glob
from scipy import interpolate
from astropy.io import fits
import aotools
os.chdir('/opt/FirstLightImaging/FliSdk/Python/demo/')
import FliSdk_V2

root_path = '/home/baldr/Documents/baldr'
data_path = root_path + '/ANU_demo_scripts/ANU_data/'
fig_path = root_path + '/figures/'

os.chdir(root_path)
from functions import baldr_demo_functions as bdf

plot_all = True

# --- timestamp
tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

#======== read in some pre-defined DM commands and deflection data 
# --- DM command to make flat DM (calibrated file provided by BMC with the DM) 
flat_dm_cmd = pd.read_csv(bdf.path_to_dm_flat_map,header=None)[0].values 
# read in DM deflection data and create interpolation functions 
deflection_data = pd.read_csv(root_path + "/DM_17DW019#053_deflection_data.csv", index_col=[0])
interp_deflection_1act = interpolate.interp1d( deflection_data['cmd'],deflection_data['1x1_act_deflection[nm]'] ) #cmd to nm deflection on DM from single actuator (from datasheet) 
interp_deflection_4x4act = interpolate.interp1d( deflection_data['cmd'],deflection_data['4x4_act_deflection[nm]'] ) #cmd to nm deflection on DM from 4x4 actuator (from datasheet) 

# ======= INIT HARDWARE
# --- setup camera
fps = 600# frames per second
camera = bdf.setup_camera(cameraIndex=0) #connect camera and init camera object
camera = bdf.set_fsp_dit( camera, fps=fps, tint=None) # set up initial frame rate, tint=None means max integration time for given FPS


# --- setup DM
dm, dm_err_code =  bdf.set_up_DM(DM_serial_number='17DW019#053')

#========= Software setup 

# can get details from most recent recon file under 'poke_images' extension
#recon_file = max(glob.glob( data_path+'BDR_RECON_*.fits' ), key=os.path.getctime)
cropping_corners = [140, 280, 90, 290]

#rows/cols to crop pupil from image
cp_x1,cp_x2,cp_y1,cp_y2 = 21, 129, 11, 119
#rows/cols to crop PSF from image
ci_x1,ci_x2,ci_y1,ci_y2 = 67, 139, 124, 196

# get new or load old reference pupils (FPM in/out)
reference_pupils_path = input('input name of reference pupil file to use, otherwise input 0 to take new data for reference pupils')
if reference_pupils_path == '0':
    
    pup_ref_name = f'PUPIL_CALIBRATION_REFERENCE_FPS-{fps}_{tstamp}.fits'
    #default_naming = input(f'save reference pupil fits as {data_path+pup_ref_name}? [input either 0 or 1]')
    if 0:#not default_naming:
        pup_ref_name = input('then give us a path + name to save reference pupil fits')

    ref_pupils = bdf.get_reference_pupils(dm, camera, fps, flat_map=flat_dm_cmd, number_images_recorded_per_cmd=5,cropping_corners=cropping_corners, save_fits = data_path + pup_ref_name)
    
    reference_pupils_path = data_path + pup_ref_name
else: # just read it in 
    ref_pupils = fits.open( reference_pupils_path )
ref_pupils

number_images_recorded_per_cmd = 10 # how many images do we take before taking median for signal processing 

modal_basis = np.eye(140)
IM_pokeamp = -0.03 # normalized DM units
# create IM
IM = []
for i in range(len(modal_basis)):
    print(f'executing cmd {i}/{len(modal_basis)}')
    cmdtmp = np.zeros(140)
    cmdtmp[i] = IM_pokeamp
    dm.send_data( flat_dm_cmd + cmdtmp )
    time.sleep(0.1)
    im = np.median( bdf.get_raw_images(camera, number_of_frames=number_images_recorded_per_cmd, cropping_corners=cropping_corners) , axis=0)
    errsig =  bdf.get_error_signal( im, reference_pupil_fits = ref_pupils, reduction_dict=None, pupil_indicies = [cp_x1,cp_x2,cp_y1,cp_y2] )

    IM.append( list(errsig.reshape(-1)) )

IM=np.array(IM)
# singular value decomposition of interaction matrix
U,S,Vt = np.linalg.svd( IM ,full_matrices=True)

if plot_all:
    plt.figure()
    plt.plot( S )
    plt.axvline( len(S) * np.pi*2**2/(4.4)**2 ,linestyle=':',color='k',label=r'$D_{DM}^2/\pi r_{pup}^2$')
    plt.ylabel('singular values',fontsize=15)
    plt.xlabel('eigenvector index',fontsize=15)
    plt.legend(fontsize=15)
    plt.gca().tick_params( labelsize=15 )
   
S_filt = S > S[ np.min( np.where( abs(np.diff(S)) < 1e-2 )[0] ) ]
Sigma = np.zeros( np.array(IM).shape, float)
np.fill_diagonal(Sigma, 1/IM_pokeamp * S[S_filt], wrap=False) #

CM = np.linalg.pinv( U @ Sigma @ Vt ) # C = A @ M

print( f'CM condition = {np.linalg.cond(CM)}' )

# ====== zonal gains

noise_level_IM = np.mean(IM)+5*np.std(IM)  #0.1

zonal_gains = np.sum( abs(IM) > noise_level_IM, axis=1)/np.max( np.sum( abs(IM) > noise_level_IM, axis=1))

#modal_gains **= 0.5 # reduce curvature
cmd_region_filt = zonal_gains > 0  # to filter where modal gains are non-zero (i.e. we can actuate here)

if plot_all: 
    plt.figure()
    plt.title('modal gains? np.sum( abs(IM) > IM_noise, axis=1)')
    plt.imshow( bdf.get_DM_command_in_2D( zonal_gains ) )
    plt.colorbar()

# ----------- include zonal gains in CM !!!!!!!!
CM *= zonal_gains


# can check we get reconstuction by sending a cmd, recording control signal and reconstructing
if plot_all:
    i=65
    cmdtmp = np.zeros(140)
    cmdtmp[i] = 0.03
    dm.send_data( flat_dm_cmd + cmdtmp )
    time.sleep(0.5)
    im = np.median( bdf.get_raw_images(camera, number_of_frames=5, cropping_corners=cropping_corners) , axis=0)
    errsig =  bdf.get_error_signal( im, reference_pupil_fits = ref_pupils, reduction_dict=None, pupil_indicies = [cp_x1,cp_x2,cp_y1,cp_y2] )

    recon_cmd = errsig.reshape(-1) @ (CM)
    fig,ax = plt.subplots(1,2)
    im0=ax[0].imshow( bdf.get_DM_command_in_2D( cmdtmp ) )
    ax[0].set_title(r'original command')
    im1=ax[1].imshow( bdf.get_DM_command_in_2D( recon_cmd ) ) 
    ax[1].set_title(r'reconstructed command')
    plt.colorbar(im0,ax=ax[0])
    plt.colorbar(im1,ax=ax[1])
    plt.show()


# ====== noise in image 
im = bdf.get_raw_images(camera, number_of_frames=500, cropping_corners=cropping_corners)
# how much of a command does this correspond to?
fake_im = np.zeros( IM[65].shape )
fake_im[np.argmax( abs( IM[65] ) )] = np.mean( np.std( im ,axis=0) ) # set where act 65 is registered to expected RMS
#plt.imshow( fake_im.reshape( [cp_x2-cp_x1, cp_y2-cp_y1] ) ); plt.show()
CM_noise = np.max(abs(fake_im @ CM)) 
print(f'detector noise level equivilant to cmd RMS =  {CM_noise}')

if plot_all:
    plt.figure()
    plt.imshow( np.std( im ,axis=0) ) ;plt.colorbar();plt.show() # we see structure due to pixelated shot noise  (std~sqrt(photons))
    print(f'expected pixel rms = {np.mean( np.std( im ,axis=0) ) }')

    plt.figure();plt.title('cmd noise RMS after injecting detector noise in act 65');plt.imshow( bdf.get_DM_command_in_2D( fake_im @ CM ) );plt.colorbar();plt.show()

# ====== modal gains
modal_gains = np.ones(len(flat_dm_cmd))


# ======= init disturbance

modes = bdf.construct_command_basis(dm , basis='Zernike', number_of_modes = 20, actuators_across_diam = 'full',flat_map=None)

mode_keys = list(modes.keys())

#zernike like disturbance
#disturbance_cmd = 0.6*cmd_region_filt * ( flat_dm_cmd - modes[mode_keys[10]] )

# square bump disturbance
disturbance_cmd = np.zeros( len( flat_dm_cmd )) 
disturbance_cmd[np.array([40,41,52,53,64,65])]=-0.1
if plot_all:
    plt.figure()
    plt.title( f'static aberration to apply to DM (std = {np.std(disturbance_cmd)} in cmd space')
    plt.imshow( bdf.get_DM_command_in_2D(disturbance_cmd, Nx_act=12) )
    plt.colorbar()
    plt.show()

# ====== PID


Kp, Ki, Kd = 1.3,0.5,0
dt = 1/fps

# notatio from PID wikipedia section "Discrete implementation". https://en.wikipedia.org/wiki/Proportional–integral–derivative_controller
A0 = Kp + Ki*dt + Kd/dt
A1 = -Kp - 2*Kd/dt
A2 = Kd/dt


# ======= Close loop
# init lists to hold data from control loop
IMG_list = [ ]
DELTA_list = [ ] #list( np.nan * np.zeros( int( (cp_x2 - cp_x1) * (cp_y2 - cp_y1) ) ) ) ]
RECO_list = [ ] #list( np.nan * flat_dm_cmd ) ]
CMD_list = [ list( flat_dm_cmd ) ]
CMDERR_list = [ list( np.zeros(len(flat_dm_cmd ))) ]
ERR_list = [ np.zeros(len(flat_dm_cmd )) ]# list( np.nan * np.zeros( int( (cp_x2 - cp_x1) * (cp_y2 - cp_y1) ) ) ) ]  # length depends on cropped pupil when flattened
RMS_list = [np.std( disturbance_cmd )] # to hold std( cmd - aber ) for each iteration
 
dm.send_data( flat_dm_cmd + disturbance_cmd )
time.sleep(1)
FliSdk_V2.Start(camera)    
time.sleep(1)
for i in range(100):

    # get new image and store it (save pupil and psf differently)
    IMG_list.append( list( np.median( bdf.get_raw_images(camera, number_of_frames=number_images_recorded_per_cmd, cropping_corners=cropping_corners) , axis=0)  ) )

    # create new error vector (remember to crop it!) with bdf.get_error_signal
    delta = bdf.get_error_signal( np.array(IMG_list[-1]), reference_pupil_fits = ref_pupils, reduction_dict=None, pupil_indicies = [cp_x1,cp_x2,cp_y1,cp_y2] ) # Note we can use recon data as long as reference pupils have FPM_ON and FPM_OFF extension names uniquely

    DELTA_list.append( delta.reshape(-1) )
    # CHECKS np.array(ERR_list[0]).shape = np.array(ERR_list[1]).shape = (cp_x2 - cp_x1) * (cp_y2 - cp_y1)

    # reconstruct phase
    #reco_modal_amps = CM.T @ RES_list[-1]  # CM.T @ (  1/Nph_obj * (sig_turb.signal - Nph_obj/Nph_cal * sig_cal_on.signal) ).reshape(-1)
    reco = list( CM.T @ DELTA_list[-1] )
    #reco_shift = reco[1:] + [np.median(reco)] # DONT KNOW WHY WE NEED THIS SHIFT!!!! ???
    #RECO_list.append( list( CM.T @ RES_list[-1] ) ) # reconstructed modal amplitudes
    RECO_list.append( reco )
   
    # to get error signal we apply modal gains
    ERR_list.append( list( np.sum( np.array([ g * a * B for g,a,B in  zip(modal_gains, RECO_list[-1], modal_basis)]) , axis=0) ) )

   
    cmderr =  A0 * np.array(ERR_list[-1]) + A1 * np.array(ERR_list[-2]) + A2 * np.array(ERR_list[-2])

    cmderr -= np.mean(cmderr) # REMOVE PISTON FORCEFULLY
   
    CMDERR_list.append( cmderr )
   
    CMD_list.append( CMD_list[-1] - cmderr ) # update the previous command with our cmd error
   
    dm.send_data( CMD_list[-1] + disturbance_cmd )
   
    RMS_list.append( np.std( np.array(CMD_list[-1]) - flat_dm_cmd + np.array(disturbance_cmd) ) )
   
    time.sleep(0.01)

dm.send_data(flat_dm_cmd)


#RMSE
plt.figure()
plt.plot( interp_deflection_4x4act( RMS_list ),'.' )
plt.ylabel('RMSE cmd space [nm RMS]')


# first 5 iterations 
iterations2plot = [0,1,2,3,4,5]
fig, ax = plt.subplots( len(iterations2plot), 5 ,figsize=(10,15))
ax[0,1].set_title('disturbance',fontsize=15)
ax[0,0].set_title('ZWFS image',fontsize=15)
ax[0,2].set_title('CMD error (feedback)',fontsize=15)
ax[0,3].set_title('DM CMD (feedback)',fontsize=15)
ax[0,4].set_title('RESIDUAL (feedback)',fontsize=15)
for i,idx in enumerate(iterations2plot):
    ax[i,0].imshow( np.array(IMG_list)[idx][cp_x1:cp_x2,cp_y1:cp_y2] ) 
    im1 = ax[i,1].imshow( bdf.get_DM_command_in_2D(disturbance_cmd) )
    plt.colorbar(im1, ax= ax[i,1])
    im2 = ax[i,2].imshow( bdf.get_DM_command_in_2D(CMDERR_list[idx]) )
    plt.colorbar(im2, ax= ax[i,2])
    im3 = ax[i,3].imshow( bdf.get_DM_command_in_2D(CMD_list[idx] ) )
    plt.colorbar(im3, ax= ax[i,3])
    im4 = ax[i,4].imshow( bdf.get_DM_command_in_2D(CMD_list[idx] + disturbance_cmd - flat_dm_cmd) )
    plt.colorbar(im4, ax= ax[i,4])

plt.show() 

# compare cmd err to measured CM noise floor
plt.figure() 
plt.ylabel('abs cmd err')
for i ,idx in enumerate(iterations2plot):
    plt.plot( abs(np.array(CMDERR_list[idx])) , alpha=0.5,color='r')
plt.axhline( CM_noise , color='k',label='CM noise floor')
plt.show()

# after N iterations where are the residuals occuring  
plt.figure()
plt.title('residuals on DM space')
plt.imshow( np.rot90( bdf.get_DM_command_in_2D(CMD_list[-3] - flat_dm_cmd + disturbance_cmd).T,2) )
plt.colorbar()
#plt.savefig(fig_path + f'static_aberration_{tstamp}.png') 
#plt.savefig(fig_path + f'dynamic_aberration_{tstamp}.png') 

# look at dist
i=0
fig,ax = plt.subplots(1,3,figsize=(15,5))

ax[0].set_title(f'disturbance {i}')
im0 = ax[0].imshow(np.rot90( bdf.get_DM_command_in_2D(disturbance_cmd).T,2))
plt.colorbar(im0,ax=ax[0])

ax[1].set_title(f'cmd err {i}')
im1 = ax[1].imshow(np.rot90( bdf.get_DM_command_in_2D(CMD_list[i]-flat_dm_cmd).T,2))
plt.colorbar(im1,ax=ax[1])

ax[2].set_title(f'residual cmd {i}')
im2 = ax[2].imshow(np.rot90( bdf.get_DM_command_in_2D(CMD_list[i]-flat_dm_cmd+disturbance_cmd).T,2))
plt.colorbar(im2,ax=ax[2])
plt.show()





# saving 
camera_info_dict = bdf.get_camera_info( camera )


IMG_fits = fits.PrimaryHDU( IMG_list )
IMG_fits.header.set('EXTNAME','IMAGES')
IMG_fits.header.set('recon_fname',recon_file.split('/')[-1])
for k,v in camera_info_dict.items(): 
    IMG_fits.header.set(k,v)   # add in some fits headers about the camera 

for i,n in zip([ci_x1,ci_x2,ci_y1,ci_y2],['ci_x1','ci_x2','ci_y1','ci_y2']):
    IMG_fits.header.set(n,i)

for i,n in zip([cp_x1,cp_x2,cp_y1,cp_y2],['cp_x1','cp_x2','cp_y1','cp_y2']):
    IMG_fits.header.set(n,i)

CMD_fits = fits.PrimaryHDU( CMD_list )
CMD_fits.header.set('EXTNAME','CMDS')
CMD_fits.header.set('WHAT_IS','DM commands')

RES_fits = fits.PrimaryHDU( RES_list )
RES_fits.header.set('EXTNAME','RES')
RES_fits.header.set('WHAT_IS','(I_t - I_CAL_FPM_ON) / I_CAL_FPM_OFF')

RECO_fits = fits.PrimaryHDU( RECO_list )
RECO_fits.header.set('EXTNAME','RECOS')
RECO_fits.header.set('WHAT_IS','CM @ ERR')

ERR_fits = fits.PrimaryHDU( ERR_list )
ERR_fits.header.set('EXTNAME','ERRS')
ERR_fits.header.set('WHAT_IS','list of modal errors to feed to PID')

RMS_fits = fits.PrimaryHDU( RMS_list )
RMS_fits.header.set('EXTNAME','RMS')
RMS_fits.header.set('WHAT_IS','std( cmd - aber_in_cmd_space )')

# add these all as fits extensions 
for f in [disturbfits, IMfits, CMfits, IMG_fits, RES_fits, RECO_fits, ERR_fits, CMD_fits, RMS_fits ]: #[Ufits, Sfits, Vtfits, CMfits, disturbfits, IMG_fits, ERR_fits, RECO_fits, CMD_fits, RMS_fits ]:
    static_ab_performance_fits.append( f ) 

#save data! 
#save_fits = data_path + f'A_FIRST_closed_loop_on_static_aberration_PID-{PID}_t-{tstamp}.fits'
#static_ab_performance_fits.writeto( save_fits )


data = static_ab_performance_fits




