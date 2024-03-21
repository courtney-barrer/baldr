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

available_recon_pupil_files = glob.glob( data_path+'PUPIL_CALIBRATION_REFERENCE*.fits' )
available_recon_pupil_files.sort(key=os.path.getmtime)
print('\n======\navailable AO reconstruction fits files:\n')
for f in available_recon_pupil_files:
    print( f ,'\n') 

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

number_images_recorded_per_cmd = 5 # how many images do we take before taking median for signal processing 

modal_basis = np.eye(140)
IM_pokeamp = -0.03 # normalized DM units
# create IM
IM = []
FliSdk_V2.Start(camera)    
time.sleep(1)
for i in range(len(modal_basis)):
    print(f'executing cmd {i}/{len(modal_basis)}')
    cmdtmp = np.zeros(140)
    cmdtmp[i] = IM_pokeamp
    dm.send_data( flat_dm_cmd + cmdtmp )
    time.sleep(0.05)
    im = np.median( bdf.get_raw_images(camera, number_of_frames=number_images_recorded_per_cmd, cropping_corners=cropping_corners) , axis=0)
    errsig =  bdf.get_error_signal( im, reference_pupil_fits = ref_pupils, reduction_dict=None, pupil_indicies = [cp_x1,cp_x2,cp_y1,cp_y2] )

    IM.append( list(errsig.reshape(-1)) )

IM=np.array(IM)
FliSdk_V2.Stop(camera)  
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
   
if plot_all: #plotting DM eigenmodes to see which to filter 
    fig,ax = plt.subplots(11,11,figsize=(15,15))
    for i,axx in enumerate( ax.reshape(-1) ) :
        axx.set_title(f'eigenmode {i}')
        axx.imshow( bdf.get_DM_command_in_2D( U[:,i] ) )
    plt.suptitle(f"rec. cutoff at i={np.min( np.where( abs(np.diff(S)) < 1e-2 )[0])}", fontsize=14)
    plt.show()

#S_filt = S > S[ np.min( np.where( abs(np.diff(S)) < 1e-2 )[0] ) ]
S_filt = S > S[30] #S[ 20 ]
#S_filt[0] = False # FILTER FIRST MODE !!!!
Sigma = np.zeros( np.array(IM).shape, float)
np.fill_diagonal(Sigma, 1/IM_pokeamp * S[S_filt], wrap=False) #

CM = np.linalg.pinv( U @ Sigma @ Vt ) # C = A @ M

print( f'CM condition = {np.linalg.cond(CM)}' )

# ====== zonal gains

noise_level_IM = np.mean(IM)+5*np.std(IM)  #0.1

zonal_gains = np.sum( abs(IM) > noise_level_IM, axis=1)/np.max( np.sum( abs(IM) > noise_level_IM, axis=1))
zonal_gains **= 1 # reduce curvature

cmd_region_filt = zonal_gains > 0.3  # to filter where modal gains are non-zero (i.e. we can actuate here)

if plot_all: 
    plt.figure()
    plt.title('modal gains? np.sum( abs(IM) > IM_noise, axis=1)')
    plt.imshow( bdf.get_DM_command_in_2D( zonal_gains ) )
    plt.colorbar()

    plt.figure()
    plt.title('cmd_region_filt')
    plt.imshow( bdf.get_DM_command_in_2D( cmd_region_filt ) )
    plt.colorbar()
    plt.show() 

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



# ======= init disturbance

#kernel = np.array([[-1., -1., -1.],[-1.,  8., -1.],[-1., -1., -1.]])

scrn_scaling_factor = 0.23
# --- create infinite phasescreen from aotools module 
Nx_act = dm.num_actuators_width()
screen_pixels = Nx_act*2**5 # some multiple of numer of actuators across DM 
D = 1.8 #m effective diameter of the telescope
scrn = aotools.infinitephasescreen.PhaseScreenVonKarman(nx_size=screen_pixels, pixel_scale=D/screen_pixels,r0=0.1,L0=12)

corner_indicies = [0, Nx_act-1, Nx_act * (Nx_act-1), -1] # Beware -1 index doesn't work if inserting in list! This is  ok for for use with create_phase_screen_cmd_for_DM function.

disturbance_cmd = cmd_region_filt * bdf.create_phase_screen_cmd_for_DM(scrn=scrn, DM=dm, flat_reference=flat_dm_cmd, scaling_factor = scrn_scaling_factor, drop_indicies = corner_indicies, plot_cmd=False)  # normalized flat_dm +- scaling_factor?
 
disturbance_cmd -= cmd_region_filt *np.mean( disturbance_cmd ) # no piston!! 
#disturbance_cmd -= cmd_region_filt * 0.8*np.min( disturbance_cmd ) # can be used to bias it negatively where we have better linearity in our WFS, or higher non-linearity, whatever you want to show. 

#HPF_dist = ndimage.convolve(bdf.get_DM_command_in_2D(disturbance_cmd), kernel).reshape(-1)
#disturbance_cmd_HPF = HPF_dist[np.isfinite( HPF_dist ) ]

rows_to_jump = 1 # how many rows to jump on initial phase screen for each Baldr loop

distance_per_correction = rows_to_jump * D/screen_pixels # effective distance travelled by turbulence per AO iteration 
print(f'{rows_to_jump} rows jumped per AO command in initial phase screen of {screen_pixels} pixels. for {D}m mirror this corresponds to a distance_per_correction = {distance_per_correction}m')



if plot_all:
    # for visualization get the 2D grid of the disturbance on DM  
    plt.figure()
    plt.imshow( bdf.get_DM_command_in_2D(disturbance_cmd, Nx_act=12) )
    plt.colorbar()
    plt.title( f'initial Kolmogorov aberration to apply to DM (rms = {round(np.std(disturbance_cmd),3)})')
    plt.show()


# ====== PID

dt = 1/fps
Ku = 1.9 #1.5 # ultimate gain to see stable oscillation in output 
Tu = 3 * dt # period of oscillations 
#apply Ziegler-Nicols methold of PI controller https://en.wikipedia.org/wiki/Ziegler%E2%80%93Nichols_method
#Kp, Ki, Kd = 0.45*Ku * zonal_gains, 0.54*Ku/Tu * zonal_gains, 0.* np.ones(len(flat_dm_cmd))
Kp, Ki, Kd = 0.45*Ku * cmd_region_filt , 0.54*Ku/Tu *cmd_region_filt, 0.* np.ones(len(flat_dm_cmd))


M2C = np.eye(len(flat_dm_cmd)) # mode to DM cmd matrix. For now we just consider poke modes so this matrix is the identity. But later we can consider Zernike or KL modes etc 


# ======= Close loop
# init lists to hold data from control loop
DIST_list = [ disturbance_cmd ]
IMG_list = [ ]
DELTA_list = [ ] #list( np.nan * np.zeros( int( (cp_x2 - cp_x1) * (cp_y2 - cp_y1) ) ) ) ]
MODE_ERR_list = [ ] #list( np.nan * flat_dm_cmd ) ]
MODE_ERR_PID_list = [ ]
CMD_ERR_list = [ list( np.zeros(len(flat_dm_cmd ))) ]
CMD_list = [ list( flat_dm_cmd ) ]
ERR_list = [ np.zeros(len(flat_dm_cmd )) ]# list( np.nan * np.zeros( int( (cp_x2 - cp_x1) * (cp_y2 - cp_y1) ) ) ) ]  # length depends on cropped pupil when flattened
RMS_list = [np.std( cmd_region_filt * disturbance_cmd )] # to hold std( cmd - aber ) for each iteration
 
dm.send_data( flat_dm_cmd + disturbance_cmd )
time.sleep(1)
FliSdk_V2.Start(camera)    
time.sleep(1)
for i in range(500):

    # get new image and store it (save pupil and psf differently)
    IMG_list.append( list( np.median( bdf.get_raw_images(camera, number_of_frames=number_images_recorded_per_cmd, cropping_corners=cropping_corners) , axis=0)  ) )

    # create new error vector (remember to crop it!) with bdf.get_error_signal
    delta = bdf.get_error_signal( np.array(IMG_list[-1]), reference_pupil_fits = ref_pupils, reduction_dict=None, pupil_indicies = [cp_x1,cp_x2,cp_y1,cp_y2] ) # Note we can use recon data as long as reference pupils have FPM_ON and FPM_OFF extension names uniquely

    DELTA_list.append( delta.reshape(-1) )
    # CHECKS np.array(ERR_list[0]).shape = np.array(ERR_list[1]).shape = (cp_x2 - cp_x1) * (cp_y2 - cp_y1)

    mode_errs = list( CM.T @ DELTA_list[-1] )
    #reco_shift = reco[1:] + [np.median(reco)] # DONT KNOW WHY WE NEED THIS SHIFT!!!! ???
    #RECO_list.append( list( CM.T @ RES_list[-1] ) ) # reconstructed modal amplitudes
    MODE_ERR_list.append( mode_errs )
    
    # apply our PID on the modal basis 
    u =  Kp * np.array(MODE_ERR_list[-1]) +  Ki * dt * np.sum( MODE_ERR_list,axis=0 ) # PID 
   
    MODE_ERR_PID_list.append( u )   
    # to get error signal we apply modal gains
    cmd_errs =  M2C @ MODE_ERR_PID_list[-1] #np.sum( np.array([ a * B for a,B in  zip( MODE_ERR_list[-1], modal_basis)]) , axis=0) # this could be matrix multiplication too
    cmd_errs -= np.mean(cmd_errs) # REMOVE PISTON FORCEFULLY
    CMD_ERR_list.append( list(cmd_errs) )

    #list( np.sum( np.array([ Kp * a * B + Ki * dt * sum(err, axis) for g,a,B in  zip(modal_gains, RECO_list[-1], modal_basis)]) , axis=0) 

    CMD_list.append( flat_dm_cmd - CMD_ERR_list[-1] ) # update the previous command with our cmd error

    # roll phase screen 
    
    for skip in range(rows_to_jump):
        scrn.add_row() 
    # get our new Kolmogorov disturbance command 
    disturbance_cmd = cmd_region_filt * bdf.create_phase_screen_cmd_for_DM(scrn=scrn, DM=dm, flat_reference=flat_dm_cmd, scaling_factor = scrn_scaling_factor, drop_indicies = corner_indicies, plot_cmd=False)
    
    disturbance_cmd -= cmd_region_filt * np.mean( disturbance_cmd )
    #disturbance_cmd -= cmd_region_filt * 0.8*np.min( disturbance_cmd ) can be used to bias into non-linear/linear regime

    DIST_list.append( disturbance_cmd )
    # we only calculate rms in our cmd region
    RMS_list.append( np.std( cmd_region_filt * ( np.array(CMD_list[-1]) - flat_dm_cmd + np.array(disturbance_cmd) ) ) )
   
    dm.send_data( CMD_list[-1] + disturbance_cmd )

    time.sleep(0.01)

dm.send_data(flat_dm_cmd)


#RMSE
plt.figure()
plt.plot( 2 * interp_deflection_4x4act( [np.std(d) for d in DIST_list]) ,'.',label='disturbance' ,alpha=0.5)
plt.plot( 2 * interp_deflection_4x4act( RMS_list ),'.',label='residual',alpha=0.5 )

plt.ylabel('RMSE wave space [nm RMS]')
plt.xlabel('iteration')
plt.legend()
#plt.savefig(data_path + f'A_FIRST_closed_loop_on_dynamic_aberration_t-{tstamp}.png')
#plt.xlim([0,50]);plt.ylim([24,26]);plt.show()
print( 'final RMSE =', interp_deflection_4x4act( RMS_list[-1] ) )
print( 'RMSE after 100 iter. =',interp_deflection_4x4act( RMS_list[100] ) );

# first 5 iterations 
iterations2plot = [0,1,2,3,4,5,-2,-1]
fig, ax = plt.subplots( len(iterations2plot), 5 ,figsize=(10,15))
ax[0,1].set_title('disturbance',fontsize=15)
ax[0,0].set_title('ZWFS image',fontsize=15)
ax[0,2].set_title('CMD error (feedback)',fontsize=15)
ax[0,3].set_title('DM CMD (feedback)',fontsize=15)
ax[0,4].set_title('RESIDUAL (feedback)',fontsize=15)
for i,idx in enumerate(iterations2plot):
    ax[i,0].imshow( np.array(IMG_list)[idx][cp_x1:cp_x2,cp_y1:cp_y2] ) 
    im1 = ax[i,1].imshow( bdf.get_DM_command_in_2D(DIST_list[idx]) )
    plt.colorbar(im1, ax= ax[i,1])
    im2 = ax[i,2].imshow( bdf.get_DM_command_in_2D(CMD_ERR_list[idx]) )
    plt.colorbar(im2, ax= ax[i,2])
    im3 = ax[i,3].imshow( bdf.get_DM_command_in_2D(CMD_list[idx] ) )
    plt.colorbar(im3, ax= ax[i,3])
    im4 = ax[i,4].imshow( bdf.get_DM_command_in_2D(np.array(CMD_list[idx]) + np.array(DIST_list[idx]) - flat_dm_cmd) )
    plt.colorbar(im4, ax= ax[i,4])

plt.show() 

fig,ax = plt.subplots( 1,2); ax[0].imshow(IMG_list[0]);
ax[0].set_title('initial ZWFS pupil');
ax[1].imshow(IMG_list[-1]);
ax[1].set_title('final ZWFS pupil'); 
#plt.savefig(data_path + f'A_FIRST_closed_loop_on_dynamic_aberration_pupils_before-after_t-{tstamp}.png')


# compare cmd err to measured CM noise floor
plt.figure() 
plt.ylabel('abs cmd err')
for i ,idx in enumerate(iterations2plot):
    plt.plot( abs(np.array(CMD_ERR_list[idx])) , alpha=0.5,color='r')
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
static_ab_performance_fits = fits.HDUList( [] )

camera_info_dict = bdf.get_camera_info( camera )

IMG_fits = fits.PrimaryHDU( IMG_list )
IMG_fits.header.set('EXTNAME','IMAGES')
#IMG_fits.header.set('recon_fname',recon_file.split('/')[-1])
for k,v in camera_info_dict.items(): 
    IMG_fits.header.set(k,v)   # add in some fits headers about the camera 

for i,n in zip([ci_x1,ci_x2,ci_y1,ci_y2],['ci_x1','ci_x2','ci_y1','ci_y2']):
    IMG_fits.header.set(n,i)

for i,n in zip([cp_x1,cp_x2,cp_y1,cp_y2],['cp_x1','cp_x2','cp_y1','cp_y2']):
    IMG_fits.header.set(n,i)

disturbfits = fits.PrimaryHDU( DIST_list )
disturbfits.header.set('EXTNAME','DIST')
disturbfits.header.set('WHAT_IS','disturbance DM command')

CM_fits = fits.PrimaryHDU( CM )
CM_fits.header.set('EXTNAME','CM')
CM_fits.header.set('WHAT_IS','control matrix (filtered)')

IM_fits = fits.PrimaryHDU( CM )
IM_fits.header.set('EXTNAME','IM')
IM_fits.header.set('WHAT_IS','unfiltered interaction matrix')

IM_fits = fits.PrimaryHDU( [list(Kp), list(Ki), list(Kd)] )
IM_fits.header.set('EXTNAME','PID_GAINS')
IM_fits.header.set('WHAT_IS','Kp,Ki,Kd (columns) for each mode (rows)')

RES_fits = fits.PrimaryHDU( DELTA_list )
RES_fits.header.set('EXTNAME','RES')
RES_fits.header.set('WHAT_IS','(I_t - I_CAL_FPM_ON) / I_CAL_FPM_OFF')

MODE_ERR_fits = fits.PrimaryHDU( MODE_ERR_list )
MODE_ERR_fits.header.set('EXTNAME','MODE_ERR')
MODE_ERR_fits.header.set('WHAT_IS','CM @ ERR')

MODE_ERRPID_fits = fits.PrimaryHDU( MODE_ERR_PID_list )
MODE_ERRPID_fits.header.set('EXTNAME','MODE_ERR_PID')
MODE_ERRPID_fits.header.set('WHAT_IS','PID applied to modal errors')

CMDERR_fits = fits.PrimaryHDU( CMD_ERR_list )
CMDERR_fits.header.set('EXTNAME','CMD_ERR')
CMDERR_fits.header.set('WHAT_IS','list of cmd errors')

CMD_fits = fits.PrimaryHDU( CMD_list )
CMD_fits.header.set('EXTNAME','CMDS')
CMD_fits.header.set('WHAT_IS','DM commands')

RMS_fits = fits.PrimaryHDU( RMS_list )
RMS_fits.header.set('EXTNAME','RMS')
RMS_fits.header.set('WHAT_IS','std( cmd - aber_in_cmd_space )')

# add these all as fits extensions 
for f in [disturbfits, IM_fits, CM_fits, IMG_fits, RES_fits, MODE_ERR_fits, MODE_ERRPID_fits,CMDERR_fits, CMD_fits, RMS_fits ]: #[Ufits, Sfits, Vtfits, CMfits, disturbfits, IMG_fits, ERR_fits, RECO_fits, CMD_fits, RMS_fits ]:
    static_ab_performance_fits.append( f ) 

#save data! 
save_fits = data_path + f'closed_loop_on_dynamic_aberration_EXAMPLE_DYNAMIC_RANGE_t-{tstamp}.fits'
static_ab_performance_fits.writeto( save_fits )





