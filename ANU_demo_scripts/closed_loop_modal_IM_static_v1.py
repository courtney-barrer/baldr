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
import pyzelda.utils.zernike as zernike

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

def shift(xs, n, m, fill_value=np.nan):
    # shifts a 2D array xs by n rows, m columns and fills the new region with fill_value

    e = xs.copy()
    if n!=0:
        if n >= 0:
            e[:n,:] = fill_value
            e[n:,:] =  e[:-n,:]
        else:
            e[n:,:] = fill_value
            e[:n,:] =  e[-n:,:]
   
       
    if m!=0:
        if m >= 0:
            e[:,:m] = fill_value
            e[:,m:] =  e[:,:-m]
        else:
            e[:,m:] = fill_value
            e[:,:m] =  e[:,-m:]
    return e

def construct_command_basis( basis='Zernike', number_of_modes = 20, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True):
    """
    returns a change of basis matrix M2C to go from modes to DM commands, where columns are the DM command for a given modal basis. e.g. M2C @ [0,1,0,...] would return the DM command for tip on a Zernike basis. Modes are normalized on command space such that <M>=0, <M|M>=1. Therefore these should be added to a flat DM reference if being applied.    

    basis = string of basis to use
    number_of_modes = int, number of modes to create
    Nx_act_DM = int, number of actuators across DM diameter
    Nx_act_basis = int, number of actuators across the active basis diameter
    act_offset = tuple, (actuator row offset, actuator column offset) to offset the basis on DM (i.e. we can have a non-centered basis)
    IM_covariance = None or an interaction matrix from command to measurement space. This only needs to be provided if you want KL modes, for this the number of modes is infered by the shape of the IM matrix. 
     
    """

   
    # shorter notations
    #Nx_act = DM.num_actuators_width() # number of actuators across diameter of DM.
    #Nx_act_basis = actuators_across_diam
    c = act_offset
    # DM BMC-3.5 is 12x12 missing corners so 140 actuators , we note down corner indicies of flattened 12x12 array.
    corner_indices = [0, Nx_act_DM-1, Nx_act_DM * (Nx_act_DM-1), -1]

    bmcdm_basis_list = []
    # to deal with
    if basis == 'Zernike':
        if without_piston:
            number_of_modes += 1 # we add one more mode since we dont include piston 

        raw_basis = zernike.zernike_basis(nterms=number_of_modes, npix=Nx_act_basis )
        for i,B in enumerate(raw_basis):
            # normalize <B|B>=1, <B>=0 (so it is an offset from flat DM shape)
            Bnorm = np.sqrt( 1/np.nansum( B**2 ) ) * B
            # pad with zeros to fit DM square shape and shift pixels as required to center
            # we also shift the basis center with respect to DM if required
            if np.mod( Nx_act_basis, 2) == 0:
                pad_width = (Nx_act_DM - B.shape[0] )//2
                padded_B = shift( np.pad( Bnorm , pad_width , constant_values=(np.nan,)) , c[0], c[1])
            else:
                pad_width = (Nx_act_DM - B.shape[0] )//2 + 1
                padded_B = shift( np.pad( Bnorm , pad_width , constant_values=(np.nan,)) , c[0], c[1])[:-1,:-1]  # we take off end due to odd numebr

            flat_B = padded_B.reshape(-1) # flatten basis so we can put it in the accepted DM command format
            np.nan_to_num(flat_B,0 ) # convert nan -> 0
            flat_B[corner_indices] = np.nan # convert DM corners to nan (so lenght flat_B = 140 which corresponds to BMC-3.5 DM)

            # now append our basis function removing corners (nan values)
            bmcdm_basis_list.append( flat_B[np.isfinite(flat_B)] )

        # our mode 2 command matrix
        if without_piston:
            M2C = np.array( bmcdm_basis_list )[1:].T #remove piston mode
        else:
            M2C = np.array( bmcdm_basis_list ).T # take transpose to make columns the modes in command space.


    elif basis == 'KL':         
        if without_piston:
            number_of_modes += 1 # we add one more mode since we dont include piston 

        raw_basis = zernike.zernike_basis(nterms=number_of_modes, npix=Nx_act_basis )
        b0 = np.array( [np.nan_to_num(b) for b in raw_basis] )
        cov0 = np.cov( b0.reshape(len(b0),-1) )
        U , S, UT = np.linalg.svd( cov0 )
        KL_raw_basis = ( b0.T @ U ).T # KL modes that diagonalize Zernike covariance matrix 
        for i,B in enumerate(KL_raw_basis):
            # normalize <B|B>=1, <B>=0 (so it is an offset from flat DM shape)
            Bnorm = np.sqrt( 1/np.nansum( B**2 ) ) * B
            # pad with zeros to fit DM square shape and shift pixels as required to center
            # we also shift the basis center with respect to DM if required
            if np.mod( Nx_act_basis, 2) == 0:
                pad_width = (Nx_act_DM - B.shape[0] )//2
                padded_B = shift( np.pad( Bnorm , pad_width , constant_values=(np.nan,)) , c[0], c[1])
            else:
                pad_width = (Nx_act_DM - B.shape[0] )//2 + 1
                padded_B = shift( np.pad( Bnorm , pad_width , constant_values=(np.nan,)) , c[0], c[1])[:-1,:-1]  # we take off end due to odd numebr

            flat_B = padded_B.reshape(-1) # flatten basis so we can put it in the accepted DM command format
            np.nan_to_num(flat_B,0 ) # convert nan -> 0
            flat_B[corner_indices] = np.nan # convert DM corners to nan (so lenght flat_B = 140 which corresponds to BMC-3.5 DM)

            # now append our basis function removing corners (nan values)
            bmcdm_basis_list.append( flat_B[np.isfinite(flat_B)] )

        # our mode 2 command matrix
        if without_piston:
            M2C = np.array( bmcdm_basis_list )[1:].T #remove piston mode
        else:
            M2C = np.array( bmcdm_basis_list ).T # take transpose to make columns the modes in command space.

    elif basis == 'Zonal': 
        M2C = np.eye(Nx_act_DM) # we just consider this over all actuators (so defaults to 140 modes) 
        # we filter zonal basis in the eigenvectors of the control matrix. 
 
    #elif basis == 'Sensor_Eigenmodes': this is done specifically in a phase_control.py function - as it needs a interaction matrix covariance first 

    return(M2C)


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

number_images_recorded_per_cmd = 5 # how many images do we take before taking median for signal processing 

# make modal basis 
M2C = construct_command_basis( basis='Zernike', number_of_modes = 20, Nx_act_DM = 12, Nx_act_basis = 8, act_offset= (0,0))

M2C_nopiston = M2C #[:,1:]

modal_basis = M2C_nopiston.T # rows are modes in command basis 

IM_pokeamp = -0.15 # to modes which are normalized <m|m>=1 
# create IM
IM = []
for i,m in enumerate(modal_basis):
    print(f'executing cmd {i}/{len(modal_basis)}')
    dm.send_data( flat_dm_cmd + IM_pokeamp * m )
    time.sleep(0.05)
    im = np.median( bdf.get_raw_images(camera, number_of_frames=number_images_recorded_per_cmd, cropping_corners=cropping_corners) , axis=0)
    errsig =  bdf.get_error_signal( im, reference_pupil_fits = ref_pupils, reduction_dict=None, pupil_indicies = [cp_x1,cp_x2,cp_y1,cp_y2] )

    IM.append( list(errsig.reshape(-1)) )

IM = np.array(IM)
# singular value decomposition of interaction matrix
U,S,Vt = np.linalg.svd( IM , full_matrices=True)

if plot_all:
    plt.figure()
    plt.plot( S )
    #plt.axvline( len(S) * np.pi*2**2/(4.4)**2 ,linestyle=':',color='k',label=r'$D_{DM}^2/\pi r_{pup}^2$')
    plt.ylabel('singular values',fontsize=15)
    plt.xlabel('eigenvector index',fontsize=15)
    plt.legend(fontsize=15)
    plt.gca().tick_params( labelsize=15 )
   
S_filt = S > 0 # S > S[ np.min( np.where( abs(np.diff(S)) < 1e-2 )[0] ) ]
Sigma = np.zeros( np.array(IM).shape, float)
np.fill_diagonal(Sigma, 1/IM_pokeamp * S[S_filt], wrap=False) #

CM = np.linalg.pinv( U @ Sigma @ Vt ) # C = A @ M

print( f'CM condition = {np.linalg.cond(CM)}' )


cmd_region_filt = M2C[:,0] > 0 #   cmd region is where we define our basis 

# ======= init disturbance

dist_modes_idx = np.array([1,3,6]) 
dist_modes = np.zeros( M2C_nopiston.shape[1] ) 
for i in dist_modes_idx:
    dist_modes[i] = np.random.randn()*0.2

disturbance_cmd = M2C_nopiston @ dist_modes
"""
scrn_scaling_factor = 0.2
# --- create infinite phasescreen from aotools module 
Nx_act = dm.num_actuators_width()
screen_pixels = Nx_act*2**5 # some multiple of numer of actuators across DM 
D = 1.8 #m effective diameter of the telescope
scrn = aotools.infinitephasescreen.PhaseScreenVonKarman(nx_size=screen_pixels, pixel_scale=D/screen_pixels,r0=0.1,L0=12)

corner_indicies = [0, Nx_act-1, Nx_act * (Nx_act-1), -1] # Beware -1 index doesn't work if inserting in list! This is  ok for for use with create_phase_screen_cmd_for_DM function.

disturbance_cmd = cmd_region_filt * bdf.create_phase_screen_cmd_for_DM(scrn=scrn, DM=dm, flat_reference=flat_dm_cmd, scaling_factor = scrn_scaling_factor, drop_indicies = corner_indicies, plot_cmd=False)  # normalized flat_dm +- scaling_factor?

disturbance_cmd -= np.mean( disturbance_cmd ) # no piston!! 
"""

if plot_all:
    # for visualization get the 2D grid of the disturbance on DM  
    plt.figure()
    plt.imshow( bdf.get_DM_command_in_2D(disturbance_cmd, Nx_act=12) )
    plt.colorbar()
    plt.title( f'initial Kolmogorov aberration to apply to DM (rms = {round(np.std(disturbance_cmd),3)})')
    plt.show()


# ====== PID

dt = 1/fps
Ku = 0.6 # ultimate gain to see stable oscillation in output 
Tu = 8*dt # period of oscillations 
#apply Ziegler-Nicols methold of PI controller https://en.wikipedia.org/wiki/Ziegler%E2%80%93Nichols_method
Kp, Ki, Kd = 0.45 * Ku , 0.54*Ku/Tu , 0.* np.ones(len(flat_dm_cmd))
# NOTE: for Ki we scale zonal_gains**3 to avoid run-away amplitification of zonal modes on edge of illuminated DM 


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
for i in range(50):

    # get new image and store it (save pupil and psf differently)
    IMG_list.append( list( np.median( bdf.get_raw_images(camera, number_of_frames=number_images_recorded_per_cmd, cropping_corners=cropping_corners) , axis=0)  ) )

    # create new error vector (remember to crop it!) with bdf.get_error_signal
    delta = bdf.get_error_signal( np.array(IMG_list[-1]), reference_pupil_fits = ref_pupils, reduction_dict=None, pupil_indicies = [cp_x1,cp_x2,cp_y1,cp_y2] ) # Note we can use recon data as long as reference pupils have FPM_ON and FPM_OFF extension names uniquely

    DELTA_list.append( delta.reshape(-1) )
    # CHECKS np.array(ERR_list[0]).shape = np.array(ERR_list[1]).shape = (cp_x2 - cp_x1) * (cp_y2 - cp_y1)

    mode_errs = list( DELTA_list[-1] @ CM )
    #reco_shift = reco[1:] + [np.median(reco)] # DONT KNOW WHY WE NEED THIS SHIFT!!!! ???
    #RECO_list.append( list( CM.T @ RES_list[-1] ) ) # reconstructed modal amplitudes
    MODE_ERR_list.append( mode_errs )
    
    # apply our PID on the modal basis 
    u =  Kp * np.array(MODE_ERR_list[-1]) +  Ki * dt * np.sum( MODE_ERR_list,axis=0 ) # PID 
   
    MODE_ERR_PID_list.append( u )   
    # to get error signal we apply modal gains
    cmd_errs =  M2C_nopiston @ MODE_ERR_PID_list[-1] #np.sum( np.array([ a * B for a,B in  zip( MODE_ERR_list[-1], modal_basis)]) , axis=0) # this could be matrix multiplication too
    cmd_errs -= np.mean(cmd_errs) # REMOVE PISTON FORCEFULLY
    CMD_ERR_list.append( list(cmd_errs) )

    #list( np.sum( np.array([ Kp * a * B + Ki * dt * sum(err, axis) for g,a,B in  zip(modal_gains, RECO_list[-1], modal_basis)]) , axis=0) 

    CMD_list.append( flat_dm_cmd - CMD_ERR_list[-1] ) # update the previous command with our cmd error
    """
    # roll phase screen 
    for skip in range(rows_to_jump):
        scrn.add_row() 
    # get our new Kolmogorov disturbance command 
    disturbance_cmd = bdf.create_phase_screen_cmd_for_DM(scrn=scrn, DM=dm, flat_reference=flat_dm_cmd, scaling_factor = scrn_scaling_factor, drop_indicies = corner_indicies, plot_cmd=False)
    """
    DIST_list.append( disturbance_cmd )
    # we only calculate rms in our cmd region
    RMS_list.append( np.std( cmd_region_filt * ( np.array(CMD_list[-1]) - flat_dm_cmd + np.array(disturbance_cmd) ) ) )
   
    dm.send_data( CMD_list[-1] + disturbance_cmd )

    time.sleep(0.01)

dm.send_data(flat_dm_cmd)


#RMSE
plt.figure()
plt.plot( interp_deflection_4x4act( RMS_list ),'.' )
plt.ylabel('RMSE cmd space [nm RMS]')


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
save_fits = data_path + f'A_FIRST_closed_loop_on_static_aberration_t-{tstamp}.fits'
static_ab_performance_fits.writeto( save_fits )





