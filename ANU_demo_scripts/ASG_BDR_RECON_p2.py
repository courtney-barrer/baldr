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


"""
ASG_BDR_RECON_p1 is the acquisition for the reconstructor, it takes a reference on/off image,
detects pupil and psf regions, then ramps through a sequence of push-pull amplitudes on the DM, recording associated images with the FPM in which is saved in a fits file. 

ASG_BDR_RECON_p2 applies the image processing to this to construct interaction and control matricies
 - reconstruct B_0 and beta_0 d\Delta / d\epsilon 
 - 
-- poking DM in open loop and creating interaction and control matricies
1. read in DM data (flat DM command matrix, deflection data etc) 
2. set up camera and DM
3. take reference pupils with FPM in/out
4. detect and define PUPIL and PSF regions in image, create respective masks 
5. create IM by poking DM,recording images and applying appropiate signal processing
6. SVD analysis of IM display and analyse results
7. create CM by pseudo inverse of IM  
8. Save in fits file
-- 

"""



# read in file 


data = fits.open(data_path+"ramp_poke_DM_ampMax-0.2_Nsamp-22_Nim_p_cmd-2_fps-600.0_imregion-['140', '280', '90', '290']_04-03-2024T11.45.00.fits")

number_amp_samples = int(data[0].header['HIERARCH #ramp steps']) 
amp_max = float(data[0].header['HIERARCH in-poke max amp'])
ramp_values = np.linspace(-amp_max, amp_max, number_amp_samples)

Nmodes_poked = int(data[0].header['HIERARCH Nmodes_poked'])
Nact =  int(data[0].header['HIERARCH Nact'])  



flat_cmd = data[1].data[0]
flat_img = data[0].data[0][0]
ref_pupils = data[2].data[0]

poke_cmds = data[1].data[1:].reshape(number_amp_samples, Nmodes_poked, Nact ) 
poke_imgs = np.array([data[0].data[i][0] for i in range(1,len(data[0].data))]).reshape( number_amp_samples, Nmodes_poked, ref_pupils['FPM_IN'].data.shape[0], ref_pupils['FPM_IN'].data.shape[1] )


plt.plot( [np.max(abs( poke_imgs[i,50] - flat_img)) for i in range(len(poke_imgs))] ,'.'); plt.show() 


plt.plot( [np.max(abs( poke_imgs[len(ramp_values)//2 + i,50] - poke_imgs[len(ramp_values)//2 - i,50])) for i in range(0,len(poke_imgs)//2)] ,'.'); plt.show() 

iregion = np.argmax((abs( poke_imgs[11,50] - flat_img ).reshape(-1) ) )

plt.figure()
plt.plot( [(poke_imgs[len(ramp_values)//2 + i,50] - poke_imgs[len(ramp_values)//2 - i,50]).reshape(-1)[iregion] for i in range(0,len(poke_imgs)//2)] ,'.'); plt.show() 

# (I(\epsilon) + I(-\epsilon) ) / 2 = I0? 
plt.figure() 
plt.imshow(  (poke_imgs[len(ramp_values)//2 + i,50] +  poke_imgs[len(ramp_values)//2 + i,50] ) /2 );plt.show()
# =====(6)
dm.send_data(flat_dm_cmd) 
# for each DM iteration we took various images (defined by "number_images_recorded_per_cmd"). Calculate median for each of these
agregated_pupils = [np.median(raw_IM_data[0].data[i],axis=0) for i in range(len(raw_IM_data[0].data))]
# get a reshaped array of the images from ramping actuators on DM, we drop index 0 since this corresponds to flat DM.
agregated_pupils_array = np.array( agregated_pupils[1:] ).reshape(number_amp_samples, modal_basis.shape[0],ref_pupils['FPM_IN'].data.shape[0], ref_pupils['FPM_IN'].data.shape[1])

# for given poke amp look at SVD and plot detector eigenmodes!

# ================================== This is what determines the poke amplitude for IM matrix
poke_amp_indx = number_amp_samples//2 + 2 # the smallest positive value 
# ==================================
print( f'======\ncalculating IM for pokeamp = {ramp_values[poke_amp_indx]}\n=====:::' )

# dont forget to crop just square pupil region
IM_unfiltered_unflat = [bdf.get_error_signal( agregated_pupils_array[poke_amp_indx][m], reference_pupil_fits=ref_pupils, reduction_dict=None, crop_indicies = [cp_x1,cp_x2,cp_y1,cp_y2]) for m in range(modal_basis.shape[0])] # [mode, x, y]

IM_unfiltered = [list(im.reshape(-1)) for im in IM_unfiltered_unflat]

# SVD
U,S,Vt = np.linalg.svd( IM_unfiltered,  full_matrices = False ) 

if verbose:
    # the eigenmodes in image space 
    fig,ax = plt.subplots(4,4,figsize=(30,30))
    plt.subplots_adjust(hspace=0.1,wspace=0.1)
    for i,axx in enumerate(ax.reshape(-1)):
        axx.imshow( Vt[i].reshape( [int(cp_x2-cp_x1), int(cp_y2-cp_y1)]) )
        axx.set_title(f'mode {i}, S={round(S[i])}')
    #plt.legend(ax=axx)
    plt.tight_layout()
    #plt.savefig( fig_path + f'ramp_poke_DM_n_analyse-SVD_modes_from_poke_IM_amp-{round(ramp_values[poke_amp_indx],3)}_t-{tstamp}.png',dpi=300)
    plt.show() 

    # the eigenmodes in DM space 
    fig,ax = plt.subplots(4,4,figsize=(30,30))
    plt.subplots_adjust(hspace=0.1,wspace=0.1)
    for i,axx in enumerate(ax.reshape(-1)):
        axx.imshow(bdf.get_DM_command_in_2D(U.T[i],Nx_act=12) )
        axx.set_title(f'mode {i}, S={round(S[i])}')
    plt.show() 

# =====(7)
# filter piston in inverse modes
invSfilt =  1/S[:] #[np.min(S)/100] + list( 1/S[1:] )
CM = Vt.T @ (np.diag( invSfilt ) )  @ U.T

# ==== (8) 
poke_amp_nm = interp_deflection_1act(0.5+ramp_values[poke_amp_indx]) - interp_deflection_1act(0.5) # get poke amp in dm that was used to create matricies 

# file name for saving final fits file [keep standard convention]
save_fits = data_path + f'BDR_RECON_amp-{round(poke_amp_nm)}nm_T-{tstamp}.fits'

recon_fits = fits.HDUList([])

camera_info_dict = bdf.get_camera_info(camera)

PRI_fits = fits.PrimaryHDU( [] )
PRI_fits.header.set('EXTNAME','HEAD')
for k,v in camera_info_dict.items(): 
    PRI_fits.header.set(k,v)   # add in some fits headers about the camera 
PRI_fits.header.set('cp_x1',cp_x1) #pupil crop coordinates
PRI_fits.header.set('cp_x2',cp_x2)
PRI_fits.header.set('cp_y1',cp_y1)
PRI_fits.header.set('cp_y2',cp_y2)
PRI_fits.header.set('ci_x1',ci_x1) # psd crop coordinates
PRI_fits.header.set('ci_x2',ci_x2)
PRI_fits.header.set('ci_y1',ci_y1)
PRI_fits.header.set('ci_y2',ci_y2)
recon_fits.append(PRI_fits)

# the reference pupil when phase mask is in beam
recon_fits.append( ref_pupils['FPM_IN'] )

# the reference pupil when phase mask is out of beam
recon_fits.append( ref_pupils['FPM_OUT'] ) 

# modal basis applied 
modalfits = fits.PrimaryHDU( modal_basis ) 
modalfits.header.set('EXTNAME','BASIS')
recon_fits.append(modalfits)

# the images after poking each mode at some amplitude  
poke_im_fits = fits.PrimaryHDU(  agregated_pupils_array[:,:,cp_x1:cp_x2,cp_y1:cp_y2] )
poke_im_fits.header.set('EXTNAME','poke_images')
poke_im_fits.header.set('WHAT_IS','images from poking DM')
poke_im_fits.header.set('dimensions','[poke_val, mode, x,y]')
for i,v in enumerate(ramp_values):
    poke_im_fits.header.set(f'poke_val_{i}',round(v,3))

recon_fits.append( poke_im_fits )

Ufits = fits.PrimaryHDU( U )
Ufits.header.set('EXTNAME','U')
Ufits.header.set('WHAT_IS','U where IM=U@S@Vt')


recon_fits.append( Ufits )

Sfits = fits.PrimaryHDU( S )
Sfits.header.set('EXTNAME','S')
Sfits.header.set('WHAT_IS','S where IM=U@S@Vt')

recon_fits.append( Sfits )

Vtfits = fits.PrimaryHDU( Vt )
Vtfits.header.set('EXTNAME','Vt')
Vtfits.header.set('WHAT_IS','Vt where IM=U@S@Vt')

recon_fits.append( Vtfits )

IMfits =  fits.PrimaryHDU( IM_unfiltered )
IMfits.header.set('EXTNAME','IM')
IMfits.header.set('WHAT IS','unfiltered interaction matrix') 
IMfits.header.set('poke_amp_cmd', round(ramp_values[poke_amp_indx],2) )
IMfits.header.set('poke_amp_nm', round(poke_amp_nm) ) 
IMfits.header.set('cp_x1',cp_x1) #pupil crop coordinates
IMfits.header.set('cp_x2',cp_x2)
IMfits.header.set('cp_y1',cp_y1)
IMfits.header.set('cp_y2',cp_y2)
IMfits.header.set('ci_x1',ci_x1) # psd crop coordinates
IMfits.header.set('ci_x2',ci_x2)
IMfits.header.set('ci_y1',ci_y1)
IMfits.header.set('ci_y2',ci_y2)
recon_fits.append( IMfits )

CMfits =  fits.PrimaryHDU( CM )
CMfits.header.set('EXTNAME','CM')
CMfits.header.set('WHAT IS','filtered control matrix') 
CMfits.header.set('poke_amp_cmd', round(ramp_values[poke_amp_indx],2)) 
CMfits.header.set('poke_amp_nm', round(poke_amp_nm) ) 

recon_fits.append( CMfits )

#WRITE IT
recon_fits.writeto( save_fits )

dm.close_dm() # close DM 
























# --- timestamp
tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
verbose = True 
plt.ioff() # not interactive mode 
# =====(1) 
# --- DM command to make flat DM (calibrated file provided by BMC with the DM) 
flat_dm_cmd = pd.read_csv(bdf.path_to_dm_flat_map,header=None)[0].values 
# read in DM deflection data and create interpolation functions 
deflection_data = pd.read_csv(root_path + "/DM_17DW019#053_deflection_data.csv", index_col=[0])
interp_deflection_1act = interpolate.interp1d( deflection_data['cmd'],deflection_data['1x1_act_deflection[nm]'] ) #cmd to nm deflection on DM from single actuator (from datasheet) 
interp_deflection_4x4act = interpolate.interp1d( deflection_data['cmd'],deflection_data['4x4_act_deflection[nm]'] ) #cmd to nm deflection on DM from 4x4 actuator (from datasheet) 


# =====(2) 
# --- setup camera
fps = float(input("how many frames per second on camera (try between 1-600)"))
camera = bdf.setup_camera(cameraIndex=0) #connect camera and init camera object
camera = bdf.set_fsp_dit( camera, fps=fps, tint=None) # set up initial frame rate, tint=None means min integration time for given FPS

user_crop_input = input( "input 1 if you want to crop the raw images, otherwise input 0" )
if  user_crop_input = 1:
    cropping_corners = []
    for prompt in ['min row index', 'max row index','min col index','max col index']:
        cropping_corners.append( input( f"input {prompt}" ) )

else:
    cropping_corners = None

# SHOULD SET UP CAMERA CROPPING HERE TOO

# --- setup DM
dm, dm_err_code = bdf.set_up_DM(DM_serial_number='17DW019#053')

# =====(3) 
# --- get reference pupils fits files. EXTNAME = 'FPM_IN' or 'FPM_OUT'
available_ref_pupil_files = glob.glob( data_path+'PUPIL_CALIBRATION_REFERENCE_*.fits' )
print('\n======\navailable reference pupil fits files:\n')
for f in available_ref_pupil_files:
    print( f ,'\n') 

reference_pupils_path = input('input name of reference pupil file to use, otherwise input 0 to take new data for reference pupils')
if reference_pupils_path == '0':
    
    pup_ref_name = f'PUPIL_CALIBRATION_REFERENCE_FPS-{fps}_{tstamp}.fits'
    #default_naming = input(f'save reference pupil fits as {data_path+pup_ref_name}? [input either 0 or 1]')
    if 0:#not default_naming:
        pup_ref_name = input('then give us a path + name to save reference pupil fits')

    ref_pupils = bdf.get_reference_pupils(dm, camera, fps, flat_map=flat_dm_cmd, number_images_recorded_per_cmd=5, save_fits = data_path + pup_ref_name)
    
    reference_pupils_path = data_path + pup_ref_name
else: # just read it in 
    ref_pupils = fits.open( reference_pupils_path )



fig,ax =  plt.subplots( 1, 2 )
ax[0].imshow(  ref_pupils['FPM_IN'].data) 
ax[0].set_title('reference pupil \nFPM_IN') 
ax[1].imshow(  ref_pupils['FPM_OUT'].data )
ax[1].set_title('reference pupil \nFPM_OUT') 
plt.show()

# =====(4) 
# --- detect Pupil and PSF regions. rough coordinate 14/2/24 PUPIL: (x,y,r) ~ (214,228,35) , PSF: (x,y,r) ~ (278,268,15)
mask_list, circles, aqc_image = bdf.detect_pupil_and_PSF_region(camera, fps = fps, plot_results = True, save_fits = None) # mask list for detected circles in image (should correspond to pupil and PSF with circles = [(x0,y0,r0),..,(xN,yN,rN)]. aqc_image is the acquisition image used for detecting the regions

print( f'{len(mask_list)} regions detected, we will assume the largest region is pupil and the smallest is PSF')
pupil_mask = mask_list[ np.argmax( [r for _,_,r in circles] ) ] 
psf_mask = mask_list[ np.argmin( [r for _,_,r in circles] ) ] 

# cropping indicies to insolate square pupil region (makes it easier to visualize eigenmodes when cropping square region)
cp_x1,cp_x2 = int(circles[0][1] - 1.2*circles[0][-1]) ,int(circles[0][1] + 1.2*circles[0][-1])
cp_y1,cp_y2 = int(circles[0][0] - 1.2*circles[0][-1]) , int(circles[0][0] + 1.2*circles[0][-1])

# PSF circle cropping indicies 
ci_x1,ci_x2 = int(circles[1][1] - 1.2*circles[1][-1]) ,int(circles[1][1] + 1.2*circles[1][-1])
ci_y1,ci_y2 = int(circles[1][0] - 1.2*circles[1][-1]) , int(circles[1][0] + 1.2*circles[1][-1])

# =====(5)
# --- setting up parameters for poking DM to create IM 
modal_basis = np.eye(len(flat_dm_cmd))#[50:90] #just look at central actuators 
number_amp_samples = 10 #int( input('number of amplitude samples to scan across (e.g. 5)') )
amp_max = 0.1
number_images_recorded_per_cmd = 2
ramp_values = np.linspace(-amp_max, amp_max, number_amp_samples)

# --- creating sequence of dm commands
_DM_command_sequence = [list(flat_dm_cmd + amp * modal_basis) for amp in ramp_values ]  
# add in flat dm command at beginning of sequence and reshape so that cmd sequence is
# [0, a0*b0,.. aN*b0, a0*b1,...,aN*b1, ..., a0*bM,...,aN*bM]
DM_command_sequence = [flat_dm_cmd] + list( np.array(_DM_command_sequence).reshape(number_amp_samples*modal_basis.shape[0],modal_basis.shape[1] ) )

# --- additional labels to append to fits file to keep information about the sequence applied 
additional_labels = [('in-poke max amp', np.max(ramp_values)),('out-poke max amp', np.min(ramp_values)),('#ramp steps',number_amp_samples), ('seq0','flatdm'), ('reshape',f'{number_amp_samples}-{modal_basis.shape[0]}-{modal_basis.shape[1]}')]

# --- poke DM in and out and record data. Extension 0 corresponds to images, extension 1 corresponds to DM commands
raw_IM_data = bdf.apply_sequence_to_DM_and_record_images(dm, camera, DM_command_sequence, number_images_recorded_per_cmd = number_images_recorded_per_cmd, save_dm_cmds = True, calibration_dict=None, additional_header_labels = additional_labels,sleeptime_between_commands=0.01, save_fits = data_path + f'ramp_poke_DM_ampMax-{amp_max}_Nsamp-{number_amp_samples}_{tstamp}.fits' ) # None



# =====(6)
dm.send_data(flat_dm_cmd) 
# for each DM iteration we took various images (defined by "number_images_recorded_per_cmd"). Calculate median for each of these
agregated_pupils = [np.median(raw_IM_data[0].data[i],axis=0) for i in range(len(raw_IM_data[0].data))]
# get a reshaped array of the images from ramping actuators on DM, we drop index 0 since this corresponds to flat DM.
agregated_pupils_array = np.array( agregated_pupils[1:] ).reshape(number_amp_samples, modal_basis.shape[0],ref_pupils['FPM_IN'].data.shape[0], ref_pupils['FPM_IN'].data.shape[1])

# for given poke amp look at SVD and plot detector eigenmodes!

# ================================== This is what determines the poke amplitude for IM matrix
poke_amp_indx = number_amp_samples//2 + 2 # the smallest positive value 
# ==================================
print( f'======\ncalculating IM for pokeamp = {ramp_values[poke_amp_indx]}\n=====:::' )

# dont forget to crop just square pupil region
IM_unfiltered_unflat = [bdf.get_error_signal( agregated_pupils_array[poke_amp_indx][m], reference_pupil_fits=ref_pupils, reduction_dict=None, crop_indicies = [cp_x1,cp_x2,cp_y1,cp_y2]) for m in range(modal_basis.shape[0])] # [mode, x, y]

IM_unfiltered = [list(im.reshape(-1)) for im in IM_unfiltered_unflat]

# SVD
U,S,Vt = np.linalg.svd( IM_unfiltered,  full_matrices = False ) 

if verbose:
    # the eigenmodes in image space 
    fig,ax = plt.subplots(4,4,figsize=(30,30))
    plt.subplots_adjust(hspace=0.1,wspace=0.1)
    for i,axx in enumerate(ax.reshape(-1)):
        axx.imshow( Vt[i].reshape( [int(cp_x2-cp_x1), int(cp_y2-cp_y1)]) )
        axx.set_title(f'mode {i}, S={round(S[i])}')
    #plt.legend(ax=axx)
    plt.tight_layout()
    #plt.savefig( fig_path + f'ramp_poke_DM_n_analyse-SVD_modes_from_poke_IM_amp-{round(ramp_values[poke_amp_indx],3)}_t-{tstamp}.png',dpi=300)
    plt.show() 

    # the eigenmodes in DM space 
    fig,ax = plt.subplots(4,4,figsize=(30,30))
    plt.subplots_adjust(hspace=0.1,wspace=0.1)
    for i,axx in enumerate(ax.reshape(-1)):
        axx.imshow(bdf.get_DM_command_in_2D(U.T[i],Nx_act=12) )
        axx.set_title(f'mode {i}, S={round(S[i])}')
    plt.show() 

# =====(7)
# filter piston in inverse modes
invSfilt =  1/S[:] #[np.min(S)/100] + list( 1/S[1:] )
CM = Vt.T @ (np.diag( invSfilt ) )  @ U.T

# ==== (8) 
poke_amp_nm = interp_deflection_1act(0.5+ramp_values[poke_amp_indx]) - interp_deflection_1act(0.5) # get poke amp in dm that was used to create matricies 

# file name for saving final fits file [keep standard convention]
save_fits = data_path + f'BDR_RECON_amp-{round(poke_amp_nm)}nm_T-{tstamp}.fits'

recon_fits = fits.HDUList([])

camera_info_dict = bdf.get_camera_info(camera)

PRI_fits = fits.PrimaryHDU( [] )
PRI_fits.header.set('EXTNAME','HEAD')
for k,v in camera_info_dict.items(): 
    PRI_fits.header.set(k,v)   # add in some fits headers about the camera 
PRI_fits.header.set('cp_x1',cp_x1) #pupil crop coordinates
PRI_fits.header.set('cp_x2',cp_x2)
PRI_fits.header.set('cp_y1',cp_y1)
PRI_fits.header.set('cp_y2',cp_y2)
PRI_fits.header.set('ci_x1',ci_x1) # psd crop coordinates
PRI_fits.header.set('ci_x2',ci_x2)
PRI_fits.header.set('ci_y1',ci_y1)
PRI_fits.header.set('ci_y2',ci_y2)
recon_fits.append(PRI_fits)

# the reference pupil when phase mask is in beam
recon_fits.append( ref_pupils['FPM_IN'] )

# the reference pupil when phase mask is out of beam
recon_fits.append( ref_pupils['FPM_OUT'] ) 

# modal basis applied 
modalfits = fits.PrimaryHDU( modal_basis ) 
modalfits.header.set('EXTNAME','BASIS')
recon_fits.append(modalfits)

# the images after poking each mode at some amplitude  
poke_im_fits = fits.PrimaryHDU(  agregated_pupils_array[:,:,cp_x1:cp_x2,cp_y1:cp_y2] )
poke_im_fits.header.set('EXTNAME','poke_images')
poke_im_fits.header.set('WHAT_IS','images from poking DM')
poke_im_fits.header.set('dimensions','[poke_val, mode, x,y]')
for i,v in enumerate(ramp_values):
    poke_im_fits.header.set(f'poke_val_{i}',round(v,3))

recon_fits.append( poke_im_fits )

Ufits = fits.PrimaryHDU( U )
Ufits.header.set('EXTNAME','U')
Ufits.header.set('WHAT_IS','U where IM=U@S@Vt')


recon_fits.append( Ufits )

Sfits = fits.PrimaryHDU( S )
Sfits.header.set('EXTNAME','S')
Sfits.header.set('WHAT_IS','S where IM=U@S@Vt')

recon_fits.append( Sfits )

Vtfits = fits.PrimaryHDU( Vt )
Vtfits.header.set('EXTNAME','Vt')
Vtfits.header.set('WHAT_IS','Vt where IM=U@S@Vt')

recon_fits.append( Vtfits )

IMfits =  fits.PrimaryHDU( IM_unfiltered )
IMfits.header.set('EXTNAME','IM')
IMfits.header.set('WHAT IS','unfiltered interaction matrix') 
IMfits.header.set('poke_amp_cmd', round(ramp_values[poke_amp_indx],2) )
IMfits.header.set('poke_amp_nm', round(poke_amp_nm) ) 
IMfits.header.set('cp_x1',cp_x1) #pupil crop coordinates
IMfits.header.set('cp_x2',cp_x2)
IMfits.header.set('cp_y1',cp_y1)
IMfits.header.set('cp_y2',cp_y2)
IMfits.header.set('ci_x1',ci_x1) # psd crop coordinates
IMfits.header.set('ci_x2',ci_x2)
IMfits.header.set('ci_y1',ci_y1)
IMfits.header.set('ci_y2',ci_y2)
recon_fits.append( IMfits )

CMfits =  fits.PrimaryHDU( CM )
CMfits.header.set('EXTNAME','CM')
CMfits.header.set('WHAT IS','filtered control matrix') 
CMfits.header.set('poke_amp_cmd', round(ramp_values[poke_amp_indx],2)) 
CMfits.header.set('poke_amp_nm', round(poke_amp_nm) ) 

recon_fits.append( CMfits )

#WRITE IT
recon_fits.writeto( save_fits )

dm.close_dm() # close DM 



