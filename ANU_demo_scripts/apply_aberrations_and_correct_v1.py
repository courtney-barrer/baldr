import os 
import datetime 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import glob
from astropy.io import fits
import aotools
os.chdir('/opt/FirstLightImaging/FliSdk/Python/demo/')
#import FliSdk_V2 

root_path = '/home/baldr/Documents/baldr'
data_path = root_path + '/ANU_demo_scripts/ANU_data/' 
fig_path = root_path + '/figures/' 

os.chdir(root_path)
from functions import baldr_demo_functions as bdf


"""
-- setup 
1. read in flat DM command matrix
2. set up camera and DM
3. take reference pupils with FPM in/out
4. detect and define PUPIL and PSF regions in image, create respective masks 
5. create IM by poking DM,recording images and applying appropiate signal processing
6. SVD analysis of IM display and analyse results
7. create CM by pseudo inverse of IM  
8. Put on static command and try correct it 
-- 

"""

# --- timestamp
tstamp = datetime.datetime.now() 

# =====(1) 
# --- DM command to make flat DM (calibrated file provided by BMC with the DM) 
flat_dm_cmd = pd.read_csv(bdf.path_to_dm_flat_map,header=None)[0].values 

# =====(2) 
# --- setup camera
fps = float(input("how many frames per second on camera (try between 1-600)"))
camera = bdf.setup_camera(cameraIndex=0) #connect camera and init camera object
camera = bdf.set_fsp_dit( camera, fps=fps, tint=None) # set up initial frame rate, tint=None means min integration time for given FPS

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
    default_naming = input(f'save reference pupil fits as {data_path+pup_ref_name}? [input either 0 or 1]')
    if not default_naming:
        pup_ref_name = input('then give us a path + name to save reference pupil fits')

    ref_pupils = bdf.get_reference_pupils(dm, camera, fps, flat_map=flat_dm_cmd, number_images_recorded_per_cmd=50, save_fits = data_path + pup_ref_name)
else: # just read it in 
    ref_pupils = fits.open( reference_pupils_path )

fig,ax =  plt.subplots( 1, 2 )
ax[0].imshow(  ref_pupils['FPM_IN'].data) 
ax[0].set_title('reference pupil \nFPM_IN') 
ax[1].imshow(  ref_pupils['FPM_OUT'].data )
ax[1].set_title('reference pupil \nFPM_OUT') 

# =====(4) 
# --- detect Pupil and PSF regions. rough coordinate 14/2/24 PUPIL: (x,y,r) ~ (214,228,35) , PSF: (x,y,r) ~ (278,268,15)
mask_list, circles, aqc_image = bdf.detect_pupil_and_PSF_region(camera, fps = fps, plot_results = True, save_fits = None) # mask list for detected circles in image (should correspond to pupil and PSF with circles = [(x0,y0,r0),..,(xN,yN,rN)]. aqc_image is the acquisition image used for detecting the regions

print( f'{len(mask_list)} regions detected, we will assume the largest region is pupil and the smallest is PSF')
pupil_mask = mask_list[ np.argmax( [r for _,_,r in circles] ) ] 
psf_mask = mask_list[ np.argmin( [r for _,_,r in circles] ) ] 

# cropping indicies to insolate square pupil region (makes it easier to visualize eigenmodes when cropping square region)
cp_x1,cp_x2 = int(circles[0][1] - 1.2*circles[0][-1]) ,int(circles[0][1] + 1.2*circles[0][-1])
cp_y1,cp_y2 = int(circles[0][0] - 1.2*circles[0][-1]) , int(circles[0][0] + 1.2*circles[0][-1])


# =====(5)
# --- setting up parameters for poking DM to create IM 
modal_basis = np.eye(len(flat_dm_cmd))#[50:90] #just look at central actuators 
number_amp_samples = 9
number_images_recorded_per_cmd = 2
ramp_values = np.linspace(-0.35,0.35,number_amp_samples)

# --- creating sequence of dm commands
_DM_command_sequence = [list(flat_dm_cmd + amp * modal_basis) for amp in ramp_values ]  
# add in flat dm command at beginning of sequence and reshape so that cmd sequence is
# [0, a0*b0,.. aN*b0, a0*b1,...,aN*b1, ..., a0*bM,...,aN*bM]
DM_command_sequence = [flat_dm_cmd] + list( np.array(_DM_command_sequence).reshape(number_amp_samples*modal_basis.shape[0],modal_basis.shape[1] ) )

# --- additional labels to append to fits file to keep information about the sequence applied 
additional_labels = [('in-poke max amp', np.max(ramp_values)),('out-poke max amp', np.min(ramp_values)),('#ramp steps',number_amp_samples), ('seq0','flatdm'), ('reshape',f'{number_amp_samples}-{modal_basis.shape[0]}-{modal_basis.shape[1]}')]

# --- poke DM in and out and record data. Extension 0 corresponds to images, extension 1 corresponds to DM commands
raw_IM_data = bdf.apply_sequence_to_DM_and_record_images(dm, camera, DM_command_sequence, number_images_recorded_per_cmd = number_images_recorded_per_cmd, save_dm_cmds = True, calibration_dict=None, additional_header_labels = additional_labels,sleeptime_between_commands=0.01, save_fits = None) #save_fits = data_path + f'ramp_poke_DM_OUTPUT1_{tstamp}.fits' )

# =====(6)
# for each DM iteration we took various images (defined by "number_images_recorded_per_cmd"). Calculate median for each of these
agregated_pupils = [np.median(raw_IM_data[0].data[i],axis=0) for i in range(len(raw_IM_data[0].data))]
# get a reshaped array of the images from ramping actuators on DM, we drop index 0 since this corresponds to flat DM.
agregated_pupils_array = np.array( agregated_pupils[1:] ).reshape(number_amp_samples, modal_basis.shape[0],ref_pupils['FPM_IN'].data.shape[0], ref_pupils['FPM_IN'].data.shape[1])

# for given poke amp look at SVD and plot detector eigenmodes!
poke_amp_indx = 5
print( f'calculating IM for pokeamp = {ramp_values[poke_amp_indx]}' )
# dont forget to crop just square pupil region
IM_unfiltered_unflat = [bdf.image_signal_processing( agregated_pupils_array[poke_amp_indx][m], reference_pupil_fits=ref_pupils, reduction_dict=None,crop_indicies = [cp_x1,cp_x2,cp_y1,cp_y2]) for m in range(modal_basis.shape[0])] # [mode, x, y]

IM_unfiltered = [list(im.reshape(-1)) for im in IM_unfiltered_unflat]
# SVD
U,S,Vt = np.linalg.svd( IM_unfiltered,  full_matrices = False ) 

# the eigenmodes in image space 
fig,ax = plt.subplots(4,4,figsize=(30,30))
plt.subplots_adjust(hspace=0.1,wspace=0.1)
for i,axx in enumerate(ax.reshape(-1)):
    axx.imshow( Vt[i].reshape( [int(cp_x2-cp_x1), int(cp_y2-cp_y1)]) )
    axx.set_title(f'mode {i}, S={round(S[i])}')
    #plt.legend(ax=axx)
plt.tight_layout()
plt.savefig( fig_path + f'ramp_poke_DM_n_analyse-SVD_modes_from_poke_IM_amp-{round(ramp_values[poke_amp_indx],3)}_t-{tstamp}.png',dpi=300)

# the eigenmodes in DM space 
fig,ax = plt.subplots(4,4,figsize=(30,30))
plt.subplots_adjust(hspace=0.1,wspace=0.1)
for i,axx in enumerate(ax.reshape(-1)):
    axx.imshow(bdf.get_DM_command_in_2D(U.T[i],Nx_act=12) )
    axx.set_title(f'mode {i}, S={round(S[i])}')

# =====(7)
#CM = U.T @ np.diag( 1/S ) @ Vt.T
# filter piston 
invSfilt = [0] + list(1/S[1:] )

IM_unfiltered @ ( Vt.T @ (np.diag( 1/S ) )  @ U.T )

plt.imshow( IM_unfiltered @ ( Vt.T @ (np.diag( invSfilt ) )  @ U.T ) ) # = (U @ S @ Vt) @ (Vt.T @ 1/S @ U.T), and Vt @ Vt.T = I, U @ U.T = I  
 
#control matrix filtering piston 
CM = Vt.T @ (np.diag( invSfilt ) )  @ U.T
"""
# =====(8)
# --- create infinite phasescreen from aotools module 
Nx_act = dm.num_actuators_width()
scrn = aotools.infinitephasescreen.PhaseScreenVonKarman(nx_size=Nx_act*2**5, pixel_scale=1.8/(Nx_act*2**5),r0=0.1,L0=12)

# --- since DM is 12x12 without corners we indicate which indicies need to be dropped in flattened array in cmd space
corner_indicies = [0, Nx_act-1, Nx_act * (Nx_act-1), -1] # Beware -1 index doesn't work if inserting in list! This is  ok for for use with create_phase_screen_cmd_for_DM function.  

# --- now roll our phase screen and aggregate it onto DM command space to create our command sequence 
dm_cmd_sequence = []
for i in range(50):
    [scrn.add_row() for i in range(5)] # propagate screen a few pixels for each iteration 
    
    dm_cmd_sequence.append( bdf.create_phase_screen_cmd_for_DM(scrn=scrn, DM=dm, flat_reference=flat_dm_cmd, scaling_factor=0.2, drop_indicies = corner_indicies, plot_cmd=False) ) 

"""















"""

#  ======== CONSTRUCT INTERACTION AND CONTROL MATRICIES 
modal_basis = np.eye(len(flat_dm_cmd))
# --- experiment parameters
number_images_recorded_per_cmd = 10
# NOTE: in_amp adds to flat command, out_amp subtracts from flat command.  
print( '\nNOTE: in_amp adds to flat command, out_amp subtracts from flat command.  \n')
in_amp = float(input("\namplitude of in-poke (try between 0-0.5)? "))
out_amp = float(input("\namplitude of out-poke (try between 0-0.5)? "))

# dm commands in array structure: [[in_act0,out_act0]...[in_actN,out_actN]]
_DM_command_sequence = [[i,j] for i,j in zip(flat_dm_cmd + in_amp * modal_basis , flat_dm_cmd - out_amp *  modal_basis )]
# add flat_dm_cmd and reshape [flat_dm, in_act0, out_act0,...,in_actN, out_actN]
DM_command_sequence = [flat_dm_cmd] + list( np.array(_DM_command_sequence).reshape(2*len(flat_dm_cmd),len(flat_dm_cmd)) )

# additional labels to append to fits file to keep information about the sequence applied 
additional_labels = [('in-poke amp', in_amp),('out-poke amp', out_amp), ('seq0','flatdm'),('seq1','in_act0'),('seq2','out_act0'),('seq3','in_act1'),('seq4','..etc..')] 

# --- poke DM in and out and record data
raw_IM_data = bdf.apply_sequence_to_DM_and_record_images(dm, camera, DM_command_sequence, number_images_recorded_per_cmd = number_images_recorded_per_cmd, save_dm_cmds = True, calibration_dict=None, additional_header_labels = additional_labels, save_fits = data_path + f'in-out_poke_DM_OUTPUT1_{tstamp}.fits' )


in_cols = [list( (pupil_mask * np.median(raw_IM_data[i].data,axis=0) )[np.isfinite(pupil_mask)] ) for i in np.arange(1,len(raw_IM_data),2)]
out_cols =  [list( (pupil_mask * np.median(raw_IM_data[i].data,axis=0) )[np.isfinite(pupil_mask)] ) for i in np.arange(2,len(raw_IM_data),2)]

U,S,Vt = np.linalg.svd( np.array(in_cols) )

plt.figure()
plt.plot( S )

# Now filter K modes in image 
red = Uk * image

 



# ======== PREP DISTURBANCE COMMANDS
# --- create infinite phasescreen from aotools module 
Nx_act = dm.num_actuators_width()
scrn = aotools.infinitephasescreen.PhaseScreenVonKarman(nx_size=Nx_act*2**5, pixel_scale=1.8/(Nx_act*2**5),r0=0.1,L0=12)

# --- since DM is 12x12 without corners we indicate which indicies need to be dropped in flattened array in cmd space
corner_indicies = [0, Nx_act-1, Nx_act * (Nx_act-1), -1] # Beware -1 index doesn't work if inserting in list! This is  ok for for use with create_phase_screen_cmd_for_DM function. 

# --- now roll our phase screen and aggregate it onto DM command space to create our command sequence 
dm_cmd_sequence = []
for i in range(50):
    [scrn.add_row() for i in range(5)] # propagate screen a few pixels for each iteration 
    
    dm_cmd_sequence.append( bdf.create_phase_screen_cmd_for_DM(scrn=scrn, DM=dm, flat_reference=flat_dm_cmd, scaling_factor=0.2, drop_indicies = corner_indicies, plot_cmd=False) ) 


"""

"""
# ======== 
record image 
signal processing to get control signal
propagate a bit more phase screen  
applyControl signal

record image 

# --- apply the command sequence to DM 
for c in dm_cmd_sequence :
    dm.send_data(c)
    time.sleep(0.5)
"""


