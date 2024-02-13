import os 
import datetime 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from astropy.io import fits
import aotools
os.chdir('/opt/FirstLightImaging/FliSdk/Python/demo/')
#import FliSdk_V2 

root_path = '/home/baldr/Documents/baldr'
data_path = root_path + '/ANU_demo_scripts/ANU_data/' 
fig_path = root_path + '/figures/' 

os.chdir(root_path)
from functions import baldr_demo_functions as bdf



# DM command to make flat DM (calibrated file provided by BMC with the DM) 
flat_dm_cmd = pd.read_csv(bdf.path_to_dm_flat_map,header=None)[0].values 

# --- timestamp
tstamp = datetime.datetime.now() 

# --- setup camera
fps = float(input("how many frames per second on camera (try between 1-600)"))
camera = bdf.setup_camera(cameraIndex=0) #conect camera and init camera object
camera = bdf.set_fsp_dit( camera, fps=fps, tint=None) # set up initial frame rate, tint=None means min integration time for given FPS

# --- setup DM
dm, dm_err_code = bdf.set_up_DM(DM_serial_number='17DW019#053')

# --- detect Pupil and PSF regions  
mask_list, circles, aqc_image = bdf.detect_pupil_and_PSF_region(camera, fps = fps, plot_results = True, save_fits = None) # mask list for detected circles in image (should correspond to pupil and PSF with circles = [(x0,y0,r0),..,(xN,yN,rN)]. aqc_image is the acquisition image used for detecting the regions

print( f'{len(mask_list)} regions detected, we will assume the largest region is pupil and the smallest is PSF')
pupil_mask = mask_list[ np.argmax( [r for _,_,r in circles] ) ] 
psf_mask = mask_list[ np.argmin( [r for _,_,r in circles] ) ] 

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


