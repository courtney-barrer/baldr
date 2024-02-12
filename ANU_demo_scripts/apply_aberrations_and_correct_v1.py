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
detected_image_regions = bdf.detect_pupil_and_PSF_region(camera, fps = 500, plot_results = True, save_fits = None) # detected circles in image (should correspond to pupil and PSF with output = [(x0,y0,r0),..,(xN,yN,rN)]

# --- create PUPIL and PSF mask 


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


