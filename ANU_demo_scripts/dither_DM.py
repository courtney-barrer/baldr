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


"""
use calibrated DM flat map and poke each actuator in DM in and out and save each respective image
[[in_data_act0, out_data_act0]... [in_data_actN, out_data_actN]]

"""



# DM command to make flat DM (calibrated file provided by BMC with the DM) 
flat_dm_cmd = pd.read_csv(bdf.path_to_dm_flat_map,header=None)[0].values 

# --- timestamp
tstamp = datetime.datetime.now() 

""" Don't use camera
# --- setup camera
fps = float(input("how many frames per second on camera (try between 1-600)"))
camera = bdf.setup_camera(cameraIndex=0) #conect camera and init camera object
camera = bdf.set_fsp_dit( camera, fps=fps, tint=None) # set up initial frame rate, tint=None means min integration time for given FPS
"""
# --- setup DM
dm, dm_err_code = bdf.set_up_DM(DM_serial_number = '17DW019#053')

# --- 
dither_period = 2 # S 
N_its = 20
dither_amp = 0.05
actuators_2_dither = np.array([5,16,28,40,52,64])


for i in range(N_its):
    
    cmdtmp = flat_dm_cmd.copy()
    cmdtmp[actuators_2_dither] += (-1)**i * dither_amp
    print(f'executing cmd {i}, with dither amp {(-1)**i * dither_amp} ' )
    #plt.imshow( bdf.get_DM_command_in_2D(cmdtmp) ) 
    #plt.show()
    dm_err_code = dm.send_data( cmdtmp )
    if dm_err_code:
        print( dm.error_string(dm_err_code) )

    time.sleep(dither_period) 

