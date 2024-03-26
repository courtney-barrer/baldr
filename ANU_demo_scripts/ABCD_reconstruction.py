# create disturbance cmd

import os 
import datetime 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import glob
from math import ceil
from scipy import interpolate
from scipy.optimize import curve_fit
from astropy.io import fits
import aotools
os.chdir('/opt/FirstLightImaging/FliSdk/Python/demo/')
import FliSdk_V2 


def func(x, A, B, F, mu):
    I = A + B * np.cos(F * x + mu)
    return I 


root_path = '/home/baldr/Documents/baldr'
data_path = root_path + '/ANU_demo_scripts/ANU_data/' 
fig_path = root_path + '/figures/' 

os.chdir(root_path)
from functions import baldr_demo_functions as bdf


debug = True # generate plots to help debug

# --- timestamp
tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
# --- 


def get_ABCD( dm, camera, dist_cmd, ABCD_cmds, flat_img, mean_pupil, cropping_corners, pupil_indicies):

    cp_x1,cp_x2, cp_y1,cp_y2 = pupil_indicies
    
    cA, cB, cC, cD = ABCD_cmds
    ABCD_intensities = []
    ABCD_SP = []
    for samp_cmd in ABCD_cmds:
    
        dm.send_data(flat_dm_cmd + dist_cmd + samp_cmd) 
        time.sleep(0.03) # wait a second

        #record + image
        I = np.median( bdf.get_raw_images(camera, number_of_frames=5,     cropping_corners=cropping_corners) , axis=0)[cp_x1:cp_x2, cp_y1:cp_y2]

        SP = mask_matrix @ ((I  - flat_img) / mean_pupil).reshape(-1)
        
        ABCD_intensities.append(I) 
        ABCD_SP.append(SP)
        

    return( ABCD_intensities, ABCD_SP ) 

# =====(1) 
# --- DM command to make flat DM (calibrated file provided by BMC with the DM) 
flat_dm_cmd = pd.read_csv(bdf.path_to_dm_flat_map, header=None)[0].values 

waffle_dm_cmd = pd.read_csv(root_path + '/DMShapes/dm_checker_pattern.csv', index_col=[0]).values.ravel() 

# phase shifts to apply
#phi_A, phi_B, phi_C, phi_D = -np.pi/2, -np.pi/2, np.pi/2, np.pi # have to be symmetric so zero total piston (aberration is measured relative to piston) 

waffle45 = np.array( [np.pi/4 / param_dict[i][2] * c if i in param_dict else 0 for i, c in enumerate(  waffle_dm_cmd ) ] )

waffle135 = np.array( [3*np.pi/4 / param_dict[i][2] * c if i in param_dict else 0 for i, c in enumerate(  waffle_dm_cmd ) ] )

ABCD_cmds = [-waffle135, -waffle45, waffle45, waffle135]

dist_cmd = np.zeros(140)
dist_cmd[65] = -0.2

ABCD_intensities, ABCD_SP = get_ABCD( dm, camera, dist_cmd, ABCD_cmds, flat_img, mean_filtered_pupil, cropping_corners,  [cp_x1,cp_x2, cp_y1,cp_y2])

dm.send_data(flat_dm_cmd) 

# phase reconstruction
plt.figure()
plt.imshow( bdf.get_DM_command_in_2D( np.arctan2(-waffle_dm_cmd *(ABCD_SP[0] - ABCD_SP[2]),(ABCD_SP[1] - ABCD_SP[3]) )  ) );plt.colorbar();plt.show()

# amplitude reconstruction 
plt.figure()
plt.imshow( bdf.get_DM_command_in_2D( ( (ABCD_SP[0] - ABCD_SP[2])**2+(ABCD_SP[1] - ABCD_SP[3])**2 )**0.5/(2*np.sum( ABCD_SP,axis=0) ) ) );plt.colorbar();plt.show()




