# testing Hubing Du paper "Random phase-shifting algorithm by constructing orthogonal phase-shifting fringe patterns 


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



# --- DM command to make flat DM (calibrated file provided by BMC with the DM) 
flat_dm_cmd = pd.read_csv(bdf.path_to_dm_flat_map, header=None)[0].values 

waffle_dm_cmd = pd.read_csv(root_path + '/DMShapes/dm_checker_pattern.csv', index_col=[0]).values.ravel() 


offset = np.array( [np.pi/10 / param_dict[i][2] * c if i in param_dict else 0 for i, c in enumerate(  waffle_dm_cmd ) ] )

offset = np.array( [np.pi/10 / param_dict[65][2] * c if i in param_dict else 0 for i, c in enumerate(  waffle_dm_cmd ) ] )


dist_cmd = np.zeros(140)
dist_cmd[64] = 0.01

# flat DM 
dm.send_data( flat_dm_cmd + dist_cmd) 

# reference intensity
I0 = np.median( bdf.get_raw_images(camera, number_of_frames=5,     cropping_corners=cropping_corners) , axis=0)[cp_x1:cp_x2, cp_y1:cp_y2] 

# modulatated intensity 
dm.send_data(flat_dm_cmd + dist_cmd + offset) 
time.sleep(0.03) # wait a second

In = np.median( bdf.get_raw_images(camera, number_of_frames=5,     cropping_corners=cropping_corners) , axis=0)[cp_x1:cp_x2, cp_y1:cp_y2] 

Is = I0 - In #subtract
Ia = I0 + In #add

dm.send_data(flat_dm_cmd) 

Is_dm = mask_matrix @ Is.reshape(-1)
Ia_dm = mask_matrix @ Ia.reshape(-1)

plt.figure();
plt.imshow( bdf.get_DM_command_in_2D( np.arctan2( Is_dm,Ia_dm )-3/2*offset ) );
plt.colorbar();
plt.show()






""""
test model 
delta = ( I(c_r+epsilon*c_i) - I(c_r-epsilon*c_i) ) / epsilon

This fits a model 
delta = A+B(sin(Fx) * cos(mu) + cos(Fx) * sin(mu) ) 

make c_r = 0, disturb_cmd = phi_res


""""
