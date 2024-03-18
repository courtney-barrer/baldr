import streamlit as st 
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


root_path = '/home/baldr/Documents/baldr'
data_path = root_path + '/ANU_demo_scripts/ANU_data/' 
fig_path = root_path + '/figures/' 

os.chdir(root_path)
from functions import baldr_demo_functions as bdf

os.chdir(root_path+'/ANU_demo_scripts')
# 
fps = st.number_input('camera_fps', min_value=1, max_value=600,value=600, step=1) 
cropping_corners = 0,280,90,290



# =====(1) 
# --- DM command to make flat DM (calibrated file provided by BMC with the DM) 
flat_dm_cmd = pd.read_csv(bdf.path_to_dm_flat_map,header=None)[0].values 
# read in DM deflection data and create interpolation functions 
deflection_data = pd.read_csv(root_path + "/DM_17DW019#053_deflection_data.csv", index_col=[0])
interp_deflection_1act = interpolate.interp1d( deflection_data['cmd'],deflection_data['1x1_act_deflection[nm]'] ) #cmd to nm deflection on DM from single actuator (from datasheet) 
interp_deflection_4x4act = interpolate.interp1d( deflection_data['cmd'],deflection_data['4x4_act_deflection[nm]'] ) #cmd to nm deflection on DM from 4x4 actuator (from datasheet) 

# =====(2) 
# --- setup camera

camera = bdf.setup_camera(cameraIndex=0) #connect camera and init camera object
camera = bdf.set_fsp_dit( camera, fps=fps, tint=None) # set up initial frame rate, tint=None means max integration time for given FPS

# --- setup DM
dm, dm_err_code =  bdf.set_up_DM(DM_serial_number='17DW019#053')


if st.button('flat DM'): 
    dm.send_data( flat_dm_cmd ) 


if st.button('send current DM shape'): 
    dm.send_data( flat_dm_cmd ) 

im = np.zeros([100,100]) 
if st.button('get image'): 
    im = np.median( bdf.get_raw_images(camera, number_of_frames=10, cropping_corners=cropping_corners) , axis=0) 

fig = plt.figure()
plt.imshow(im)
st.pyplot(fig)


