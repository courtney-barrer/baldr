import os 
import datetime 
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
os.chdir('/opt/FirstLightImaging/FliSdk/Python/demo/')
#import FliSdk_V2 

root_path = '/home/baldr/Documents/baldr'
data_path = root_path + '/ANU_demo_scripts/ANU_data/' 
fig_path = root_path + '/figures/' 

os.chdir(root_path)
from functions import baldr_demo_functions as bdf

"""
iterate through different camera frame rates and record a series of images for each
this can be used for building darks or flats or general analysis of detector noise properties.
"""

# --- timestamp
tstamp = datetime.datetime.now() 

# --- setup camera
camera = bdf.setup_camera(cameraIndex=0) #conect camera and init camera object
camera = bdf.set_fsp_dit( camera, fps=500, tint=None) # set up initial frame rate, tint=None means min integration time for given FPS

# --- experiment parameters
fps = [1,2,4,6,10,20,40,60,100,200,400,600] #frames per second to iterate over 
number_images_recorded_per_cmd = 50 
# --- scan FPS and record data
data = bdf.scan_detector_framerates(camera, frame_rates = fps, number_images_recorded_per_cmd = number_images_recorded_per_cmd, save_fits = data_path+f'cred3_noise_vs_fps_OUTPUT1_{tstamp}.fits')

# --- test reading it back in
data = fits.open(data_path+f'cred3_noise_vs_fps_OUTPUT1_{tstamp}.fits')

# --- analyse 
# we take median of 50 images taken at same frame rate 
# then look at standard deviation over that aggregated image 
fpsarray =  [data[i].header['camera_fps'] for i in range(len(data))]
tintarray =  [float(data[i].header['camera_tint']) for i in range(len(data))]
#if sum(np.array(fpsarray) != np.array(fps)):
#    print('FPS mismatch- are we reading in correctly?')
rms = [np.std( np.median(data[i].data,axis=0) ) for i in range(len(data))]

# --- plot 
fig,ax = plt.subplots(1,1)
ax.semilogx(fpsarray, rms, linestyle='--', marker='o', color='k')
ax.set_xlabel('frames per second')
ax.set_ylabel(r'$\sigma$ [adu]')
#ax.set_title(f'using FliSdk_V2.GetProcessedImageGrayscale16bNumpyArray\nstd of median over {number_images_recorded_per_cmd} images')
ax.set_title(f'using FliSdk_V2.GetRawImageAsNumpyArray\nstd of median over {number_images_recorded_per_cmd} images')
def fps2int(x):
    return(1/x)

def int2fps(x):
    return(1/x)

ax2 = ax.secondary_xaxis('top', functions=(fps2int,int2fps))
ax2.set_xlabel('integration time [s]')

#ax2 = ax.twiny()
#ax2.semilogx(1/fpsarray, rms, linestyle='--', marker='o', color='k')
#ax2.set_xlabel('integration time [ms]')

plt.tight_layout()
plt.savefig(fig_path+f'cred3_noise_vs_fps_OUTPUT1_{tstamp}.png',dpi=150)
plt.show()


