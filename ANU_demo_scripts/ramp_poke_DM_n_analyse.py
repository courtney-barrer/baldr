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
want to see predicted sinusoidal change in intensity with ramping phase on single actuator. Can we fit model? 
"""


# DM command to make flat DM (calibrated file provided by BMC with the DM) 
flat_dm_cmd = pd.read_csv(bdf.path_to_dm_flat_map,header=None)[0].values 

# --- timestamp string for saving files 
tstamp = str(datetime.datetime.now()).replace(':','.').replace(' ','T')

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
modal_basis = np.eye(len(flat_dm_cmd))#[50:90] #just look at central actuators 
number_amp_samples = 9
# --- experiment parameters
number_images_recorded_per_cmd = 2
ramp_values = np.linspace(-0.35,0.35,number_amp_samples)
# NOTE: in_amp adds to flat command, out_amp subtracts from flat command.  
#print( '\nNOTE: in_amp adds to flat command, out_amp subtracts from flat command.  \n')
#in_amp = float(input("\namplitude of in-poke (try between 0-0.5)? "))
#out_amp = float(input("\namplitude of out-poke (try between 0-0.5)? "))

# dm commands in array structure: [[in_act0,out_act0]...[in_actN,out_actN]]
_DM_command_sequence = [list(flat_dm_cmd + amp * modal_basis) for amp in ramp_values ]  #[[i,j] for i,j in zip(flat_dm_cmd + 
DM_command_sequence = [flat_dm_cmd] + list( np.array(_DM_command_sequence).reshape(number_amp_samples*modal_basis.shape[0],modal_basis.shape[1] ) )

# additional labels to append to fits file to keep information about the sequence applied 
additional_labels = [('in-poke max amp', np.max(ramp_values)),('out-poke max amp', np.min(ramp_values)),('#ramp steps',number_amp_samples), ('seq0','flatdm'), ('reshape',f'{number_amp_samples}-{modal_basis.shape[0]}-{modal_basis.shape[1]}')]

# --- poke DM in and out and record data
raw_IM_data = bdf.apply_sequence_to_DM_and_record_images(dm, camera, DM_command_sequence, number_images_recorded_per_cmd = number_images_recorded_per_cmd, save_dm_cmds = True, calibration_dict=None, additional_header_labels = additional_labels,sleeptime_between_commands=0.01, save_fits = None) #save_fits = data_path + f'ramp_poke_DM_OUTPUT1_{tstamp}.fits' )

# --- 

agregated_pupils = [pupil_mask *np.median(raw_IM_data[0].data[i],axis=0) for i in range(len(raw_IM_data[0].data))] #median images for each ramp iteration

ref_pupil = agregated_pupils[0] # with no aberration applied 

# remove reference pupil and get an array of the images from ramping actuators on DM
agregated_pupils_array = np.array( agregated_pupils[1:] ).reshape(number_amp_samples, modal_basis.shape[0],ref_pupil.shape[0], ref_pupil.shape[1])
"""
fig, ax = plt.subplots(2,2);
ax[0,0].imshow(agregated_pupils_array[0][15]); ax[1,0].imshow(agregated_pupils_array[1][15]);
ax[0,1].imshow(agregated_pupils_array[2][15]);
ax[1,1].imshow(agregated_pupils_array[3][15]);
plt.show()
"""
# get std of intensity over each pixels ramp when in small amp range
pixelwise_std_matrix = np.std(agregated_pupils_array[2:4,:,:,:],axis=0)
# now for each cmd find x,y where pixelwise std was max
cmd2pix_registration = [np.unravel_index(pixelwise_std_matrix[i].argmax(),pixelwise_std_matrix[i].shape) for i in range(len(modal_basis))]

kwargs = {'fontsize':15}
# look at pixel registration
plt.figure(figsize=(8,5)) 
for i,(x,y) in enumerate(cmd2pix_registration):
    plt.scatter(x,y,c='k') 
plt.xlabel('x [pixel]',**kwargs)
plt.ylabel('y [pixel]',**kwargs)
plt.show() 

# look at linearity, do we see sinusoidal variation (how does this look with FPM out?)
plt.figure(figsize=(8,5)) 
for i,(x,y) in enumerate(cmd2pix_registration):
    plt.plot(ramp_values,  agregated_pupils_array[:,i,x,y] ,color='k',alpha=0.1) 
plt.ylabel('Intensity [adu]',**kwargs)
plt.xlabel('DM actuator offset [normalized]',**kwargs)
plt.tight_layout()
plt.savefig( fig_path + f'ramp_poke_DM_n_analyse-zwfs_linearity_{tstamp}.png',dpi=300)
plt.show()

# for give poke amp look at SVD and plot detector eigenmodes!
poke_amp_indx = 5
#IM_unfiltered = agregated_pupils_array[poke_amp_indx, : ,pupil_mask!=0] #shape=(len(modal_basis,-1)_
# REALIZED IT MIGHT JUST BE BEST TO CROP REGION SO WE CAN RECONSTRUCT EIGENMODES OF DETECTOR IMAGE EASILY. E.G:

# IM very important to subtract off reference 
IM_unfiltered = (agregated_pupils_array[poke_amp_indx] - ref_pupil).reshape(len(modal_basis),-1)
# cropping
cx1,cx2 = int(circles[0][1] - 1.2*circles[0][-1]) ,int(circles[0][1] + 1.2*circles[0][-1])
cy1,cy2 = int(circles[0][0] - 1.2*circles[0][-1]) , int(circles[0][0] + 1.2*circles[0][-1])

# SVD
U,S,Vt = np.linalg.svd( IM_unfiltered,  full_matrices=False ) 
# to get iamge to detector basis multiply image by Vt.T. 
"""
TEST we image perfectly detector modal basis M = Vt[i]. 
then (M @ Vt.T)[i] = 1, (M @ Vt.T)[j!=i] = 0. e.g. i=3=> M @ Vt.T = [0,0,0,1,..,0]
Here in this vector space we can set a particular sensed mode index to zero or 
apply some particular gain, and then 
to go detector basis to DM basis we multiply by U.T. e.g. U.T @ M @ Vt.T
TEST if measurement is a particular actuator poke of actuator i,  if we invert it we should see  U.T @ (agregated_pupils_array[poke_amp_indx][3] -  ref_pupil) @ Vt.T) ~ [0,0,0,1,..,0]?

plt.plot(((agregated_pupils_array[poke_amp_indx][6] -  ref_pupil).reshape(-1) @ Vt.T  )/ S);plt.show()

"""

fig,ax = plt.subplots(4,4,figsize=(30,30))
plt.subplots_adjust(hspace=0.1,wspace=0.1)
for i,axx in enumerate(ax.reshape(-1)):
    axx.imshow( Vt[i].reshape(ref_pupil.shape)[cx1:cx2,cy1:cy2] )
    axx.set_title(f'mode {i}, S={round(S[i])}')
    #plt.legend(ax=axx)
plt.tight_layout()
#plt.savefig( data_path + 'ramp_poke_DM_n_analyse-SVD_modes_reference_pupil_FPM-in_{tstamp}.png',dpi=300)
plt.savefig( fig_path + f'ramp_poke_DM_n_analyse-SVD_modes_from_poke_IM_{tstamp}.png',dpi=300)
 
plt.show()
# filter out piston 
S[0] = 0 



"""
for i in dm_indx:
    np.median(raw_IM_data[i].data ) #median over each amplitude modulation iteration (10 images taken)
    pixelwise_std_matrix # calculate pixel wise std over amplitude modulation axis 
 
    np.unravel_index(pixelwise_std_matrix.argmax(), pixelwise_std_matrix.shape)
"""





