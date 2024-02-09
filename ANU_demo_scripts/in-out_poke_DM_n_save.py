import os 
import datetime 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
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

# --- setup camera
fps = float(input("how many frames per second on camera (try between 1-600)"))
camera = bdf.setup_camera(cameraIndex=0) #conect camera and init camera object
camera = bdf.set_fsp_dit( camera, fps=fps, tint=None) # set up initial frame rate, tint=None means min integration time for given FPS

# --- setup DM
dm, dm_err_code = bdf.set_up_DM(DM_serial_number='17DW019#053')

# --- experiment parameters
number_images_recorded_per_cmd = 1
# NOTE: in_amp adds to flat command, out_amp subtracts from flat command.  
print( '\nNOTE: in_amp adds to flat command, out_amp subtracts from flat command.  \n')
in_amp = float(input("amplitude of in-poke (try between 0-0.5)? "))
out_amp = float(input("amplitude of out-poke (try between 0-0.5)? "))

# dm commands in array structure: [[in_act0,out_act0]...[in_actN,out_actN]]
_DM_command_sequence = [[i,j] for i,j in zip(flat_dm_cmd + in_amp * np.eye(len(flat_dm_cmd)), flat_dm_cmd - out_amp * np.eye(len(flat_dm_cmd)) )]
# add flat_dm_cmd and reshape [flat_dm, in_act0, out_act0,...,in_actN, out_actN]
DM_command_sequence = [flat_dm_cmd] + list( np.array(_DM_command_sequence).reshape(2*len(flat_dm_cmd),len(flat_dm_cmd)) )

# additional labels to append to fits file to keep information about the sequence applied 
additional_labels = [('in-poke amp', in_amp),('out-poke amp', out_amp), ('seq0','flatdm'),('seq1','in_act0'),('seq2','out_act0'),('seq3','in_act1'),('seq4','..etc..')] 

# --- poke DM in and out and record data
data = bdf.apply_sequence_to_DM_and_record_images(dm, camera, DM_command_sequence, number_images_recorded_per_cmd = number_images_recorded_per_cmd, save_dm_cmds = True, calibration_dict=None, additional_header_labels = additional_labels, save_fits = data_path + f'in-out_poke_DM_OUTPUT1_{tstamp}.fits' )

# --- test reading it back in
data = fits.open(data_path+f'in-out_poke_DM_OUTPUT1_{tstamp}.fits')

# --- analyse 
ref0 = np.median(data[0].data[0],axis=0) #median image with flat DM 
poke_amp_in, poke_amp_out = float( data[0].header['in-poke amp'] ), float( data[0].header['out-poke amp'] )

p2v = []
p2v_max = []
bias = []

for i in np.arange(1,len(DM_command_sequence),2):

    I_in = np.median(data[0].data[i],axis=0) #median intensity when poking actuator i in 
    I_out = np.median(data[0].data[i+1],axis=0) #median intensity when poking actuator i out
   
    p2v.append( (I_in - I_out)/(poke_amp_in + poke_amp_out) ) # peak-to-valley intensity normalized by peak-to-valley poke amp
    
    #p2v_max.append( np.max( abs( (I_in - I_out)/(poke_amp_in + poke_amp_out) ) ) ) 
    bias.append( ref0 - (I_in + I_out)/2  ) #distance between midpoint of intensity with push_pull relative to flat DM
    

# save them as fits to analyse on weekend
p2v_fits = fits.PrimaryHDU( p2v )
bias_fits = fits.PrimaryHDU( bias )
p2v_fits.header.set('what is', '(I_in - I_out)/(poke_amp_in + poke_amp_out)' )
bias_fits.header.set('what is', 'ref0 - (I_in + I_out)/2 from in-out poke script')

p2v_fits.writeto(data_path+f'p2v_{tstamp}.fits')
bias_fits.writeto(data_path+f'bias_{tstamp}.fits')

"""
    #reconstruct WFS gains on illuminated DM (actuator_gain_on_DM)
    # rough idea
    
    actuator_gain_on_DM = np.array(p2v_max+[np.nan for i in range(4)]).reshape(12,12) #add corners back in at end (not in order.. need to work this out! 
    
    plt.imshow( actuator_gain_on_DM ) # [adu / dm_cmd];plt.show()

    
"""

    
   





# center 





