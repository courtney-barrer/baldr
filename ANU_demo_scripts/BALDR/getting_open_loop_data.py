#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import time 
import datetime
from astropy.io import fits
import sys
import pickle
import aotools 

sys.path.insert(1, '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/')

from baldr_control import ZWFS
from baldr_control import phase_control
from baldr_control import pupil_control
from baldr_control import utilities as util

fig_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/figures/' 
data_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/data/' 

tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

debug = True # plot some intermediate results 
fps = 400 
DIT = 2e-3 #s integration time 

sw = 8 # 8 for 12x12, 16 for 6x6 
pupil_crop_region = [157-sw, 269+sw, 98-sw, 290+sw ] #[165-sw, 261+sw, 106-sw, 202+sw ] #one pixel each side of pupil.  #tight->[165, 261, 106, 202 ]  #crop region around ZWFS pupil [row min, row max, col min, col max] 
#readout_mode = '12x12' # '6x6'
#pupil_crop_region = pd.read_csv('/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/' + f'T1_pupil_region_{readout_mode}.csv',index_col=[0])['0'].values

#pupil_crop_region = [None, None, None, None]

#init our ZWFS (object that interacts with camera and DM)
zwfs = ZWFS.ZWFS(DM_serial_number='17DW019#053', cameraIndex=0, DMshapes_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/DMShapes/', pupil_crop_region=pupil_crop_region ) 

# ,------------------ AVERAGE OVER 8X8 SUBWIDOWS SO 12X12 PIXELS IN PUPIL
#zwfs.pixelation_factor = sw #8 # sum over 8x8 pixel subwindows in image
# HAVE TO PROPAGATE THIS TO PUPIL COORDINATES 
#zwfs._update_image_coordinates( )

zwfs.set_camera_fps(fps) # set the FPS 
zwfs.set_camera_dit(DIT) # set the DIT 

#zwfs.set_camera_cropping(r1=152, r2=267, c1=96, c2=223)
#zwfs.enable_frame_tag(tag = False) # first 1-3 pixels count frame number etc

# TO EXIT CAMERA AND DM SO THEY CAN BE RE-INITIALIZED 
#zwfs.exit_camera()
#zwfs.exit_dm()

##
##    START CAMERA 
zwfs.start_camera()
# ----------------------
# look at the image for a second
util.watch_camera(zwfs)


amp_max = 0.3
number_amp_samples = 30

flat_dm_cmd = zwfs.dm_shapes['flat_dm']

#zonal
modal_basis_zonal = np.eye(len(flat_dm_cmd))

#zernike
modal_basis_zernike = util.construct_command_basis( basis='Zernike', number_of_modes = 140, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True)

#normal
normal_random_zonal = 0.1 * np.random.randn(10000,140) # this gives like 4 signal at +-0.5 (which is good when were working around the DM flat)

#random realizations of Kolmogorov screen 
# --- create infinite phasescreen from aotools module 
Nx_act = 12
screen_pixels = Nx_act*2**5 # some multiple of numer of actuators across DM 
D = 1.8 #m effective diameter of the telescope
corner_indicies = [0, Nx_act-1, Nx_act * (Nx_act-1), -1] # Beware -1 index doesn't work if inserting in list! This is  ok for for use with create_phase_screen_cmd_for_DM function.
kolmogorov_random = []
scrn_scaling_factor =  0.22 #1.2*( np.random.rand() - 0.5 )

scrn = aotools.infinitephasescreen.PhaseScreenVonKarman(nx_size=screen_pixels, pixel_scale=D/screen_pixels,r0=0.1,L0=12)

for _ in range(10000): # 1 per second, ~17 minutes for 1000...
    for _ in range(20):
        scrn.add_row()
    kol_cmd = util.create_phase_screen_cmd_for_DM(scrn,  scaling_factor=scrn_scaling_factor   , drop_indicies = corner_indicies, plot_cmd=False)
    kolmogorov_random.append( kol_cmd )

# TO POKE CMDS WHILE ROLLING PHASE SCREEN 
j=0 # to count when to poke next actuator
act_basis = np.eye(140) # actuator basis 
scrn_scaling_factor =  0.25 #1.2*( np.random.rand() - 0.5 )
kolmogorov_random = []
#print( act_basis[140] )
for i in range(15000): # 1 per second, ~17 minutes for 1000...
    for _ in range(40): # roll it more agressively 
        scrn.add_row()

    if np.mod(i,100)==0:
        if j<140:
            poke_cmd = -0.04 * act_basis[j]
            j+=1
        else:
            poke_cmd = np.zeros(140)

    kol_cmd = util.create_phase_screen_cmd_for_DM(scrn,  scaling_factor=scrn_scaling_factor   , drop_indicies = corner_indicies, plot_cmd=False)
    kolmogorov_random.append( kol_cmd + poke_cmd )

"""
fig, ax = plt.subplots( 1,2)
ax[0].imshow( util.get_DM_command_in_2D( kolmogorov_random[20] ) )
ax[1].imshow( util.get_DM_command_in_2D( kolmogorov_random[22] ) )
plt.show()
#plt.imshow( util.get_DM_command_in_2D( kol_cmd ) );plt.colorbar();plt.show()
"""
#plt.imshow( util.get_DM_command_in_2D( modal_basis_2[:,2] )) ;plt.show()

cp_x1,cp_x2,cp_y1,cp_y2 = zwfs.pupil_crop_region
number_amp_samples = 30
ramp_values = np.linspace(-amp_max, amp_max, number_amp_samples)


modal_basis_1 = np.array(kolmogorov_random) #normal_random_zonal #modal_basis_zernike #modal_basis_zonal

descr_mode = f'LONG_rolling_kolmogorov_scaling-{round(scrn_scaling_factor ,3)}' #f'rolling_kolmogorov_scaling-{round(scrn_scaling_factor ,3)}' #'random_normal' #'zonal_pokes', 'zernike_pokes'

_DM_command_sequence_1 = [list(flat_dm_cmd + amp * modal_basis_1.T) for amp in ramp_values ]  
#_DM_command_sequence_2 = [list(flat_dm_cmd + amp * modal_basis_2.T) for amp in ramp_values ]  
    # add in flat dm command at beginning of sequence and reshape so that cmd sequence is
    # [0, a0*b0,.. aN*b0, a0*b1,...,aN*b1, ..., a0*bM,...,aN*bM]

DM_command_sequence = [flat_dm_cmd] + list( np.array(_DM_command_sequence_1).reshape(number_amp_samples*modal_basis_1.shape[0],modal_basis_1.shape[1] ) )

if ('kolmog' in descr_mode) or (descr_mode=='random_normal'):
    DM_command_sequence = [flat_dm_cmd] + list(modal_basis_1)
    number_amp_samples = 0
# +
#list( np.array(_DM_command_sequence_2).reshape(number_amp_samples*modal_basis_2.shape[0],modal_basis_2.shape[1] ) ) +
#list( 

# to check
plt.imshow( util.get_DM_command_in_2D( DM_command_sequence[100] ));plt.colorbar();plt.show()

# --- additional labels to append to fits file to keep information about the sequence applied 
additional_labels = [('mode',descr_mode), ('cp_x1',cp_x1),('cp_x2',cp_x2),('cp_y1',cp_y1),('cp_y2',cp_y2),('in-poke max amp', np.max(ramp_values)),('out-poke max amp', np.min(ramp_values)),('#ramp steps',number_amp_samples), ('seq0','flatdm'), ('reshape',f'{number_amp_samples}-{modal_basis_1.shape[0]}-{modal_basis_1.shape[1]}'),('Nmodes_poked',len(modal_basis_1.T)),('Nact',140)]

# --- poke DM in and out and record data. Extension 0 corresponds to images, extension 1 corresponds to DM commands
#raw_recon_data = apply_sequence_to_DM_and_record_images(zwfs, DM_command_sequence, number_images_recorded_per_cmd = number_images_recorded_per_cmd, take_median_of_images=True, save_dm_cmds = True, calibration_dict=None, additional_header_labels = additional_labels,sleeptime_between_commands=0.03, cropping_corners=None,  save_fits = None ) # None

a = util.apply_sequence_to_DM_and_record_images(zwfs, DM_command_sequence, number_images_recorded_per_cmd = 3, take_median_of_images=True, save_dm_cmds = True, calibration_dict=None, additional_header_labels=None, sleeptime_between_commands=0.03, cropping_corners=additional_labels, save_fits = data_path+f'open_loop_data_{descr_mode}_{tstamp}.fits')



#check 
a = fits.open( data_path + f'open_loop_data_{descr_mode}_{tstamp}.fits')
# check
plt.imshow( a[0].data[100][0]  ); plt.show()

plt.figure()
plt.imshow( util.get_DM_command_in_2D(a[1].data[100])  );plt.colorbar(); plt.show()
