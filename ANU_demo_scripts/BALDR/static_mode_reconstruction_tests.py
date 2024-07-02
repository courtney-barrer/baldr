# mode reconstruction 

import numpy as np
import matplotlib.pyplot as plt 
import time 
import datetime
from astropy.io import fits
import sys

sys.path.insert(1, '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/')

from baldr_control import ZWFS
from baldr_control import phase_control
from baldr_control import pupil_control
from baldr_control import utilities as util

fig_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/figures/' 
data_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/data/' 


# THIS NEEDS TO BE CONSISTENT EVERY WHERE !!
# IT IS HOW IM IS BUILT - AND HOW WE PROCESS NEW SIGNALS 
def err_signal(I, I0, N0, bias):
    e = ( (I-bias) - (I0-bias) ) / np.sum( (N0-bias) )
    return( e )


# timestamp
tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

reco_file = f'RECONSTRUCTORS_{usr_label}_DIT-{round(float(info_fits.header["camera_tint"]),6)}_gain_{info_fits.header["camera_gain"]}_{tstamp}.fits' #'RECONSTRUCTORS_try2_DIT-0.002003_gain_medium_04-06-2024T12.40.05.fits' #'RECONSTRUCTORS_3-june_DIT-0.002003_gain_medium_04-06-2024T11.24.40.fits'#'RECONSTRUCTORS_test2_DIT-0.002003_gain_medium_31-05-2024T16.54.43.fits' #'RECONSTRUCTORS_test2_DIT-0.002003_gain_medium_31-05-2024T15.44.44.fits'
reco_fits = fits.open(data_path + reco_file  ) 


R_TT = reco_fits['R_TT'].data #tip-tilt reconstructor
R_HO = reco_fits['R_HO'].data #higher-oder reconstructor
CM = reco_fits['CM'].data # full control matrix 

U = reco_fits['U'].data 
S = reco_fits['S'].data
Vt = reco_fits['Vt'].data

smat = np.zeros((U.shape[0], Vt.shape[0]), dtype=float)
smat[:len(S),:len(S)] = np.diag(S)

IM = U @ smat @ Vt 
CM_unfilt = np.linalg.pinv( IM )


I0 = reco_fits['FPM_IN'].data # reference image FPM in
N0 = reco_fits['FPM_OUT'].data # reference image FPM out
Bi = reco_fits['BIAS'].data # bias reference used to construct IM

pupil_pixels = reco_fits['pupil_pixels'].data # pixels inside pupil
secondary_pixels = reco_fits['secondary_pixels'].data # pixels in secondary obstruction
outside_pixels = reco_fits['outside_pixels'].data # pixels outside pupil

# CM settings
DIT  = float( reco_fits['INFO'].header['camera_tint']) 
fps = float( reco_fits['INFO'].header['camera_fps'] )
gain = reco_fits['INFO'].header['camera_gain']

# cropping region of pupil
cp_x1 = reco_fits['INFO'].header['cropping_corners_r1']
cp_x2 = reco_fits['INFO'].header['cropping_corners_r2']
cp_y1 = reco_fits['INFO'].header['cropping_corners_c1']
cp_y2 = reco_fits['INFO'].header['cropping_corners_c2']

pupil_crop_region = [cp_x1, cp_x2, cp_y1, cp_y2]
#[None, None, None, None] #

# HO basis on DM used for filtering tip/tilt & HO
G =reco_fits['G'].data  # rows are DM actuators, columns modes

debug = True # plot some intermediate results 

zwfs = ZWFS.ZWFS(DM_serial_number='17DW019#053', cameraIndex=0, DMshapes_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/DMShapes/', pupil_crop_region=pupil_crop_region ) 

# ,------------------ AVERAGE OVER 8X8 SUBWIDOWS SO 12X12 PIXELS IN PUPIL
#zwfs.pixelation_factor = sw #8 # sum over 8x8 pixel subwindows in image
# HAVE TO PROPAGATE THIS TO PUPIL COORDINATES 
#zwfs._update_image_coordinates( )

zwfs.set_camera_fps(fps) # set the FPS 
zwfs.set_camera_dit(DIT) # set the DIT 
# set gain 

##
##    START CAMERA 
zwfs.start_camera()
# ----------------------
# look at the image for a second
util.watch_camera(zwfs)

"""
# CHECK THAT OUR PUPIL REGIONS MAKE SENSE (BRIGHT INSIDE PUPIL?)
I = zwfs.get_image() 

Iin = I.reshape(-1)[pupil_pixels]
Iout = I.reshape(-1)[outside_pixels]

plt.figure()
plt.hist( Iout , bins=np.linspace(0,5000,100) ,alpha=0.5)
plt.hist( Iin , bins=np.linspace(0,5000,100) ,alpha=0.5, label='inside')
plt.legend()
plt.show()
"""

# get an updated reference image
zwfs.dm.send_data(zwfs.dm_shapes['flat_dm'])

time.sleep(0.1)
raw_img_list = []
for i in range( 10 ) :
    raw_img_list.append( zwfs.get_image() ) # @D, remember for control_phase method this needs to be flattened and filtered for pupil region
raw_img = np.median( raw_img_list, axis = 0) 

I0_new = raw_img


# mode index 
mode_indx = 5 
# put a mode on DM and reconstruct it with our CM 
amp = 0.2
#mode_indx = 11
mode_aberration = G.T[mode_indx]

dm_cmd_aber = zwfs.dm_shapes['flat_dm'] + amp * mode_aberration

#plt.figure()
#plt.imshow( util.get_DM_command_in_2D(mode_aberration));plt.colorbar()
#plt.show() 
plt.figure()
plt.imshow( util.get_DM_command_in_2D(dm_cmd_aber));plt.colorbar()
plt.show() 

zwfs.dm.send_data( dm_cmd_aber )

time.sleep(0.1)
raw_img_list = []
for i in range( 10 ) :
    raw_img_list.append( zwfs.get_image() ) # @D, remember for control_phase method this needs to be flattened and filtered for pupil region
raw_img = np.median( raw_img_list, axis = 0) 


I = raw_img.reshape(-1)[pupil_pixels]
# use the new reference !!! 
i0 = I0_new.reshape(-1)[pupil_pixels]
n0 = N0.reshape(-1)[pupil_pixels]
bi = Bi.reshape(-1)[pupil_pixels]

if debug:
    plt.figure()
    plt.imshow( raw_img-Bi - (I0-Bi) );plt.colorbar()
    plt.show() 

err_img = err_signal(I, i0, n0, bi)

TT = R_TT @ err_img 
HO = R_HO @ err_img


plt.figure()
plt.imshow(util.get_DM_command_in_2D( CM.T @ err_img  ) );plt.colorbar();plt.show()

plt.figure()
plt.imshow(util.get_DM_command_in_2D( R_TT @ err_img  ) );plt.colorbar();plt.show()

plt.figure()
plt.imshow(util.get_DM_command_in_2D( R_HO @ err_img  ) );plt.colorbar(); plt.show()

plt.figure()
plt.imshow(util.get_DM_command_in_2D( CM_unfilt.T @ err_img  ) );plt.show()




#
#zwfs.dm.send_data(G.T[0])


#%% OLDER BELOW


import pandas as pd

#sw = 8 # 8 for 12x12, 16 for 6x6 
#pupil_crop_region = [157-sw, 269+sw, 98-sw, 210+sw ] #[165-sw, 261+sw, 106-sw, 202+sw ] #one pixel each side of pupil.  #tight->[165, 261, 106, 202 ]  #crop region around ZWFS pupil [row min, row max, col min, col max] 
# save to csv files 
# pd.Series( pupil_crop_region, index = ['r1','r2','c1','c2'] ).to_csv('T1_pupil_region_6x6.csv')

#pupil_crop_region = [157-sw, 269+sw, 98-sw, 210+sw ] #[165-sw, 261+sw, 106-sw, 202+sw ] #one pixel each side of pupil.  #tight->[165, 261, 106, 202 ]  #crop region around ZWFS pupil [row min, row max, col min, col max] 
readout_mode = '12x12' # '6x6'
pupil_crop_region = pd.read_csv('/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/' + f'T1_pupil_region_{readout_mode}.csv',index_col=[0])['0'].values.astype(int)

#init our ZWFS (object that interacts with camera and DM)
zwfs = ZWFS.ZWFS(DM_serial_number='17DW019#053', cameraIndex=0, DMshapes_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/DMShapes/', pupil_crop_region=pupil_crop_region ) 

# ,------------------ AVERAGE OVER 8X8 SUBWIDOWS SO 12X12 PIXELS IN PUPIL
#zwfs.pixelation_factor = sw #8 # sum over 8x8 pixel subwindows in image
# HAVE TO PROPAGATE THIS TO PUPIL COORDINATES 
#zwfs._update_image_coordinates( )
fps = 400
DIT = 2e-3
zwfs.set_camera_fps(fps) # set the FPS 
zwfs.set_camera_dit(DIT) # set the DIT 

##
##    START CAMERA 
zwfs.start_camera()
# ----------------------
# look at the image for a second
util.watch_camera(zwfs)

#init our phase controller (object that processes ZWFS images and outputs DM commands)
phase_ctrl = phase_control.phase_controller_1(config_file = None) 
#phase_ctrl.change_control_basis_parameters(number_of_controlled_modes=140, basis_name='Zonal' , dm_control_diameter = 10 ) 

#plt.figure(); plt.imshow( util.get_DM_command_in_2D( phase_ctrl.config['M2C'].T[0] ) )
#init our pupil controller (object that processes ZWFS images and outputs VCM commands)

pupil_ctrl = pupil_control.pupil_controller_1(config_file = None)


# 1.2) analyse pupil and decide if it is ok
pupil_report = pupil_control.analyse_pupil_openloop( zwfs, debug = True, return_report = True)

zwfs.update_reference_regions_in_img( pupil_report )

# 1.3) builds our control model with the zwfs
#control_model_report
ctrl_method_label = 'ctrl_1'
# for zonal basis 
#phase_ctrl.build_control_model( zwfs , poke_amp = -0.05, label=ctrl_method_label, debug = True)  
# for modal basis
phase_ctrl.build_control_model( zwfs , poke_amp = -0.15, label=ctrl_method_label, debug = True)  

#pupil_ctrl tells phase_ctrl where the pupil is

# double check DM is flat 
zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] )

#modal_overshoot_factor = [] 
#reco = []

# Try reconstruct modes on a different basis to the one developed in the IM 
ab_basis = util.construct_command_basis( basis='Zernike', number_of_modes = 50, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True)

M2C = phase_ctrl.config['M2C'] # readability 
# unfiltered CM
CM = phase_ctrl.ctrl_parameters[ctrl_method_label]['CM'] # readability 

""" #FOR ZONAL BASIS
# try filtering here
IM = phase_ctrl.ctrl_parameters[ctrl_method_label]['IM']
U, S, Vt = np.linalg.svd( IM ) 
S_filt = np.array([i < np.pi * (10/2)**2 for i,_ in enumerate(S) ]) # we consider the highest eigenvalues/vectors up to the number_of_controlled_modes
Sigma = np.zeros( np.array(IM).shape, float)
np.fill_diagonal(Sigma, S[S_filt], wrap=False) 
#filtered CM 
CM =  np.linalg.pinv( U @ Sigma @ Vt )
"""

# put a mode on DM and reconstruct it with our CM 
amp = -0.1
#mode_indx = 11

for mode_indx in range(1,6) :  

    mode_aberration = ab_basis.T[mode_indx]
    #plt.imshow( util.get_DM_command_in_2D(amp*mode_aberration));plt.colorbar();plt.show()
    
    dm_cmd_aber = zwfs.dm_shapes['flat_dm'] + amp * mode_aberration 

    zwfs.dm.send_data( dm_cmd_aber )
    time.sleep(0.1)
    raw_img_list = []
    for i in range( 10 ) :
        raw_img_list.append( zwfs.get_image() ) # @D, remember for control_phase method this needs to be flattened and filtered for pupil region
    raw_img = np.median( raw_img_list, axis = 0) 
 
    err_img = phase_ctrl.get_img_err( 1/np.mean(raw_img) *  raw_img.reshape(-1)[zwfs.pupil_pixel_filter]  ) 

    mode_res = CM.T @ err_img 

    #plt.figure(); plt.plot( mode_res ); plt.show()

    cmd_res = M2C @ mode_res
    
    time.sleep(0.01) 
    #reco.append( util.get_DM_command_in_2D( cmd_res ) )

    # =============== plotting 
    if mode_indx < 5:
    
        im_list = [util.get_DM_command_in_2D( mode_aberration ), raw_img.T/np.max(raw_img), util.get_DM_command_in_2D( cmd_res)  ]
        xlabel_list = [None, None, None]
        ylabel_list = [None, None, None]
        title_list = ['Aberration on DM', 'ZWFS Pupil', 'reconstructed DM cmd']
        cbar_label_list = ['DM command', 'Normalized intensity' , 'DM command' ] 
        savefig = None #fig_path + f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'

        util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)

    #[np.nansum( abs( f * cmd_res - ( amp * mode_aberration ) ) ) for f in np.logspace( -6,1,50 )]

    #modal_overshoot_factor.append( cmd_res - ( amp * mode_aberration ) ) 

#plt.figure()
#plt.plot( modal_overshoot_factor )

"""
zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] )

#update to WFS_Eigenmodes modes (DM modes that diagonalize the systems interaction matrix) 
phase_ctrl.change_control_basis_parameters( controller_label = ctrl_method_label, number_of_controlled_modes=phase_ctrl.config['number_of_controlled_modes'], basis_name='WFS_Eigenmodes' , dm_control_diameter=None, dm_control_center=None)

# now build control model on KL modes 
phase_ctrl.build_control_model( zwfs , poke_amp = -0.15, label='ctrl_2', debug = True) 

plt.figure()
plt.imshow( util.get_DM_command_in_2D( phase_ctrl.config['M2C'].T[2] ) );plt.show()


"""















