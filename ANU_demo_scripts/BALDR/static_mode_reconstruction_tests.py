# mode reconstruction 

from baldr_control import ZWFS
from baldr_control import phase_control
from baldr_control import pupil_control
from baldr_control import utilities as util

import numpy as np
import matplotlib.pyplot as plt 
import time 
import datetime
from astropy.io import fits

fig_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/figures/' 
data_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/data/' 

# timestamp
tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

debug = True # plot some intermediate results 
fps = 400 
DIT = 2e-3 #s integration time 

sw = 8 # 8 for 12x12, 16 for 6x6 
pupil_crop_region = [157-sw, 269+sw, 98-sw, 210+sw ] #[165-sw, 261+sw, 106-sw, 202+sw ] #one pixel each side of pupil.  #tight->[165, 261, 106, 202 ]  #crop region around ZWFS pupil [row min, row max, col min, col max] 
# save to csv files 
# pd.Series( pupil_crop_region, index = ['r1','r2','c1','c2'] ).to_csv('T1_pupil_region_6x6.csv')

pupil_crop_region = pd.read_csv('T1_pupil_region_12x12.csv',index_col=[0])['0'].values


#init our ZWFS (object that interacts with camera and DM)
zwfs = ZWFS.ZWFS(DM_serial_number='17DW019#053', cameraIndex=0, DMshapes_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/DMShapes/', pupil_crop_region=pupil_crop_region ) 

# ,------------------ AVERAGE OVER 8X8 SUBWIDOWS SO 12X12 PIXELS IN PUPIL
#zwfs.pixelation_factor = sw #8 # sum over 8x8 pixel subwindows in image
# HAVE TO PROPAGATE THIS TO PUPIL COORDINATES 
#zwfs._update_image_coordinates( )

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
#phase_ctrl.change_control_basis_parameters(number_of_controlled_modes=20, basis_name='Zernike' , dm_control_diameter = 10 ) 

#plt.figure(); plt.imshow( util.get_DM_command_in_2D( phase_ctrl.config['M2C'].T[0] ) )
#init our pupil controller (object that processes ZWFS images and outputs VCM commands)

pupil_ctrl = pupil_control.pupil_controller_1(config_file = None)


# 1.2) analyse pupil and decide if it is ok
pupil_report = pupil_control.analyse_pupil_openloop( zwfs, debug = True, return_report = True)

zwfs.update_reference_regions_in_img( pupil_report )

# 1.3) builds our control model with the zwfs
#control_model_report
ctrl_method_label = 'ctrl_1'
phase_ctrl.build_control_model( zwfs , poke_amp = -0.15, label=ctrl_method_label, debug = True)  

#pupil_ctrl tells phase_ctrl where the pupil is

# double check DM is flat 
zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] )

#modal_overshoot_factor = [] 
#reco = []
for mode_indx in range(1,20) :  

    M2C = phase_ctrl.config['M2C'] # readability 
    CM = phase_ctrl.ctrl_parameters[ctrl_method_label]['CM'] # readability 

    # put a mode on DM and reconstruct it with our CM 
    amp = -0.1
    #mode_indx = 11
    mode_aberration = phase_ctrl.config['M2C'].T[mode_indx]
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















