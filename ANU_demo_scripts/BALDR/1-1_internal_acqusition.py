from baldr_control import ZWFS
from baldr_control import phase_control
from baldr_control import pupil_control
from baldr_control import utilities as util

import numpy as np
import matplotlib.pyplot as plt 
import time 


fig_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/figures/' 
data_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/data/' 


debug = True # plot some intermediate results 
fps = 400 
DIT = 2e-3 #s integration time 

sw = 8 # 8 for 12x12, 16 for 6x6 
pupil_crop_region = [157-sw, 269+sw, 98-sw, 210+sw ] #[165-sw, 261+sw, 106-sw, 202+sw ] #one pixel each side of pupil.  #tight->[165, 261, 106, 202 ]  #crop region around ZWFS pupil [row min, row max, col min, col max] 

#init our ZWFS (object that interacts with camera and DM)
zwfs = ZWFS.ZWFS(DM_serial_number='17DW019#053', cameraIndex=0, DMshapes_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/DMShapes/', pupil_crop_region=pupil_crop_region ) 

# ,------------------ AVERAGE OVER 8X8 SUBWIDOWS SO 12X12 PIXELS IN PUPIL
zwfs.pixelation_factor = sw #8 # sum over 8x8 pixel subwindows in image
# HAVE TO PROPAGATE THIS TO PUPIL COORDINATES 
zwfs._update_image_coordinates( )

zwfs.set_camera_fps(fps) # set the FPS 
zwfs.set_camera_dit(DIT) # set the DIT 

##
##    START CAMERA 
zwfs.start_camera()
# ----------------------

#init our phase controller (object that processes ZWFS images and outputs DM commands)
phase_ctrl = phase_control.phase_controller_1(config_file = None) 

# have a look at one of the modes on the DM. Modes are normalized in cmd space <m|m> = 1
if debug:
    mode_indx = 0
    plt.figure() 
    plt.imshow( util.get_DM_command_in_2D(phase_ctrl.config['M2C'].T[mode_indx],Nx_act=12))
    plt.title(f'example: mode {mode_indx} on DM') 
    plt.show()
    print( f'number of controlled modes = {phase_ctrl.config["number_of_controlled_modes"]}')


#init our pupil controller (object that processes ZWFS images and outputs VCM commands)
pupil_ctrl = pupil_control.pupil_controller_1(config_file = None)


# ========== PROCEEDURES ON INTERNAL SOURCE 
 
# 1.1) center source on DM 
pup_err_x, pup_err_y = pupil_ctrl.measure_dm_center_offset( zwfs, debug=True  )

#pupil_ctrl.move_pupil_relative( pup_err_x, pup_err_y ) 

# repeat until within threshold 

# 1.2) analyse pupil and decide if it is ok
pupil_report = pupil_control.analyse_pupil_openloop( zwfs, debug = True, return_report = True)

if pupil_report['pupil_quality_flag'] == 1: 
    # I think this needs to become attribute of ZWFS as the ZWFS object is always passed to pupil and phase control as an argunment to take pixtures and ctrl DM. The object controlling the camera should provide the info on where a controller object should look to apply control algorithm. otherwise pupil and phase controller would always need to talk to eachother. Also we will have 4 controllers in total

    zwfs.update_reference_regions_in_img( pupil_report )
    # this function just adds the following attributes 
    #zwfs.pupil_pixel_filter = pupil_report['pupil_pixel_filter']
    #zwfs.pupil_pixels = pupil_report['pupil_pixels']  
    #zwfs.pupil_center_ref_pixels = pupil_report['pupil_center_ref_pixels']
    #zwfs.dm_center_ref_pixels = pupil_report['dm_center_ref_pixels']

else:
    print('implement proceedure X1') 

"""
if debug:
    plt.figure() 
    r1,r2,c1,c2 = zwfs.pupil_crop_region
    plt.imshow( zwfs.pupil_pixel_filter.reshape( [14, 14] ) )
    plt.title(f'example: mode {mode_indx} on DM') 
    plt.show()
    print( f'number of controlled modes = {phase_ctrl.config["number_of_controlled_modes"]}')
"""


# 1.3) builds our control model with the zwfs
#control_model_report


ctrl_method_label = 'ctrl_1'
phase_ctrl.build_control_model( zwfs , poke_amp = -0.15, label=ctrl_method_label, debug = True)  
#pupil_ctrl tells phase_ctrl where the pupil is

# double check DM is flat 
zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] )


#update to WFS_Eigenmodes modes (DM modes that diagonalize the systems interaction matrix) 
phase_ctrl.change_control_basis_parameters( controller_label = ctrl_method_label, number_of_controlled_modes=phase_ctrl.config['number_of_controlled_modes'], basis_name='WFS_Eigenmodes' ,dm_control_diameter=None, dm_control_center=None)

# now build control model on KL modes 
phase_ctrl.build_control_model( zwfs , poke_amp = -0.15, label='ctrl_2', debug = True) 

plt.figure()
plt.imshow( util.get_DM_command_in_2D( phase_ctrl.config['M2C'].T[2] ) );plt.show()

if debug: 
    # put a mode on DM and reconstruct it with our CM 
    amp = -0.15
    mode_indx = 11
    mode_aberration = phase_ctrl.config['M2C'].T[mode_indx]
    
    dm_cmd_aber = zwfs.dm_shapes['flat_dm'] + amp * mode_aberration 
    zwfs.dm.send_data( dm_cmd_aber )
    time.sleep(0.1)
    raw_img_list = []
    for i in range( 10 ) :
        raw_img_list.append( zwfs.get_image() ) # @D, remember for control_phase method this needs to be flattened and filtered for pupil region
    raw_img = np.median( raw_img_list, axis = 0) 
    err_img = phase_ctrl.get_img_err( raw_img.reshape(-1)[zwfs.pupil_pixels]  ) 

    #NEED TO IMPLEMENT THIS TO STANDARDIZE phase_ctrl.get_error_intensity( raw_img.reshape(-1)[zwfs.pupil_pixels] ) 
    # amplitudes of modes sensed
    reco_modal_basis = phase_ctrl.control_phase( err_img  , controller_name = ctrl_method_label)

    M2C = phase_ctrl.config['M2C'] # readability 
    CM = phase_ctrl.ctrl_parameters[ctrl_method_label]['CM'] # readability 

    dm_cmd_reco_2D_cmdbasis = util.get_DM_command_in_2D( M2C @ reco_modal_basis ) 

    im_list = [util.get_DM_command_in_2D( mode_aberration ), raw_img.T/np.max(raw_img), dm_cmd_reco_2D_cmdbasis ]
    xlabel_list = [None, None, None]
    ylabel_list = [None, None, None]
    title_list = ['Aberration on DM', 'ZWFS Pupil', 'reconstructed DM cmd']
    cbar_label_list = ['DM command','Normalized intensity' , 'DM command' ] 
    savefig = None #fig_path + f'phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}_readout_mode-12x12.png'

    util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)

    del M2C, CM


if debug: # plot covariance of interaction matrix 
    #plt.title('Covariance of Interaciton Matrix')
    im_list = [np.cov( phase_ctrl.ctrl_parameters[ctrl_method_label]['IM'] ) / np.max( abs( np.cov( phase_ctrl.ctrl_parameters[ctrl_method_label]['IM'] ) ) ) ]
    xlabel_list = [f'{phase_ctrl.config["basis"]} mode index i']
    ylabel_list = [f'{phase_ctrl.config["basis"]} mode index j']
    title_list = [None]
    cbar_label_list = [r'$\sigma^2_{i,j}$']
    savefig =  None #fig_path + f'IM_covariance_matrix_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}_readout_mode-12x12.png'

    util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=False, cbar_orientation = 'right', savefig=savefig)



if debug: 
    im_list = [phase_ctrl.N0_2D/np.mean(phase_ctrl.N0 ), phase_ctrl.I0_2D/np.mean(phase_ctrl.N0 )]
    xlabel_list = ['x [pixels]','x [pixels]']
    ylabel_list = ['y [pixels]','y [pixels]']
    title_list = ['FPM OUT', 'FPM IN']
    cbar_label_list = ['Normalized pupil intensity','Normalized pupil intensity']
    savefig =  None #fig_path + f'pupil_FPM_IN-OUT_readout_mode-12x12.png'

    util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)



if debug: 
    im_list = [phase_ctrl.I0_2D/np.mean(phase_ctrl.N0 ) - phase_ctrl.N0_2D/np.mean(phase_ctrl.N0 ) ]
    xlabel_list = ['x [pixels]']
    ylabel_list = ['y [pixels]']
    title_list = ['FPM IN - FPM OUT']
    cbar_label_list = ['Normalized pupil intensity']
    savefig =  None #fig_path + f'pupil_FPM_IN-OUT_readout_mode-12x12.png'

    util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)



"""
# ---- if we were going to close the loop 
for i in range( 10 ) :
    raw_img_list.append( zwfs.get_image() ) # @D, remember for control_phase method this needs to be flattened and filtered for pupil region
raw_img = np.median( raw_img_list, axis = 0 ) 

reco_modes = phase_ctrl.control_phase( err_img  , controller_name = ctrl_method_label)
reco_dm_cmds = M2C @ mode_reco

delta_cmd.append( 0.9*delta_cmd[-1] + reco_dm_cmds )

zwfs.dm.send_data( dist[-1] + flat_dm + delta_cmd[-1] )
time.sleep(0.1)
"""

# --- Tune gains in closed loop

# 1.4) analyse pupil and decide if it is ok
if control_model_report.header['model_quality_flag']:
    # commit the model to the ZWFS attributes so it can SHOULD THE MODEL BE 
    
else: 
    print('implement proceedure X2')


zwfs.stop_camera()

"""
for i in range(10):
    # how to best deal with different phase controllers that require two images? 
    img = zwfs.get_image()

    cmd = phase_ctrl.process_img(img)

    zwfs.send_cmd(cmd)
    time.time.sleep(0.005)
"""
