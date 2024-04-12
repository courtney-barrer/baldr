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
pupil_crop_region = [165-sw, 261+sw, 106-sw, 202+sw ] #one pixel each side of pupil.  #tight->[165, 261, 106, 202 ]  #crop region around ZWFS pupil [row min, row max, col min, col max] 

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

#control Zernike modes 
phase_ctrl_Z_70 = phase_control.phase_controller_1(config_file = None) 
#phase_ctrl_Z_20 = phase_control.phase_controller_1(config_file = None) 
#phase_ctrl_Z_20.change_control_basis_parameters(number_of_controlled_modes = 20, basis_name = 'Zernike', dm_control_diameter=None, dm_control_center=None, controller_label=None)

# control KL modes
phase_ctrl_KL_70 = phase_control.phase_controller_1(config_file = None) 
phase_ctrl_KL_70.change_control_basis_parameters(number_of_controlled_modes = 70, basis_name = 'KL', dm_control_diameter=None, dm_control_center=None, controller_label=None)

# have a look at one of the modes on the DM. Modes are normalized in cmd space <m|m> = 1
if debug:
    mode_indx = 0
    plt.figure() 
    plt.imshow( util.get_DM_command_in_2D(phase_ctrl_KL_70.config['M2C'].T[mode_indx],Nx_act=12))
    plt.title(f'example: mode {mode_indx} on DM') 
    plt.show()
    #print( f'number of controlled modes = {phase_ctrl.config["number_of_controlled_modes"]}')


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


ctrl_method_label = 'Zernike'
phase_ctrl_Z_70.build_control_model( zwfs , poke_amp = -0.15, label='Zernike_70', debug = True)  
# double check DM is flat 
zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] )

#phase_ctrl_Z_20.build_control_model( zwfs , poke_amp = -0.15, label='Zernike_20', debug = True)  

# double check DM is flat 
#zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] )

phase_ctrl_KL_70.build_control_model( zwfs , poke_amp = -0.15, label='KL_70', debug = True) 
# double check DM is flat 
zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] )

#phase_ctrl_KL_20.build_control_model( zwfs , poke_amp = -0.15, label='KL_20', debug = True) 


# double check DM is flat 
zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] )



phase_ctrl = phase_ctrl_Z_70
ctrl_method_label = 'Zernike_70'
if debug: # plot covariance of interaction matrix 
    #plt.title('Covariance of Interaciton Matrix')
    im_list = [np.cov( phase_ctrl.ctrl_parameters[ctrl_method_label]['IM'] ) / np.max( abs( np.cov( phase_ctrl.ctrl_parameters[ctrl_method_label]['IM'] ) ) ) ]
    xlabel_list = [f'{phase_ctrl.config["basis"]} mode index i']
    ylabel_list = [f'{phase_ctrl.config["basis"]} mode index j']
    title_list = [None]
    cbar_label_list = [r'$\sigma^2_{i,j}$']
    savefig = None #fig_path + f'IM_covariance_matrix_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}_readout_mode-6x6.png'

    util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=False, cbar_orientation = 'right', savefig=savefig)




#update to Eigenmodes 
phase_ctrl_Z_70.change_control_basis_parameters(  number_of_controlled_modes=phase_ctrl.config['number_of_controlled_modes'], basis_name='WFS_Eigenmodes' ,dm_control_diameter=None, dm_control_center=None, controller_label = 'Zernike_70')

# now build control model on eigenmodes modes 
phase_ctrl_Z_70.build_control_model( zwfs , poke_amp = -0.15, label='WFS_Eigenmodes_70', debug = True) 

# double check DM is flat 
zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] )

phase_ctrl = phase_ctrl_Z_70
ctrl_method_label = 'WFS_Eigenmodes_70'
if debug: # plot covariance of interaction matrix 
    #plt.title('Covariance of Interaciton Matrix')
    im_list = [np.cov( phase_ctrl.ctrl_parameters[ctrl_method_label]['IM'] ) / np.max( abs( np.cov( phase_ctrl.ctrl_parameters[ctrl_method_label]['IM'] ) ) ) ]
    xlabel_list = [f'{phase_ctrl.config["basis"]} mode index i']
    ylabel_list = [f'{phase_ctrl.config["basis"]} mode index j']
    title_list = [None]
    cbar_label_list = [r'$\sigma^2_{i,j}$']
    savefig =  None #fig_path + f'IM_covariance_matrix_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}_readout_mode-6x6.png'

    util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=False, cbar_orientation = 'right', savefig=savefig)


# Now look at eigenvalues of IM Eigenmodes to see how many modes we are really sensitive to in 10x10 actuators on DM in 12x12 readout. 
cov = np.cov( phase_ctrl.ctrl_parameters['WFS_Eigenmodes_70']['IM'] ) 
U, S, UT = np.linalg.svd( cov ) 

fig, ax = plt.subplots(1,1,figsize=(8,5))
fs = 15
ax.semilogy( S/np.max(S) )
ax.set_ylabel('WFS Eigenvalues',fontsize=fs)
ax.set_xlabel('Mode index',fontsize=fs) 
ax.tick_params(labelsize=fs)
savefig =  None #fig_path + 'Eigenvalues_WFS_eigenbasis_ctrl_modes-70_readout_mode-6x6.png'
#plt.savefig( savefig , bbox_inches='tight', dpi=300) 

plt.show() 

"""
# write to csv 
import pandas as pd
pd.Series( S ).to_csv( data_path + 'WFS_eigenvalues_12x12.csv' , index=False, header=False)
pd.DataFrame( phase_ctrl.config['M2C'] ).to_csv(data_path + 'WFS_eigenvectors_12x12.csv', index=False,header=False)
"""
S_6x6 = pd.read_csv( data_path + 'WFS_eigenvalues_6x6.csv',header=None).values.ravel()
S_12x12 = pd.read_csv( data_path + 'WFS_eigenvalues_12x12.csv',header=None).values.ravel()

fig, ax = plt.subplots(1,1,figsize=(8,5))
fs = 15
ax.semilogy( S_12x12/np.max(S_12x12) , label='12x12 pixels/pupil')
ax.semilogy( S_6x6/np.max(S_6x6) , label='6x6 pixels/pupil')
ax.set_ylabel('WFS Eigenvalues (normalized)',fontsize=fs)
ax.set_xlabel('Mode index',fontsize=fs) 
ax.tick_params(labelsize=fs)
savefig =  fig_path + 'Eigenvalues_WFS_eigenbasis_READOUT_COMPARISON_ctrl_modes-70.png'
#plt.savefig( savefig , bbox_inches='tight', dpi=300) 
plt.legend(fontsize=fs)
plt.show() 



 



