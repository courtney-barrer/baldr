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
# look at the image for a second
util.watch_camera(zwfs)

#init our phase controller (object that processes ZWFS images and outputs DM commands)
phase_ctrl = phase_control.phase_controller_1(config_file = None) 

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
for mode_indx in range(1,50) :  

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

    err_img = phase_ctrl.get_img_err( raw_img.reshape(-1)[zwfs.pupil_pixel_filter]  ) 

    mode_res = CM.T @ err_img 

    #plt.figure(); plt.plot( mode_res ); plt.show()

    cmd_res = M2C @ mode_res
    
    time.sleep(0.01) 
    #reco.append( util.get_DM_command_in_2D( cmd_res ) )

    # =============== plotting 
    if mode_indx < 4:
    
        im_list = [util.get_DM_command_in_2D( mode_aberration ), raw_img.T/np.max(raw_img), util.get_DM_command_in_2D( cmd_res)  ]
        xlabel_list = [None, None, None]
        ylabel_list = [None, None, None]
        title_list = ['Aberration on DM', 'ZWFS Pupil', 'reconstructed DM cmd']
        cbar_label_list = ['DM command', 'Normalized intensity' , 'DM command' ] 
        savefig = None #fig_path + f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}_readout_mode-12x12.png'

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





# --------------------- JUST COPIED FROM closed_loop_modal_IM_static_v1.py
M2C = phase_ctrl.config['M2C']
CM = phase_ctrl.ctrl_parameters[ctrl_method_label]['CM']
# ======= init disturbance

dist_modes_idx = np.array([1,3,6]) 
dist_modes = np.zeros( M2C.shape[1] ) 
for i in dist_modes_idx:
    dist_modes[i] = np.random.randn()*0.08

disturbance_cmd = M2C @ dist_modes
"""
scrn_scaling_factor = 0.2
# --- create infinite phasescreen from aotools module 
Nx_act = dm.num_actuators_width()
screen_pixels = Nx_act*2**5 # some multiple of numer of actuators across DM 
D = 1.8 #m effective diameter of the telescope
scrn = aotools.infinitephasescreen.PhaseScreenVonKarman(nx_size=screen_pixels, pixel_scale=D/screen_pixels,r0=0.1,L0=12)

corner_indicies = [0, Nx_act-1, Nx_act * (Nx_act-1), -1] # Beware -1 index doesn't work if inserting in list! This is  ok for for use with create_phase_screen_cmd_for_DM function.

disturbance_cmd = cmd_region_filt * bdf.create_phase_screen_cmd_for_DM(scrn=scrn, DM=dm, flat_reference=flat_dm_cmd, scaling_factor = scrn_scaling_factor, drop_indicies = corner_indicies, plot_cmd=False)  # normalized flat_dm +- scaling_factor?

disturbance_cmd -= np.mean( disturbance_cmd ) # no piston!! 
"""

if 1:
    # for visualization get the 2D grid of the disturbance on DM  
    plt.figure()
    plt.imshow( util.get_DM_command_in_2D(disturbance_cmd, Nx_act=12) )
    plt.colorbar()
    plt.title( f'initial Kolmogorov aberration to apply to DM (rms = {round(np.std(disturbance_cmd),3)})')
    plt.show()






dt = 1/fps
Ku = 0.8 # ultimate gain to see stable oscillation in output 
Tu = 1e8 #4*dt # period of oscillations 
#apply Ziegler-Nicols methold of PI controller https://en.wikipedia.org/wiki/Ziegler%E2%80%93Nichols_method
Kp, Ki, Kd = 0.45 * Ku , 0.54*Ku/Tu , 0.* np.ones(len(flat_dm_cmd))
# NOTE: for Ki we scale zonal_gains**3 to avoid run-away amplitification of zonal modes on edge of illuminated DM 


# ======= Close loop
# init lists to hold data from control loop
flat_dm_cmd = zwfs.dm_shapes['flat_dm']

DIST_list = [ disturbance_cmd ]
IMG_list = [ ]
DELTA_list = [ ] #list( np.nan * np.zeros( int( (cp_x2 - cp_x1) * (cp_y2 - cp_y1) ) ) ) ]
MODE_ERR_list = [ ] #list( np.nan * flat_dm_cmd ) ]
MODE_ERR_PID_list = [ ]
CMD_ERR_list = [ list( np.zeros(len(flat_dm_cmd ))) ]
CMD_list = [ list( flat_dm_cmd ) ]
ERR_list = [ np.zeros(len(flat_dm_cmd )) ]# list( np.nan * np.zeros( int( (cp_x2 - cp_x1) * (cp_y2 - cp_y1) ) ) ) ]  # length depends on cropped pupil when flattened
RMS_list = [np.std(  disturbance_cmd )] # to hold std( cmd - aber ) for each iteration
 
zwfs.dm.send_data( flat_dm_cmd + disturbance_cmd )

time.sleep(1)

for i in range(50):

    # get new image and store it (save pupil and psf differently)
    IMG_list.append( list( np.median( zwfs.get_image() , axis=0)  ) )

    # create new error vector (remember to crop it!) with bdf.get_error_signal
    delta = phase_ctrl.get_img_err( raw_img.reshape(-1)[zwfs.pupil_pixel_filter]  )

    DELTA_list.append( delta.reshape(-1) )
    # CHECKS np.array(ERR_list[0]).shape = np.array(ERR_list[1]).shape = (cp_x2 - cp_x1) * (cp_y2 - cp_y1)

    mode_errs = list( CM.T @ DELTA_list[-1] )
    #reco_shift = reco[1:] + [np.median(reco)] # DONT KNOW WHY WE NEED THIS SHIFT!!!! ???
    #RECO_list.append( list( CM.T @ RES_list[-1] ) ) # reconstructed modal amplitudes
    MODE_ERR_list.append( mode_errs )
    
    # apply our PID on the modal basis 
    u =  Kp * np.array(MODE_ERR_list[-1]) +  Ki * dt * np.sum( MODE_ERR_list,axis=0 ) # PID 
   
    MODE_ERR_PID_list.append( u )   
    # to get error signal we apply modal gains
    cmd_errs =  M2C @ MODE_ERR_PID_list[-1] #np.sum( np.array([ a * B for a,B in  zip( MODE_ERR_list[-1], modal_basis)]) , axis=0) # this could be matrix multiplication too
    cmd_errs -= np.mean(cmd_errs) # REMOVE PISTON FORCEFULLY
    CMD_ERR_list.append( list(cmd_errs) )

    #list( np.sum( np.array([ Kp * a * B + Ki * dt * sum(err, axis) for g,a,B in  zip(modal_gains, RECO_list[-1], modal_basis)]) , axis=0) 

    CMD_list.append( flat_dm_cmd - CMD_ERR_list[-1] ) # update the previous command with our cmd error
    """
    # roll phase screen 
    for skip in range(rows_to_jump):
        scrn.add_row() 
    # get our new Kolmogorov disturbance command 
    disturbance_cmd = bdf.create_phase_screen_cmd_for_DM(scrn=scrn, DM=dm, flat_reference=flat_dm_cmd, scaling_factor = scrn_scaling_factor, drop_indicies = corner_indicies, plot_cmd=False)
    """
    DIST_list.append( disturbance_cmd )
    # we only calculate rms in our cmd region
    RMS_list.append( np.std(  ( np.array(CMD_list[-1]) - flat_dm_cmd + np.array(disturbance_cmd) ) ) )
   
    zwfs.dm.send_data( CMD_list[-1] + disturbance_cmd )

    time.sleep(0.01)

zwfs.dm.send_data(flat_dm_cmd)


#RMSE
plt.figure()
plt.plot(  RMS_list ,'.' )
plt.ylabel('RMSE cmd space [nm RMS]')





# first 5 iterations 
iterations2plot = [0,1,2,3,4,5,-2,-1]
fig, ax = plt.subplots( len(iterations2plot), 5 ,figsize=(10,15))
ax[0,1].set_title('disturbance',fontsize=15)
ax[0,0].set_title('ZWFS image',fontsize=15)
ax[0,2].set_title('CMD error (feedback)',fontsize=15)
ax[0,3].set_title('DM CMD (feedback)',fontsize=15)
ax[0,4].set_title('RESIDUAL (feedback)',fontsize=15)
for i,idx in enumerate(iterations2plot):
    ax[i,0].imshow( np.array(IMG_list) ) 
    im1 = ax[i,1].imshow( util.get_DM_command_in_2D(DIST_list[idx]) )
    plt.colorbar(im1, ax= ax[i,1])
    im2 = ax[i,2].imshow( util.get_DM_command_in_2D(CMD_ERR_list[idx]) )
    plt.colorbar(im2, ax= ax[i,2])
    im3 = ax[i,3].imshow( util.get_DM_command_in_2D(CMD_list[idx] ) )
    plt.colorbar(im3, ax= ax[i,3])
    im4 = ax[i,4].imshow( util.get_DM_command_in_2D(np.array(CMD_list[idx]) + np.array(DIST_list[idx]) - flat_dm_cmd) )
    plt.colorbar(im4, ax= ax[i,4])

plt.show() 













