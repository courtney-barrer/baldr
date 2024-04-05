from baldr_control import ZWFS
from baldr_control import phase_control
from baldr_control import pupil_control

fps = 400 
DIT = 2e-3 #s integration time 
pupil_crop_region = [140+21,140+129,90+11,90+119] #crop region around ZWFS pupil [row min, row max, col min, col max] 

#init our ZWFS (object that interacts with camera and DM)
zwfs = ZWFS.ZWFS(DM_serial_number='17DW019#053', cameraIndex=0, DMshapes_path = '/home/baldr/Documents/baldr/DMShapes/' ) 

zwfs.set_camera_fps(fps) # set the FPS 
zwfs.set_camera_dit(DIT) # set the DIT 

##
##    START CAMERA 
zwfs.start_camera()
# ----------------------

#init our phase controller (object that processes ZWFS images and outputs DM commands)
#phase_ctrl = phase_control.phase_contoller_1()

#init our pupil controller (object that processes ZWFS images and outputs VCM commands)
pupil_ctrl = pupil_control.pupil_controller_1(pupil_crop_region=pupil_crop_region, config_file = None)


# ========== PROCEEDURES ON INTERNAL SOURCE 
 
# 1.1) center source on DM 
pup_err_x, pup_err_y = pupil_ctrl.measure_dm_center_offset( zwfs, debug=True  )

pupil_ctrl.move_pupil_relative( pup_err_x, pup_err_y ) 

# repeat until within threshold 

# 1.2) analyse pupil and decide if it is ok
pupil_report = pupil_ctrl.analyse_pupil( zwfs, crop_region , return_report = True)

if pupil_report['pupil_quality_flag'] == 1:
    pupil_ctrl.set_pupil_reference_pixels( ) #measure and store pupil center pixels 
    pupil_ctrl.set_pupil_filter( )  # measure and store pixel indicies where the pupil is
else:
    print('implement proceedure X1') 


# 1.3) builds our control model with the zwfs
control_model_report = phase_ctrl.build_control_model( zwfs, pupil_ctrl, return_report = True)  
#pupil_ctrl tells phase_ctrl where the pupil is

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
