 
"""
Testing pupil control offset

set up ZWFS , pupil controller with classified regions 

for range 
    move pupil
    measure image filtered for secondary peak field 
    measure and record cuad diode photometric center  
    repeat


# questions 
- with noisy atm how does std in peak ref field change with number of images taken
- sensitivity 

"""

from baldr_control import ZWFS
from baldr_control import phase_control
from baldr_control import pupil_control
from baldr_control import utilities as util

import numpy as np
import matplotlib.pyplot as plt 
import time 
import datetime
from astropy.io import fits
import pandas as pd

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

# put four torres on DM for centering before hand and watch 
#zwfs.dm.send_data(0.2* zwfs.dm_shapes['four_torres_2'] ) 
#util.watch_camera(zwfs, frames_to_watch = 10, time_between_frames=0.01,cropping_corners=None)
#zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] ) 

#init our pupil controller (object that processes ZWFS images and outputs VCM commands)
pupil_ctrl = pupil_control.pupil_controller_1(config_file = None)

pupil_report = pupil_control.analyse_pupil_openloop( zwfs, debug = True, return_report = True)

# update zwfs with control regions 
zwfs.update_reference_regions_in_img( pupil_report ) 





#get reference at current position (zero offset)
delta_im = (abs(zwfs.get_image() - zwfs.N0))
quad = delta_im.reshape(-1)[zwfs.refpeak_pixel_filter].reshape(2,2)

ex_ref = np.diff(np.sum(  quad, axis=0) )[0] 
ey_ref = np.diff(np.sum(  quad, axis=1) )[0] 

#init things for our expierment 
x= np.array( [-1,1] )
y = np.array( [-1,1] )
ex_list = []
ey_list = [] 

plt.figure() 
for i in range(10):

    delta_im_list = []
    for i in range(50):    
        delta_im_list.append( abs(zwfs.get_image() - zwfs.N0) )
        time.sleep(0.005)

    delta_im = np.median( delta_im_list ,axis=0 ) 

    plt.plot( delta_im[len(delta_im)//2, : ] , label=f'{i}')
    plt.plot( delta_im[:,len(delta_im)//2] )

    # get 4 pixels around classified reference field peak 
    quad = delta_im.reshape(-1)[zwfs.refpeak_pixel_filter].reshape(2,2)

    ex = np.diff(np.sum(  quad, axis=0) )[0] #np.sum( x * quad, axis=0)/np.sum( quad ,axis=0) 
    ey = np.diff(np.sum(  quad, axis=1) )[0]#np.sum( y * quad, axis=1)/np.sum( quad ,axis=1)

    ex_list.append( ex - ex_ref )
    ey_list.append( ey - ey_ref)


    _ = input('move pupil 1 tick, press enter when ready ')
plt.show() 

#want to save images to! keep I, N0, filter to create movie?

# we moved it 1mm over 10 0.05mm steps with a 4mm pupil diameter
offsets = np.linspace(0, 1/4, len(ex_list) )

plt.figure()
plt.plot( offsets , ex_list ,label='x error') 
plt.plot(offsets,  ey_list ,label='y error') 
plt.legend()
plt.xlabel('pupil offset [%]')
plt.ylabel('error cmd')

plt.savefig(fig_path + f'first_pupil_ctrl_on_secondary_f{tstamp}.png', bbox_inches='tight', dpi=300) 
plt.show()

df = pd.DataFrame( np.array([offsets, ex_list, ey_list]).T, columns = ['offset[pc]', 'err_x', 'err_y'] )
df.to_csv(data_path + f'first_pupil_ctrl_on_secondary_f{tstamp}.csv')









