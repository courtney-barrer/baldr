#

from baldr_control import ZWFS
from baldr_control import phase_control
from baldr_control import pupil_control
from baldr_control import utilities as util

import numpy as np
import matplotlib.pyplot as plt 
import time 

import numpy as np
from scipy.optimize import curve_fit
import scipy.special
import pandas as pd 

def airy_disk_model(xy, A, a, B, e):
    """
    Airy disk model function.
    
    Parameters:
        r (float or numpy array): Radial distance from the center.
        a (float): Amplitude parameter.
        b (float): Radius parameter.
        
    Returns:
        float or numpy array: Intensity at radial distance r.
    """
    x,y = xy
    r = np.sqrt(x**2 + y**2)
    
    airy = A/(1-e**2)**2 * (2 * (scipy.special.j1(a*r) - e * scipy.special.j1(e * a * r) )/ (a*r) )**2 + B
    #idx = np.where( ~np.isfinite(airy) )[0]
    #for i in idx:
    #    airy.reshape(-1)[i] = airy.reshape(-1)[i+1] # just use the next value (shitty but good for now)
    return (airy)




def fit_airy_disk(xy, z):
    """
    Fit an Airy disk to x, y, z data.
    
    Parameters:
        xy (tuple): Tuple containing x and y coordinates.
        z (array-like): Intensity values.
        
    Returns:
        tuple: Optimal parameters (A, a, B, e) for the Airy disk model.
    """
    
    x, y = xy
    # Convert x, y to radial distance from the center
    r = np.sqrt(x**2 + y**2)
    
    # Initial guess for parameters
    initial_guess = [np.max(z), np.max(1/r), 0.0, 0.6] # airydisk peak (A), airydisk radius (a), airydisk offset (B), central obstruction ratio (e).
    
    # Fit the model to the data
    params, _ = curve_fit(airy_disk_model, xy , z, p0=initial_guess)
    
    return params





fig_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/figures/' 
data_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/data/' 


debug = True # plot some intermediate results 
fps = 400 
DIT = 2e-3 #s integration time 

sw = 16 # 8 for 12x12, 16 for 6x6 
pupil_crop_region = [157-4*8-sw, 269+4*8+sw, 98-4*8-sw, 210+4*8+sw ] #[157-sw, 269+sw, 98-sw, 210+sw ] #[165-sw, 261+sw, 106-sw, 202+sw ] #one pixel each side of pupil.  #tight->[165, 261, 106, 202 ]  #crop region around ZWFS pupil [row min, row max, col min, col max] 

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

#init our pupil controller (object that processes ZWFS images and outputs VCM commands)
pupil_ctrl = pupil_control.pupil_controller_1(config_file = None)


# ========== PROCEEDURES ON INTERNAL SOURCE 
 
# 1.1) center source on DM 
pup_err_x, pup_err_y = pupil_ctrl.measure_dm_center_offset( zwfs, debug=False  )


# 1.2) analyse pupil and decide if it is ok
pupil_report = pupil_control.analyse_pupil_openloop( zwfs, debug = True, return_report = True)

if pupil_report['pupil_quality_flag'] == 1: 
    # I think this needs to become attribute of ZWFS as the ZWFS object is always passed to pupil and phase control as an argunment to take pixtures and ctrl DM. The object controlling the camera should provide the info on where a controller object should look to apply control algorithm. otherwise pupil and phase controller would always need to talk to eachother. Also we will have 4 controllers in total

    zwfs.update_reference_regions_in_img( pupil_report )


else:
    print('implement proceedure X1') 



#init our phase controller (object that processes ZWFS images and outputs DM commands)
phase_ctrl = phase_control.phase_controller_1(config_file = None) 

phase_ctrl.measure_FPM_OUT_reference(zwfs)

phase_ctrl.measure_FPM_IN_reference(zwfs) 

Z = ((phase_ctrl.I0_2D - phase_ctrl.N0_2D)/phase_ctrl.N0_2D )

plt.figure()
plt.pcolormesh( zwfs.row_coords, zwfs.col_coords, Z ) ;
plt.show()

plt.figure()
plt.plot( zwfs.row_coords, Z[len(phase_ctrl.I0_2D)//2,:] , label = 'measured') 

X,Y = np.meshgrid( zwfs.row_coords,zwfs.row_coords )
plt.plot( zwfs.row_coords, airy_disk_model( (X,Y), 1, 1.1, 0.,0.6)[len(X)//2,:] , label = 'fit'); plt.show()
plt.legend()


params = fit_airy_disk((X, Y) ,  Z )

label = "5loD_coldstop_6x6"
pd.DataFrame( phase_ctrl.I0_2D ).to_csv( data_path +f'I0_2D_{label}.csv', header=None, index=False)
pd.DataFrame( phase_ctrl.N0_2D ).to_csv( data_path +f'N0_2D_{label}.csv', header=None, index=False)
pd.DataFrame( Z ).to_csv( data_path +f'I0-N0overN0_{label}.csv', header=None, index=False)


im_list = [phase_ctrl.N0_2D/np.max(phase_ctrl.N0_2D), phase_ctrl.I0_2D/np.max(phase_ctrl.I0_2D), Z ]
xlabel_list = [r'$f\lambda/D=5$ coldstop',r'$f\lambda/D=5$ coldstop',r'$f\lambda/D=5$ coldstop']
ylabel_list = ['', '', '']
title_list = [r'FPM OUT','FPM IN',r'$\Delta$']
cbar_label_list = ['Normalized Intensity','Normalized Intensity',r'$\frac{IN - OUT}{OUT}$']
savefig = None # fig_path + f'pupil_intensity_{label}'
util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)


# Check int (|psi_C|^2 - |psi_A|^2 - bias)  = 0
np.sum( Z - np.mean( Z[:20,:20] ) ) 




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


Z = ((phase_ctrl.I0_2D - phase_ctrl.N0_2D)/phase_ctrl.N0_2D )**2 

plt.figure()
plt.pcolormesh( zwfs.row_coords, zwfs.col_coords, Z ) ;
plt.show()

plt.figure()
plt.plot( zwfs.row_coords, Z[len(phase_ctrl.I0_2D)//2,:] , label = 'measured') 

X,Y = np.meshgrid( zwfs.row_coords,zwfs.row_coords )
plt.plot( zwfs.row_coords, airy_disk_model( (X,Y), 1, 1.1, 0.,0.6)[len(X)//2,:] , label = 'fit'); plt.show()
plt.legend()


params = fit_airy_disk((X, Y) ,  Z )


#pd.DataFrame( phase_ctrl.I0_2D ).to_csv( data_path +'I0_2D_16-4-24.csv', header=None, index=False)
#pd.DataFrame( phase_ctrl.N0_2D ).to_csv( data_path +'N0_2D_16-4-24.csv', header=None, index=False)
#pd.DataFrame( Z ).to_csv( data_path +'I0-N0overN0.csv', header=None, index=False)









