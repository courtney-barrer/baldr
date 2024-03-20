#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 12:35:33 2024

@author: bencb
"""

import os 
import datetime 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import glob
from math import ceil
from scipy import interpolate
from scipy.optimize import curve_fit
from astropy.io import fits
import aotools
os.chdir('/opt/FirstLightImaging/FliSdk/Python/demo/')
import FliSdk_V2 


root_path = '/home/baldr/Documents/baldr'
data_path = root_path + '/ANU_demo_scripts/ANU_data/' 
fig_path = root_path + '/figures/' 

os.chdir(root_path)
from functions import baldr_demo_functions as bdf

# NEED TO MAKE SURE RECO DATA COVERS +-0.2 amp range with around 20 samples! 

#==== OUR FUNCTIONS TO FIT TO GO FROM COMMAND TO INTENSITY 
def func(x, A, B, F, mu):
    I = A + B * np.cos(F * x + mu)
    return I 

def invfunc(I, A, B, F, mu):
    if abs( (I-A)/B ) < 1: 
        x = (np.arccos( (I-A)/B ) - mu) / F  #(np.sqrt( 2*( 1-(I-A)/B) ) - mu)/F  #
    else:
        x = valtmp = np.sign( (I-A)/B ) # round to +/-1
        #x = (np.arccos( valtmp ) - mu) / F

    #else: # linear estimator
    #    x = (np.pi/2 - (I-A)/B - mu ) / F #linear estimator
    return x

def reconstruct_delta_command( im , param_dict, pixel_indx_dict ):
    im = im.reshape(-1) # ensure flattened , note im should be same dimensions as input to model
    cmd = np.zeros(140)
    for i in range(140):
        if i in param_dict:
            
            I = im[ pixel_indx_dict[i] ]
            
            A_opt, B_opt, F_opt, mu_opt = param_dict[i]
            
            delta_c_i = invfunc(I, A_opt, B_opt, F_opt, mu_opt)
            
            cmd[i] = delta_c_i 
            
        else:
            cmd[i] = 0 # flat_cmd[i]
     
    return(cmd)
            

"""
-- setup 
1. setup files & parameters
2. init camera and DM 
3. Put static phase screen on DM 
4. Try correct it and record data  
-- 

-- to do
- seperate aqcuisition template, output is reference pupil and control model fits files 
- these can then be read in 

"""

# --- timestamp
tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
# --- 

# =====(1) 
# --- DM command to make flat DM (calibrated file provided by BMC with the DM) 
flat_dm_cmd = pd.read_csv(bdf.path_to_dm_flat_map,header=None)[0].values 
# read in DM deflection data and create interpolation functions 
deflection_data = pd.read_csv(root_path + "/DM_17DW019#053_deflection_data.csv", index_col=[0])
interp_deflection_1act = interpolate.interp1d( deflection_data['cmd'],deflection_data['1x1_act_deflection[nm]'] ) #cmd to nm deflection on DM from single actuator (from datasheet) 
interp_deflection_4x4act = interpolate.interp1d( deflection_data['cmd'],deflection_data['4x4_act_deflection[nm]'] ) #cmd to nm deflection on DM from 4x4 actuator (from datasheet) 

# --- read in recon file 
available_recon_pupil_files = glob.glob( data_path+'BDR_RECON_*.fits' )
available_recon_pupil_files.sort(key=os.path.getctime) # sort by most recent 
print('\n======\navailable AO reconstruction fits files:\n')
for f in available_recon_pupil_files:
    print( f ,'\n') 

recon_file = input('input file to use for interaction or control matricies. Input 0 if you want to create a new one')

if recon_file != '0':
    recon_data = fits.open( recon_file )
else:
    os.chdir(root_path + '/ANU_demo_scripts') 
    import ASG_BDR_RECON # run RECON script to calibrate matricies in open loop
    
    time.sleep(1) 

    new_available_recon_pupil_files = glob.glob( data_path+'BDR_RECON*.fits' )
    recon_file = max(new_available_recon_pupil_files, key=os.path.getctime) #latest_file
    recon_data = fits.open( recon_file ) 
    os.chdir(root_path)

print(f'\n\nusing:{recon_file}')
# in ASG_BDR_RECON we define way to initially crop image, we stored this in title
if 'cropping_corners_r1' in recon_data['poke_images'].header:
    r1 = int(recon_data['poke_images'].header['cropping_corners_r1'])
    r2 = int(recon_data['poke_images'].header['cropping_corners_r2'])
    c1 = int(recon_data['poke_images'].header['cropping_corners_c1'])
    c2 = int(recon_data['poke_images'].header['cropping_corners_c2'])
    cropping_corners = [r1,r2,c1,c2]
else:
    cropping_corners = None

print(f'=====\n\n\n=========hey\n\n======\n\ncropping corners = {cropping_corners}\n\n======') 

#--- create pixel intensity models and store in actuator keyed dictionaries

# pupil cropping coordinates
cp_x1,cp_x2,cp_y1,cp_y2 = int(recon_data[0].header['cp_x1']),int(recon_data[0].header['cp_x2']),int(recon_data[0].header['cp_y1']),int(recon_data[0].header['cp_y2'])
# PSF cropping coordinates 
ci_x1,ci_x2,ci_y1,ci_y2 = int(recon_data[0].header['ci_x1']),int(recon_data[0].header['ci_x2']),int(recon_data[0].header['ci_y1']),int(recon_data[0].header['ci_y2'])

# poke values used in linear ramp
No_ramps = int(recon_data['poke_images'].header['#ramp steps'])
max_ramp = float( recon_data['poke_images'].header['in-poke max amp'] )
min_ramp = float( recon_data['poke_images'].header['out-poke max amp'] ) 
ramp_values = np.linspace( min_ramp, max_ramp, No_ramps)

Nmodes_poked = 140 # int(recon_data[0].header['HIERARCH Nmodes_poked']) # can also see recon_data[0].header['RESHAPE']
Nact =  140 #int(recon_data[0].header['HIERARCH Nact'])  


pupil = recon_data['FPM_OUT'].data[ cp_x1 : cp_x2, cp_y1 : cp_y2]
P = np.sqrt( pupil ) # 
flat_img = recon_data['FPM_IN'].data[ cp_x1 : cp_x2, cp_y1 : cp_y2]

poke_cmds = recon_data['BASIS'].data
poke_imgs = recon_data['poke_images'].data[:,:, cp_x1 : cp_x2, cp_y1 : cp_y2]
poke_imgs = poke_imgs[1:].reshape(No_ramps, 140, flat_img.shape[0], flat_img.shape[1])

# ========================== !!!! =====================
#  == define the region of influence on DM where we correct (define a threshold for I(epsilon)_max -I_0) 

fig,ax= plt.subplots( 4, 4, figsize=(10,10))
num_pixels = []
candidate_thresholds = np.linspace(100,3000,16)
for axx, thresh in zip(ax.reshape(-1),candidate_thresholds):
    
    dm_pupil_filt = thresh < np.array( [np.max( abs( poke_imgs[len(poke_imgs)//2-2][i] - flat_img) ) for i in range(140)] ) 
    axx.imshow( bdf.get_DM_command_in_2D( dm_pupil_filt  ) ) 
    axx.set_title('threshold = {}'.format(round( thresh )),fontsize=12) 
    axx.axis('off')
    num_pixels.append(sum(dm_pupil_filt)) 
    # we could use this to automate threshold decision.. look for where 
    # d num_pixels/ d threshold ~ 0.. np.argmin( abs( np.diff( num_pixels ) )[:10])
plt.show()

recommended_threshold = candidate_thresholds[np.argmin( abs( np.diff( num_pixels ) )[:11]) + 1 ]
print( f'\n\nrecommended threshold ~ {round(recommended_threshold)} \n(check this makes sense with the graph by checking the colored area is stable around changes in threshold about this value)\n\n')

pupil_filt_threshold = float(input('input threshold of peak differences'))

dm_pupil_filt = pupil_filt_threshold < np.array( [np.max( abs( poke_imgs[len(poke_imgs)//2-1][i] - flat_img) ) for i in range(140)] ) 


param_dict = {}
pixel_indx_dict = {} 
cov_dict = {}
residuals=[]
mask = np.nan * np.ones(len(flat_img.reshape(-1))) 

# ======= FITTING MODELS FOR EACH FILTERED PIXEL
plot_fits = True
if plot_fits:
    Nrows = ceil( sum( dm_pupil_filt )**0.5)
    fig,ax = plt.subplots(Nrows,Nrows,figsize=(20,20))
    axx = ax.reshape(-1)
    j=0 #axx index
for act_indx in range(140): 
    if dm_pupil_filt[act_indx]:
        try:
            amp_indx = len(poke_imgs)//2-2 # CHOOSE THE AMPLITUDE TO USE FOR FINDING THE PEAKS! WORKS BEST FOR COMMANDS < FLAT DM WHERE WE HAVE BETTER LINERARITY 
            indx_at_peak = np.argmax( abs( poke_imgs[amp_indx][act_indx] - flat_img ) )
            pixel_indx_dict[act_indx] = indx_at_peak
            mask[indx_at_peak] = 1
            #plt.figure()
            #plt.imshow(poke_imgs[amp_indx][act_indx])
            
            I_v_ramp = np.array( [ poke_imgs[a][act_indx].reshape(-1)[indx_at_peak] for a in range(len(ramp_values)) ] )
            
            x_data = ramp_values.copy()[2:-2]
            y_data = I_v_ramp.copy()[2:-2]
            
            # Fit the curve to the data (set to median of distributions)
            initial_guess = [8918, -5773, 12, 1.7]  # Initial guess for parameters A, B, F, mu
            popt, pcov = curve_fit(func, x_data, y_data, p0=initial_guess)
            
            param_dict[act_indx] = popt
            cov_dict[act_indx] = pcov 
            Pi = P.reshape(-1)[indx_at_peak]

            # Extract the optimized parameters
            A_opt, B_opt, F_opt, mu_opt = popt
            
            # Print the optimized parameters
            print("Optimized parameters:")
            print("A:", A_opt)
            print("B:", B_opt)
            print("F:", F_opt)
            print("mu:", mu_opt)
            
            residuals.append( 1/(I_v_ramp / Pi**2) * (I_v_ramp / Pi**2 - func(ramp_values, A_opt, B_opt, F_opt, mu_opt)/ Pi**2 ) )
            if plot_fits:
                
                axx[j].plot( ramp_values, func(ramp_values, A_opt, B_opt, F_opt, mu_opt)/ Pi**2 ,label=f'fit (act{act_indx})') 
                axx[j].plot( ramp_values, I_v_ramp / Pi**2 ,label=f'measured (act{act_indx})' )
                axx[j].set_xlabel( 'normalized DM command')
                axx[j].set_ylabel( 'normalized Intensity')
                #axx[j].legend()
                ins = axx[j].inset_axes([0.15,0.15,0.25,0.25])
                ins.imshow(poke_imgs[3][act_indx] )
                j+=1
                """
                fig,ax = plt.subplots(1,2)
                ax[0].sact_indx = 0
amp_indx = 8
delta_c = reconstruct_delta_command( poke_imgs[amp_indx][act_indx], param_dict, pixel_indx_dict )
et_title(act_indx)
                ax[0].plot( ramp_values, func(ramp_values, A_opt, B_opt, F_opt, mu_opt)/ Pi**2 ,label='fit') 
                ax[0].plot( ramp_values, I_v_ramp / Pi**2 ,label='measured')
                ax[0].set_xlabel( 'normalized DM command')
                ax[0].set_ylabel( 'normalized Intensity')
                ax[0].legend()
                ax[1].imshow( poke_imgs[3][act_indx] )
                """
        except:
            print('failed for act', act_indx)
if plot_fits:
    plt.show()


# TEST AGAIN CAN WE RECONSTRUCT THE INPUT DATA
act_indx = 60
amp_indx = 5
delta_c = reconstruct_delta_command( poke_imgs[amp_indx][act_indx], param_dict, pixel_indx_dict )
plt.plot( delta_c );plt.show()

I_meas = np.array( [poke_imgs[a][act_indx].reshape(-1)[pixel_indx_dict[act_indx]] for a in range(len( poke_imgs))])
A_opt, B_opt, F_opt, mu_opt = param_dict[act_indx]
I_theory = func(ramp_values, A_opt, B_opt, F_opt, mu_opt)
plt.plot( ramp_values, I_meas ); plt.plot(ramp_values,I_theory)
plt.show() 


for i in pixel_indx_dict:
    delta_c = reconstruct_delta_command( poke_imgs[amp_indx][i], param_dict, pixel_indx_dict )
    plt.plot(delta_c)


# =====(2) 
# --- setup camera
fps = float( recon_data[1].header['camera_fps'] ) # WHY CAMERA FPS IN header [0] is zero????
camera = bdf.setup_camera(cameraIndex=0) #connect camera and init camera object
camera = bdf.set_fsp_dit( camera, fps=fps, tint=None) # set up initial frame rate, tint=None means max integration time for given FPS

# --- setup DM
dm, dm_err_code =  bdf.set_up_DM(DM_serial_number='17DW019#053')

# --- setup DM interaction and control matricies
modal_basis = recon_data['BASIS'].data #
# check modal basis dimensions are correct
if modal_basis.shape[1] != 140:
    raise TypeError( 'modal_basis.shape[1] != 140 => not right shape. Maybe we should transpose it?\nmodal_basis[i] should have a 140 length command for the DM corresponding to mode i')

# =====(3)
# set up parameters 

number_images_recorded_per_cmd = 5 #NDITs to take median over 

save_fits = data_path + f'closed_loop_on_static_aberration_disturb-kolmogorov_t-{tstamp}.fits'


#corner_indicies = [0, Nx_act-1, Nx_act * (Nx_act-1), -1] # Beware -1 index doesn't work if inserting in list! This is  ok for for use with create_phase_screen_cmd_for_DM function.

# NOTE WE ONLY APPLY DISTURVBANCE WITHIN DM INFLUENCE REGION (using dm_pupil_filt)

#disturbance_cmd = np.zeros( len( flat_dm_cmd ))  
#disturbance_cmd[64] = 0.02

modes = bdf.construct_command_basis(dm , basis='Zernike', number_of_modes = 20, actuators_across_diam = 'full',flat_map=None)

mode_keys = list(modes.keys())

disturbance_cmd = 0.4 * ( flat_dm_cmd - modes[mode_keys[10]] ) 


#disturbance_cmd[np.array([5,16,28,40,52,64])]=0.06
#disturbance_cmd += flat_dm_cmd.copy()



# for visualization get the 2D grid of the disturbance on DM  
plt.figure()
plt.imshow( bdf.get_DM_command_in_2D(disturbance_cmd, Nx_act=12) )
plt.colorbar()
plt.title( 'static aberration to apply to DM')
plt.show()
#plt.close() 



# ======================= LETS GO! =====

static_ab_performance_fits = fits.HDUList([])


# init lists to hold data from control loop

IMG_list = [ ]
CMD_list = [ list( flat_dm_cmd ) ] 
ERR_list = [ ]# list( np.nan * np.zeros( int( (cp_x2 - cp_x1) * (cp_y2 - cp_y1) ) ) ) ]  # length depends on cropped pupil when flattened 
RMS_list = [ np.std( - disturbance_cmd ) ] # to hold std( cmd - aber ) for each iteration
 

# start with flat DM
dm.send_data( flat_dm_cmd + disturbance_cmd)

FliSdk_V2.Start(camera)    
time.sleep(1)

#===========================

for i in range(10): 

    IMG_list.append( np.median( bdf.get_raw_images(camera, number_of_frames=number_images_recorded_per_cmd, cropping_corners=cropping_corners) , axis=0) ) 

    delta_c = reconstruct_delta_command( IMG_list[-1][cp_x1:cp_x2, cp_y1:cp_y2], param_dict, pixel_indx_dict )
    
    plt.figure();plt.plot(delta_c); plt.plot(disturbance_cmd,label='dist');plt.legend();plt.show()

    ERR_list.append( delta_c ) 
    
    CMD_list.append( list(flat_dm_cmd) - delta_c  )

    dm.send_data(  CMD_list[-1] + disturbance_cmd ) 
    # record RMS in command space 
    RMS_list.append( np.std( delta_c - disturbance_cmd ) )

    time.sleep(0.05)



fig,ax = plt.subplots(1,2); 
im0=ax[0].imshow( bdf.get_DM_command_in_2D( ERR_list[0] ) ) ;plt.colorbar(im0,ax=ax[0]); im1=ax[1].imshow( bdf.get_DM_command_in_2D( disturbance_cmd) ) ;plt.colorbar(im1,ax=ax[1]); 
plt.show()

# now flatten once finished
dm.send_data( flat_dm_cmd ) 


camera_info_dict = bdf.get_camera_info( camera )

FliSdk_V2.Stop(camera)


disturbfits = fits.PrimaryHDU( disturbance_cmd )
disturbfits.header.set('EXTNAME','DISTURBANCE')
disturbfits.header.set('WHAT_IS','disturbance in cmd space')

IMG_fits = fits.PrimaryHDU( IMG_list )
IMG_fits.header.set('EXTNAME','IMAGES')
IMG_fits.header.set('recon_fname',recon_file.split('/')[-1])

for k,v in camera_info_dict.items(): 
    IMG_fits.header.set(k,v)   # add in some fits headers about the camera 

for i,n in zip([ci_x1,ci_x2,ci_y1,ci_y2],['ci_x1','ci_x2','ci_y1','ci_y2']):
    IMG_fits.header.set(n,i)

for i,n in zip([cp_x1,cp_x2,cp_y1,cp_y2],['cp_x1','cp_x2','cp_y1','cp_y2']):
    IMG_fits.header.set(n,i)

CMD_fits = fits.PrimaryHDU( CMD_list )
CMD_fits.header.set('EXTNAME','CMDS')
CMD_fits.header.set('WHAT_IS','DM commands')

ERR_fits = fits.PrimaryHDU( ERR_list )
ERR_fits.header.set('EXTNAME','ERRS')
ERR_fits.header.set('WHAT_IS','list of modal errors to feed to PID')

RMS_fits = fits.PrimaryHDU( RMS_list )
RMS_fits.header.set('EXTNAME','RMS')
RMS_fits.header.set('WHAT_IS','std( cmd - aber_in_cmd_space )')

# add these all as fits extensions 
for f in [disturbfits, IMG_fits, ERR_fits, CMD_fits, RMS_fits ]: #[Ufits, Sfits, Vtfits, CMfits, disturbfits, IMG_fits, ERR_fits, RECO_fits, CMD_fits, RMS_fits ]:
    static_ab_performance_fits.append( f ) 

#save data! 
static_ab_performance_fits.writeto( save_fits )



