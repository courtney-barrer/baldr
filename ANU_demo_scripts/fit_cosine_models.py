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


debug = True # generate plots to help debug

# --- timestamp
tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
# --- 


#==== OUR FUNCTIONS TO FIT TO GO FROM COMMAND TO INTENSITY 

def taylor_series_arccos(x, mu=0):
    """
    Taylor Series approximation of arcos around mu to 5th order
    """
    den = np.sqrt(1-mu**2)
    res = np.arccos(mu) - (x-mu)/den - mu*(x-mu)**2/(2*den**3)- (2*mu**2+1)*(x-mu)**3/(6*den**5)- mu*(2*mu**2+3)*(x-mu)**4/(8*den**7) - (8*mu**4+24*mu**2+3)*(x-mu)**5/(40*den**9)

    """
    to test
    x = np.linspace(-1.3,1.3,100)
    plt.plot( x,taylor_series_arccos(x),label='approximation'); 
    plt.plot(x[abs(x)<1],np.arccos(x[abs(x)<1]),label='real')
    plt.legend(); plt.show()
    """
    return(res) 



def func(x, A, B, F, mu):
    I = A + B * np.cos(F * x + mu)
    return I 

def invfunc(I, A, B, F, mu):

    if abs( (I-A)/B ) < 1: 
        x = (np.arccos( (I-A)/B ) - mu) / F
    else:
        x = ( taylor_series_arccos( (I-A)/B ) - mu ) / F

    return x


def reconstruct_delta_command( S , param_dict ):
    # ensure S is flattened , note im should be same dimensions as input to model
    cmd = np.zeros(140)
    for i in range(140):
        if i in param_dict:
            
            A_opt, B_opt, F_opt, mu_opt = param_dict[i]
            

            delta_c_i = invfunc(S[i], A_opt, B_opt, F_opt, mu_opt)
            cmd[i] = delta_c_i 
            
        else:
            cmd[i] = 0 # flat_cmd[i]
     
    return(cmd)
            

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


# =====(2) 
# --- setup camera
fps = float( recon_data[1].header['camera_fps'] ) # WHY CAMERA FPS IN header [0] is zero????
camera = bdf.setup_camera(cameraIndex=0) #connect camera and init camera object
camera = bdf.set_fsp_dit( camera, fps=fps, tint=None) # set up initial frame rate, tint=None means max integration time for given FPS

# --- setup DM
dm, dm_err_code =  bdf.set_up_DM(DM_serial_number='17DW019#053')


# ========================== !!!! =====================
#  == define the region of influence on DM where we correct (define a threshold for I(epsilon)_max -I_0) 

i0 = len(ramp_values)//2 - 1 # which poke values do we want to consider for finding region of influence. Pick a value near the center where in a linear regime. 

fig,ax= plt.subplots( 4, 4, figsize=(10,10))
num_pixels = []
candidate_thresholds = np.linspace(100,2000,16)
for axx, thresh in zip(ax.reshape(-1),candidate_thresholds):
    
    dm_pupil_filt = thresh < np.array( [np.max( abs( poke_imgs[i0][act] - flat_img) ) for act in range(140)] ) 
    axx.imshow( bdf.get_DM_command_in_2D( dm_pupil_filt ) ) 
    axx.set_title('threshold = {}'.format(round( thresh )),fontsize=12) 
    axx.axis('off')
    num_pixels.append(sum(dm_pupil_filt)) 
    # we could use this to automate threshold decision.. look for where 
    # d num_pixels/ d threshold ~ 0.. np.argmin( abs( np.diff( num_pixels ) )[:10])
plt.show()

recommended_threshold = candidate_thresholds[np.argmin( abs( np.diff( num_pixels ) )[:11]) + 1 ]
print( f'\n\nrecommended threshold ~ {round(recommended_threshold)} \n(check this makes sense with the graph by checking the colored area is stable around changes in threshold about this value)\n\n')

pupil_filt_threshold = float(input('input threshold of peak differences'))

dm_pupil_filt = pupil_filt_threshold < np.array( [np.max( abs( poke_imgs[i0][act] - flat_img) ) for act in range(140)] ) 

if debug:
   plt.figure()
   plt.imshow( bdf.get_DM_command_in_2D( dm_pupil_filt ) )
   plt.title('influence region on DM where we will fit intensity models per actuator')
   plt.show()
# ========================== !!!! =====================
# want a mask that maps [Px x Py] array to [Sw x Nact] where Sw is a subwindow of Sx x Sy pixels centered around a given actuators peak region of influence. We then sum over Sw dimension to get Nact array which we fit our model to for each actuator. 

Sw_x, Sw_y = 3,3 #+- pixels taken around region of peak influence. PICK ODD NUMBERS SO WELL CENTERED!   
act_img_mask = {}
act_flag = {}
act_img_idx = {}
#act2pix_idx = []


for act_idx in range(len(flat_dm_cmd)):
    delta =  poke_imgs[i0][act_idx]-flat_img 

    mask = np.zeros( flat_img.shape )
   
    if dm_pupil_filt[act_idx]:
        i,j = np.unravel_index( np.argmax( abs(delta) ),flat_img.shape )

        #act2pix_idx.append( (i,j) ) 
        mask[i-Sw_x-1: i+Sw_x, j-Sw_y-1:j+Sw_y] = 1 # keep centered, normalize by #pixels in window 
        mask *= 1/np.sum(mask[i-Sw_x-1: i+Sw_x, j-Sw_y-1:j+Sw_y])
        act_img_mask[act_idx] = mask 
        act_img_idx[act_idx] = (i,j)  
        act_flag[act_idx] = 1 
    else :
        act_img_mask[act_idx] = mask 
        act_flag[act_idx] = 0 
if debug:
    plt.title('masked regions of influence per actuator')
    plt.imshow( np.sum( list(act_img_mask.values()), axis = 0 ) )
    plt.show()

# turn our dictionary to a big matrix 
mask_matrix = np.array([list(act_img_mask[act_idx].reshape(-1)) for act_idx in range(140)])


# now lets plot sum of intensities for each poke-pull amplitude in the respecitve actuator subwindow
if debug:
    act_idx = 66
    plt.plot(ramp_values, [np.sum( act_img_mask[act_idx] * poke_imgs[i][act_idx]) for i in range(len(ramp_values))] ) 
    plt.xlabel('DM cmd aberration');plt.ylabel('mean windowed intensity')
    plt.show()



# ======= FITTING MODELS FOR EACH FILTERED PIXEL


param_dict = {}
cov_dict = {}
fit_residuals = []

if debug:
    Nrows = ceil( sum( dm_pupil_filt )**0.5)
    fig,ax = plt.subplots(Nrows,Nrows,figsize=(20,20))
    axx = ax.reshape(-1)
    j=0 #axx index


mean_filtered_pupil = np.mean( mask_matrix @ pupil.reshape(-1) )

for act_idx in range(len(flat_dm_cmd)): 
    if dm_pupil_filt[act_idx]:
        # -- we do this with matrix multiplication using  mask_matrix
        #P_i = np.sum( act_img_mask[act_idx] * pupil ) #Flat DM with FPM OUT 
        P_i = mean_filtered_pupil.copy() # just consider mean pupil! 
        
        I_i = np.array( [np.sum( act_img_mask[act_idx] * poke_imgs[i][act_idx] ) for i in range(len(ramp_values))] ) #spatially filtered sum of intensities per actuator cmds 
        I_0 = np.sum( act_img_mask[act_idx] * flat_img ) # Flat DM with FPM IN  
        
        # ================================
        #   THIS IS OUR MODEL!! S=A+B*cos(F*x + mu)  
        S = (I_i - I_0) / P_i # signal to fit!
        # ================================

        #re-label and filter to capture best linear range 
        x_data = ramp_values[2:-2].copy()
        y_data = S[2:-2].copy()

        initial_guess = [0.5, 0.5, 15, 2.4]  #A_opt, B_opt, F_opt, mu_opt  ( S = A+B*cos(F*x + mu) )
        # FIT 
        popt, pcov = curve_fit(func, x_data, y_data, p0=initial_guess)

        # Extract the optimized parameters explictly to measure residuals
        A_opt, B_opt, F_opt, mu_opt = popt

        # STORE FITS 
        param_dict[act_idx] = popt
        cov_dict[act_idx] = pcov 
        # also record fit residuals 
        fit_residuals.append( S - func(ramp_values, A_opt, B_opt, F_opt, mu_opt) )


        if debug: 
            # Print the optimized parameters
            #print(f"Optimized parameters for act {act_idx}:")
            #print("A:", A_opt)
            #print("B:", B_opt)
            #print("F:", F_opt)
            #print("mu:", mu_opt)

            axx[j].plot( ramp_values, func(ramp_values, A_opt, B_opt, F_opt, mu_opt) ,label=f'fit (act{act_idx})') 
            axx[j].plot( ramp_values, S ,label=f'measured (act{act_idx})' )
            #axx[j].set_xlabel( 'normalized DM command')
            #axx[j].set_ylabel( 'normalized Intensity')
            #axx[j].legend()
            axx[j].set_title(act_idx)
            ins = axx[j].inset_axes([0.15,0.15,0.25,0.25])
            ins.imshow(poke_imgs[3][act_idx] )
            j+=1

if debug:
    plt.figure()
    plt.title('histogram of mean fit residuals')
    plt.hist( np.mean( fit_residuals, axis=1) ) 
    plt.show() 



# convert act_img_mask[act_idx] to matrix 


# CREATE IM ( S = IM @ delta_c ) to compare results to sinusoid fits
IM = []
for act_idx in range(140):
    print(f'executing cmd {act_idx}/140')
    delta_c = np.zeros(140)
    delta_c[act_idx] = -0.03
    #send aberration to DM
    dm.send_data(flat_dm_cmd + delta_c) 
    time.sleep(0.02) # wait a second
    #record image
    im = np.median( bdf.get_raw_images(camera, number_of_frames=5,     cropping_corners=cropping_corners) , axis=0)[cp_x1:cp_x2, cp_y1:cp_y2]
    #create signal
    S =  ((im  - flat_img) / mean_filtered_pupil).reshape(-1) 
    IM.append(S) 

IM = np.array(IM) 

##### ----- TO DO COMPARISON !!! 

# ======= APPLY STATIC ABERRATION, MEASURE AND RECONSTRUCT 

if debug:
    cmd_aber = 0.1*np.random.randn(100)
    reco_cmd_aber = []
    sig_list = []
    act_idx = 65
    for i in cmd_aber:
        #create aberration command
        delta_c = np.zeros(140)
        delta_c[act_idx] = i
        cmd = delta_c + flat_dm_cmd
        #send aberration to DM
        dm.send_data(cmd) 
        time.sleep(0.03) # wait a second
        #record image
        im = np.median( bdf.get_raw_images(camera, number_of_frames=5,     cropping_corners=cropping_corners) , axis=0)[cp_x1:cp_x2, cp_y1:cp_y2]
        #create signal
        S = mask_matrix @ ((im  - flat_img) / mean_filtered_pupil).reshape(-1) 
        sig_list.append(S)
        #reconstruct command 
        delta_c_reco = reconstruct_delta_command( S , param_dict )

        reco_cmd_aber.append( delta_c_reco[act_idx] )    
    sig_list=np.array(sig_list)
    fig,ax = plt.subplots(1,2)     
    ax[0].plot(cmd_aber, np.array(sig_list)[:,act_idx],'.',label='measured' )
    A,B,F,mu = param_dict[act_idx]
    ax[0].plot( cmd_aber, func(cmd_aber, A, B, F, mu),'.', label='model') 
    ax[0].legend() 
    
    ax[1].plot( cmd_aber, reco_cmd_aber, '.')
    ax[1].plot( cmd_aber, cmd_aber ,color='r',label='1:1')
    ax[1].legend()
    ax[1].set_xlabel('real cmd')
    ax[1].set_ylabel('reconstructed cmd')
    plt.show()
    """
    fig,ax = plt.subplots(1,2)
    im0=ax[0].imshow( bdf.get_DM_command_in_2D(delta_c, Nx_act=12) )
    ax[0].set_title('original aberration')
    im1=ax[1].imshow( bdf.get_DM_command_in_2D(delta_c_reco, Nx_act=12) )
    ax[1].set_title('reconstructed aberration')
    plt.colorbar(im0, ax= ax[0])
    plt.colorbar(im1, ax= ax[1])

    plt.show()
    """



# CORRECT STATIC ABERRATION 

# ======= init disturbance

""" 
## STATIC
disturbance_cmd = np.zeros( len( flat_dm_cmd )) 
for i in np.array(list(param_dict.keys())[::2] ): 
    disturbance_cmd[i] =  -0.15

for i in np.array(list(param_dict.keys())[::3] ): 
    disturbance_cmd[i] = 0.05

#disturbance_cmd[np.array([40,41,52,53,64,65])]=-0.1
"""
## DYNAMIC 
scrn_scaling_factor = 0.32
# --- create infinite phasescreen from aotools module 
Nx_act = dm.num_actuators_width()
screen_pixels = Nx_act*2**5 # some multiple of numer of actuators across DM 
D = 1.8 #m effective diameter of the telescope
scrn = aotools.infinitephasescreen.PhaseScreenVonKarman(nx_size=screen_pixels, pixel_scale=D/screen_pixels,r0=0.1,L0=12)

corner_indicies = [0, Nx_act-1, Nx_act * (Nx_act-1), -1] # Beware -1 index doesn't work if inserting in list! This is  ok for for use with create_phase_screen_cmd_for_DM function.

disturbance_cmd = dm_pupil_filt * bdf.create_phase_screen_cmd_for_DM(scrn=scrn, DM=dm, flat_reference=flat_dm_cmd, scaling_factor = scrn_scaling_factor, drop_indicies = corner_indicies, plot_cmd=False)  # normalized flat_dm +- scaling_factor?

disturbance_cmd -= np.mean( disturbance_cmd ) # no piston!! 

rows_to_jump = 1 # how many rows to jump on initial phase screen for each Baldr loop

distance_per_correction = rows_to_jump * D/screen_pixels # effective distance travelled by turbulence per AO iteration 
print(f'{rows_to_jump} rows jumped per AO command in initial phase screen of {screen_pixels} pixels. for {D}m mirror this corresponds to a distance_per_correction = {distance_per_correction}m')

if debug:
    # for visualization get the 2D grid of the disturbance on DM  
    plt.figure()
    plt.imshow( bdf.get_DM_command_in_2D(disturbance_cmd, Nx_act=12) )
    plt.colorbar()
    plt.title( f'initial Kolmogorov aberration to apply to DM (rms = {round(np.std(disturbance_cmd),3)})')
    plt.show()


# ====== PID

dt = 1/fps
Ku = 3. # ultimate gain to see stable oscillation in output 
Tu = 3*dt # period of oscillations 
#apply Ziegler-Nicols methold of PI controller https://en.wikipedia.org/wiki/Ziegler%E2%80%93Nichols_method
unity_dm_array = np.ones(len(flat_dm_cmd))
Kp, Ki, Kd = 0.45 * Ku* unity_dm_array , 0.54*Ku/Tu *unity_dm_array, 0.*unity_dm_array 
# NOTE: for Ki we scale zonal_gains**3 to avoid run-away amplitification of zonal modes on edge of illuminated DM 


# ======= Close loop
# init lists to hold data from control loop
DIST_list = [ disturbance_cmd ]
IMG_list = [ ]
DELTA_list = [ ] #list( np.nan * np.zeros( int( (cp_x2 - cp_x1) * (cp_y2 - cp_y1) ) ) ) ]
MODE_ERR_list = [ ] #list( np.nan * flat_dm_cmd ) ]
MODE_ERR_PID_list = [ ]
CMD_ERR_list = [ list( np.zeros(len(flat_dm_cmd ))) ]
CMD_list = [ list( flat_dm_cmd ) ]

RMS_list = [np.std( dm_pupil_filt * disturbance_cmd )] # to hold std( cmd - aber ) for each iteration
 
dm.send_data( flat_dm_cmd + disturbance_cmd )

M2C = np.eye(len(flat_dm_cmd)) # mode to DM cmd matrix. For now we just consider poke modes so this matrix is the identity. But later we can consider Zernike or KL modes etc 
number_images_recorded_per_cmd = 5
time.sleep(1)
FliSdk_V2.Start(camera)    
time.sleep(1)

for i in range(150):
    im = np.median( bdf.get_raw_images(camera, number_of_frames=number_images_recorded_per_cmd, cropping_corners=cropping_corners) , axis=0)[cp_x1:cp_x2, cp_y1:cp_y2] 
    # get new image and store it (save pupil and psf differently)
    IMG_list.append( list( im ) ) 

    #create signal
    S = mask_matrix @ ((IMG_list[-1]  - flat_img) / pupil).reshape(-1) 
    DELTA_list.append( S )

    delta_c_reco = reconstruct_delta_command( S , param_dict )
    MODE_ERR_list.append( delta_c_reco )

    u =  Kp * np.array(MODE_ERR_list[-1]) +  Ki * dt * np.sum( MODE_ERR_list,axis=0 )

    MODE_ERR_PID_list.append( u )   

    cmd_errs =  M2C @ MODE_ERR_PID_list[-1] 
    cmd_errs -= np.mean(cmd_errs) # REMOVE PISTON FORCEFULLY
    CMD_ERR_list.append( list(cmd_errs) )

    CMD_list.append( flat_dm_cmd - CMD_ERR_list[-1] )

    DIST_list.append( disturbance_cmd )

    # roll phase screen 
    for skip in range(rows_to_jump):
        scrn.add_row() 
    # get our new Kolmogorov disturbance command 
    disturbance_cmd = dm_pupil_filt * bdf.create_phase_screen_cmd_for_DM(scrn=scrn, DM=dm, flat_reference=flat_dm_cmd, scaling_factor = scrn_scaling_factor, drop_indicies = corner_indicies, plot_cmd=False)

    disturbance_cmd -= np.mean( disturbance_cmd )

    # we only calculate rms in our cmd region
    RMS_list.append( np.std( dm_pupil_filt * ( np.array(CMD_list[-1]) - flat_dm_cmd + np.array(disturbance_cmd) ) ) )
   
    dm.send_data( CMD_list[-1] + disturbance_cmd )

dm.send_data(flat_dm_cmd)

#RMSE
plt.figure()
plt.plot( 2 * interp_deflection_4x4act( RMS_list ),'.',label='residual' )
plt.plot( 2 * interp_deflection_4x4act( [np.std(d) for d in DIST_list]) ,'.',label='disturbance' )
plt.ylabel('RMSE wave space [nm RMS]')
plt.xlabel('iteration')
plt.legend()
#plt.savefig(data_path + f'A_FIRST_closed_loop_on_dynamic_aberration_t-{tstamp}.png')
#plt.xlim([0,50]);plt.ylim([24,26]);plt.show()
print( 'final RMSE =', interp_deflection_4x4act( RMS_list[-1] ) )
print( 'RMSE after 100 iter. =',interp_deflection_4x4act( RMS_list[100] ) );


# first 5 iterations 
iterations2plot = [0,1,2,3,4,5,-2,-1]
fig, ax = plt.subplots( len(iterations2plot), 5 ,figsize=(10,15))
ax[0,1].set_title('disturbance',fontsize=15)
ax[0,0].set_title('ZWFS image',fontsize=15)
ax[0,2].set_title('CMD error (feedback)',fontsize=15)
ax[0,3].set_title('DM CMD (feedback)',fontsize=15)
ax[0,4].set_title('RESIDUAL (feedback)',fontsize=15)
for i,idx in enumerate(iterations2plot):
    ax[i,0].imshow( np.array(IMG_list)[idx] ) 
    im1 = ax[i,1].imshow( bdf.get_DM_command_in_2D(DIST_list[idx]) )
    plt.colorbar(im1, ax= ax[i,1])
    im2 = ax[i,2].imshow( bdf.get_DM_command_in_2D(CMD_ERR_list[idx]) )
    plt.colorbar(im2, ax= ax[i,2])
    im3 = ax[i,3].imshow( bdf.get_DM_command_in_2D(CMD_list[idx] ) )
    plt.colorbar(im3, ax= ax[i,3])
    im4 = ax[i,4].imshow( bdf.get_DM_command_in_2D(np.array(CMD_list[idx]) + np.array(DIST_list[idx]) - flat_dm_cmd) )
    plt.colorbar(im4, ax= ax[i,4])

plt.show() 






# saving 
static_ab_performance_fits = fits.HDUList( [] )

camera_info_dict = bdf.get_camera_info( camera )

IMG_fits = fits.PrimaryHDU( IMG_list )
IMG_fits.header.set('EXTNAME','IMAGES')
#IMG_fits.header.set('recon_fname',recon_file.split('/')[-1])
for k,v in camera_info_dict.items(): 
    IMG_fits.header.set(k,v)   # add in some fits headers about the camera 

for i,n in zip([ci_x1,ci_x2,ci_y1,ci_y2],['ci_x1','ci_x2','ci_y1','ci_y2']):
    IMG_fits.header.set(n,i)

for i,n in zip([cp_x1,cp_x2,cp_y1,cp_y2],['cp_x1','cp_x2','cp_y1','cp_y2']):
    IMG_fits.header.set(n,i)

disturbfits = fits.PrimaryHDU( DIST_list )
disturbfits.header.set('EXTNAME','DIST')
disturbfits.header.set('WHAT_IS','disturbance DM command')


RES_fits = fits.PrimaryHDU( DELTA_list )
RES_fits.header.set('EXTNAME','RES')
RES_fits.header.set('WHAT_IS','(I_t - I_CAL_FPM_ON) / I_CAL_FPM_OFF')

MODE_ERR_fits = fits.PrimaryHDU( MODE_ERR_list )
MODE_ERR_fits.header.set('EXTNAME','MODE_ERR')
MODE_ERR_fits.header.set('WHAT_IS','CM @ ERR')

MODE_ERRPID_fits = fits.PrimaryHDU( MODE_ERR_PID_list )
MODE_ERRPID_fits.header.set('EXTNAME','MODE_ERR_PID')
MODE_ERRPID_fits.header.set('WHAT_IS','PID applied to modal errors')

CMDERR_fits = fits.PrimaryHDU( CMD_ERR_list )
CMDERR_fits.header.set('EXTNAME','CMD_ERR')
CMDERR_fits.header.set('WHAT_IS','list of cmd errors')

CMD_fits = fits.PrimaryHDU( CMD_list )
CMD_fits.header.set('EXTNAME','CMDS')
CMD_fits.header.set('WHAT_IS','DM commands')

RMS_fits = fits.PrimaryHDU( RMS_list )
RMS_fits.header.set('EXTNAME','RMS')
RMS_fits.header.set('WHAT_IS','std( cmd - aber_in_cmd_space )')

# add these all as fits extensions 
for f in [disturbfits,  IMG_fits, RES_fits, MODE_ERR_fits, MODE_ERRPID_fits,CMDERR_fits, CMD_fits, RMS_fits ]: #[Ufits, Sfits, Vtfits, CMfits, disturbfits, IMG_fits, ERR_fits, RECO_fits, CMD_fits, RMS_fits ]:
    static_ab_performance_fits.append( f ) 

#save data! 
save_fits = data_path + f'A_FIRST_COSINE_MODEL_closed_loop_on_dynamic_aberration_t-{tstamp}.fits'
static_ab_performance_fits.writeto( save_fits )








