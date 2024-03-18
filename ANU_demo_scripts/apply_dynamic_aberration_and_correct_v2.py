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
        x = 0
        #valtmp = np.sign( (I-A)/B ) # round to +/-1
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
    
    dm_pupil_filt = thresh < np.array( [np.max( abs( poke_imgs[len(poke_imgs)//2-3][i] - flat_img) ) for i in range(140)] ) 
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

dm_pupil_filt = pupil_filt_threshold < np.array( [np.max( abs( poke_imgs[len(poke_imgs)//2-3][i] - flat_img) ) for i in range(140)] ) 


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
            amp_indx = len(poke_imgs)//2-1 # CHOOSE THE AMPLITUDE TO USE FOR FINDING THE PEAKS! WORKS BEST FOR COMMANDS < FLAT DM WHERE WE HAVE BETTER LINERARITY 
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
                ax[0].set_title(act_indx)
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

scrn_scaling_factor = 0.15
number_images_recorded_per_cmd = 20 #NDITs to take median over 
PID = [0.9, 0.0, 0.0] # proportional, integator, differential gains  
Nint = 1 # used for integral term.. should be calcualted later. TO DO 
dt_baldr = 1  # used for integral term.. should be calcualted later. TO DO  

save_fits = data_path + f'closed_loop_on_dynanic_aberration_disturb-kolmogorov_amp-{scrn_scaling_factor}_PID-{PID}_t-{tstamp}.fits'


# --- create infinite phasescreen from aotools module 
Nx_act = dm.num_actuators_width()
screen_pixels = Nx_act*2**5 # some multiple of numer of actuators across DM 
D = 1.8 #m effective diameter of the telescope
scrn = aotools.infinitephasescreen.PhaseScreenVonKarman(nx_size=screen_pixels, pixel_scale=D/screen_pixels,r0=0.1,L0=12)

corner_indicies = [0, Nx_act-1, Nx_act * (Nx_act-1), -1] # Beware -1 index doesn't work if inserting in list! This is  ok for for use with create_phase_screen_cmd_for_DM function.

# NOTE WE ONLY APPLY DISTURVBANCE WITHIN DM INFLUENCE REGION (using dm_pupil_filt)
disturbance_cmd = dm_pupil_filt * bdf.create_phase_screen_cmd_for_DM(scrn=scrn, DM=dm, flat_reference=flat_dm_cmd, scaling_factor = scrn_scaling_factor, drop_indicies = corner_indicies, plot_cmd=False)  # normalized flat_dm +- scaling_factor?

rows_to_jump = 2 # how many rows to jump on initial phase screen for each Baldr loop

distance_per_correction = rows_to_jump * D/screen_pixels # effective distance travelled by turbulence per AO iteration 
print(f'{rows_to_jump} rows jumped per AO command in initial phase screen of {screen_pixels} pixels. for {D}m mirror this corresponds to a distance_per_correction = {distance_per_correction}m')

# for visualization get the 2D grid of the disturbance on DM  
plt.figure()
plt.imshow( bdf.get_DM_command_in_2D(disturbance_cmd, Nx_act=12) )
plt.colorbar()
plt.title( 'initial Kolmogorov aberration to apply to DM')
plt.show()
#plt.close() 



# ======================= LETS GO! =====

dynamic_ab_performance_fits = fits.HDUList([])


# init lists to hold data from control loop
DISTURB_list = [ np.zeros( len( flat_dm_cmd ) )  ] # begin with no disturbance (add it after first iteration)
IMG_list = [ ]
CMD_list = [ list( flat_dm_cmd ) ] 
ERR_list = [ ]# list( np.nan * np.zeros( int( (cp_x2 - cp_x1) * (cp_y2 - cp_y1) ) ) ) ]  # length depends on cropped pupil when flattened 
RMS_list = [ ] # to hold std( cmd - aber ) for each iteration
 

# start with flat DM
dm.send_data( flat_dm_cmd )

FliSdk_V2.Start(camera)    
time.sleep(1)

Nits = 100
keep_loop_open_for = 30 
#===========================
#OPEN LOOP (dont feedback a command to the DM besides the disturbance command)
for i in range(keep_loop_open_for): 

    IMG_list.append( np.median( bdf.get_raw_images(camera, number_of_frames=number_images_recorded_per_cmd, cropping_corners=cropping_corners) , axis=0) ) 

    delta_c = reconstruct_delta_command( IMG_list[-1][cp_x1:cp_x2, cp_y1:cp_y2], param_dict, pixel_indx_dict )
    
    ERR_list.append( delta_c ) 
    
    CMD_list.append( list(flat_dm_cmd) )

    # propagate our phase screen a few rows  
    for skip in range(rows_to_jump):
        scrn.add_row() 
    # get our new Kolmogorov disturbance command (normalized between [-scrn_scaling_factor,scrn_scaling_factor]
    # NOTE WE ONLY APPLY DISTURVBANCE WITHIN DM INFLUENCE REGION
    disturbance_cmd = dm_pupil_filt * bdf.create_phase_screen_cmd_for_DM(scrn=scrn, DM=dm, flat_reference=flat_dm_cmd, scaling_factor = scrn_scaling_factor, drop_indicies = corner_indicies, plot_cmd=False)

    DISTURB_list.append( disturbance_cmd )
    # apply our flat DM + disturbance for open loop 
    dm.send_data(  CMD_list[-1] + DISTURB_list[-1] ) 
    # record RMS in command space 
    RMS_list.append( np.std(  dm_pupil_filt * (np.array(CMD_list[-1]) - flat_dm_cmd - np.array(DISTURB_list[-1]) ) ) )

fig,ax = plt.subplots(1,2); 
im0=ax[0].imshow( bdf.get_DM_command_in_2D(ERR_list[-1]) ) ;plt.colorbar(im0,ax=ax[0]); im1=ax[1].imshow( bdf.get_DM_command_in_2D(DISTURB_list[-1]) ) ;plt.colorbar(im1,ax=ax[1]); 
plt.show()

#===========================
#CLOSED LOOP (feedback ZWFS control command)
for i in range(Nits - keep_loop_open_for):  
    # get new image and store it (save pupil and psf differently)
    IMG_list.append( np.median( bdf.get_raw_images(camera, number_of_frames=number_images_recorded_per_cmd, cropping_corners=cropping_corners) , axis=0) )  

    delta_c = reconstruct_delta_command( IMG_list[-1][cp_x1:cp_x2, cp_x1:cp_x2], param_dict, pixel_indx_dict )
    
    ERR_list.append( delta_c )

    # PID control 
    if len( ERR_list ) < Nint:
        cmd = PID[0] * np.array(ERR_list[-1]) + PID[1] * np.sum( ERR_list ) * dt_baldr 
    else:
        cmd = PID[0] * np.array(ERR_list[-1]) + PID[1] * np.sum( ERR_list[-Nint:] , axis = 0 ) * dt_baldr 
            
    #cmdtmp =  cmd - np.mean(cmd) # REMOVE PISTON FORCEFULLY 
    """
    fig,ax = plt.subplots( 1,2 ) 
    ax[0].imshow(  bdf.get_DM_command_in_2D(disturbance_cmd, Nx_act=12) )
    ax[0].set_title( 'static aberration') 
    ax[1].imshow( bdf.get_DM_command_in_2D(cmdtmp, Nx_act=12) )
    ax[1].set_title( 'cmd vector') 
    """

    # check dm commands are within limits  # THIS IS WRONG SINCE cmdtmp is an error signal not absolute! c
    #if np.max(cmdtmp)>1:
    #    print(f'WARNING {sum(cmdtmp>1)} DM commands exceed max value of 1') 
    #    cmdtmp[cmdtmp>1] = 1 #force clip
    #if np.min(cmdtmp)<0:
    #    print(f'WARNING {sum(cmdtmp<0)} DM commands exceed min value of 0') 
    #    cmdtmp[cmdtmp<0] = 0 # force clip
    # finally append it:
    CMD_list.append( CMD_list[-1] - cmd )
    
    # propagate our phase screen a few rows  
    for skip in range(rows_to_jump):
        scrn.add_row() 
    # get our new Kolmogorov disturbance command 
    disturbance_cmd = dm_pupil_filt * bdf.create_phase_screen_cmd_for_DM(scrn=scrn, DM=dm, flat_reference=flat_dm_cmd, scaling_factor = scrn_scaling_factor, drop_indicies = corner_indicies, plot_cmd=False)

    DISTURB_list.append( disturbance_cmd )
    # apply control command to DM + our static disturbance 
    dm.send_data( CMD_list[-1] + DISTURB_list[-1] ) 
    # record RMS in command space 
    RMS_list.append( np.nanstd( dm_pupil_filt * (np.array(CMD_list[-1]) - flat_dm_cmd - np.array(DISTURB_list[-1]) ) ))
    # rest a bit
    time.sleep(0.05)

# now flatten once finished
dm.send_data( flat_dm_cmd ) 

camera_info_dict = bdf.get_camera_info( camera )

FliSdk_V2.Stop(camera)


disturbfits = fits.PrimaryHDU( DISTURB_list )
disturbfits.header.set('EXTNAME','DISTURBANCE')
disturbfits.header.set('WHAT_IS','disturbance in cmd space')

IMG_fits = fits.PrimaryHDU( IMG_list )
IMG_fits.header.set('EXTNAME','IMAGES')
IMG_fits.header.set('recon_fname',recon_file.split('/')[-1])

IMG_fits.header.set('open_loop_iter', keep_loop_open_for)
IMG_fits.header.set('close_loop_iter', Nits - keep_loop_open_for)

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
    dynamic_ab_performance_fits.append( f ) 

#save data! 
dynamic_ab_performance_fits.writeto( save_fits )



fig,ax = plt.subplots(3,2,figsize=(10,10))
im1 = ax[0,0].imshow( bdf.get_DM_command_in_2D(ERR_fits.data[3] ))
im2 = ax[1,0].imshow( bdf.get_DM_command_in_2D(ERR_fits.data[Nits - keep_loop_open_for-2] ))
im3 = ax[2,0].imshow( bdf.get_DM_command_in_2D(ERR_fits.data[-1] ))

im1 = ax[0,1].imshow( bdf.get_DM_command_in_2D(disturbfits.data[3] ))
im2 = ax[1,1].imshow( bdf.get_DM_command_in_2D(disturbfits.data[Nits - keep_loop_open_for-2] ))
im3 = ax[2,1].imshow( bdf.get_DM_command_in_2D(disturbfits.data[-1] ))

# PLOTTING SOME RESULTS 
psf_max = np.array( [ np.max( psf[ci_x1:ci_x2,ci_y1:ci_y2] )  for psf in dynamic_ab_performance_fits['IMAGES'].data] )

psf_max_ref = np.max( recon_data['FPM_OUT'].data[ci_x1:ci_x2,ci_y1:ci_y2] )

fig,ax = plt.subplots( 4,2,figsize=(7,30))
ax[0,0].plot( dynamic_ab_performance_fits['RMS'].data )
ax[0,0].set_ylabel('RMS in cmd space')
ax[0,0].set_xlabel('iteration')

ax[0,1].plot( psf_max/psf_max_ref )
ax[0,1].set_ylabel('max(I) / max(I_ref)')
ax[0,1].set_xlabel('iteration')


ax[1,0].imshow( recon_data['FPM_IN'].data[cp_x1:cp_x2,cp_y1:cp_y2] )
ax[1,0].set_title('reference pupil (FPM IN)')

ax[2,0].imshow( dynamic_ab_performance_fits['IMAGES'].data[1][cp_x1:cp_x2,cp_y1:cp_y2]  )
ax[2,0].set_title('initial pupil with disturbance')

ax[3,0].imshow( dynamic_ab_performance_fits['IMAGES'].data[-1][cp_x1:cp_x2,cp_y1:cp_y2]  ) 
ax[3,0].set_title('final pupil after 10 iterations')


ax[1,1].imshow( recon_data['FPM_OUT'].data[ci_x1:ci_x2,ci_y1:ci_y2] )
ax[1,1].set_title('reference PSF')

ax[2,1].imshow( dynamic_ab_performance_fits['IMAGES'].data[0][ci_x1:ci_x2,ci_y1:ci_y2]  )
ax[2,1].set_title('initial PSF with disturbance')

ax[3,1].imshow( dynamic_ab_performance_fits['IMAGES'].data[-1][ci_x1:ci_x2,ci_y1:ci_y2]  ) 
ax[3,1].set_title('final PSF after 10 iterations')

plt.subplots_adjust(wspace=0.5, hspace=0.5)
#plt.savefig(fig_path + f'closed_loop_on_static_aberration_RESULTS_disturb-kolmogorov_amp-{scrn_scaling_factor}_PID-{PID}_t-{tstamp}.png') 
plt.show() 

