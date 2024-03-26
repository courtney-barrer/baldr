"""
idea is that we have a set point signal 
S = Mask @ (I - I0)/n
where I0 is reference intensity of ZWFS with the FPM in on near perfect calibration source, I is the current measured intensity in ZWFS and n is some proportional estimate of the number photons in pupil (e.g. <I> when FPM is out). Mask is a matrix model we may construct to aggregate the signal over a given number of pixels or selectively pick certain pixels. We then model 
S = A + B * cos(F*x + phi + mu) 
where phi is phase aberration relative to pupil piston (mean phase). x is difference between the command c and a reference command (i.e flat DM) c0 (x=c-c0). Using calibration source we can fit A,B, F, mu and from phase mask design we can compare to theoretical values of A, B, mu for a given flux and known Strehl. 

The issue with reconstruction is that measurement S can only be used to solve for phi uniquely over a range of pi radians. However if we know the derivative we can extend this out to a 2pi range. BUT HOW DO WE KNOW WHAT SIDE WE ARE ON? 

using x = C +/-epsilon, where C is our correction command we measure Delta = (S+ - S-)/epsilon. where S+ =   A + B * cos(F*(C+epsilon) + phi + mu) , S- =   A + B * cos(F*(C-epsilon) + phi + mu) 
=> Delta = -2 * B * sin(F*epsilon)*sin(F*C + phi + mu)/epsilon ~  -2 * B * F *sin(F*C + phi + mu). which is the derivative of S at F*C + phi



Steps
======
1. setup, and get or load push pull DM data on calibration source
2. Calculate S and Fit Models 
3. Apply a large poke on a single actuator to push phase aberration beyond pi range
4. apply push, pull waffle to measure delta 
5. plot the cosine model vs phase with x points marking where actuators are, then show colormap of the modelled DM surface with applied aberration, then show colormap of measured gradients in each aggregated pixels from delta measurement.  


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


def func(x, A, B, F, mu):
    I = A + B * np.cos(F * x + mu)
    return I 


root_path = '/home/baldr/Documents/baldr'
data_path = root_path + '/ANU_demo_scripts/ANU_data/' 
fig_path = root_path + '/figures/' 

os.chdir(root_path)
from functions import baldr_demo_functions as bdf


debug = True # generate plots to help debug

# --- timestamp
tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
# --- 



# =====(1) 
# --- DM command to make flat DM (calibrated file provided by BMC with the DM) 
flat_dm_cmd = pd.read_csv(bdf.path_to_dm_flat_map, header=None)[0].values 

waffle_dm_cmd = pd.read_csv(root_path + '/DMShapes/dm_checker_pattern.csv', index_col=[0]).values.ravel() 

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



# =======(2) FITTING MODELS FOR EACH FILTERED PIXEL
 
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

# ======= (3) Apply a large poke on a single actuator to push phase aberration beyond pi range


def get_delta( dm, camera, delta_c, epsilon, flat_img, mean_pupil, cropping_corners, pupil_indicies):
    cp_x1,cp_x2, cp_y1,cp_y2 = pupil_indicies

    # send +epsilon command
    dm.send_data(delta_c + epsilon * waffle_dm_cmd) 
    time.sleep(0.03) # wait a second

    #record + image
    im_plus = np.median( bdf.get_raw_images(camera, number_of_frames=5,     cropping_corners=cropping_corners) , axis=0)[cp_x1:cp_x2, cp_y1:cp_y2]

    S_plus = mask_matrix @ ((im_plus  - flat_img) / mean_pupil).reshape(-1)

    # send -epsilon command
    dm.send_data(delta_c - epsilon * waffle_dm_cmd) 
    time.sleep(0.03) # wait a second

    #record - image
    im_minus = np.median( bdf.get_raw_images(camera, number_of_frames=5,      cropping_corners=cropping_corners) , axis=0)[cp_x1:cp_x2, cp_y1:cp_y2]

    S_minus = mask_matrix @ ((im_minus  - flat_img) / mean_pupil).reshape(-1)

    # get out delta! 
    delta = ( S_plus - S_minus ) / (2*abs(epsilon))

    return( delta ) 


act_idx = 65
disturb =  np.pi/2.2 #radians
epsilon = 0.01 # DM command unit (0-1)
A_opt, B_opt, F_opt, mu_opt = param_dict[act_idx]
delta_c = np.zeros(140)
delta_c[act_idx] = disturb/F_opt # F_opt * delta_c = disturb


delta0 = get_delta( dm, camera, np.zeros(140), epsilon, flat_img, mean_filtered_pupil, cropping_corners, [cp_x1,cp_x2, cp_y1,cp_y2])

delta = get_delta( dm, camera, delta_c, epsilon, flat_img, mean_filtered_pupil, cropping_corners, [cp_x1,cp_x2, cp_y1,cp_y2])


# now plot 



from mpl_toolkits.axes_grid1 import make_axes_locatable

fig = plt.figure(figsize=(7, 7))
ax1 = fig.add_subplot(111)
im1 = ax1.imshow( bdf.get_DM_command_in_2D( F_opt * delta_c ))
ax1.set_title('DM aberration',fontsize=15)
divider = make_axes_locatable(ax1)
cax = divider.append_axes('bottom', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='horizontal',label='radians')
ax1.axis('off')
plt.tight_layout()
plt.savefig( fig_path + 'dither_gradient_dm_aberration_cmap.png',dpi=300)


fig = plt.figure(figsize=(7, 7))
ax2 = fig.add_subplot(111)
ax2.plot( F_opt*ramp_values[2:-2], A_opt + B_opt * np.cos( F_opt * ramp_values[2:-2]+mu_opt), linestyle=':',color='k',label='set point model' ) 
ax2.plot( F_opt*delta_c[act_idx] , A_opt + B_opt * np.cos( F_opt *delta_c[act_idx] +mu_opt), 'x',color='red',label='poked actuator') 
ax2.plot( F_opt*0 , A_opt + B_opt * np.cos( F_opt *0 +mu_opt), 'x',color='green',label='rest actuators') 
ax2.legend(fontsize=15)
ax2.set_xlabel('phase [radians]',fontsize=15)
ax2.set_ylabel(r'$SP=\frac{I-I_0}{N_{ph}}$ [adu]',fontsize=15)

plt.tight_layout()
plt.savefig( fig_path + 'dither_gradient_setpoint_plot.png',dpi=300)

fig = plt.figure(figsize=(7, 7))
ax3 = fig.add_subplot(111)
im3 = ax3.imshow(bdf.get_DM_command_in_2D( -np.sign( 0.6+delta ) ),cmap='gray', interpolation='None') #'bicubic')
ax3.axis('off')
ax3.set_title('Measured Gradient Sign',fontsize=15)
divider = make_axes_locatable(ax3)
cax = divider.append_axes('bottom', size='5%', pad=0.05)
fig.colorbar(im3, cax=cax, orientation='horizontal',label='gradient sign')
plt.tight_layout()
plt.savefig( fig_path + 'dither_gradient_sign_cmap.png',dpi=300)

plt.show()






















