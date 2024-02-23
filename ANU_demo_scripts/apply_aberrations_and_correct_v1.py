import os 
import datetime 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import glob
from scipy import interpolate
from astropy.io import fits
import aotools
os.chdir('/opt/FirstLightImaging/FliSdk/Python/demo/')
import FliSdk_V2 

root_path = '/home/baldr/Documents/baldr'
data_path = root_path + '/ANU_demo_scripts/ANU_data/' 
fig_path = root_path + '/figures/' 

os.chdir(root_path)
from functions import baldr_demo_functions as bdf


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
print('\n======\navailable AO reconstruction fits files:\n')
for f in available_recon_pupil_files:
    print( f ,'\n') 

recon_file = input('input file to use for interaction or control matricies. Input 0 if you want to create a new one')

if recon_file != '0':
    recon_data = fits.open( recon_file )
else:
    os.chdir(root_path + '/ANU_demo_scripts') 
    import ASG_BDR_RECON # run RECON script to calibrate matricies in open loop

    new_available_recon_pupil_files = glob.glob( data_path+'BDR_RECON_*.fits' )
    recon_file = max(new_available_recon_pupil_files, key=os.path.getctime) #latest_file
    recon_data = fits.open( recon_file ) 
    os.chdir(root_path)

# =====(2) 
# --- setup camera
fps = float( recon_data[0].header['camera_fps'] )
camera = bdf.setup_camera(cameraIndex=0) #connect camera and init camera object
camera = bdf.set_fsp_dit( camera, fps=fps, tint=None) # set up initial frame rate, tint=None means max integration time for given FPS


# --- setup DM
dm, dm_err_code = bdf.set_up_DM(DM_serial_number='17DW019#053')

# --- setup DM interaction and control matricies
modal_basis = recon_data['BASIS'].data #
# check modal basis dimensions are correct
if modal_basis.shape[1] != 140:
    raise TypeError( 'modal_basis.shape[1] != 140 => not right shape. Maybe we should transpose it?\nmodal_basis[i] should have a 140 length command for the DM corresponding to mode i') 
IM = recon_data['IM'].data # unfiltered
CM = recon_data['CM'].data # filtered


# poke amplitude in DM command space used to generate IM 
IM_pokeamp = float( recon_data['IM'].header['poke_amp_cmd'] )
# pupil cropping coordinates
cp_x1,cp_x2,cp_y1,cp_y2 = recon_data[0].header['cp_x1'],recon_data[0].header['cp_x2'],recon_data[0].header['cp_y1'],recon_data[0].header['cp_y2']
# PSF cropping coordinates 
ci_x1,ci_x2,ci_y1,ci_y2 = recon_data[0].header['ci_x1'],recon_data[0].header['ci_x2'],recon_data[0].header['ci_y1'],recon_data[0].header['ci_y2']

# have a look at one of the interaction images for a particular modal actuation
# plt.imshow( IM[100].reshape(cp_x2-cp_x1,cp_y2-cp_y1) );plt.show()

# =====(3)
# create static phase screen on DM 

scrn_scaling_factor = 0.1
number_images_recorded_per_cmd = 10 #NDITs to take median over 
PID = [1, 0, 0] # proportional, integator, differential gains  
Nint = 1 # used for integral term.. should be calcualted later. TO DO 
dt_baldr = 1  # used for integral term.. should be calcualted later. TO DO  

save_fits = data_path + f'closed_loop_on_static_aberration_disturb-kolmogorov_amp-{scrn_scaling_factor}_PID-{PID}_t-{tstamp}.fits'

# --- create infinite phasescreen from aotools module 
Nx_act = dm.num_actuators_width()
scrn = aotools.infinitephasescreen.PhaseScreenVonKarman(nx_size=Nx_act*2**5, pixel_scale=1.8/(Nx_act*2**5),r0=0.1,L0=12)

corner_indicies = [0, Nx_act-1, Nx_act * (Nx_act-1), -1] # Beware -1 index doesn't work if inserting in list! This is  ok for for use with create_phase_screen_cmd_for_DM function.

disturbance_cmd = bdf.create_phase_screen_cmd_for_DM(scrn=scrn, DM=dm, flat_reference=flat_dm_cmd, scaling_factor = scrn_scaling_factor, drop_indicies = corner_indicies, plot_cmd=False)  # normalized flat_dm +- scaling_factor?

# for visualization get the 2D grid of the disturbance on DM  
#plt.imshow( bdf.get_DM_command_in_2D(disturbance_cmd, Nx_act=12) ); plt.show()
#plt.close() 

print(' \n\n applying static aberration and closing loop')
# apply DISTURBANCE 
dm.send_data( disturbance_cmd )

# =====(6)
# NOW try to correct it! 

# init main fits file for saving telemetry
static_ab_performance_fits = fits.HDUList([])

disturbfits = fits.PrimaryHDU( disturbance_cmd )
disturbfits.header.set('EXTNAME','DISTURBANCE')
disturbfits.header.set('WHAT_IS','disturbance in cmd space')

CMfits =  fits.PrimaryHDU( CM )
CMfits.header.set('EXTNAME','CM')
CMfits.header.set('WHAT IS','CM_filtered') 

# init lists to hold data from control loop
IMG_list = []
RES_list = [ ] #list( np.nan * np.zeros( int( (cp_x2 - cp_x1) * (cp_y2 - cp_y1) ) ) ) ]
RECO_list = [ ] #list( np.nan * flat_dm_cmd ) ]
CMD_list = [ list( flat_dm_cmd ) ] 
ERR_list = [ ]# list( np.nan * np.zeros( int( (cp_x2 - cp_x1) * (cp_y2 - cp_y1) ) ) ) ]  # length depends on cropped pupil when flattened 
RMS_list = [] # to hold std( cmd - aber ) for each iteration
  

modal_gains = IM_pokeamp * np.ones(len(modal_basis[0]))
"""
setting PI parameters https://www.zhinst.com/ch/en/resources/principles-of-pid-controllers?gclid=CjwKCAiApaarBhB7EiwAYiMwqi06BUUcq6C11e3tHueyTd7x1DqVrk9gi8xLmtLwUBRCT4nW7EsJnxoCz4oQAvD_BwE&hsa_acc=8252128723&hsa_ad=665555823596&hsa_cam=14165786829&hsa_grp=126330066395&hsa_kw=pid%20controller&hsa_mt=p&hsa_net=adwords&hsa_src=g&hsa_tgt=kwd-354990109332&hsa_ver=3&utm_campaign=PID%20Group&utm_medium=ppc&utm_source=adwords&utm_term=pid%20controller
    https://apmonitor.com/pdc/index.php/Main/ProportionalIntegralControl#:~:text=Discrete%20PI%20Controller,the%20integral%20of%20the%20error.
    1. Set the P,I, and D gain to zero
    2. Increase the proportional (P) gain until the system starts to show consistent and stable oscillation. This value is known as the ultimate gain (Ku).
    3. Measure the period of the oscillation (Tu).
    4. Depending on the desired type of control loop (P, PI or PID) set the gains to the following values:
   
Nint = 2
dt_baldr = number_images_recorded_per_cmd/fps
Tu = Nint * dt_baldr
PID[1] = 0.54 * Ku/Tu #0.9    #0.75 #0.0
PID[0] = 0.45 * Ku #1 #0.45 * Ku # 1.1 #1. #2.
""" 

#modal_reco_list = CM.T @ (  1/Nph_obj * (sig_turb.signal - Nph_obj/Nph_cal * sig_cal_on.signal) ).reshape(-1) #list of amplitudes of the modes measured by the ZWFS
#        modal_gains = -1. * S_filt  / np.max(S_filt) 


FliSdk_V2.Start(camera)        
for i in range(10):

    # get new image and store it (save pupil and psf differently)
    IMG_list.append( list( np.median( [FliSdk_V2.GetRawImageAsNumpyArray(camera,-1)  for i in range(number_images_recorded_per_cmd)] , axis=0) ) ) 

    # create new error vector (remember to crop it!) with bdf.get_error_signal
    err_2d = bdf.get_error_signal( np.array(IMG_list[-1]), reference_pupil_fits = recon_data, reduction_dict=None, crop_indicies = [cp_x1,cp_x2,cp_y1,cp_y2] ) # Note we can use recon data as long as reference pupils have FPM_ON and FPM_OFF extension names uniquely
    """
    fig,ax = plt.subplots( 1,2 ) 
    ax[0].imshow(  bdf.get_DM_command_in_2D(disturbance_cmd, Nx_act=12) )
    ax[0].set_title( 'static aberration') 
    ax[1].imshow( err_2d )
    ax[1].set_title( 'error vector') 
    """
    RES_list.append( err_2d.reshape(-1) ) 
    # CHECKS np.array(ERR_list[0]).shape = np.array(ERR_list[1]).shape = (cp_x2 - cp_x1) * (cp_y2 - cp_y1)

    
    
    # reconstruct phase 
    #reco_modal_amps = CM.T @ RES_list[-1]  # CM.T @ (  1/Nph_obj * (sig_turb.signal - Nph_obj/Nph_cal * sig_cal_on.signal) ).reshape(-1)
    RECO_list.append( list( CM.T @ RES_list[-1] ) ) # reconstructed modal amplitudes
    
    # to get error signal we apply modal gains
    ERR_list.append( list( np.sum( np.array([ g * a * B for g,a,B in  zip(modal_gains, RECO_list[-1], modal_basis)]) , axis=0) ) )

    # PID control 
    if len( ERR_list ) < Nint:
        cmd = PID[0] * ERR_list[-1] +  PID[1] * np.sum( ERR_list ) * dt_baldr 
    else:
        cmd = PID[0] * ERR_list[-1] +  PID[1] * np.sum( ERR_list[-Nint:] , axis = 0 ) * dt_baldr 
            
    cmdtmp =  cmd - np.mean(cmd) # REMOVE PISTON FORCEFULLY 
    """
    fig,ax = plt.subplots( 1,2 ) 
    ax[0].imshow(  bdf.get_DM_command_in_2D(disturbance_cmd, Nx_act=12) )
    ax[0].set_title( 'static aberration') 
    ax[1].imshow( bdf.get_DM_command_in_2D(cmdtmp, Nx_act=12) )
    ax[1].set_title( 'cmd vector') 
    """

    # check dm commands are within limits 
    if np.max(cmdtmp)>1:
        print(f'WARNING {sum(cmdtmp>1)} DM commands exceed max value of 1') 
        cmdtmp[cmdtmp>1] = 1 #force clip
    if np.min(cmdtmp)<0:
        print(f'WARNING {sum(cmdtmp<0)} DM commands exceed min value of 0') 
        cmdtmp[cmdtmp<0] = 0 # force clip
    # finally append it:
    CMD_list.append( CMD_list[-1] - cmdtmp )
    RMS_list.append( np.std( np.array(CMD_list[-1]) - np.array(disturbance_cmd) ) )
    # apply control command to DM + our static disturbance 
    dm.send_data( CMD_list[-1] + disturbance_cmd ) 

# now flatten once finished
dm.send_data( flat_dm_cmd ) 

camera_info_dict = bdf.get_camera_info( camera )

FliSdk_V2.Stop(camera)



IMG_fits = fits.PrimaryHDU( IMG_list )
IMG_fits.header.set('EXTNAME','IMAGES')
IMG_fits.header.set('recon_fname',recon_file.split('/')[-1])
for k,v in camera_info_dict.items(): 
    IMG_fits.header.set(k,v)   # add in some fits headers about the camera 

CMD_fits = fits.PrimaryHDU( CMD_list )
CMD_fits.header.set('EXTNAME','CMDS')
CMD_fits.header.set('WHAT_IS','DM commands')

RES_fits = fits.PrimaryHDU( RES_list )
RES_fits.header.set('EXTNAME','RES')
RES_fits.header.set('WHAT_IS','(I_t - I_CAL_FPM_ON) / I_CAL_FPM_OFF')

RECO_fits = fits.PrimaryHDU( RECO_list )
RECO_fits.header.set('EXTNAME','RECOS')
RECO_fits.header.set('WHAT_IS','CM @ ERR')

ERR_fits = fits.PrimaryHDU( ERR_list )
ERR_fits.header.set('EXTNAME','ERRS')
ERR_fits.header.set('WHAT_IS','list of modal errors to feed to PID')

RMS_fits = fits.PrimaryHDU( RMS_list )
RMS_fits.header.set('EXTNAME','RMS')
RMS_fits.header.set('WHAT_IS','std( cmd - aber_in_cmd_space )')

# add these all as fits extensions 
for f in [disturbfits, IMG_fits, RES_fits, RECO_fits, ERR_fits, CMD_fits, RMS_fits ]: #[Ufits, Sfits, Vtfits, CMfits, disturbfits, IMG_fits, ERR_fits, RECO_fits, CMD_fits, RMS_fits ]:
    static_ab_performance_fits.append( f ) 

#save data! 
static_ab_performance_fits.writeto( save_fits )




# PLOTTING SOME RESULTS 
psf_max = np.array( [ np.max( psf[ci_x1:ci_x2,ci_y1:ci_y2] )  for psf in static_ab_performance_fits['IMAGES'].data] )

psf_max_ref = np.max( recon_data['FPM_OUT'].data[ci_x1:ci_x2,ci_y1:ci_y2] )

fig,ax = plt.subplots( 4,2,figsize=(7,30))
ax[0,0].plot( static_ab_performance_fits['RMS'].data )
ax[0,0].set_ylabel('RMS in cmd space')
ax[0,0].set_xlabel('iteration')

ax[0,1].plot( psf_max/psf_max_ref )
ax[0,1].set_ylabel('max(I) / max(I_ref)')
ax[0,1].set_xlabel('iteration')


ax[1,0].imshow( recon_data['FPM_OUT'].data[cp_x1:cp_x2,cp_y1:cp_y2] )
ax[1,0].set_title('reference pupil')

ax[2,0].imshow( static_ab_performance_fits['IMAGES'].data[0][cp_x1:cp_x2,cp_y1:cp_y2]  )
ax[2,0].set_title('initial pupil with disturbance')

ax[3,0].imshow( static_ab_performance_fits['IMAGES'].data[-1][cp_x1:cp_x2,cp_y1:cp_y2]  ) 
ax[3,0].set_title('final pupil after 10 iterations')


ax[1,1].imshow( recon_data['FPM_OUT'].data[ci_x1:ci_x2,ci_y1:ci_y2] )
ax[1,1].set_title('reference PSF')

ax[2,1].imshow( static_ab_performance_fits['IMAGES'].data[0][ci_x1:ci_x2,ci_y1:ci_y2]  )
ax[2,1].set_title('initial PSF with disturbance')

ax[3,1].imshow( static_ab_performance_fits['IMAGES'].data[-1][ci_x1:ci_x2,ci_y1:ci_y2]  ) 
ax[3,1].set_title('final PSF after 10 iterations')

plt.subplots_adjust(wspace=0.5, hspace=0.5)
#plt.savefig(fig_path + f'closed_loop_on_static_aberration_RESULTS_disturb-kolmogorov_amp-{scrn_scaling_factor}_PID-{PID}_t-{tstamp}.png') 
plt.show() 






