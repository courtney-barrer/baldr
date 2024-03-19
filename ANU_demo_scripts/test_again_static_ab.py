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


from scipy.ndimage import gaussian_filter


# --- timestamp
tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")


# =====(1) 
# --- DM command to make flat DM (calibrated file provided by BMC with the DM) 
flat_dm_cmd = pd.read_csv(bdf.path_to_dm_flat_map,header=None)[0].values 
# read in DM deflection data and create interpolation functions 
deflection_data = pd.read_csv(root_path + "/DM_17DW019#053_deflection_data.csv", index_col=[0])
interp_deflection_1act = interpolate.interp1d( deflection_data['cmd'],deflection_data['1x1_act_deflection[nm]'] ) #cmd to nm deflection on DM from single actuator (from datasheet) 
interp_deflection_4x4act = interpolate.interp1d( deflection_data['cmd'],deflection_data['4x4_act_deflection[nm]'] ) #cmd to nm deflection on DM from 4x4 actuator (from datasheet) 

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


#recon_file = data_path + 'BDR_RECON_18-03-2024T14.14.29.fits'
#recon_data = fits.open( recon_file )

#image cropping coordinates 
if 'cropping_corners_r1' in recon_data['poke_images'].header:
    r1 = int(recon_data['poke_images'].header['cropping_corners_r1'])
    r2 = int(recon_data['poke_images'].header['cropping_corners_r2'])
    c1 = int(recon_data['poke_images'].header['cropping_corners_c1'])
    c2 = int(recon_data['poke_images'].header['cropping_corners_c2'])
    cropping_corners = [r1,r2,c1,c2]
else:
    cropping_corners = None

# pupil cropping coordinates
cp_x1,cp_x2,cp_y1,cp_y2 = recon_data[0].header['cp_x1'],recon_data[0].header['cp_x2'],recon_data[0].header['cp_y1'],recon_data[0].header['cp_y2']
# PSF cropping coordinates 
ci_x1,ci_x2,ci_y1,ci_y2 = recon_data[0].header['ci_x1'],recon_data[0].header['ci_x2'],recon_data[0].header['ci_y1'],recon_data[0].header['ci_y2']



raw_IM_data = recon_data['poke_images']
number_amp_samples = int( raw_IM_data.header['#ramp steps']) 
max_ramp = float(raw_IM_data.header['HIERARCH in-poke max amp']) 
min_ramp = float(raw_IM_data.header['HIERARCH out-poke max amp']) 
ramp_values = np.linspace( min_ramp, max_ramp, number_amp_samples)
modal_basis = recon_data['BASIS'].data

# ================================== This is what determines the poke amplitude for IM matrix
poke_amp_indx = number_amp_samples//2 - 2 # the smallest positive value 
# ==================================
print( f'======\ncalculating IM for pokeamp = {ramp_values[poke_amp_indx]}\n=====:::' )

# poke amplitude in DM command space used to generate IM 
IM_pokeamp = ramp_values[poke_amp_indx]
# dont forget to crop just square pupil region

agregated_pupils = [np.median(raw_IM_data.data[i],axis=0) for i in range(len(raw_IM_data.data))]

agregated_pupils_array = np.array( agregated_pupils[1:] ).reshape(number_amp_samples, modal_basis.shape[0],recon_data['FPM_IN'].data.shape[0], recon_data['FPM_IN'].data.shape[1])

IM_unfiltered_unflat = [bdf.get_error_signal( agregated_pupils_array[poke_amp_indx][m], reference_pupil_fits=recon_data, reduction_dict=None, pupil_indicies = [cp_x1,cp_x2,cp_y1,cp_y2]) for m in range(modal_basis.shape[0])] # [mode, x, y]

IM = np.array([list(im.reshape(-1)) for im in IM_unfiltered_unflat])


#IM_filt = IM.copy()
#IM_filt[abs(IM)<0.1]=0 # forcefully filter out noise in the IM 

#plt.imshow( np.array(IM)[65].reshape([cp_x2-cp_x1,cp_y2-cp_y1]));
#plt.colorbar();
#plt.title('row 65 from unfiltered IM constructed ')
#plt.show()

""" OLD ISSUE OF SHIFT IN IM/CM SUCH THAT AN RECONSTRUCTION OF COMMAND FROM KNOWN IMAGE GAVE A SHIFTED COMMAND - 

the issue was that DM hadn't updated before image taken for IM construction, so there were repeats etc..

Code below can be used as a check 
##========== CHECK THAT A NEW IMAGE MATCHES ROW IN IM
# (issue of repeating err signals in IM) 

i=75
cmdtmp = np.zeros(140)
cmdtmp[i] = IM_pokeamp
dm.send_data( flat_dm_cmd + cmdtmp ) #send dm data
time.sleep(0.5)
im = np.median( bdf.get_raw_images(camera, number_of_frames=5, cropping_corners=cropping_corners) , axis=0)

errsig =  bdf.get_error_signal( im, reference_pupil_fits = recon_data, reduction_dict=None, pupil_indicies = [cp_x1,cp_x2,cp_y1,cp_y2] )

fig,ax = plt.subplots(1,3,figsize=(15,5))

ax[0].set_title(f'errsign {i}')
im0 = ax[0].imshow(errsig)
plt.colorbar(im0,ax=ax[0])

ax[1].set_title(f'IM {i}')
im1 = ax[1].imshow( IM[i].reshape([cp_x2-cp_x1, cp_y2-cp_y1] ))
plt.colorbar(im1,ax=ax[1])

ax[2].set_title(f'residual {i}')
im2 = ax[2].imshow( errsig - IM[i].reshape([cp_x2-cp_x1, cp_y2-cp_y1]))
plt.colorbar(im2,ax=ax[2])
plt.show()

# OK this is the problem--- lets just take a new IM here

IM_new = []
for i in range(len(IM)):
    cmdtmp = np.zeros(140)
    cmdtmp[i] = IM_pokeamp
    dm.send_data( flat_dm_cmd + cmdtmp )
    time.sleep(0.2) 
    im = np.median( bdf.get_raw_images(camera, number_of_frames=5, cropping_corners=cropping_corners) , axis=0)
    errsig =  bdf.get_error_signal( im, reference_pupil_fits = recon_data, reduction_dict=None, pupil_indicies = [cp_x1,cp_x2,cp_y1,cp_y2] )

    IM_new.append( list(errsig.reshape(-1)) ) 

# now re-plot same as above with IM_new
i=65
cmdtmp = np.zeros(140)
cmdtmp[i] = IM_pokeamp
dm.send_data( flat_dm_cmd + cmdtmp ) #send dm data
time.sleep(0.5)
im = np.median( bdf.get_raw_images(camera, number_of_frames=5, cropping_corners=cropping_corners) , axis=0)

errsig =  bdf.get_error_signal( im, reference_pupil_fits = recon_data, reduction_dict=None, pupil_indicies = [cp_x1,cp_x2,cp_y1,cp_y2] )

fig,ax = plt.subplots(1,3,figsize=(15,5))

ax[0].set_title(f'errsign {i}')
im0 = ax[0].imshow(errsig)
plt.colorbar(im0,ax=ax[0])

ax[1].set_title(f'IM {i}')
im1 = ax[1].imshow( np.array(IM_new[i]).reshape([cp_x2-cp_x1, cp_y2-cp_y1] ))
plt.colorbar(im1,ax=ax[1])

ax[2].set_title(f'residual {i}')
im2 = ax[2].imshow( errsig - np.array(IM_new[i]).reshape([cp_x2-cp_x1, cp_y2-cp_y1]))
plt.colorbar(im2,ax=ax[2])
plt.show()
"""

#IM_filt = np.array(IM_new.copy())
#IM_filt[abs(np.array(IM_new))<0.1]=0 # forcefully filter out noise in the IM 


# =====(2) 
# --- setup camera
fps = float( recon_data[1].header['camera_fps'] )
camera = bdf.setup_camera(cameraIndex=0) #connect camera and init camera object
camera = bdf.set_fsp_dit( camera, fps=fps, tint=None) # set up initial frame rate, tint=None means max integration time for given FPS


# --- setup DM
dm, dm_err_code =  bdf.set_up_DM(DM_serial_number='17DW019#053')

# --- setup DM interaction and control matricies
modal_basis = recon_data['BASIS'].data #
# check modal basis dimensions are correct
if modal_basis.shape[1] != 140:
    raise TypeError( 'modal_basis.shape[1] != 140 => not right shape. Maybe we should transpose it?\nmodal_basis[i] should have a 140 length command for the DM corresponding to mode i')

# poke amplitude in DM command space used to generate IM 
#IM_pokeamp = float( recon_data['IM'].header['poke_amp_cmd'] )


#IM = recon_data['IM'].data # unfiltered


U,S,Vt = np.linalg.svd( IM ,full_matrices=True)

plt.figure()
plt.plot( S )
plt.axvline( len(S) * np.pi*2**2/(4.4)**2 ,linestyle=':',color='k',label=r'$D_{DM}^2/\pi r_{pup}^2$')
plt.ylabel('singular values',fontsize=15)
plt.xlabel('eigenvector index',fontsize=15)
plt.legend(fontsize=15)
plt.gca().tick_params( labelsize=15 )


Sfilt = S > S[ np.min( np.where( abs(np.diff(S)) < 1e-2 )[0] ) ]
Sigma = np.zeros( np.array(IM).shape, float)
np.fill_diagonal(Sigma, 1/abs(IM_pokeamp) * S[Sfilt], wrap=False) #

CM = np.linalg.pinv( U @ Sigma @ Vt ) # C = A @ M 

#plt.imshow( CM[:,52].reshape(cp_x2-cp_x1,cp_y2-cp_y1) );plt.colorbar();plt.show()

number_images_recorded_per_cmd = 5 #NDITs to take median over 
# using Ziegler-Nichols method
Ku = 2.
Tu = 2 #period of oscillations (samples) with ultimate gain 
PID = [0.45*Ku, 0.54*Ku/Tu, 0] # proportional, integator, differential gains  
Nint = 4 # used for integral term.. should be calcualted later. TO DO 
dt_baldr = 1  # used for integral term.. should be calcualted later. TO DO  


noise_level_IM = np.mean(IM)+5*np.std(IM)  #0.1

plt.figure()
plt.title('modal gains? np.sum( abs(IM) > IM_noise, axis=1)')
plt.imshow( bdf.get_DM_command_in_2D( np.sum( abs(IM) > noise_level_IM, axis=1) ))
plt.colorbar()
plt.show() 

# ====== modal gains 

#modal_gains = 1 * np.ones(140) #
modal_gains = np.sum( abs(IM) > noise_level_IM, axis=1)/np.max( np.sum( abs(IM) > noise_level_IM, axis=1)) 

modal_gains **= 0.5 # reduce curvature

#modal_gains = 1.0 * ( np.sum( abs(IM) > noise_level_IM, axis=1) > 0)
#0.5 * np.ones(len(modal_basis[0]))
#modal_gains = np.array([1/m if m!=0 else 0 for m in modal_gains])

cmd_region_filt = modal_gains > 0  # to filter where modal gains are non-zero (i.e. we can actuate here) 

# ======= disturbance 
#disturbance_cmd = np.zeros( len( flat_dm_cmd )) 
#disturbance_cmd[np.array([40,41,52,53,64,65])]=-0.13
#disturbance_cmd[np.array([75,76,85])]=-0.1
 
#disturbance_cmd -= 0.1 * pd.read_csv('/home/baldr/Documents/baldr/DMShapes/Crosshair140.csv').values.T.ravel()
#disturbance_cmd[np.array([65])]=0.08
#disturbance_cmd[np.array([5,16,28,40,52,64])]=0.05 # disturbance is ~zero mean! 

modes = bdf.construct_command_basis(dm , basis='Zernike', number_of_modes = 20, actuators_across_diam = 'full',flat_map=None)

mode_keys = list(modes.keys())

disturbance_cmd = 0.6*cmd_region_filt * ( flat_dm_cmd - modes[mode_keys[10]] ) 

plt.figure()
plt.title( f'static aberration to apply to DM (std = {np.std(disturbance_cmd)} in cmd space')
plt.imshow( bdf.get_DM_command_in_2D(disturbance_cmd, Nx_act=12) )
plt.colorbar()
plt.show()


# init main fits file for saving telemetry
static_ab_performance_fits = fits.HDUList([])

disturbfits = fits.PrimaryHDU( disturbance_cmd )
disturbfits.header.set('EXTNAME','DISTURBANCE')
disturbfits.header.set('WHAT_IS','disturbance in cmd space')

IMfits =  fits.PrimaryHDU( IM )
IMfits.header.set('EXTNAME','IM')
IMfits.header.set('WHAT IS','IM_filtered') 

CMfits =  fits.PrimaryHDU( CM )
CMfits.header.set('EXTNAME','CM')
CMfits.header.set('WHAT IS','CM_filtered') 

# init lists to hold data from control loop
IMG_list = [ ]
RES_list = [ ] #list( np.nan * np.zeros( int( (cp_x2 - cp_x1) * (cp_y2 - cp_y1) ) ) ) ]
RECO_list = [ ] #list( np.nan * flat_dm_cmd ) ]
CMD_list = [ list( flat_dm_cmd ) ] 
ERR_list = [ ]# list( np.nan * np.zeros( int( (cp_x2 - cp_x1) * (cp_y2 - cp_y1) ) ) ) ]  # length depends on cropped pupil when flattened 
RMS_list = [np.std( disturbance_cmd )] # to hold std( cmd - aber ) for each iteration
  

dm.send_data( flat_dm_cmd + disturbance_cmd )
time.sleep(1)
FliSdk_V2.Start(camera)    
time.sleep(1)
for i in range(100):

    # get new image and store it (save pupil and psf differently)
    IMG_list.append( list( np.median( bdf.get_raw_images(camera, number_of_frames=number_images_recorded_per_cmd, cropping_corners=cropping_corners) , axis=0)  ) ) 

    # create new error vector (remember to crop it!) with bdf.get_error_signal
    err_2d = bdf.get_error_signal( np.array(IMG_list[-1]), reference_pupil_fits = recon_data, reduction_dict=None, pupil_indicies = [cp_x1,cp_x2,cp_y1,cp_y2] ) # Note we can use recon data as long as reference pupils have FPM_ON and FPM_OFF extension names uniquely
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
    reco = list( CM.T @ RES_list[-1] )
    #reco_shift = reco[1:] + [np.median(reco)] # DONT KNOW WHY WE NEED THIS SHIFT!!!! ???
    #RECO_list.append( list( CM.T @ RES_list[-1] ) ) # reconstructed modal amplitudes
    RECO_list.append( reco )
    
    # to get error signal we apply modal gains
    ERR_list.append( list( np.sum( np.array([ g * a * B for g,a,B in  zip(modal_gains, RECO_list[-1], modal_basis)]) , axis=0) ) )

    # PID control 
    if len( ERR_list ) < Nint:
        cmd = PID[0] * np.array(ERR_list[-1]) +  PID[1] * np.sum( ERR_list,axis=0 ) * dt_baldr 
    else:
        cmd = PID[0] * np.array(ERR_list[-1]) +  PID[1] * np.sum( ERR_list[-Nint:] , axis = 0 ) * dt_baldr 
            
    cmderr =  cmd - np.mean(cmd) # REMOVE PISTON FORCEFULLY 
    """
    fig,ax = plt.subplots( 1,2 ) 
    ax[0].imshow(  bdf.get_DM_command_in_2D(disturbance_cmd, Nx_act=12) )
    ax[0].set_title( 'static aberration') 
    ax[1].imshow( bdf.get_DM_command_in_2D(cmdtmp, Nx_act=12) )
    ax[1].set_title( 'cmd vector') 
    """

    # check dm commands are within limits 
    if np.max(cmderr+flat_dm_cmd)>1:
        print(f'WARNING {sum(cmdtmp>1)} DM commands exceed max value of 1') 
        cmdtmp[cmderr+flat_dm_cmd>1] = 0.4 #force clip
    if np.min(cmderr+flat_dm_cmd)<0:
        print(f'WARNING {sum(cmdtmp<0)} DM commands exceed min value of 0') 
        cmdtmp[cmderr+flat_dm_cmd<0] = -0.4 # force clip
    # finally append it:
    #CMD_list.append( CMD_list[-1] + cmderr ) # CMD_list begins as flat DM 
    CMD_list.append( flat_dm_cmd + cmderr )
    RMS_list.append( np.std( np.array(CMD_list[-1]) - flat_dm_cmd + np.array(disturbance_cmd) ) )
    # apply control command to DM + our static disturbance 
    dm.send_data( CMD_list[-1] + disturbance_cmd ) 

# now flatten once finished
dm.send_data( flat_dm_cmd ) 

camera_info_dict = bdf.get_camera_info( camera )


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
for f in [disturbfits, IMfits, CMfits, IMG_fits, RES_fits, RECO_fits, ERR_fits, CMD_fits, RMS_fits ]: #[Ufits, Sfits, Vtfits, CMfits, disturbfits, IMG_fits, ERR_fits, RECO_fits, CMD_fits, RMS_fits ]:
    static_ab_performance_fits.append( f ) 

#save data! 
#save_fits = data_path + f'A_FIRST_closed_loop_on_static_aberration_PID-{PID}_t-{tstamp}.fits'
#static_ab_performance_fits.writeto( save_fits )


data = static_ab_performance_fits


plt.figure()
plt.plot( interp_deflection_4x4act( RMS_list ),'.' )
plt.ylabel('RMSE cmd space [nm RMS]')
plt.show() 


plt.figure()
plt.imshow( bdf.get_DM_command_in_2D(CMD_list[-1]));plt.colorbar();plt.show()
# we see that in open loop noise propagates badly outside of pupil
# need to filter better in IM 

# after N iterations where are the residuals occuring  
plt.figure()
plt.title('residuals on DM space')
plt.imshow( np.rot90( bdf.get_DM_command_in_2D(CMD_list[-3] - flat_dm_cmd + data['DISTURBANCE'].data).T,2) )
plt.colorbar()
#plt.savefig(fig_path + f'static_aberration_{tstamp}.png') 
#plt.savefig(fig_path + f'dynamic_aberration_{tstamp}.png') 
plt.show()


fig,ax = plt.subplots( data['IMAGES'].data.shape[0]+1,3,figsize=(5,1.5*data['IMAGES'].data.shape[0]) )
plt.subplots_adjust(hspace=0.5,wspace=0.5)
im00=ax[0,0].imshow( np.rot90( bdf.get_DM_command_in_2D(flat_dm_cmd + disturbance_cmd ).T,2 ) )
plt.colorbar(im00, ax=ax[0,0])
ax[0,0].set_title('disturbance + flat cmd')
for i, (im, err, cmd) in enumerate( zip(data['IMAGES'].data, data['ERRS'].data, data['CMDS'].data[1:])):
    imA = ax[i+1,0].imshow( im )
    errA = ax[i+1,1].imshow(np.rot90( bdf.get_DM_command_in_2D(err).T,2))
    cmdA = ax[i+1,2].imshow(np.rot90( bdf.get_DM_command_in_2D(cmd).T,2))
    for A,axx in zip([imA,errA,cmdA], [ax[i+1,0],ax[i+1,1],ax[i+1,2]]): 
        plt.colorbar(A, ax=axx)
        axx.axis('off')
    ax[i+1,0].set_title(f'image {i}')
    ax[i+1,1].set_title(f'err {i}') 
    ax[i+1,2].set_title(f'cmd {i}') 
#plt.savefig(fig_path + f'closed_loop_iterations_{tstamp}.png') 
plt.show()


i=23
fig,ax = plt.subplots(1,3,figsize=(15,5))

ax[0].set_title(f'disturbance {i}')
im0 = ax[0].imshow(np.rot90( bdf.get_DM_command_in_2D(disturbance_cmd).T,2))
plt.colorbar(im0,ax=ax[0])

ax[1].set_title(f'cmd err {i}')
im1 = ax[1].imshow(np.rot90( bdf.get_DM_command_in_2D(CMD_list[i]-flat_dm_cmd).T,2))
plt.colorbar(im1,ax=ax[1])

ax[2].set_title(f'residual cmd {i}')
im2 = ax[2].imshow(np.rot90( bdf.get_DM_command_in_2D(CMD_list[i]-flat_dm_cmd+disturbance_cmd).T,2))
plt.colorbar(im2,ax=ax[2])
plt.show()


# is there overlap with the err signal in intensity space 
fig,ax = plt.subplots(1,2,figsize=(15,5))
ax[0].set_title('image err')
im0 = ax[0].imshow( np.array(RES_list[i]).reshape( [cp_x2-cp_x1, cp_y2-cp_y1]) );
plt.colorbar(im0,ax=ax[0])

ax[1].set_title('image')
im1 = ax[1].imshow( (np.array(IMG_list[i])-recon_data['FPM_IN'].data)[cp_x1:cp_x2, cp_y1:cp_y2] );
plt.colorbar(im1,ax=ax[1])

plt.show()




