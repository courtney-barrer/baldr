
from baldr_control import ZWFS
from baldr_control import phase_control
from baldr_control import pupil_control
from baldr_control import utilities as util

import numpy as np
import matplotlib.pyplot as plt 
import time 
import datetime
import aotools
from astropy.io import fits

fig_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/figures/' 
data_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/data/telemetry/'


def close_loop(zwfs, phase_ctrl, disturbance_coes, Kp, Ki, iterations, closed_loop=False, close_after=np.inf, update_dist_every=2, telemetry=True, psf_crop_region=[10,20,10,20]):

    # ============ INITIALIZATION 
    #telemetry = True
    #closed_loop = True
    flat_dm_cmd = zwfs.dm_shapes['flat_dm']


    safety_lock = False # keeps loop open if DM cmd is too large     
    #control matrix 
    CM = phase_ctrl.ctrl_parameters[ctrl_method_label]['CM']
    # mode to DM command matrix 
    M2C = phase_ctrl.config['M2C']
    # filter for DM command region (where basis is defined) 
    cmd_region_filt = phase_ctrl.config['active_actuator_filter'] 

    #make our disturbance command from the coefcients 
    disturbance_cmd = M2C @ disturbance_coes


    # gains 
    phase_ctrl.config['Ki'] = Ki # np.zeros( M2C.shape[1]  ) 
    phase_ctrl.config['Kp'] = Kp #np.zeros( M2C.shape[1] )  #np.logspace( -3, 0, M2C.shape[1])[::-1] #
    #phase_ctrl.config['Kp'][[0,1]] = 1/10 
    zeros = np.zeros( len(phase_ctrl.config['Ki'] ) )
    # Save fits?
    if telemetry:
        phase_scrn_tele = [] # input disturbance
        disturbance_dm_tele = [] # input disturbance 
        img_tele = [] # raw image 
        img_filtered_tele = [] # normalized image filtered within defined pupil
        to_lock_buffer_tele = [] # intensity average of pixels registered at seoncary obstruction normalized by mean intensity in the pupil readout. Can be used as an instantaneous proxy to Strehl. 
        img_err_tele = [] # error in the image space (subtracting set point etc) 
        modal_err_tele = [] # using control matrix to put in modal space 
        u_tele = [] # applying PI gains to modal errors 
        dm_cmd_tele = [] # converting modes to DM commands 
        psf_img_tele = [] #image of PSF

        RMS_tele = [] 
        Kp_tele = [phase_ctrl.config['Kp']]
        Ki_tele = [phase_ctrl.config['Ki']]
        closed_loop_state = [] # 1 is closed 


        #tele_list = [ phase_scrn_tele, disturbance_dm_tele, img_tele, img_filtered_tele , to_lock_buffer_tele , img_err_tele, modal_err_tele, u_tele, dm_cmd_tele, psf_img_tele , RMS_tele ,Kp_tele,Ki_tele, closed_loop_state]

        #tele_label_list = ['PHASE_SCRN', 'DISTURBANCE', 'IMG','IMG_FILTERED','I_SECONDARY','IMG_ERR', 'MODE_ERR', 'MODE_PI_ERR','DM_CMD','PSF','RMS','Kp','Ki','CLOSED_LOOP']

        tele_dict = {'PHASE_SCRN':phase_scrn_tele, 'DISTURBANCE':disturbance_dm_tele, 'IMG':img_tele,'IMG_FILTERED': img_filtered_tele,'I_SECONDARY':to_lock_buffer_tele,'IMG_ERR':img_err_tele, 'MODE_ERR': modal_err_tele, 'MODE_PI_ERR':u_tele,'DM_CMD':dm_cmd_tele,'PSF':psf_img_tele ,'RMS':RMS_tele,'Kp':Kp_tele,'Ki':Ki_tele,'CLOSED_LOOP':closed_loop_state} 

    # ======== Ibuffers to hold control information 
    img_buffer = [] # to hold raw images

    imgs2average = [] # to hold images to average to update 'b' gains 

    img_filtered_buffer = [] # average normalized by mean intensity in the pupil readout and filtered for pixels registered in the telescope pupil.

    to_lock_buffer = [] # intensity average of pixels registered at seoncary obstruction normalized by mean intensity in the pupil readout. Can be used as an instantaneous proxy to Strehl. 

    img_err_buffer = [] # tp hold err in pixel space (calculated from setpoint) 

    modal_err_buffer = [] #to hold modal error residuals to be reconstructed 

    u_buffer = [list(np.zeros(len(phase_ctrl.config['Kp'])))] # after applying gains to modal errors (PI controller) 

    dm_cmd_buffer = [] # the DM command to apply
       
    # START WITH A FLAT DM (SO REFERENCE IMAGE) 
    zwfs.dm.send_data ( flat_dm_cmd ) 

    # ======== START 
    for i in range( iterations) : 

        if (i > close_after) and not (safety_lock) :
            closed_loop = True

        time.sleep(0.005) 
        # raw ZWFS image
        tmpimg_list = []
        for _ in range(5):
            tmpimg_list.append( zwfs.get_image() )
            time.sleep(0.005)
        imgtmp = np.median(  tmpimg_list, axis=0 ) 
        img_buffer.insert( 0, imgtmp ) # pixel space

        # images to average to update set points
        imgs2average.insert(0, img_buffer[0] )

        # normalized intensity in center pixels
        to_lock_buffer.insert(0, np.mean( img_buffer[0].reshape(-1)[zwfs.refpeak_pixel_filter] ) / np.mean(img_buffer[0])  )

        #normalize over entire image and filter for pixels in the pupil region
        img_filtered_buffer.insert(0, 1/np.mean( img_buffer[0] ) * img_buffer[0].reshape(-1)[zwfs.pupil_pixel_filter] ) 

        # phase errors in pixel space 
        img_err_buffer.insert( 0, phase_ctrl.get_img_err(img_filtered_buffer[0]) ) #pixel space

        # modal residuals coefficients from the ZWFS
        modal_err_buffer.insert( 0, CM.T @ img_err_buffer[0] ) #DM modal space

        # safety / anti windup
        # modes normalized <M|M>=1 in cmd basis (clamping anti windup https://www.youtube.com/watch?app=desktop&v=UMit8mVCJ_I)
        # Ki[ abs( modal_err_buffer[0] ) > 0.4  ] = 0 #any modes over some threshold we cut off the integration in the PID
         
        # output in modal space
        if closed_loop:
            u_buffer.insert( 0, phase_ctrl.config['Ki'] * np.array( u_buffer[0] ) + phase_ctrl.config['Kp'] * np.array( modal_err_buffer[0] ) ) #DM modal space
        else: 
            u_buffer.insert( 0, zeros ) 

        # convert to delta DM command
        dm_res = M2C @ u_buffer[0]
        dm_res -= np.mean( dm_res ) # force to be piston free over DM ! 

        if np.any(abs(dm_res) > 0.5):
            closed_loop = False
            safety_lock = True
            print(f'!!! DM command too large!!!\nopenning loop on iteration {i}')

        if closed_loop:
            dm_cmd_buffer.insert(0, flat_dm_cmd - dm_res )  # DM command space
        else:
            dm_cmd_buffer.insert(0, flat_dm_cmd )
        closed_loop_state.append( int(closed_loop)  ) # keep track if we're opened or closed 

        # function to update input phase 
        if np.mod(update_dist_every, i+1):
            disturbance_coes = update_disturbance_modal_coefficient(disturbance_coes) #applies AR(1) model to modal coefficients 
            disturbance_cmd = M2C @ disturbance_coes # rebuild our disturbance command 
        
        """
        for skip in range(rows_to_jump):
        scrn.add_row() 
        # get our new Kolmogorov disturbance command 
        disturbance_cmd = cmd_region_filt * cmd_region_filt * util.create_phase_screen_cmd_for_DM(scrn=scrn, scaling_factor = scrn_scaling_factor, drop_indicies = corner_indicies, plot_cmd=False)
        disturbance_cmd -= np.mean( disturbance_cmd )
        """

        # apply DM command if in closed loop
        if closed_loop:
            zwfs.dm.send_data( dm_cmd_buffer[0] + disturbance_cmd  )

        else:
            zwfs.dm.send_data( flat_dm_cmd + disturbance_cmd  )

        if telemetry:
            img_tele.append( img_buffer[0] ) # raw image 
            to_lock_buffer_tele.append( to_lock_buffer[0] ) 
            img_filtered_tele.append( img_filtered_buffer[0] )  # normalized image filtered within defined pupil
            img_err_tele.append( img_err_buffer[0] )  # error in the image space (subtracting set point etc) 
            modal_err_tele.append( modal_err_buffer[0] ) # using control matrix to put in modal space 
            u_tele.append( u_buffer[0] )  # applying PI gains to modal errors 
            dm_cmd_tele.append( dm_cmd_buffer[0] ) # converting modes to DM commands 
        #average_img_tele = [] #

            # get the PSF 
            psf_img = zwfs.get_image_in_another_region( crop_region = psf_crop_region )

            psf_img_tele.append( psf_img ) 

        #phase_scrn_tele.append( scrn.scrn ) # input disturbance
            disturbance_dm_tele.append( disturbance_cmd  )
            if closed_loop:
                RMS_tele.append( np.std( cmd_region_filt * ( flat_dm_cmd - np.array(dm_cmd_buffer[0]) + np.array(disturbance_cmd) ) ) )
            else :
                RMS_tele.append( np.std( cmd_region_filt * (  np.array(disturbance_cmd) ) ) )

        # pop out the last elements of all the lists
        if len( img_buffer ) > 3:
            img_buffer.pop()
            img_filtered_buffer.pop() 
            to_lock_buffer.pop()
            img_err_buffer.pop()
            modal_err_buffer.pop()
            u_buffer.pop()
            dm_cmd_buffer.pop()

    if telemetry:

        telemetry_fits = fits.HDUList([])
        print('savefits') 
        for label, data in tele_dict.items(): #zip(tele_label_list, tele_list):
            fitsTMP = fits.PrimaryHDU( data )
            fitsTMP.header.set('EXTNAME', label )
            telemetry_fits.append( fitsTMP )  

            fname =f'TELEMETRY_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}_ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}readout_mode-12x12_{tstamp}.fits'

            #telemetry_fits.writeto( data_path + fname )


        return( telemetry_fits ) 

    else:
        return( dm_cmd_buffer )



def update_disturbance_modal_coefficient(coes):
    #checking psf imaging and tip commands are seen - we get about 16 pixels per tip DM cmd, 0.1 amp -> 1.6 pixel shift in centroid (without pixelation)
    for i,c in enumerate(coes):
        if c!=0:
            coes[i] = coes[i] + 0.005 * np.random.randn()
            if coes[i] < 0.2:
                coes[i] = coes[i] + 0.002
    return(coes) 


# ============= SETUP 
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

psf_crop_region = [215,275,230,290]


##
##    START CAMERA 
zwfs.start_camera()

# init our phase controller 
phase_ctrl = phase_control.phase_controller_1(config_file = None) 
#phase_ctrl.change_control_basis_parameters(number_of_controlled_modes=20, basis_name='Zernike' , dm_control_diameter = 10 ) 

# inti our pupil controller
pupil_ctrl = pupil_control.pupil_controller_1(config_file = None)

pupil_report = pupil_control.analyse_pupil_openloop( zwfs, debug = True, return_report = True)

if pupil_report['pupil_quality_flag'] == 1: 
    zwfs.update_reference_regions_in_img( pupil_report ) # 
else:
    print('implement proceedure X1') 

# ============= BUILD CONTROL MODEL  
ctrl_method_label = 'ctrl_1'
phase_ctrl.build_control_model( zwfs , poke_amp = -0.1, label=ctrl_method_label, debug = True)  


# ========== NOISE ANALYSIS 
#if we put zero disturbance in the system what noise do we get? 

zwfs.dm.send_data( flat_dm_cmd ) 

# DISTURBANCE
# disturbance_cmd = 0.3* phase_ctrl.config['M2C'].T[0]
disturbance_coes = np.zeros( phase_ctrl.config['M2C'].shape[1]  ) 

cmd_region_filt = phase_ctrl.config['active_actuator_filter'] 

# DM cmd to radians etc 
radpercmd = 12.5 # radians on wavefront per DM cmd 
wvl0 = 1290 #nm 
M2C = phase_ctrl.config['M2C'] 
Ki = np.zeros( M2C.shape[1]  )
Kp = np.zeros( M2C.shape[1]  )

tele_fits = close_loop(zwfs, phase_ctrl,  disturbance_coes, Kp, Ki, iterations=500, closed_loop=False, close_after = 1e4, update_dist_every=5, telemetry=True, psf_crop_region = psf_crop_region)

fig,ax = plt.subplots( 1, 1 )
#plt.plot( np.mean( tele_fits['MODE_ERR'].data, axis=0) )
vartmp =np.var( tele_fits['MODE_ERR'].data, axis=0)
n = tele_fits['MODE_ERR'].data.shape[0]
ax.errorbar( np.arange(1, len( M2C[0] )+1), vartmp, yerr=vartmp * np.sqrt( 2/(n+1)) ,marker='o',ls='none') 
ax.set_ylabel(r'$\sigma^2(a_i)$' , fontsize=15 ) 
ax.set_xlabel(r'mode index $(i)$' , fontsize=15 ) 
ax.tick_params(labelsize=15)
#plt.savefig( data_path + f'MODE_VAR_ZERO_DISTURB_NOISE_ANALYSIS_TELEMETRY_readout_mode-12x12_{tstamp}.png' , bbox_inches='tight', dpi=300) 

fig,ax = plt.subplots( 1, 1 )
#plt.plot( np.mean( tele_fits['MODE_ERR'].data, axis=0) )
meantmp =np.mean( tele_fits['MODE_ERR'].data, axis=0)
n = tele_fits['MODE_ERR'].data.shape[0]
ax.errorbar( np.arange(1, len( M2C[0] )+1), meantmp, yerr=vartmp * np.sqrt( 1/n) ,marker='o',ls='none') 
ax.set_ylabel(r'$<a_i>$' , fontsize=15 ) 
ax.set_xlabel(r'mode index $(i)$' , fontsize=15 ) 
ax.tick_params(labelsize=15)
#plt.savefig( data_path + f'MODE_MEAN_ZERO_DISTURB_NOISE_ANALYSIS_TELEMETRY_readout_mode-12x12_{tstamp}.png' , bbox_inches='tight', dpi=300) 

plt.show()





# ============= INIT PHASE SCREEN  
flat_dm_cmd = zwfs.dm_shapes['flat_dm']

zwfs.dm.send_data( flat_dm_cmd ) 

#  DISTURBANCE
#disturbance_cmd = 0.3* phase_ctrl.config['M2C'].T[0]
disturbance_coes = np.zeros( phase_ctrl.config['M2C'].shape[1]  ) 
disturbance_coes[0] = 0.3 # only put tip

cmd_region_filt = phase_ctrl.config['active_actuator_filter'] 

# DM cmd to radians etc 
radpercmd = 12.5 # radians on wavefront per DM cmd 
wvl0 = 1290 #nm 
M2C = phase_ctrl.config['M2C'] 
Ki = np.zeros( M2C.shape[1]  )
Kp = np.zeros( M2C.shape[1]  )


#checking psf imaging and tip commands are seen - we get about 16 pixels per tip DM cmd, 0.1 amp -> 1.6 pixel shift in centroid (without pixelation)

#Kp[[0,1]] = 1/15, 0*1/10 #1/18, 1/18 # 1/12, 1/12 #pp_grid[iopt]
#Ki[[0,1]] = 0.4, 0.   # ii_grid[jopt]

tele_fits = close_loop(zwfs, phase_ctrl,  disturbance_coes, Kp, Ki, iterations=500, closed_loop=False, close_after = 1e4, update_dist_every=5, telemetry=True, psf_crop_region=psf_crop_region)

#fname =f'ZERO_DISTURB_NOISE_ANALYSIS_TELEMETRY_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}_ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}readout_mode-12x12_{tstamp}.fits'

#tele_fits.writeto( data_path + fname )



#RMSE

plt.figure()
#plt.plot( 2 * interp_deflection_4x4act( RMS_tele ),'.',label='residual' )
plt.plot( wvl0/(2*np.pi)  * radpercmd * np.array( tele_fits['RMS'].data ) ,'-',label='residual' )
plt.plot( wvl0/(2*np.pi)  * radpercmd * np.array([np.std(cmd_region_filt * d) for d in tele_fits['DISTURBANCE'].data]) ,'-',label='disturbance' )
plt.ylabel('tip/tilt [nm RMS]')
plt.xlabel('iteration')
plt.legend()
#plt.savefig(data_path + f'E2_TIP-TILT_CLOSED_DYNAMIC_TELEMETRY_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}_ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}readout_mode-12x12_{tstamp}.png')
#plt.xlim([0,50]);plt.ylim([24,26]);plt.show()
#print( 'final RMSE =', interp_deflection_4x4act(RMS_tele[-1] ) )
#print( 'RMSE after 100 iter. =',interp_deflection_4x4act( RMS_tele[100] ) );
plt.show()



# first 5 iterations 
iterations2plot = [0,1,2,3,4,5,-2,-1]
fig, ax = plt.subplots( len(iterations2plot), 4 ,figsize=(10,15))
ax[0,1].set_title('disturbance',fontsize=15)
ax[0,0].set_title(r'$I - I_0$',fontsize=15)
ax[0,2].set_title('CMD',fontsize=15)
#ax[0,3].set_title('DM CMD (feedback)',fontsize=15)
ax[0,3].set_title('RESIDUAL',fontsize=15)
for i,idx in enumerate(iterations2plot):
    im0 = ax[i,0].imshow(tele_fits['IMG'].data[idx]/np.mean(tele_fits['IMG'].data[idx]) - phase_ctrl.I0_2D  )
    plt.colorbar(im0, ax= ax[i,0]) 
    im1 = ax[i,1].imshow( util.get_DM_command_in_2D(tele_fits['DISTURBANCE'].data[idx]) )
    plt.colorbar(im1, ax= ax[i,1])
    #im2 = ax[i,2].imshow( util.get_DM_command_in_2D(u_tele[idx]) )
    #plt.colorbar(im2, ax= ax[i,2])
    im3 = ax[i,2].imshow( util.get_DM_command_in_2D(flat_dm_cmd - tele_fits['DM_CMD'].data[idx] )) 
    plt.colorbar(im3, ax= ax[i,2])
    im4 = ax[i,3].imshow( util.get_DM_command_in_2D(flat_dm_cmd - tele_fits['DM_CMD'].data[idx]  + tele_fits['DISTURBANCE'].data[idx]) )
    plt.colorbar(im4, ax= ax[i,3])

plt.show() 




plt.figure() 
for i in [2,3,-2,-1]:
    plt.plot( tele_fits['MODE_ERR'].data[i], label=i) 
plt.xlabel('modes')
plt.ylabel('reconstruction coefficient')
plt.legend() 
plt.show()

plt.figure() 
for i in [2,3,-2,-1]:
    plt.plot( tele_fits['MODE_PI_ERR'].data[i], label=i) 
plt.xlabel('modes')
plt.ylabel('reconstruction coefficient')
plt.legend() 
plt.show()






"""

scrn_scaling_factor = 0.2
# --- create infinite phasescreen from aotools module 
Nx_act = 12
screen_pixels = Nx_act*2**5 # some multiple of numer of actuators across DM 
D = 1.8 #m effective diameter of the telescope
scrn = aotools.infinitephasescreen.PhaseScreenVonKarman(nx_size=screen_pixels, pixel_scale=D/screen_pixels,r0=0.1,L0=12)

corner_indicies = [0, Nx_act-1, Nx_act * (Nx_act-1), -1] # Beware -1 index doesn't work if inserting in list! This is  ok for for use with create_phase_screen_cmd_for_DM function.

disturbance_cmd = cmd_region_filt * util.create_phase_screen_cmd_for_DM(scrn=scrn, scaling_factor = scrn_scaling_factor, drop_indicies = corner_indicies, plot_cmd=True)  # normalized flat_dm +- scaling_factor?

disturbance_cmd -= np.mean( disturbance_cmd ) # no piston!! 

rows_to_jump = 1 # how many rows to jump on initial phase screen for each Baldr loop

distance_per_correction = rows_to_jump * D/screen_pixels # effective distance travelled by turbulence per AO iteration 
print(f'{rows_to_jump} rows jumped per AO command in initial phase screen of {screen_pixels} pixels. for {D}m mirror this corresponds to a distance_per_correction = {distance_per_correction}m')

#plt.figure();plt.imshow( util.get_DM_command_in_2D(disturbance_cmd ));plt.show() #to check on filtered cmd space
"""







""" 
EXPERIMENTS 

Ki = np.zeros( M2C.shape[1]  )
Kp = np.zeros( M2C.shape[1]  )
pp_grid = np.logspace(-2,0,12)
ii_grid = np.logspace( -2,-0.1,10)
rms_grid = np.zeros( [len(pp_grid),len(ii_grid)] )
for i,pp in enumerate(pp_grid):
    print( f'{i}/{len(pp_grid)}' )
    for j,ii in enumerate(ii_grid):
        Kp[[0,1]] = pp,pp
        Ki[[0,1]] = ii,ii

        tele_fits = close_loop(zwfs, phase_ctrl, disturbance_cmd, Kp, Ki, iterations=10, closed_loop=True, telemetry=True)

        rms_grid[i,j] = np.mean( wvl0/(2*np.pi)  * radpercmd * np.array( tele_fits['RMS'].data ) ) 

        zwfs.dm.send_data( flat_dm_cmd ) 
        time.sleep(0.01)
#['PHASE_SCRN', 'DISTURBANCE', 'IMG','IMG_FILTERED','I_SECONDARY','IMG_ERR', 'MODE_ERR', 'MODE_PI_ERR','DM_CMD','RMS','Kp','Ki','CLOSED_LOOP']
iopt, jopt = np.unravel_index( np.argmin( rms_grid ), rms_grid.shape )

#checking psf imaging and tip commands are seen - we get about 16 pixels per tip DM cmd, 0.1 amp -> 1.6 pixel shift in centroid (without pixelation)
amp_grid = np.linspace(0,0.6,10)
psf_dist = []
for a in np.linspace(0,0.6,10):
    zwfs.dm.send_data( flat_dm_cmd ) 
    time.sleep(0.1)
    img_0 = np.log10( zwfs.get_image_in_another_region(psf_crop_region))

    zwfs.dm.send_data( flat_dm_cmd + a * phase_ctrl.config['M2C'].T[0]) 
    time.sleep(0.1)
    img_1 = np.log10( zwfs.get_image_in_another_region(psf_crop_region))
    #plt.figure();plt.imshow(img_0 ) ;plt.title('before')
    #plt.figure();plt.imshow( img_1 ) ;plt.title('after')
    #plt.show()

    d = np.array( np.unravel_index(np.argmax( img_0 ),img_0.shape) ) - np.array(list(np.unravel_index(np.argmax( img_1 ),img_1.shape)) )
    psf_dist.append( (d @ d )**.5 )

plt.plot( amp_grid, psf_dist )
plt.xlabel('DM tip amplitude')
plt.ylabel('centroid difference (pixels)')
plt.show()
print( f'on average we move roughly {np.mean( np.diff(psf_dist)/np.diff(amp_grid) )} pixels per tip cmd')  
"""












if 1:
    # ============ INITIALIZATION 
    telemetry = True
    closed_loop = True
    flat_dm_cmd = zwfs.dm_shapes['flat_dm']
    #control matrix 
    CM = phase_ctrl.ctrl_parameters[ctrl_method_label]['CM']
    # mode to DM command matrix 
    M2C = phase_ctrl.config['M2C']
    # gains 
    phase_ctrl.config['Ki'] =  np.zeros( M2C.shape[1]  ) 
    phase_ctrl.config['Kp'] = np.zeros( M2C.shape[1] )  #np.logspace( -3, 0, M2C.shape[1])[::-1] #
    phase_ctrl.config['Kp'][[0,1]] = 1/20
    #phase_ctrl.config['Ki'][[0,1]] = 1

    # Save fits?
    if telemetry:
        phase_scrn_tele = [] # input disturbance
        disturbance_dm_tele = [] # input disturbance 
        img_tele = [] # raw image 
        img_filtered_tele = [] # normalized image filtered within defined pupil
        to_lock_buffer_tele = [] # intensity average of pixels registered at seoncary obstruction normalized by mean intensity in the pupil readout. Can be used as an instantaneous proxy to Strehl. 
        img_err_tele = [] # error in the image space (subtracting set point etc) 
        modal_err_tele = [] # using control matrix to put in modal space 
        u_tele = [] # applying PI gains to modal errors 
        dm_cmd_tele = [] # converting modes to DM commands 
        RMS_tele = [] 
        Kp_tele = [phase_ctrl.config['Kp']]
        Ki_tele = [phase_ctrl.config['Ki']]
        closed_loop_state = [] # 1 is closed 
        #average_img_tele = [] #
        #pup_err_tele = [] 
        #open_loop_tele = []
        #b_tele = []
        #A_tele = []
        tele_list = [ phase_scrn_tele, disturbance_dm_tele, img_tele, img_filtered_tele ,\
     to_lock_buffer_tele , img_err_tele, modal_err_tele, u_tele, dm_cmd_tele, RMS_tele ,Kp_tele,Ki_tele, closed_loop_state]
        tele_label_list = ['PHASE_SCRN', 'DISTURBANCE', 'IMG','IMG_FILTERED','I_SECONDARY','IMG_ERR', 'MODE_ERR', 'MODE_PI_ERR','DM_CMD','RMS','Kp','Ki','CLOSED_LOOP']

    # ======== Ibuffers to hold control information 
    img_buffer = [] # to hold raw images

    imgs2average = [] # to hold images to average to update 'b' gains 

    img_filtered_buffer = [] # average normalized by mean intensity in the pupil readout and filtered for pixels registered in the telescope pupil.

    to_lock_buffer = [] # intensity average of pixels registered at seoncary obstruction normalized by mean intensity in the pupil readout. Can be used as an instantaneous proxy to Strehl. 

    img_err_buffer = [] # tp hold err in pixel space (calculated from setpoint) 

    modal_err_buffer = [] #to hold modal error residuals to be reconstructed 

    u_buffer = [list(np.zeros(len(phase_ctrl.config['Kp'])))] # after applying gains to modal errors (PI controller) 

    dm_cmd_buffer = [] # the DM command to apply
       
    # START WITH A FLAT DM (SO REFERENCE IMAGE) 
    zwfs.dm.send_data ( flat_dm_cmd ) 

    # ======== START 
    for i in range(10) : 
        time.sleep(0.005) 
        # raw ZWFS image
        tmpimg_list = []
        for _ in range(5):
            tmpimg_list.append( zwfs.get_image() )
            time.sleep(0.005)
        imgtmp = np.median(  tmpimg_list, axis=0 ) 
        img_buffer.insert( 0, imgtmp ) # pixel space

        # images to average to update set points
        imgs2average.insert(0, img_buffer[0] )

        # normalized intensity in center pixels
        to_lock_buffer.insert(0, np.mean( img_buffer[0].reshape(-1)[zwfs.refpeak_pixel_filter] ) / np.mean(img_buffer[0])  )

        #normalize over entire image and filter for pixels in the pupil region
        img_filtered_buffer.insert(0, 1/np.mean( img_buffer[0] ) * img_buffer[0].reshape(-1)[zwfs.pupil_pixel_filter] ) 

        # phase errors in pixel space 
        img_err_buffer.insert( 0, phase_ctrl.get_img_err(img_filtered_buffer[0]) ) #pixel space

        # modal residuals coefficients from the ZWFS
        modal_err_buffer.insert( 0, CM.T @ img_err_buffer[0] ) #DM modal space

        # safety / anti windup
        # modes normalized <M|M>=1 in cmd basis (clamping anti windup https://www.youtube.com/watch?app=desktop&v=UMit8mVCJ_I)
        # Ki[ abs( modal_err_buffer[0] ) > 0.4  ] = 0 #any modes over some threshold we cut off the integration in the PID
         
        # output in modal space
        u_buffer.insert( 0, phase_ctrl.config['Ki'] * np.array( u_buffer[0] ) + phase_ctrl.config['Kp'] * np.array( modal_err_buffer[0] ) ) #DM modal space

        # convert to DM command
        dm_res = M2C @ u_buffer[0]
        dm_res -= np.mean( dm_res ) # force to be piston free over DM ! 

        if np.any(abs(dm_res) > 0.5):
            closed_loop = False
            print(f'!!! DM command too large!!!\nopenning loop on iteration {i}')

        dm_cmd_buffer.insert(0, flat_dm_cmd - dm_res )  # DM command space

        closed_loop_state.append( int(closed_loop)  ) # keep track if we're opened or closed 

        # function to update input phase 
        """
        for skip in range(rows_to_jump):
        scrn.add_row() 
        # get our new Kolmogorov disturbance command 
        disturbance_cmd = cmd_region_filt * cmd_region_filt * util.create_phase_screen_cmd_for_DM(scrn=scrn, scaling_factor = scrn_scaling_factor, drop_indicies = corner_indicies, plot_cmd=False)
        disturbance_cmd -= np.mean( disturbance_cmd )
        """

        # apply DM command if in closed loop
        if closed_loop:
            zwfs.dm.send_data( dm_cmd_buffer[0] + disturbance_cmd  )

        else:
            zwfs.dm.send_data( flat_dm_cmd + disturbance_cmd  )

        if telemetry:
            img_tele.append( img_buffer[0] ) # raw image 
            to_lock_buffer_tele.append( to_lock_buffer[0] ) 
            img_filtered_tele.append( img_filtered_buffer[0] )  # normalized image filtered within defined pupil
            img_err_tele.append( img_err_buffer[0] )  # error in the image space (subtracting set point etc) 
            modal_err_tele.append( modal_err_buffer[0] ) # using control matrix to put in modal space 
            u_tele.append( u_buffer[0] )  # applying PI gains to modal errors 
            dm_cmd_tele.append( dm_cmd_buffer[0] ) # converting modes to DM commands 
        #average_img_tele = [] #

        #phase_scrn_tele.append( scrn.scrn ) # input disturbance
            disturbance_dm_tele.append( disturbance_cmd  )

            RMS_tele.append( np.std( cmd_region_filt * ( np.array(dm_cmd_buffer[0]) - flat_dm_cmd + np.array(disturbance_cmd) ) ) )


        # pop out the last elements of all the lists
        if len( img_buffer ) > 3:
            img_buffer.pop()
            img_filtered_buffer.pop() 
            to_lock_buffer.pop()
            img_err_buffer.pop()
            modal_err_buffer.pop()
            u_buffer.pop()
            dm_cmd_buffer.pop()

    if telemetry:

        telemetry_fits = fits.HDUList([])
        print('savefits') 
        for label,data in zip(tele_label_list, tele_list):
            fitsTMP = fits.PrimaryHDU( data )
            fitsTMP.header.set('EXTNAME', label )
            telemetry_fits.append( fitsTMP )  

            fname =f'TELEMETRY_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}_ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}readout_mode-12x12_{tstamp}.fits'

            #telemetry_fits.writeto( data_path + fname )



"""
# SAVING FITS 
if telemetry:
    telemetry_fits = fits.HDUList([])
    print('savefits') 
    for label,data in zip(tele_label_list, tele_list):
        fitsTMP = fits.PrimaryHDU( data )
        fitsTMP.header.set('EXTNAME', label )
        telemetry_fits.append( fitsTMP )  

    fname =f'TELEMETRY_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}_ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}readout_mode-12x12_{tstamp}.fits'

    telemetry_fits.writeto( data_path + fname )
"""

# plotting 
import pandas as pd 
from scipy import interpolate

deflection_data = pd.read_csv("/home/baldr/Documents/baldr/DM_17DW019#053_deflection_data.csv", index_col=[0])
interp_deflection_1act = interpolate.interp1d( deflection_data['cmd'],deflection_data['1x1_act_deflection[nm]'] ) #cmd to nm deflection on DM from single actuator (from datasheet) 
interp_deflection_4x4act = interpolate.interp1d( deflection_data['cmd'],deflection_data['4x4_act_deflection[nm]'] ) #cmd to nm deflection on DM from 4x4 actuator (from datasheet) 

#RMSE
plt.figure()
plt.plot( 2 * interp_deflection_4x4act( RMS_tele ),'.',label='residual' )
plt.plot( 2 * interp_deflection_4x4act( [np.std(d) for d in disturbance_dm_tele]) ,'.',label='disturbance' )
plt.ylabel('RMSE wave space [nm RMS]')
plt.xlabel('iteration')
plt.legend()
#plt.savefig(data_path + f'A_FIRST_closed_loop_on_dynamic_aberration_t-{tstamp}.png')
#plt.xlim([0,50]);plt.ylim([24,26]);plt.show()
print( 'final RMSE =', interp_deflection_4x4act(RMS_tele[-1] ) )
#print( 'RMSE after 100 iter. =',interp_deflection_4x4act( RMS_tele[100] ) );
plt.show()

# first 5 iterations 
iterations2plot = [0,1,2,3,4,5,-2,-1]
fig, ax = plt.subplots( len(iterations2plot), 4 ,figsize=(10,15))
ax[0,1].set_title('disturbance',fontsize=15)
ax[0,0].set_title('ZWFS image',fontsize=15)
ax[0,2].set_title('CMD',fontsize=15)
#ax[0,3].set_title('DM CMD (feedback)',fontsize=15)
ax[0,3].set_title('RESIDUAL',fontsize=15)
for i,idx in enumerate(iterations2plot):
    ax[i,0].imshow( np.array(img_tele)[idx] ) 
    im1 = ax[i,1].imshow( util.get_DM_command_in_2D(disturbance_dm_tele[idx]) )
    plt.colorbar(im1, ax= ax[i,1])
    #im2 = ax[i,2].imshow( util.get_DM_command_in_2D(u_tele[idx]) )
    #plt.colorbar(im2, ax= ax[i,2])
    im3 = ax[i,2].imshow( util.get_DM_command_in_2D(dm_cmd_tele[idx]- flat_dm_cmd)) 
    plt.colorbar(im3, ax= ax[i,2])
    im4 = ax[i,3].imshow( util.get_DM_command_in_2D(flat_dm_cmd - np.array(dm_cmd_tele[idx])  + np.array(disturbance_dm_tele[idx]) ) )
    plt.colorbar(im4, ax= ax[i,3])

plt.show() 


idx=3
im_list = [util.get_DM_command_in_2D( disturbance_dm_tele[idx] ), np.array(img_tele)[idx].T/np.max(np.array(img_tele)[idx] ), util.get_DM_command_in_2D( dm_cmd_tele[idx]- flat_dm_cmd )  ]
xlabel_list = [None, None, None]
ylabel_list = [None, None, None]
title_list = ['Aberration on DM', 'ZWFS Pupil', 'reconstructed DM cmd']
cbar_label_list = ['DM command', 'Normalized intensity' , 'DM command' ] 
savefig = None #fig_path + f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'

util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)




"""
if len( imgs2average ) > 100:
    im = np.mean( imgs2average , axis = 0 )
   
    pup_err = pup_ctrl.get_pupil_err( zwfs, im )
   
    phase_ctrl.update_b( im )
   
    if telemetry:
        average_img_tele.insert( 0, im)
        pup_err_tele.insert(0, pup_err)
        b_tele.insert( 0, zwfs.b ) # or what ever gets updated in update setpoint
        A_tele.insert( 0, zwfs.A ) # or what ever gets updated in update setpoint
       
    imgs2average = [] # re-initialize
   
"""

