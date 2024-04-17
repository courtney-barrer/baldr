import os
import numpy as np 
import matplotlib.pyplot as plt 
import pyzelda.utils.zernike as zernike
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime
import time 
from astropy.io import fits 
from scipy import ndimage
from scipy import signal
from scipy.optimize import curve_fit
import itertools
import corner

import sys 
sys.path.insert(1, '/opt/FirstLightImaging/FliSdk/Python/demo/')
import FliSdk_V2 

# ============== UTILITY FUNCTIONS





def construct_command_basis( basis='Zernike', number_of_modes = 20, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True):
    """
    returns a change of basis matrix M2C to go from modes to DM commands, where columns are the DM command for a given modal basis. e.g. M2C @ [0,1,0,...] would return the DM command for tip on a Zernike basis. Modes are normalized on command space such that <M>=0, <M|M>=1. Therefore these should be added to a flat DM reference if being applied.    

    basis = string of basis to use
    number_of_modes = int, number of modes to create
    Nx_act_DM = int, number of actuators across DM diameter
    Nx_act_basis = int, number of actuators across the active basis diameter
    act_offset = tuple, (actuator row offset, actuator column offset) to offset the basis on DM (i.e. we can have a non-centered basis)
    IM_covariance = None or an interaction matrix from command to measurement space. This only needs to be provided if you want KL modes, for this the number of modes is infered by the shape of the IM matrix. 
     
    """

   
    # shorter notations
    #Nx_act = DM.num_actuators_width() # number of actuators across diameter of DM.
    #Nx_act_basis = actuators_across_diam
    c = act_offset
    # DM BMC-3.5 is 12x12 missing corners so 140 actuators , we note down corner indicies of flattened 12x12 array.
    corner_indices = [0, Nx_act_DM-1, Nx_act_DM * (Nx_act_DM-1), -1]

    bmcdm_basis_list = []
    # to deal with
    if basis == 'Zernike':
        if without_piston:
            number_of_modes += 1 # we add one more mode since we dont include piston 

        raw_basis = zernike.zernike_basis(nterms=number_of_modes, npix=Nx_act_basis )
        for i,B in enumerate(raw_basis):
            # normalize <B|B>=1, <B>=0 (so it is an offset from flat DM shape)
            Bnorm = np.sqrt( 1/np.nansum( B**2 ) ) * B
            # pad with zeros to fit DM square shape and shift pixels as required to center
            # we also shift the basis center with respect to DM if required
            if np.mod( Nx_act_basis, 2) == 0:
                pad_width = (Nx_act_DM - B.shape[0] )//2
                padded_B = shift( np.pad( Bnorm , pad_width , constant_values=(np.nan,)) , c[0], c[1])
            else:
                pad_width = (Nx_act_DM - B.shape[0] )//2 + 1
                padded_B = shift( np.pad( Bnorm , pad_width , constant_values=(np.nan,)) , c[0], c[1])[:-1,:-1]  # we take off end due to odd numebr

            flat_B = padded_B.reshape(-1) # flatten basis so we can put it in the accepted DM command format
            np.nan_to_num(flat_B,0 ) # convert nan -> 0
            flat_B[corner_indices] = np.nan # convert DM corners to nan (so lenght flat_B = 140 which corresponds to BMC-3.5 DM)

            # now append our basis function removing corners (nan values)
            bmcdm_basis_list.append( flat_B[np.isfinite(flat_B)] )

        # our mode 2 command matrix
        if without_piston:
            M2C = np.array( bmcdm_basis_list )[1:].T #remove piston mode
        else:
            M2C = np.array( bmcdm_basis_list ).T # take transpose to make columns the modes in command space.


    elif basis == 'KL':         
        if without_piston:
            number_of_modes += 1 # we add one more mode since we dont include piston 

        raw_basis = zernike.zernike_basis(nterms=number_of_modes, npix=Nx_act_basis )
        b0 = np.array( [np.nan_to_num(b) for b in raw_basis] )
        cov0 = np.cov( b0.reshape(len(b0),-1) )
        U , S, UT = np.linalg.svd( cov0 )
        KL_raw_basis = ( b0.T @ U ).T # KL modes that diagonalize Zernike covariance matrix 
        for i,B in enumerate(KL_raw_basis):
            # normalize <B|B>=1, <B>=0 (so it is an offset from flat DM shape)
            Bnorm = np.sqrt( 1/np.nansum( B**2 ) ) * B
            # pad with zeros to fit DM square shape and shift pixels as required to center
            # we also shift the basis center with respect to DM if required
            if np.mod( Nx_act_basis, 2) == 0:
                pad_width = (Nx_act_DM - B.shape[0] )//2
                padded_B = shift( np.pad( Bnorm , pad_width , constant_values=(np.nan,)) , c[0], c[1])
            else:
                pad_width = (Nx_act_DM - B.shape[0] )//2 + 1
                padded_B = shift( np.pad( Bnorm , pad_width , constant_values=(np.nan,)) , c[0], c[1])[:-1,:-1]  # we take off end due to odd numebr

            flat_B = padded_B.reshape(-1) # flatten basis so we can put it in the accepted DM command format
            np.nan_to_num(flat_B,0 ) # convert nan -> 0
            flat_B[corner_indices] = np.nan # convert DM corners to nan (so lenght flat_B = 140 which corresponds to BMC-3.5 DM)

            # now append our basis function removing corners (nan values)
            bmcdm_basis_list.append( flat_B[np.isfinite(flat_B)] )

        # our mode 2 command matrix
        if without_piston:
            M2C = np.array( bmcdm_basis_list )[1:].T #remove piston mode
        else:
            M2C = np.array( bmcdm_basis_list ).T # take transpose to make columns the modes in command space.

    elif basis == 'Zonal': 
        M2C = np.eye(Nx_act_DM) # we just consider this over all actuators (so defaults to 140 modes) 
        # we filter zonal basis in the eigenvectors of the control matrix. 
 
    #elif basis == 'Sensor_Eigenmodes': this is done specifically in a phase_control.py function - as it needs a interaction matrix covariance first 

    return(M2C)



def get_DM_command_in_2D(cmd,Nx_act=12):
    # function so we can easily plot the DM shape (since DM grid is not perfectly square raw cmds can not be plotted in 2D immediately )
    #puts nan values in cmd positions that don't correspond to actuator on a square grid until cmd length is square number (12x12 for BMC multi-2.5 DM) so can be reshaped to 2D array to see what the command looks like on the DM.
    corner_indices = [0, Nx_act-1, Nx_act * (Nx_act-1), Nx_act*Nx_act]
    cmd_in_2D = list(cmd.copy())
    for i in corner_indices:
        cmd_in_2D.insert(i,np.nan)
    return( np.array(cmd_in_2D).reshape(Nx_act,Nx_act) )


def shift(xs, n, m, fill_value=np.nan):
    # shifts a 2D array xs by n rows, m columns and fills the new region with fill_value

    e = xs.copy()
    if n!=0:
        if n >= 0:
            e[:n,:] = fill_value
            e[n:,:] =  e[:-n,:]
        else:
            e[n:,:] = fill_value
            e[:n,:] =  e[-n:,:]
   
       
    if m!=0:
        if m >= 0:
            e[:,:m] = fill_value
            e[:,m:] =  e[:,:-m]
        else:
            e[:,m:] = fill_value
            e[:,:m] =  e[:,-m:]
    return e


def line_intersection(line1, line2):
    """
    find intersection of lines given by their endpoints, 
       line1 = (A,B)
       line2 = (C,D)
       where A=[x1_1, y1_1], B=[x1_2,y1_2], are end points of line1 
             C=[x2_1, y2_1], D=[x2_2, y2_2], are end points of line2
        
    """
 
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return( x, y )


def move_fpm( tel, pos = 0):
    # in real life this will command a motor - but for now i do this manually
    """
    # SETS FOCAL PLANE MASK - TO IMPLIMENT ONCE WE HAVE FPM MOTORS
    if pos == 0:    
        print( 'move FPM to position X')
    elif pos == 1:  
        print( 'move FPM to position Y') #etc
    """
    return(None)


def watch_camera(zwfs, frames_to_watch = 10, time_between_frames=0.01,cropping_corners=None) :
  
    print( f'{frames_to_watch} frames to watch with ~{time_between_frames}s wait between frames = ~{5*time_between_frames*frames_to_watch}s watch time' )

    #t0= datetime.datetime.now() 
    plt.figure(figsize=(15,15))
    plt.ion() # turn on interactive mode 
    #FliSdk_V2.Start(camera)     
    seconds_passed = 0
    if type(cropping_corners)==list: 
        x1,x2,y1,y2 = cropping_corners #[row min, row max, col min, col max]

    for i in range(int(frames_to_watch)): 
        
        a = zwfs.get_image()
        if type(cropping_corners)==list: 
            plt.imshow(a[x1:x2,y1:y2])
        else: 
            plt.imshow(a)
        plt.pause( time_between_frames )
        #time.sleep( time_between_frames )
        plt.clf() 
    """
    while seconds_passed < seconds_to_watch:
        a=FliSdk_V2.GetRawImageAsNumpyArray(camera,-1)
        plt.imshow(a)
        plt.pause( time_between_frames )
        time.sleep( time_between_frames )
        plt.clf() 
        t1 = datetime.datetime.now() 
        seconds_passed = (t1 - t0).seconds"""

    #FliSdk_V2.Stop(camera) 
    plt.ioff()# turn off interactive mode 
    plt.close()







def create_phase_screen_cmd_for_DM(scrn, flat_reference, scaling_factor=0.1, drop_indicies = None, plot_cmd=False):
    """
    aggregate a scrn (aotools.infinitephasescreen object) onto a DM command space. phase screen is normalized by
    between +-0.5 and then scaled by scaling_factor and offset by flat_reference command. Final DM command values should
    always be between 0-1. phase screens are usually a NxN matrix, while DM is MxM with some missing pixels (e.g. 
    corners). drop_indicies is a list of indicies in the flat MxM DM array that should not be included in the command space. 
    """

    #print('----------\ncheck phase screen size is multiple of DM\n--------')
    
    Nx_act = 12 #number of actuators across DM diameter
    
    scrn_array = ( scrn.scrn - np.min(scrn.scrn) ) / (np.max(scrn.scrn) - np.min(scrn.scrn)) - 0.5 # normalize phase screen between -0.5 - 0.5 
    
    size_factor = int(scrn_array.shape[0] / Nx_act) # how much bigger phase screen is to DM shape in x axis. Note this should be an integer!!
    
    # reshape screen so that axis 1,3 correspond to values that should be aggregated 
    scrn_to_aggregate = scrn_array.reshape(scrn_array.shape[0]//size_factor, size_factor, scrn_array.shape[1]//size_factor, size_factor)
    
    # now aggreagate and apply the scaling factor 
    scrn_on_DM = scaling_factor * np.mean( scrn_to_aggregate, axis=(1,3) ).reshape(-1) 

    #If DM is missing corners etc we set these to nan and drop them before sending the DM command vector
    #dm_cmd =  scrn_on_DM.to_list()
    if drop_indicies != None:
        for i in drop_indicies:
            scrn_on_DM[i]=np.nan
             
    if plot_cmd: #can be used as a check that the command looks right!
        fig,ax = plt.subplots(1,2,figsize=(12,6))
        im0 = ax[0].imshow(np.mean(flat_reference) + scrn_on_DM.reshape([Nx_act,Nx_act]) )
        ax[0].set_title('DM command (averaging offset)')
        im1 = ax[1].imshow(scrn.scrn)
        ax[1].set_title('original phase screen')
        plt.colorbar(im0, ax=ax[0])
        plt.colorbar(im1, ax=ax[1]) 
        plt.show() 

    dm_cmd =  list( scrn_on_DM[np.isfinite(scrn_on_DM)] ) #drop non-finite values which should be nan values created from drop_indicies array
    return(dm_cmd) 






def block_sum(ar, fact): # sums over subwindows of a 2D array
    # ar is the 2D array, fact is the factor to reduce it by 
    # obviously  fact should be factor of ar.shape 
    assert isinstance(fact, int), type(fact)
    sx, sy = ar.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    regions = sy//fact * (X//fact) + Y//fact
    res = ndimage.sum(ar, labels=regions, index=np.arange(regions.max() + 1))
    res.shape = (sx//fact, sy//fact)
    return res




def nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list,cbar_label_list, fontsize=15, cbar_orientation = 'bottom', axis_off=True, savefig=None):

    n = len(im_list)
    fs = fontsize
    fig = plt.figure(figsize=(5*n, 5))

    for a in range(n) :
        ax1 = fig.add_subplot(int(f'1{n}{a+1}'))
        ax1.set_title(title_list[a] ,fontsize=fs)

        im1 = ax1.imshow(  im_list[a] )
        ax1.set_title( title_list[a] ,fontsize=fs)
        ax1.set_xlabel( xlabel_list[a] ,fontsize=fs) 
        ax1.set_ylabel( ylabel_list[a] ,fontsize=fs) 
        ax1.tick_params( labelsize=fs ) 

        if axis_off:
            ax1.axis('off')
        divider = make_axes_locatable(ax1)
        if cbar_orientation == 'bottom':
            cax = divider.append_axes('bottom', size='5%', pad=0.05)
            cbar = fig.colorbar( im1, cax=cax, orientation='horizontal')
        elif cbar_orientation == 'top':
            cax = divider.append_axes('top', size='5%', pad=0.05)
            cbar = fig.colorbar( im1, cax=cax, orientation='horizontal')
        else: # we put it on the right 
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar( im1, cax=cax, orientation='vertical')           
        cbar.set_label( cbar_label_list[a], rotation=0,fontsize=fs)
        cbar.ax.tick_params(labelsize=fs)
    if savefig!=None:
        plt.savefig( savefig , bbox_inches='tight', dpi=300) 

    plt.show() 



def test_controller_in_cmd_space( zwfs, phase_controller, Vw =None , D=None , AO_lag=None  ):
    #outputs fits file with telemetry  
    # applies open loop for X iterations, Closed loop for Y iterations
    # ability to ramp number of controlled modes , how to deal with gains? gain = 1/number controlled modes for Zernike basis

    #open loop 
    for i in range(X) : 

        for i in range( 3 ) :
            raw_img_list.append( zwfs.get_image() ) # @D, remember for control_phase method this needs to be flattened and filtered for pupil region

        raw_img.append( np.median( raw_img_list, axis = 0 ) ) 

        # ALSO MANUALLY GET PSF IMAGE 
        
        err_img.append(  phase_controller.get_img_err( raw_img[-1].reshape(-1)[zwfs.pupil_pixels]  ) )

        reco_modes.append( phase_controller.control_phase( err_img[-1]  , controller_name = ctrl_method_label) )

        reco_dm_cmds.append( phase_controller.config['M2C'] @ mode_reco[-1] )

        delta_cmd.append( Ki * delta_cmd[-1] - Kp * reco_dm_cmds[-1] )

        # propagate_phase_screen one step 

        # don't yet apply the correction cmd while in open loop
        zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] + dist[-1] ) 

        time.sleep(0.1)


    #close loop 
    for i in range(X) : 

        for i in range( 3 ) :
            raw_img_list.append( zwfs.get_image() ) # @D, remember for control_phase method this needs to be flattened and filtered for pupil region

        raw_img.append( np.median( raw_img_list, axis = 0 ) ) 

        err_img.append(  phase_controller.get_img_err( raw_img[-1].reshape(-1)[zwfs.pupil_pixels]  ) )

        reco_modes.append( phase_controller.control_phase( err_img[-1]  , controller_name = ctrl_method_label) )

        reco_dm_cmds.append( phase_controller.config['M2C'] @ mode_reco[-1] )

        delta_cmd.append( Ki * delta_cmd[-1] - Kp * reco_dm_cmds[-1] )

        # propagate_phase_screen one step 

        # then apply the correction 
        zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] + delta_cmd[-1] + dist[-1] )

        time.sleep(0.1)




def apply_sequence_to_DM_and_record_images(zwfs, DM_command_sequence, number_images_recorded_per_cmd = 1, take_median_of_images=False, save_dm_cmds = True, calibration_dict=None, additional_header_labels=None, sleeptime_between_commands=0.01, cropping_corners=None, save_fits = None):
    """
    

    Parameters
    ----------

    zwfs is a zwfs object from the ZWFS class that must have zwfs.dm and zwfs.camera attributes:
        dm : TYPE
            DESCRIPTION. DM Object from BMC SDK. Initialized  (this is zwfs.camera object) 
        camera : TYPE camera objection from FLI SDK. (this is zwfs.camera object)
            DESCRIPTION. Camera context from FLISDK. Initialized from context = FliSdk_V2.Init(), 

    DM_command_sequence : TYPE list 
        DESCRIPTION. Nc x Na matrix where Nc is number of commands to send in sequence (rows)
        Na is number actuators on DM (columns).   
    number_images_recorded_per_cmd : TYPE, optional
        DESCRIPTION. The default is 1. puting a value >= 0 means no images are recorded.
    take_median_of_images: TYPE, optional
        DESCRIPTION. The default is False. if True we take the median image of number_images_recorded_per_cmd such that there is only one image per command (that is the aggregated image)
    calibration_dict: TYPE, optional
        DESCRIPTION. The default is None meaning saved images don't get flat fielded. 
        if flat fielding is required a dictionary must be supplied that contains 
        a bias, dark and flat frame under keys 'bias', 'dark', and 'flat' respectively
    additional_header_labels : TYPE, optional
        DESCRIPTION. The default is None which means no additional header is appended to fits file 
        otherwise a tuple (header, value) or list of tuples [(header_0, value_0)...] can be used. 
        If list, each item in list will be added as a header. 
    cropping_corners: TYPE, optional
        DESCRIPTION. list of length 4 holding [row min, row max, col min, col max] to crop raw data frames.
    save_fits : TYPE, optional
        DESCRIPTION. The default is None which means images are not saved, 
        if a string is provided images will be saved with that name in the current directory
    sleeptime_between_commands : TYPE, optional float
        DESCRIPTION. time to sleep between sending DM commands and recording images in seconds. default is 0.01s
    Returns
    -------
    fits file with images corresponding to each DM command in sequence
    first extension is images
    second extension is DM commands

    """

    
    should_we_record_images = True
    try: # foce to integer
        number_images_recorded_per_cmd = int(number_images_recorded_per_cmd)
        if number_images_recorded_per_cmd <= 0:
            should_we_record_images = False
    except:
        raise TypeError('cannot convert "number_images_recorded_per_cmd" to a integer. Check input type')
    
    image_list = [] #init list to hold images

    # NOTE THE CAMERA SHOULD ALREADY BE STARTED BEFORE BEGINNING - No checking here yet
    for cmd_indx, cmd in enumerate(DM_command_sequence):
        print(f'executing cmd_indx {cmd_indx} / {len(DM_command_sequence)}')
        # wait a sec        
        time.sleep(sleeptime_between_commands)
        # ok, now apply command
        zwfs.dm.send_data(cmd)
        # wait a sec        
        time.sleep(sleeptime_between_commands)

        if should_we_record_images: 
            if take_median_of_images:
                ims_tmp = [np.median([zwfs.get_image() for _ in range(number_images_recorded_per_cmd)] , axis=0)] #keep as list so it is the same type as when take_median_of_images=False
            else:
                ims_tmp = zwfs.get_image() #get_raw_images(camera, number_images_recorded_per_cmd, cropping_corners) 
            image_list.append( ims_tmp )

    
    #FliSdk_V2.Stop(camera) # stop camera
    
    # init fits files if necessary
    if should_we_record_images: 
        #cmd2pix_registration
        data = fits.HDUList([]) #init main fits file to append things to
        
        # Camera data
        cam_fits = fits.PrimaryHDU( image_list )
        
        cam_fits.header.set('EXTNAME', 'SEQUENCE_IMGS' )
        #camera headers
        camera_info_dict = get_camera_info(zwfs.camera)
        for k,v in camera_info_dict.items():
            cam_fits.header.set(k,v)
        cam_fits.header.set('#images per DM command', number_images_recorded_per_cmd )
        cam_fits.header.set('take_median_of_images', take_median_of_images )
        
        cam_fits.header.set('cropping_corners_r1', zwfs.pupil_crop_region[0] )
        cam_fits.header.set('cropping_corners_r2', zwfs.pupil_crop_region[1] )
        cam_fits.header.set('cropping_corners_c1', zwfs.pupil_crop_region[2] )
        cam_fits.header.set('cropping_corners_c2', zwfs.pupil_crop_region[3] )

        #if user specifies additional headers using additional_header_labels
        if (additional_header_labels!=None): 
            if type(additional_header_labels)==list:
                for i,h in enumerate(additional_header_labels):
                    cam_fits.header.set(h[0],h[1])
            else:
                cam_fits.header.set(additional_header_labels[0],additional_header_labels[1])

        # add camera data to main fits
        data.append(cam_fits)
        
        if save_dm_cmds:
            # put commands in fits format
            dm_fits = fits.PrimaryHDU( DM_command_sequence )
            #DM headers 
            dm_fits.header.set('timestamp', str(datetime.datetime.now()) )
            dm_fits.header.set('EXTNAME', 'DM_CMD_SEQUENCE' )
            #dm_fits.header.set('DM', DM.... )
            #dm_fits.header.set('#actuators', DM.... )

            # append to the data
            data.append(dm_fits)
        
        if save_fits!=None:
            if type(save_fits)==str:
                data.writeto(save_fits)
            else:
                raise TypeError('save_images needs to be either None or a string indicating where to save file')
            
            
        return(data)
    
    else:
        return(None)



def get_camera_info(camera):
    camera_info_dict = {} 
    # query camera settings 
    fps_res, fps_response = FliSdk_V2.FliSerialCamera.GetFps(camera)
    tint_res, tint_response = FliSdk_V2.FliSerialCamera.SendCommand(camera, "tint raw")
    
    #camera headers
    camera_info_dict['timestamp'] = str(datetime.datetime.now()) 
    camera_info_dict['camera'] = FliSdk_V2.GetCurrentCameraName(camera) 
    camera_info_dict['camera_fps'] = fps_response
    camera_info_dict['camera_tint'] = tint_response
    
    return(camera_info_dict)

    
def scan_detector_framerates(camera, frame_rates, number_images_recorded_per_cmd = 50, cropping_corners=None, save_fits = None): 
    """
    iterate through different camera frame rates and record a series of images for each
    this can be used for building darks or flats.

    Parameters
    ----------
    camera : TYPE camera objection from FLI SDK.
        DESCRIPTION. Camera context from FLISDK. Initialized from context = FliSdk_V2.Init() 
    frame_rates : TYPE list like 
        DESCRIPTION. array holding different frame rates to iterate through
    number_images_recorded_per_cmd : TYPE, optional
        DESCRIPTION. The default is 50. puting a value >= 0 means no images are recorded.
    save_fits : TYPE, optional
        DESCRIPTION. The default is None which means images are not saved, 
        if a string is provided images will be saved with that name in the current directory

    Raises
    ------
    TypeError
        DESCRIPTION.

    Returns
    -------
    fits file with each extension corresponding to a different camera frame rate 

    """
    
    data = fits.HDUList([]) 
    for fps in frame_rates:
        
        camera = set_fsp_dit( camera, fps, tint=None) # set max dit (tint=None) for given fps
        
        FliSdk_V2.Start(camera) # start camera
	
        time.sleep(1) # wait 1 second
        #tmp_fits = fits.PrimaryHDU( [FliSdk_V2.GetProcessedImageGrayscale16bNumpyArray(camera,-1)  for i in range(number_images_recorded_per_cmd)] )

        tmp_fits = fits.PrimaryHDU(  get_raw_images(camera, number_images_recorded_per_cmd, cropping_corners) )

        
        camera_info_dict = get_camera_info(camera)
        for k,v in camera_info_dict.items():
            tmp_fits.header.set(k,v)     

        data.append( tmp_fits )

        FliSdk_V2.Stop(camera) # stop camera
    
    if save_fits!=None:
        if type(save_fits)==str:
            data.writeto(save_fits)
        else:
            raise TypeError('save_images needs to be either None or a string indicating where to save file')
        
    return(data)




def GET_BDR_RECON_DATA_INTERNAL(zwfs, number_amp_samples = 18, amp_max = 0.2, number_images_recorded_per_cmd = 2, save_fits = None) :

    flat_dm_cmd = zwfs.dm_shapes['flat_dm']

    modal_basis = np.eye(len(flat_dm_cmd))
    cp_x1,cp_x2,cp_y1,cp_y2 = zwfs.pupil_crop_region

    ramp_values = np.linspace(-amp_max, amp_max, number_amp_samples)

    # ======== reference image with FPM OUT

    zwfs.dm.send_data(flat_dm_cmd) 
    _ = input('MANUALLY MOVE PHASE MASK OUT OF BEAM, PRESS ENTER TO BEGIN' )
    watch_camera(zwfs, frames_to_watch = 50, time_between_frames=0.05)

    N0_list = []
    for _ in range(number_images_recorded_per_cmd):
        N0_list.append( zwfs.get_image( ) ) #REFERENCE INTENSITY WITH FPM IN
    N0 = np.median( N0_list, axis = 0 ) 

    #make_fits

    # ======== reference image with FPM IN
    _ = input('MANUALLY MOVE PHASE MASK BACK IN, PRESS ENTER TO BEGIN' )
    watch_camera(zwfs, frames_to_watch = 50, time_between_frames=0.05)

    I0_list = []
    for _ in range(number_images_recorded_per_cmd):
        I0_list.append( zwfs.get_image(  ) ) #REFERENCE INTENSITY WITH FPM IN
    I0 = np.median( I0_list, axis = 0 ) 
        
    #====== make references fits files
    I0_fits = fits.PrimaryHDU( I0 )
    N0_fits = fits.PrimaryHDU( N0 )
    I0_fits.header.set('EXTNAME','FPM_IN')
    N0_fits.header.set('EXTNAME','FPM_OUT')

    flat_DM_fits = fits.PrimaryHDU( flat_dm_cmd )
    flat_DM_fits.header.set('EXTNAME','FLAT_DM_CMD')

    #make_fits

    # ======== RAMPING ACTUATORS  
    # --- creating sequence of dm commands
    _DM_command_sequence = [list(flat_dm_cmd + amp * modal_basis) for amp in ramp_values ]  
    # add in flat dm command at beginning of sequence and reshape so that cmd sequence is
    # [0, a0*b0,.. aN*b0, a0*b1,...,aN*b1, ..., a0*bM,...,aN*bM]
    DM_command_sequence = [flat_dm_cmd] + list( np.array(_DM_command_sequence).reshape(number_amp_samples*modal_basis.shape[0],modal_basis.shape[1] ) )

    # --- additional labels to append to fits file to keep information about the sequence applied 
    additional_labels = [('cp_x1',cp_x1),('cp_x2',cp_x2),('cp_y1',cp_y1),('cp_y2',cp_y2),('in-poke max amp', np.max(ramp_values)),('out-poke max amp', np.min(ramp_values)),('#ramp steps',number_amp_samples), ('seq0','flatdm'), ('reshape',f'{number_amp_samples}-{modal_basis.shape[0]}-{modal_basis.shape[1]}'),('Nmodes_poked',len(modal_basis)),('Nact',140)]

    # --- poke DM in and out and record data. Extension 0 corresponds to images, extension 1 corresponds to DM commands
    raw_recon_data = apply_sequence_to_DM_and_record_images(zwfs, DM_command_sequence, number_images_recorded_per_cmd = number_images_recorded_per_cmd, take_median_of_images=True, save_dm_cmds = True, calibration_dict=None, additional_header_labels = additional_labels,sleeptime_between_commands=0.03, cropping_corners=None,  save_fits = None ) # None

    zwfs.dm.send_data(flat_dm_cmd) 

    # append FPM IN and OUT references (note FPM in reference is also first entry in recon_data so we can compare if we want!) 
    raw_recon_data.append( I0_fits ) 
    raw_recon_data.append( N0_fits ) 
    raw_recon_data.append( flat_DM_fits )

    if save_fits != None:  
        if type(save_fits)==str:
            raw_recon_data.writeto(save_fits)
        else:
            raise TypeError('save_images needs to be either None or a string indicating where to save file')


    return(raw_recon_data) 


def Ic_model_constrained(x, A, B, F, mu):
    penalty = 0
    if (F < 0) or (mu < 0): # F and mu can be correlated so constrain the quadrants 
        penalty = 1e3
    I = A + 2 * B * np.cos(F * x + mu) + penalty
    return I 

def Ic_model_constrained_3param(x, A,  F, mu): 
    #penalty = 0
    #if (F < 0) or (mu < 0): # F and mu can be correlated so constrain the quadrants 
    #    penalty = 1e3
    # 
    I = A**2 + 2 * A * np.cos(F * x + mu) + penalty
    return I 


# should this be free standing or a method? ZWFS? controller? - output a report / fits file
def PROCESS_BDR_RECON_DATA_INTERNAL(data, debug=True) :
    """
    # calibration of our ZWFS: 
    # this will fit M0, b0, mu, F which can be appended to a phase_controller,
    # f and/or mu can be be provided (because may haave been previously characterise) in which case only b0/M0 is fitted.   
    # also P2C_1x1, P2C_3x3 matrix which corresponds to 1x1 and 3x3 pixels around where peak influence for an actuator was found. 
    # Also pupil regions to filter 
       - active pupil pixels 
       - secondary obstruction pixels
       - outside pupil pixels 

    # plot fits, histograms of values (corner plot?), also image highlighting the regions (coloring pixels) 
    # estimate center of DM in pixels and center of pupil in pixels (using previous method)   
    # note A^2 can be measured with FPM out, M can be sampled with FPM in where A^2 = 0. 
    """    


    # ========================== !! 0 !! =====================
    # -- prelims of reading in and labelling data 

    # poke values used in linear ramp
    No_ramps = int(recon_data['SEQUENCE_IMGS'].header['#ramp steps'])
    max_ramp = float( recon_data['SEQUENCE_IMGS'].header['in-poke max amp'] )
    min_ramp = float( recon_data['SEQUENCE_IMGS'].header['out-poke max amp'] ) 
    ramp_values = np.linspace( min_ramp, max_ramp, No_ramps)

    flat_dm_cmd = recon_data['FLAT_DM_CMD'].data

    Nmodes_poked = int(recon_data[0].header['HIERARCH Nmodes_poked']) # can also see recon_data[0].header['RESHAPE']

    Nact =  int(recon_data[0].header['HIERARCH Nact'])  

    N0 = recon_data['FPM_OUT'].data
    #P = np.sqrt( pupil ) # 
    I0 = recon_data['FPM_IN'].data

    # the first image is another reference I0 with FPM IN and flat DM
    poke_imgs = recon_data['SEQUENCE_IMGS'].data[1:].reshape(No_ramps, 140, I0.shape[0], I0.shape[1])
    #poke_imgs = poke_imgs[1:].reshape(No_ramps, 140, I0.shape[0], I0.shape[1])


    # ========================== !! 1 !! =====================
    #  == define the region of influence on DM where we will fit out models (define a threshold for I(epsilon)_max -I_0). This is important because the quality of the fits can go to shit around pupil/DM edges, we only need some good samples around the center to reconstruct what we need, setting this threshold is semi automated here  

    a0 = len(ramp_values)//2 - 2 # which poke values do we want to consider for finding region of influence. Pick a value near the center where in a linear regime. 

    fig,ax= plt.subplots( 4, 4, figsize=(10,10))
    num_pixels = []
    candidate_thresholds = np.linspace(4 * np.std(abs(poke_imgs[a0,:,:,:] - I0)),np.max(abs(poke_imgs[a0,:,:,:] - I0)),16)
    for axx, thresh in zip(ax.reshape(-1),candidate_thresholds):
    
        dm_pupil_filt = thresh < np.array( [np.max( abs( poke_imgs[a0][act] - I0) ) for act in range(140)] ) 
        axx.imshow( get_DM_command_in_2D( dm_pupil_filt ) ) 
        axx.set_title('threshold = {}'.format(round( thresh )),fontsize=12) 
        axx.axis('off')
        num_pixels.append(sum(dm_pupil_filt)) 
        # we could use this to automate threshold decision.. look for where 
        # d num_pixels/ d threshold ~ 0.. np.argmin( abs( np.diff( num_pixels ) )[:10])
    plt.show()

    recommended_threshold = candidate_thresholds[np.argmin( abs( np.diff( num_pixels ) )[2:11]) + 1 ]
    print( f'\n\nrecommended threshold ~ {round(recommended_threshold)} \n(check this makes sense with the graph by checking the colored area is stable around changes in threshold about this value)\n\n')

    pupil_filt_threshold = float(input('input threshold of peak differences'))

    ### <---- THIS FILTER DETERMINES WHERE WE FIT THE MODELS (ONLY FIT WHERE DM HAS GOOD INFLUENCE!)
    dm_pupil_filt = pupil_filt_threshold < np.array( [np.max( abs( poke_imgs[a0][act] - I0) ) for act in range(Nact)] ) 

    if debug:
       plt.figure()
       plt.imshow( get_DM_command_in_2D( dm_pupil_filt ) )
       plt.title('influence region on DM where we will fit intensity models per actuator')
       plt.show()


    # ========================== !! 2 !! =====================
    # ======== P2C 

    Sw_x, Sw_y = 3,3 #+- pixels taken around region of peak influence. PICK ODD NUMBERS SO WELL CENTERED!   
    act_img_mask = {}
    act_flag = {}
    act_img_idx = {}
    #act2pix_idx = []


    for act_idx in range(Nact):
        delta =  poke_imgs[a0][act_idx] - I0

        mask = np.zeros( I0.shape )
   
        if dm_pupil_filt[act_idx]: #  if we decided actuator has strong influence on ZWFS image, we 
            i,j = np.unravel_index( np.argmax( abs(delta) ), I0.shape )

            #act2pix_idx.append( (i,j) ) 
            mask[i-Sw_x-1: i+Sw_x, j-Sw_y-1:j+Sw_y] = 1 # keep centered, 
            #mask *= 1/np.sum(mask[i-Sw_x-1: i+Sw_x, j-Sw_y-1:j+Sw_y]) #normalize by #pixels in window 
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

    # turn our dictionary to a big pixel to command matrix 
    P2C = np.array([list(act_img_mask[act_idx].reshape(-1)) for act_idx in range(Nact)])

    # we can look at filtering a particular actuator in image P2C[i].reshape(zwfs.get_image().shape)
    # we can also interpolate signals in 3x3 grid to if DM actuator not perfectly registered to center of pixel. 


    # INSIDE PUPIL JUST DO MATCHED FILTER OF FPM OUT IMAGE WITH CIRCULAR PUPIL OF DIFFERENT SIZE AND CENTER 
    # OUTSIDE PUPIL JUST INVERT P2C 0<->1 as long as subwindows are large enough to overlap with pupil. OR USE INVERSE OF INSIDE PUPIL... VARIOUS WAYS OF DOING THIS 

    # argmax( |FPM_IN - FPM_OUT| ) is center of secondary obstruction  

    i_sec, j_sec = np.unravel_index( np.argmax( abs(I0-N0).reshape(-1) ), I0.shape )
 
    # careful here, we don't know if we should take indicies above or below peak center etc..
    # we should take 3x3 grid and 2D convolve with ones to see where peak is and take those points 
    subgrid = abs(I0 - N0)[ i_sec - 1 : i_sec + 2, j_sec - 1 : j_sec + 2 ] # isec,jsec at center of subgrid (index = 1,1) 

    conv = signal.convolve2d(subgrid, np.ones([2,2]), mode='valid')
    di, dj = np.unravel_index( np.argmax(conv), conv.shape )
    si_rows = [i_sec + di, i_sec + di - 1]
    si_cols = [j_sec + dj , j_sec + dj - 1 ] 

    secondary_indicies = list(itertools.product(si_rows, si_cols))



    if debug: 
        im_list = [(I0 - N0) / N0 ]
        xlabel_list = ['x [pixels]']
        ylabel_list = ['y [pixels]']
        title_list = ['']
        cbar_label_list = [r'$\frac{|\psi_C|^2 - |\psi_A|^2}{|\psi_A|^2}$']
        savefig = None# fig_path + f'pupil_FPM_IN-OUT_readout_mode-FULL_t{tstamp}.png'

        util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)




# ========================== !! 3 !! =====================
    # ======== FITTING 
    # what do we fit ( I - N0 ) / N0 


    param_dict = {}
    cov_dict = {}
    fit_residuals = []

    if debug:
        Nrows = np.ceil( sum( dm_pupil_filt )**0.5).astype(int)
        fig,ax = plt.subplots(Nrows,Nrows,figsize=(20,20))
        axx = ax.reshape(-1)
        for aaa in axx:
            aaa.axis('off')
        j=0 #axx index

    mean_filtered_pupil = 1/(3*3)  * np.mean( P2C @ N0.reshape(-1) )

    for act_idx in range(len(flat_dm_cmd)): 
        if dm_pupil_filt[act_idx]:
            # -- we do this with matrix multiplication using  mask_matrix
            #P_i = np.sum( act_img_mask[act_idx] * pupil ) #Flat DM with FPM OUT 
            #P_i = mean_filtered_pupil.copy() # just consider mean pupil! 
        
            I_i = np.array( [P2C[act_idx] @ poke_imgs[i][act_idx].reshape(-1) for i in  range(len(ramp_values))] ) #np.array( [np.sum( act_img_mask[act_idx] * poke_imgs[i][act_idx] ) for i in range(len(ramp_values))] ) #spatially filtered sum of intensities per actuator cmds 
            I_0 = 1/(3*3) * P2C[act_idx] @ I0.reshape(-1) #np.sum( act_img_mask[act_idx] * I0 ) # Flat DM with FPM IN  
            N_0 = 1/(3*3) *P2C[act_idx] @ N0.reshape(-1) #np.sum( act_img_mask[act_idx] * N0 )
            # ================================
            #   THIS IS OUR MODEL!S=A+B*cos(F*x + mu)  
            #S = (I_i - I_0) / P_i # signal to fit!
            S = I_i  # A 
            #S = (I_i - N_0) / N_0 # signal to fit! <- should I take mean of total pupil?? 
            # THEORETICALLY THIS SIGNAL IS: S = |M0|^2/|A|^2 + M0/|A| * cos(F.c + mu)  
            # ================================

            #re-label and filter to capture best linear range 
            x_data = ramp_values[2:-2].copy()
            y_data = S[2:-2].copy()

            initial_guess = [np.mean(S), (np.max(S)-np.min(S))/2,  15, 2.4]
            #initial_guess = [7, 2, 15, 2.4] #[0.5, 0.5, 15, 2.4]  #A_opt, B_opt, F_opt, mu_opt  ( S = A+B*cos(F*x + mu) )
            # FIT 
            popt, pcov = curve_fit(Ic_model_constrained, x_data, y_data, p0=initial_guess)

            # Extract the optimized parameters explictly to measure residuals
            A_opt, B_opt, F_opt, mu_opt = popt

            # STORE FITS 
            param_dict[act_idx] = popt
            cov_dict[act_idx] = pcov 
            # also record fit residuals 
            fit_residuals.append( S - Ic_model_constrained(ramp_values, A_opt, B_opt, F_opt, mu_opt) )


            if debug: 

                axx[j].plot( ramp_values, Ic_model_constrained(ramp_values, A_opt, B_opt, F_opt, mu_opt) ,label=f'fit (act{act_idx})') 
                axx[j].plot( ramp_values, S ,label=f'measured (act{act_idx})' )
                #axx[j].set_xlabel( 'normalized DM command')
                #axx[j].set_ylabel( 'normalized Intensity')
                axx[j].legend(fontsize=6)
                #axx[j].set_title(act_idx,fontsize=5)
                #ins = axx[j].inset_axes([0.15,0.15,0.25,0.25])
                #ins.imshow(poke_imgs[3][act_idx] )
                #axx[j].axis('off')
                j+=1
    

    if debug:
        #plt.savefig( fig_path + f'cosine_interation_model_fits_ALL_{tstamp}.png' , bbox_inches='tight', dpi=500) 
        plt.show() 

    if debug:
        buffcorners = np.array( [list( param_dict.values() ) for _ in range(10)]).reshape(-1,4)

        corner.corner( buffcorners , quantiles=[0.16,0.5,0.84], show_titles=True, labels = ['Q [adu]', 'W [adu/cos(rad)]', 'F [rad/cmd]', r'$\mu$ [rad]'], range=[(-10, 10), (0, 5), (5, 20), (-3,3)])

        plt.show()
        
    # get an estimate for things that should not ideally have spatial variability
    _, _, F_est, mu_est = np.median( list( param_dict.values() ) ,axis = 0)

    # from mu measure theta - our estimated phase shift  
    #   (using result: mu = tan^-1(sin(theta)/(cos(theta)-1) = 0.5*(theta - pi) ) 
    theta_est  = 2*mu_est + np.pi
        
    # THEORETICALLY THIS SIGNAL: S = (I - N0)/N0 = |M0|^2/|A|^2 + 2*M0/|A| * cos(F.c + mu) 
    # Q =  |M0|^2/|A|^2 , W = |M0|/|A|. 
    # therefore M0 = np.sqrt( Q * |A|^2 ) => W = Q
        

    # things that can have spatial variability
    Q_est =  np.array(list( param_dict.values() ))[:, 0]
    W_est = np.array(list( param_dict.values() ))[:, 1] 


    # ========== GET A and B from these ... check against measured values within pupil obstruction and fit gaussian for b within pupil.   

    A_est = np.sqrt( Q_est + np.sqrt( Q_est**2 + W_est**2 ) ) / np.sqrt(2) 

    M_est = W_est /  (np.sqrt(2) * np.sqrt( Q_est + np.sqrt( Q_est**2 - W_est**2 ) )  ) 
    
    b_est = M_est / (2*(1-np.cos(theta_est)))**0.5

    bsamples = np.nan * np.zeros( dm_pupil_filt.shape)
    bsamples[dm_pupil_filt] = b_est

    plt.figure()
    plt.imshow( util.get_DM_command_in_2D( bsamples )); plt.colorbar()
    plt.show()
    

    data





