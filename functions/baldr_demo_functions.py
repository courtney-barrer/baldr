# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

We want generic structure for:
    0) initializing and setting up DM and camera parameters
    1) actuating DM, 
    2) recording one or many images for each DM action
    3) processing the image 
    4) sending new command to DM 
    
For this we create generic structures: 
 - DM and camera configurtation file / object
 - RECORD FULL IMAGE, THEN DO IMAGE RECOGNITION TO IDENTIY OBJECTS AND CROP FRAME RATE
 
 PRINT MAX FRAME RATE / MIN DIT AVAILABLE WITH CROPPED REGION 
 
 flatfiedling https://en.wikipedia.org/wiki/Flat-field_correction 
 subtracting a bias frame from a dark frame
"""


import os 
import numpy as np
import pandas as pd
import copy
import time 
import matplotlib.pyplot as plt
#import time
import datetime
import cv2
from scipy.interpolate import interp2d, interp1d
from scipy.stats import poisson
from astropy.io import fits
import aotools
import pyzelda.utils.zernike as zernike
import bmc
os.chdir('/opt/FirstLightImaging/FliSdk/Python/demo/')
import FliSdk_V2 
#import astropy.constants as cst
#import astropy.units as u

path_to_dm_flat_map = "/opt/Boston Micromachines/Shapes/17DW019#053_FLAT_MAP_COMMANDS.txt"

#cal_dict = {bi}
def calibrate_raw_image(im,cal_dict):
    #flat fiedling https://en.wikipedia.org/wiki/Flat-field_correction 
    # bias frames https://en.wikipedia.org/wiki/Bias_frame
    bias = cal_dict['bias']
    dark = cal_dict['dark']
    flat = cal_dict['flat']
    
    cal_dark = cal_dict['dark'] - cal_dict['bias'] 
    cal_im = ( im - cal_dict['flat'] ) / ( cal_dict['flat'] - cal_dark )
    
    return( cal_im )
    
def setup_camera(cameraIndex=0):
    context = FliSdk_V2.Init() # init camera object
    listOfGrabbers = FliSdk_V2.DetectGrabbers(context)
    listOfCameras = FliSdk_V2.DetectCameras(context)
    # print some info and exit if nothing detected 
    if len(listOfGrabbers) == 0:
        print("No grabber detected, exit.")
        exit()
    if len(listOfCameras) == 0:
        print("No camera detected, exit.")
        exit()
    for i,s in enumerate(listOfCameras):
        print("- " + str(i) + " -> " + s)
    print('note we always default to cameraIndex=0 if cameraIndex is not provided')
    # set the camera 
    ok = FliSdk_V2.SetCamera(context, listOfCameras[cameraIndex])
    if not ok:
        print("Error while setting camera.")
        exit()
    print("Setting mode full.")
    FliSdk_V2.SetMode(context, FliSdk_V2.Mode.Full)
    print("Updating...")
    ok = FliSdk_V2.Update(context)
    if not ok:
        print("Error while updating SDK.")
        exit()
    return(context) 

def set_fsp_dit( context, fps, tint=None): 
    """
    

    Parameters
    ----------
    context : TYPE
        DESCRIPTION. camera context
    fps : TYPE integer
        DESCRIPTION. camera frames per second 
    tint : TYPE, optional
        DESCRIPTION. The default is None which automatically sets the integration time to the 
        minimum given the provided fps. otherwise a value (float) can be specified.

    Raises
    ------
    TypeError
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if FliSdk_V2.IsCredThree(context):
        if np.isfinite(fps):
            if (fps<601) & (fps > 0.1):
                FliSdk_V2.FliSerialCamera.SetFps(context, float(fps))
                res, response = FliSdk_V2.FliSerialCamera.GetFps(context)
                print("FPS set to : " + str(response))
            else:
                raise TypeError('fps not valid, check in script for valid values') 
        else: 
            raise TypeError('fps not finite. fps needs to be finite (max~600)') 


        res, response = FliSdk_V2.FliSerialCamera.SendCommand(context, "mintint raw")
        minTint = float(response) #s

        res, response = FliSdk_V2.FliSerialCamera.SendCommand(context, "maxtint raw")
        maxTint = float(response) #s

        print(f'for selected FPS integration time (tint) must be between {minTint*1e3} - {maxTint*1e3}ms')
        
        if tint == None:
            print(f'setting integration time to maximum ({maxTint*1e3}) given the provided frame rate')
            FliSdk_V2.FliSerialCamera.SendCommand(context, "set tint " + str(maxTint) )
            
        elif (tint<=maxTint*1e3) & (tint >= minTint*1e3):
        
            FliSdk_V2.FliSerialCamera.SendCommand(context, "set tint " + str(float(tint)/1000))

        else: 
            raise TypeError('tint not valid. Check (tint<=maxTint) & (tint >= minTint)') 

        res, response = FliSdk_V2.FliSerialCamera.SendCommand(context, "tint raw")
        print("camera tint set to : " + str(float(response)*1000) + "ms")

        FliSdk_V2.ImageProcessing.EnableAutoClip(context, -1, True)
        
        ok = FliSdk_V2.Update(context)
        if not ok:
            print("Error while updating SDK.")
            exit()
        return(context) 

 
def get_camera_info(camera):
    camera_info_dict = {} 
    # query camera and DM settings 
    fps_res, fps_response = FliSdk_V2.FliSerialCamera.GetFps(camera)
    tint_res, tint_response = FliSdk_V2.FliSerialCamera.SendCommand(camera, "tint raw")
    #no_actuators
    
    #camera headers
    camera_info_dict['timestamp'] = str(datetime.datetime.now()) 
    camera_info_dict['camera'] = FliSdk_V2.GetCurrentCameraName(camera) 
    camera_info_dict['camera_fps'] = fps_response
    camera_info_dict['camera_tint'] = tint_response
    
    return(camera_info_dict)
   

def get_raw_images(camera, number_of_frames, cropping_corners=None):

    camera_started = FliSdk_V2.IsStarted(camera)

    #check if camera has started 
    if not camera_started:
        FliSdk_V2.Start(camera)
        time.sleep(1) # wait 1 second to start 
    if type(cropping_corners)==type([]):
        if len(cropping_corners)==4:
            x1,x2,y1,y2 = cropping_corners
            images = [FliSdk_V2.GetRawImageAsNumpyArray(camera,-1)[int(x1):int(x2), int(y1):int(y2)] for i in range(number_of_frames)]
        else:
            raise TypeError('cropping_corners must be list of integers of length 4 corresponding to [row min, row max, col min, col max]')

    else:
        images = [FliSdk_V2.GetRawImageAsNumpyArray(camera,-1) for i in range(number_of_frames)]
        

    if not camera_started: # if the camera wasn't started when we started, then we stop it again before finishing
        FliSdk_V2.Stop(camera)  
    
    
    return(images)



def watch_camera(camera, frames_to_watch = 10, time_between_frames=0.01,cropping_corners=None) :
  
    print( f'{frames_to_watch} frames to watch with ~{time_between_frames}s wait between frames = ~{5*time_between_frames*frames_to_watch}s watch time' )

    t0= datetime.datetime.now() 
    plt.figure(figsize=(15,15))
    plt.ion() # turn on interactive mode 
    FliSdk_V2.Start(camera)     
    seconds_passed = 0
    if type(cropping_corners)==list: 
        x1,x2,y1,y2 = cropping_corners #[row min, row max, col min, col max]

    for i in range(int(frames_to_watch)): 
        
        a=FliSdk_V2.GetRawImageAsNumpyArray(camera,-1)
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

    FliSdk_V2.Stop(camera) 
    plt.ioff()# turn off interactive mode 
    plt.close()

def measure_bias():
    print('to do')
    
def apply_bias_correction():
    print('to do')
    
def measure_flat():
    print('to do')
    
def apply_flat_correction():
    print('to do')    
    
def set_up_DM(DM_serial_number='17DW019#053'):
    dm = bmc.BmcDm() # init DM object
    err_code = dm.open_dm(DM_serial_number) # open DM
    if err_code:
        print('Error initializing DM')
        raise Exception(dm.error_string(err_code))
    
    return(dm, err_code)
   

   
def apply_sequence_to_DM_and_record_images(DM, camera, DM_command_sequence, number_images_recorded_per_cmd = 1, take_median_of_images=False, save_dm_cmds = True, calibration_dict=None, additional_header_labels=None, sleeptime_between_commands=0.01, cropping_corners=None, save_fits = None):
    """
    

    Parameters
    ----------
    DM : TYPE
        DESCRIPTION. DM Object from BMC SDK. Initialized 
    camera : TYPE camera objection from FLI SDK.
        DESCRIPTION. Camera context from FLISDK. Initialized from context = FliSdk_V2.Init() 
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
    FliSdk_V2.Start(camera) # start camera
    for cmd_indx, cmd in enumerate(DM_command_sequence):
        print(f'executing cmd_indx {cmd_indx} / {len(DM_command_sequence)}')
        # wait a sec        
        time.sleep(sleeptime_between_commands)
        # ok, now apply command
        DM.send_data(cmd)
        # wait a sec        
        time.sleep(sleeptime_between_commands)

        if should_we_record_images: 
            if take_median_of_images:
                ims_tmp = [np.median(get_raw_images(camera, number_images_recorded_per_cmd, cropping_corners) , axis=0)] #keep as list so it is the same type as when take_median_of_images=False
            else:
                ims_tmp = get_raw_images(camera, number_images_recorded_per_cmd, cropping_corners) 
            image_list.append( ims_tmp )

    
    FliSdk_V2.Stop(camera) # stop camera
    
    # init fits files if necessary
    if should_we_record_images: 
        #cmd2pix_registration
        data = fits.HDUList([]) #init main fits file to append things to
        
        # Camera data
        cam_fits = fits.PrimaryHDU( image_list )
        
        #camera headers
        camera_info_dict = get_camera_info(camera)
        for k,v in camera_info_dict.items():
            cam_fits.header.set(k,v)
        cam_fits.header.set('#images per DM command', number_images_recorded_per_cmd )
        cam_fits.header.set('take_median_of_images', take_median_of_images )
        if cropping_corners!=None:
            cam_fits.header.set('cropping_corners_r1', cropping_corners[0] )
            cam_fits.header.set('cropping_corners_r2', cropping_corners[1] )
            cam_fits.header.set('cropping_corners_c1', cropping_corners[2] )
            cam_fits.header.set('cropping_corners_c2', cropping_corners[3] )

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




def create_phase_screen_cmd_for_DM(scrn, DM, flat_reference, scaling_factor=0.1, drop_indicies = None, plot_cmd=False):
    """
    aggregate a scrn (aotools.infinitephasescreen object) onto a DM command space. phase screen is normalized by
    between +-0.5 and then scaled by scaling_factor and offset by flat_reference command. Final DM command values should
    always be between 0-1. phase screens are usually a NxN matrix, while DM is MxM with some missing pixels (e.g. 
    corners). drop_indicies is a list of indicies in the flat MxM DM array that should not be included in the command space. 
    """

    #print('----------\ncheck phase screen size is multiple of DM\n--------')
    
    Nx_act = DM.num_actuators_width() #number of actuators across DM diameter
    
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



def _set_regions_manually(xyr_list):
    x = float(input('center x'))
    y = float(input('center y'))
    r = float(input('cirlce radius'))
    xyr_list.append((x,y,r))
    repeat = int(input('add another regions? input 1 if yes, 0 if no'))
    if repeat:
        _set_regions_manually(xyr_list)
        
    return(xyr_list) 

def detect_pupil_and_PSF_region(camera, fps = 600 , plot_results = True, cropping_corners=None, save_fits = None): 

    data = scan_detector_framerates(camera, [fps], number_images_recorded_per_cmd = 50, cropping_corners=cropping_corners, save_fits = None)
     
    im = np.median( data[0].data, axis=0 ) #take median of however many images we recorded
    gray_scale_image = np.array( 2**8 * (im - np.min(im)) / (np.max(im) - np.min(im)) , dtype = np.uint8 )
    
    # detect circles, we can play with minDist, param1, param2 to optimize if re-alligned 
    #circles = cv2.HoughCircles(gray_scale_image, method=cv2.HOUGH_GRADIENT, dp=1,minDist=50,param1=5,param2=16,minRadius=0,maxRadius=0)
    try:
        circles = cv2.HoughCircles(gray_scale_image, method=cv2.HOUGH_GRADIENT, dp=1,minDist=50,param1=11,param2=36,minRadius=10,maxRadius=100)[0]
    #detect circles in image
    #circles = np.uint16(np.around(circles)) #[[(x0,y0,r0),..,(xN,yN,rN)]]
    except:
        print("no circles detected\n")
        circles = []
    if (len(circles) > 5) or (len(circles) == 0) :
        print( '\n to many circles detected in image - detection failed\n')

    if plot_results:
        plt.figure(figsize=(8,5))
        plt.imshow( np.array(gray_scale_image,dtype=float) )
        
        if (len(circles) < 5) or (len(circles) == 0) :
            pltcircle = []
            for x,y,r in circles:
                pltcircle.append( plt.Circle((x,y), r, facecolor='None', edgecolor='r', lw=1,label='detected region'))
            for c in pltcircle:
                plt.gca().add_patch(c)
                plt.legend()
        plt.show() 
        #time.sleep(5)
        #plt.close()
    
    
    set_regions_manually = float(input('\nif you are happy with detected regions input 0, otherwise input 1 to manually set masked regions. \n\n---(try (x,y,r) = 65,75,45 for pupil, 160,103,30 for PSF)\n'))

            
    if set_regions_manually:
        circles = [] #re-initialize circles 
        _set_regions_manually(circles)

    if plot_results:
        plt.figure(figsize=(8,5))
        plt.imshow( np.array(gray_scale_image,dtype=float) )
        
        if len(circles) < 5:
            pltcircle = []
            for x,y,r in circles:
                pltcircle.append( plt.Circle((x,y), r, facecolor='None', edgecolor='r', lw=1,label='detected region'))
            for c in pltcircle:
                plt.gca().add_patch(c)
                plt.legend()
        plt.show() 
        #time.sleep(5)
        #plt.close()


    if save_fits!=None:
        fits2save = fits.PrimaryHDU( data )
        #write camera info to headers 
        camera_info_dict = get_camera_info(camera)
        for k,v in camera_info_dict.items():
            data.header.set(k,v) 
        #write dected circles to headers
        for i, (x,y,r) in enumerate(circles[0]):
            data.header.set(f'circle_{i}_xyr',f'({x},{y},{r})') 
        #try write fits 
        if type(save_fits)==str:
            fits2save.writeto(save_fits)
        else:
            raise TypeError('save_images needs to be either None or a string indicating where to save file')
    
    buf = float(input('set radius buffer (%)? 0 if none, othersize enter buffer. Final radius will be r+r*buffer'))
    mask_list = []
    for x,y,r in circles:
        mask_tmp = aotools.pupil.circle(radius = r + r*buf, size = np.max(gray_scale_image.shape), circle_centre = (x,y), origin='corner')
        mask_list.append( mask_tmp[ :gray_scale_image.shape[0], :gray_scale_image.shape[1] ] )
        
 
    return(mask_list, circles, gray_scale_image)





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



def construct_command_basis( basis='Zernike', number_of_modes = 20, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0)):
    """
    returns a change of basis matrix M2C to go from modes to DM commands, where columns are the DM command for a given modal basis. e.g. M2C @ [0,1,0,...] would return the DM command for tip on a Zernike basis. Modes are normalized on command space such that <M>=0, <M|M>=1. Therefore these should be added to a flat DM reference if being applied.     

    basis = string of basis to use
    number_of_modes = int, number of modes to create
    Nx_act_DM = int, number of actuators across DM diameter 
    Nx_act_basis = int, number of actuators across the active basis diameter 
    act_offset = tuple, (actuator row offset, actuator column offset) to offset the basis on DM (i.e. we can have a non-centered basis)
     
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
        raw_basis = zernike.zernike_basis(nterms=number_of_modes, npix=Nx_act_basis )
        for i,B in enumerate(raw_basis):
            # normalize <B|B>=1, <B>=0 (so it is an offset from flat DM shape)
            Bnorm = np.sqrt( 1/np.nansum( B**2 ) ) * B
            # pad with zeros to fit DM square shape and shift pixels as required to center
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
    M2C = np.array( bmcdm_basis_list ).T # take transpose to make columns the modes in command space.

    return(M2C)


def get_DM_command_in_2D(cmd,Nx_act=12):
    #puts nan values in cmd positions that don't correspond to actuator on a square grid until cmd length is square number (12x12 for BMC multi-2.5 DM) so can be reshaped to 2D array to see what the command looks like on the DM. 
    corner_indices = [0, Nx_act-1, Nx_act * (Nx_act-1), Nx_act*Nx_act]
    cmd_in_2D = list(cmd.copy())
    for i in corner_indices:
        cmd_in_2D.insert(i,np.nan)
    return( np.array(cmd_in_2D).reshape(12,12) )
    

def get_reference_pupils(DM, camera, fps, flat_map, number_images_recorded_per_cmd=50, cropping_corners=None,  save_fits=None):
    """
    beam aligned on phase dot, we grab some images of FPM in ( I(theta=phi/2 ) )
    then apply offset so that FPM is out ( I(theta=0 ) ) and record these two images 

    """
    # make our low order basis in command space 
    #basis = construct_command_basis(DM , basis='Zernike', number_of_modes = 3, actuators_across_diam = 'full',flat_map=None)
    
    #flat DM 
    DM.send_data( flat_map )  
    
    #define command sequence
    DM_command_sequence = [flat_map]

    print('\n=======\n ADJUST FOCAL PLANE MASK SO THAT DOT IS OFF AXIS (NO PHASE SHIFT APPLIED!)') 

    frames_to_watch = float(input('input how many frames do you need to watch the camera for to make manual adjustment to FPM?'))

    watch_camera(camera, frames_to_watch = frames_to_watch, time_between_frames=0.05)

    # RECORD FPM-OUT IMAGE 
    data_fpm_out = apply_sequence_to_DM_and_record_images(DM, camera, DM_command_sequence, number_images_recorded_per_cmd = 50, save_dm_cmds = True, calibration_dict=None, additional_header_labels=[('FPM_status','OUT')],sleeptime_between_commands=0.01,cropping_corners=cropping_corners, save_fits = None)

    print('\n=======\n ADJUST FOCAL PLANE MASK SO THAT DOT IS ON AXIS (PHASE SHIFT APPLIED!)') 

    frames_to_watch = float(input('input how many frames do you need to watch the camera for to make manual adjustment to FPM?'))

    watch_camera(camera, frames_to_watch = frames_to_watch, time_between_frames=0.05)


    # RECORD FPM-IN IMAGE 
    data_fpm_in = apply_sequence_to_DM_and_record_images(DM, camera, DM_command_sequence, number_images_recorded_per_cmd = 50, save_dm_cmds = True, calibration_dict=None, additional_header_labels=[('FPM_status','IN')],sleeptime_between_commands=0.01,cropping_corners=cropping_corners, save_fits = None)

    # take median frame over each image set
    fpm_in_fits = fits.PrimaryHDU( np.median( data_fpm_in[0].data[0] , axis = 0) )
    fpm_out_fits = fits.PrimaryHDU( np.median( data_fpm_out[0].data[0] , axis = 0) )
    # SET FPM EXTNAMES headers
    fpm_in_fits.header.set('EXTNAME','FPM_IN')
    fpm_out_fits.header.set('EXTNAME','FPM_OUT')
    #camera headers
    camera_info_dict = get_camera_info(camera)
    for k,v in camera_info_dict.items():
       fpm_in_fits.header.set(k,v)
       fpm_out_fits.header.set(k,v)

    # init and append calibration frames to main fits list
    data = fits.HDUList([])
    data.append( fpm_in_fits )
    data.append( fpm_out_fits )
    if save_fits!=None:
        if type(save_fits)==str:
            data.writeto(save_fits)
        else:
            raise TypeError('save_images needs to be either None or a string indicating where to save file')
            
        
    return(data) 




def get_error_signal( image, reference_pupil_fits, reduction_dict=None, pupil_indicies = None ): 
    """
    signal processing for getting error vector from an input image from the ZWFS. 
    This error vector can be fed back with PID gains  for closed loop operation   
    - image is the image to turn into control signal
    - reference_pupil_fits is a fits file with images of the pupil with FPM in and out, it is the output of get_reference_pupils() function 
    - reduction_dict is a dictionary with darks biases etc if we want to reduce signals before processing 
    - pupil_indicies is list of indicies to crop pupil in the image. i.e pupil_indicies=[x1,x2,y1,y2]. For no cropping pupil_indicies=None, which is default
    """
    if reduction_dict==None:
        if pupil_indicies == None: # no cropping of pupil in images 
            Serr = ( image - reference_pupil_fits['FPM_IN'].data ) /  np.mean( reference_pupil_fits['FPM_OUT'].data )
        else : 
            x1,x2,y1,y2 = pupil_indicies
            Serr = ( image[x1:x2,y1:y2] - reference_pupil_fits['FPM_IN'].data[x1:x2,y1:y2] ) /  np.mean( reference_pupil_fits['FPM_OUT'].data[x1:x2,y1:y2] )
    else:
        raise TypeError('to do: include reduction from deduction dictionary') 
    
    return(Serr) 


"""
# how to deal with corners 
# we have an image and a control matrix which is pseudo inverse of IM

# zernikes need to be defined relative to flat surface 
zernikes normalized between -0.5-0.5, flat surface is 0.5

mode = flat + Normed_Zernike 



#corner indices in flattened array
corner_indices = 0, len(BMC_dm_reco)-1, len(BMC_dm_reco) * (len(BMC_dm_reco)-1), -1
flat_BMC_dm_cmd = BMC_dm_reco.reshape(-1).copy()
for i in corner_indices:
    flat_BMC_dm_cmd[i]=np.nan

# note cmds need to be 140, not 144 since corners of 12x12 DM not in command space
dm.send_data( flat_BMC_dm_cmd[np.isfinite(flat_BMC_dm_cmd)] )

    
def indentify_valid_camera_regions():
    #poke DM randomly recording images and take difference to non-poked reference image
    
"""    

"""
#%% 
dm_name = '17DW019#053'
# zero_DM 
flat_map = pd.read_csv("/home/heimdallr/Documents/17DW019#053_FLAT_MAP_COMMANDS.txt",header=None)[0].values

dm.send_data(flat_map) #send the flatmap cmds to DM 

# get noise properties of camera
# apply flat DM and change frame rate and apply min DIT to see how noise changes 



for b in zwfs.control_variables['control_20_zernike_modes']['control_basis']:
    bmcdm_basis.append( (b - b.min()) / (b.max() - b.min()) )  #normalized between 0-1
  
    
  
# +++++++ setup camera 
fps = 300 #frames per sec 
tint = 1.1 # integration time (ms) 

camera_context = setup_camera()
camera_context = set_fsp_dit( context, fps, tint )

dm, dm_errcode = set_up_DM(DM_serial_number='')

# start camera 
FliSdk_V2.Start(context)

"""
