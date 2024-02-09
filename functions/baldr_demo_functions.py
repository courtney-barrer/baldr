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
import matplotlib.pyplot as plt
import time
from scipy.interpolate import interp2d, interp1d
from scipy.stats import poisson
from astropy.io import fits
import pyzelda.utils.zernike as zernike
import bmc
os.chdir('/opt/FirstLightImaging/FliSdk/Python/demo/')
import FliSdk_V2 
#import astropy.constants as cst
#import astropy.units as u


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
            print(f'setting integration time to minimum ({minTint*1e3}) given the provided frame rate')
            FliSdk_V2.FliSerialCamera.SendCommand(context, "set tint " + str(minTint) )
            
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
    
def measure_bias():
    print('to do')
    
def apply_bias_correction():
    print('to do')
    
def measure_flat():
    print('to do')
    
def apply_flat_correction():
    print('to do')    
    
def set_up_DM(DM_serial_number=''):
    dm = bmc.BmcDm() # init DM object
    err_code = dm.open_dm(DM_serial_number) # open DM
    if err_code:
        print('Error initializing DM')
        raise Exception(dm.error_string(err_code))
    
    return(dm, err_code)
    
def get_camera_info(camera):
    camera_info_dict = {} 
    # query camera and DM settings 
    fps_res, fps_response = FliSdk_V2.FliSerialCamera.GetFps(camera)
    tint_res, tint_response = FliSdk_V2.FliSerialCamera.SendCommand(context, "tint raw")
    #no_actuators
    
    #camera headers
    camera_info_dict['timestamp'] = str(datetime.datetime.now()) 
    camera_info_dict['camera'] = FliSdk_V2.GetCurrentCameraName(camera) 
    camera_info_dict['camera_fps'] = fps_res 
    camera_info_dict['camera_tint'] = tint_res*1e-3 
    
    return(camera_info_dict)
    
    
def apply_sequence_to_DM_and_record_images(DM, camera, DM_command_sequence, number_images_recorded_per_cmd = 1, save_dm_cmds = True, calibration_dict=None, save_fits = None):
    """
    

    Parameters
    ----------
    DM : TYPE
        DESCRIPTION. DM Object from BMC SDK. Initialized 
    camera : TYPE camera objection from FLI SDK.
        DESCRIPTION. Camera context from FLISDK. Initialized from context = FliSdk_V2.Init() 
    DM_command_sequence : TYPE
        DESCRIPTION.
    number_images_recorded_per_cmd : TYPE, optional
        DESCRIPTION. The default is 1. puting a value >= 0 means no images are recorded.
    calibration_dict: TYPE, optional
        DESCRIPTION. The default is None meaning saved images don't get flat fielded. 
        if flat fielding is required a dictionary must be supplied that contains 
        a bias, dark and flat frame under keys 'bias', 'dark', and 'flat' respectively
    save_fits : TYPE, optional
        DESCRIPTION. The default is None which means images are not saved, 
        if a string is provided images will be saved with that name in the current directory

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
    
    image_list = []
    for cmd in DM_command_sequence:
    
        DM.apply_command(cmd)
    
        if should_we_record_images: 
            
            image_list.append( [FliSdk_V2.GetRawImageAsNumpyArray(context,-1)  for i in range(number_images_recorded_per_cmd)] )

    # init fits files if necessary
    if should_we_record_images: 
        
        data = fits.HDUList([]) 
        
        # Camera data
        cam_fits = fits.PrimaryHDU( image_list )
        if save_dm_cmds:
            dm_fits = fits.PrimaryHDU( DM_command_sequence )
        
        # query camera and DM settings 
        fps_res, fps_response = FliSdk_V2.FliSerialCamera.GetFps(camera)
        tint_res, tint_response = FliSdk_V2.FliSerialCamera.SendCommand(context, "tint raw")
        #no_actuators
        
        #camera headers
        camera_info_dict = get_camera_info(camera)
        for k,v in items( camera_info_dict ):
            cam_fits.header.set(k,v)
        cam_fits.header.set('#images per DM command', number_images_recorded_per_cmd )
        #cam_fits.header.set('timestamp', str(datetime.datetime.now()) )
        #cam_fits.header.set('camera', FliSdk_V2.GetCurrentCameraName(camera) )
        #cam_fits.header.set('camera_fps', fps_res )
        #cam_fits.header.set('camera_tint', tint_res*1e-3 )
        
             
        # add camera data to main fits
        data.append(cam_fits)
        
        if save_dm_cmds:
            #DM headers 
            dm_fits.header.set('timestamp', str(datetime.datetime.now()) )
            #dm_fits.header.set('DM', DM.... )
            #dm_fits.header.set('#actuators', DM.... )
            data.append(dm_fits)
        
        if save_fits!=None:
            if type(save_fits)==str:
                data.writeto(save_fits)
            else:
                raise TypeError('save_images needs to be either None or a string indicating where to save file')
            
            
        return(data)
    
    else:
        return(None)


    
def scan_detector_framerates(camera, frame_rates, number_images_recorded_per_cmd = 50, save_fits = None): 
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
        
        set_fsp_dit( camera, fps, tint=None) # set mimimu dit (tint=None) for given fps
        
        tmp_fits = fits.PrimaryHDU( [FliSdk_V2.GetRawImageAsNumpyArray(context,-1)  for i in range(number_images_recorded_per_cmd)] )
        tmp_fits.header.set('frames per second' , fps)
        
    
    for i in image_list: 
        
        tmp_fits = fits.PrimaryHDU( i )
        
        camera_info_dict = get_camera_info(camera)
        for k,v in items( camera_info_dict ):
            tmp_fits.header.set(k,v)

    data.append( tmp_fits )
    
    if save_fits!=None:
        if type(save_images)==str:
            data.writeto(save_fits)
        else:
            raise TypeError('save_images needs to be either None or a string indicating where to save file')
        
    return(data)



def construct_command_basis(DM ,flat_map, basis='Zernike', number_of_modes = 20):
    
    Nx_act = dm.num_actuators_width() # number of actuators across diameter of DM.
    
    # to deal with
    if np.mod( Nx_act,2 )==0:
        basis = zernike.zernike_basis(nterms=number_of_modes, npix=Nx_act + 4 )
    else:
        basis = zernike.zernike_basis(nterms=number_of_modes, npix=Nx_act + 4 )

    basis_name2i = {zernike.zern_name(i):i for i in range(1,30)}
    basis_i2name = {v:k for k,v in basis_name2i.items()}
    
    
    for b in zwfs.control_variables['control_20_zernike_modes']['control_basis']:
        bmcdm_basis.append( 0.5 - (b - b.min()) / (b.max() - b.min()) )  #normalized between 0-1 
        
    # DM is 12x12 without the corners, so mask them with nan and drop them
    corner_indices = 0, Nx_act-1, Nx_act * (Nx_act-1), -1

flat_map = pd.read_csv("/opt/Boston Micromachines/Shapes/17DW019#053_FLAT_MAP_COMMANDS.txt",header=None)[0].values 


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