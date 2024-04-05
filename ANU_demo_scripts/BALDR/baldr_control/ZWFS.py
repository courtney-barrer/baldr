#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:53:02 2024

@author: bencb

ZWFS class can only really get an image and send a command to the DM and/or update the camera settings

it does have an state machine, any processes interacting with ZWFS object
must do the logic to check and update state

"""

import bmc
import numpy as np
import os
import glob 
import bmc
os.chdir('/opt/FirstLightImaging/FliSdk/Python/demo/')
import FliSdk_V2 
import pandas as pd 


                


class ZWFS():
    # for now one camera, one DM per object - real system will have 1 camera, 4 DMs!
    def __init__(self, DM_serial_number, cameraIndex=0, DMshapes_path='/home/baldr/Documents/baldr/DMShapes/' ):
       
        # connecting to camera
        camera = FliSdk_V2.Init() # init camera object
        listOfGrabbers = FliSdk_V2.DetectGrabbers(camera)
        listOfCameras = FliSdk_V2.DetectCameras(camera)
        # print some info and exit if nothing detected
        if len(listOfGrabbers) == 0:
            print("No grabber detected, exit.")
            exit()
        if len(listOfCameras) == 0:
            print("No camera detected, exit.")
            exit()
        for i,s in enumerate(listOfCameras):
            print("- index:" + str(i) + " -> " + s)
       
        print(f'--->using cameraIndex={cameraIndex}')
        # set the camera
        camera_err_flag = FliSdk_V2.SetCamera(camera, listOfCameras[cameraIndex])
        if not camera_err_flag:
            print("Error while setting camera.")
            exit()
        print("Setting mode full.")
        FliSdk_V2.SetMode(camera, FliSdk_V2.Mode.Full)
        print("Updating...")
        camera_err_flag = FliSdk_V2.Update(camera)
        if not camera_err_flag:
            print("Error while updating SDK.")
            exit()
           
        # connecting to DM
       
        dm = bmc.BmcDm() # init DM object
        dm_err_flag  = dm.open_dm(DM_serial_number) # open DM
        if dm_err_flag :
            print('Error initializing DM')
            raise Exception(dm.error_string(dm_err_flag))
       
        # ========== CAMERA & DM
        self.camera = camera
        self.dm = dm
        self.dm_number_of_actuators = 140

        # ========== DM shapes
        shapes_dict = {}
        if os.path.isdir(DMshapes_path):
            filenames = glob.glob(DMshapes_path+'*.csv')
            if filenames == []:
                print('no valid files in path. Empty dictionary appended to ZWFS.dm_shapes')
            else:
                for file in filenames:
                
                    try:
                        shape_name = file.split('/')[-1].split('.csv')[0]
                        shape = pd.read_csv(file, header=None)[0].values
                        if len(shape)==self.dm_number_of_actuators:
                            shapes_dict[shape_name] = np.array( (shape) )
                        else: 
                            print(f'file: {file}\n has {len(shape)} values which does not match the number of actuators on the DM ({self.dm_number_of_actuators})')

                    except:
                        print(f'falided to read and/or append shape corresponding to file: {file}')
        else:
            print('DMshapes path does not exist. Empty dictionary appended to ZWFS.dm_shapes')

        self.dm_shapes = shapes_dict
           
            
        #self.dm_err_flag = dm_err_flag
        #self.camera_err_flag = camera_err_flag
        #self.flat_dm_cmd =  pd.read_csv(DMshapes_path+'17DW019#053_FLAT_MAP_COMMANDS.txt',header=None)[0].values
        #self.four_torres = 
        
 
        # ========== CONTROLLERS
        self.phase_controllers = [] # initiate with no controllers
        self.pupil_controllers = [] # initiate with no controllers
        # ========== STATES
        """
        Notes:
        - Any process that takes ZWFS object as input is required to update ZWFS states
        - before using any controller, the ZWFS states should be compared with controller
        configuration file to check for consistency (i.e. can controller be applied in the current state?)
        """
        self.states = {}
       
        self.states['simulation_mode'] = 0 # 1 if we have in simulation mode
       
        self.states['telescopes'] = ['AT1'] #
        self.states['phase_ctrl_state'] = 0 # 0 open loop, 1 closed loop
        self.states['pupil_ctrl_state'] = 0 # 0 open loop, 1 closed loop
        self.states['source'] = 0 # 0 for no source, 1 for internal calibration source, 2 for stellar source
        self.states['sky'] = 0 # 1 if we are on a sky (background), 0 otherwise
        self.states['fpm'] = 0 # 0 for focal plance mask out, positive number for in depending on which one)
        self.states['busy'] = 0 # 1 if the ZWFS is doing something
       
    def send_cmd(self, cmd):
       
        self.dm_err_flag = self.dm.send_data(cmd)
       
       
   
    def propagate_states(self, simulation_mode = False):
       
        if not self.states['simulation_mode']:
           
            for state, value in self.states.items():
                try:
                    print('check the systems state relative to current ZWFS state and \
                      rectify (e.g. send command to some motor) if any discrepency')
                except:
                    print('raise an error or implement some workaround if the requested state cannot be realised')

    def get_image(self):
        # I do not check if the camera is running. Users should check this 
        # gets the last image in the buffer
        img = FliSdk_V2.GetRawImageAsNumpyArray( self.camera , -1)       
        return(img)    

    def start_camera(self):
        FliSdk_V2.Start(self.camera)

    def stop_camera(self):
        FliSdk_V2.Stop(self.camera)

    def get_camera_dit(self):
        camera_err_flag, DIT = FliSdk_V2.FliSerialCamera.SendCommand(self.camera, "tint raw")
        return( DIT ) 

    def get_camera_fps(self):
        camera_err_flag, fps = FliSdk_V2.FliSerialCamera.GetFps(self.camera)
        return( fps ) 

    def get_dit_limits(self):
               
        camera_err_flag, minDIT = FliSdk_V2.FliSerialCamera.SendCommand(self.camera, "mintint raw")
        self.camera_err_flag = camera_err_flag
       
        camera_err_flag, maxDIT = FliSdk_V2.FliSerialCamera.SendCommand(self.camera, "maxtint raw")
        self.camera_err_flag = camera_err_flag
       
        return(minDIT, maxDIT)   

    def set_camera_dit(self, DIT):
        # set detector integration time (DIT). input in seconds
        minDit, maxDit = self.get_dit_limits()
        if (DIT >= float(minDit)) & (DIT <= float(maxDit)):
            FliSdk_V2.FliSerialCamera.SendCommand(self.camera, "set tint " + str(float(DIT)))
        else:
            print(f"requested DIT {1e3*DIT}ms outside DIT limits {(1e3*minDit,1e3*maxDit)}ms.\n Cannot change DIT to this value")
    
    def set_camera_fps(self, fps):
        FliSdk_V2.FliSerialCamera.SetFps(self.camera, fps)


    def crop_camera( self, region ):
        print('')
       
       
       
       
