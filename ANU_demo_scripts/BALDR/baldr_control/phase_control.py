from . import utilities as util 
from . import hardware 

import matplotlib.pyplot as plt 
import numpy as np
import time 
class phase_controller_1():
    """
    linear interaction model on internal calibration source
    """
   
    def __init__(self, config_file = None):
       
        if type(config_file)==str:
            if config_file.split('.')[-1] == 'json':
                with open('data.json') as json_file:
                    self.config  = json.load(json_file)
            else:
                raise TypeError('input config_file is not a json file')
               
        elif config_file==None:
            # class generic controller config parameters
            self.config = {}
            self.config['telescopes'] = ['AT1']
            self.config['fpm'] = 1  #0 for off, positive integer for on a particular phase dot
            self.config['basis'] = 'Zernike' # either Zernike, Zonal, KL, or WFS_Eigenmodes
            self.config['number_of_controlled_modes'] = 70 # number of controlled modes
            self.config['source'] = 1
           
            self.ctrl_parameters = {} # empty dictionary cause we have none
            #for tel in self.config['telescopes']:
            
            self.config['Kp'] = [0 for _ in range( self.config['number_of_controlled_modes'] )] # proportional gains
            self.config['Ki'] = [0 for _ in range( self.config['number_of_controlled_modes'] )] # integral gains

            self.config['dm_control_diameter'] = 12 # diameter (actuators) of active actuators
            self.config['dm_control_center'] = [0,0] # in-case of mis-alignments we can offset control basis on DM
            # all reference intensities normalized by sum of pixels in cropping region. 
            self.I0 = None #reference intensity filtered over defined pupil with FPM IN (1D array)
            self.N0 = None #reference intensity filtered over defined pupil with FPM OUT (1D array)
            self.I0_2D = None # reference intensity with FPM IN without filtering (2D array)
            self.N0_2D = None # reference intensity with FPM OUT without filtering (2D array)
            self.b = None #ZWFS gain in pixel space filtered over defined pupil - needs to be calculated with I0, N0 measurement 
            self.b_2D = None #ZWFS gain in pixel space filtered over defined pupil - needs to be calculated with I0, N0 measurement 

            self.theta = 3.14/180 * 110 # (radian) phase shift of phase mask 
            self.FPM_diam = 49e-6 # (m) - focal plane mask dot diameter 
            self.FPM_depth = 0.932e-6 # (m) - focal plane mask dot diameter 
            self.fratio = 21.2 # focal length / D of ZWFS.
            self.wvl0 = 1.29e-6 # (m) - central wavelength of system (default is center of SLS202 L/M thermal source measured in CRED3 detector)           

            # mode to command matrix
            M2C = util.construct_command_basis( basis=self.config['basis'] , \
                    number_of_modes = self.config['number_of_controlled_modes'],\
                        Nx_act_DM = 12,\
                           Nx_act_basis = self.config['dm_control_diameter'],\
                               act_offset= self.config['dm_control_center'], without_piston=True) # cmd = M2C @ mode_vector, i.e. mode to command matrix
           
            self.config['M2C'] = M2C # 
                #self.config[tel]['control_parameters'] = [] # empty list cause we have none
            self.config['active_actuator_filter'] = (abs(np.sum( self.config['M2C'], axis=1 )) > 0 ).astype(bool)
           

            self.config['active_actuator_indicies'] = np.where( abs(np.sum( self.config['M2C'], axis=1 )) > 0 )[0]
             
#plt.imshow( util.get_DM_command_in_2D( abs(np.sum( phase_ctrl.config['M2C'], axis=1 )) > 0 ) ); plt.show() #<-this can be used to check the actuator filter makes sense. 

        else:
            raise TypeError( 'config_file type is wrong. It must be None or a string indicating the config file to read in' )


    def change_control_basis_parameters(self,  number_of_controlled_modes, basis_name ,dm_control_diameter=None, dm_control_center=None,controller_label=None):
        # standize updating of control basis parameters so no inconsistencies
        # warning no error checking here! 
        # controller_label only needs to be provided if updating to 'WFS_Eigenmodes' because it needs the covariance of the IM which is stored in self.ctrl_parameters[controller_label]['IM']

        if basis_name != 'WFS_Eigenmodes':
            self.config['basis'] = basis_name
            self.config['number_of_controlled_modes'] = number_of_controlled_modes

            if dm_control_diameter!=None: 
                self.config['dm_control_diameter'] =  dm_control_diameter
            if dm_control_center!=None:
                self.config['dm_control_center']  =  dm_control_center
        
            # mode to command matrix
            M2C = util.construct_command_basis( basis=self.config['basis'] , \
                        number_of_modes = self.config['number_of_controlled_modes'],\
                            Nx_act_DM = 12,\
                               Nx_act_basis = self.config['dm_control_diameter'],\
                                   act_offset= self.config['dm_control_center'], without_piston=True)

            self.config['M2C'] = M2C 

            self.config['Kp'] = [0 for _ in range( self.config['number_of_controlled_modes'] )] # proportional gains
            self.config['Ki'] = [0 for _ in range( self.config['number_of_controlled_modes'] )] # integral gains
            self.config['active_actuator_filter'] = (abs(np.sum( self.config['M2C'], axis=1 )) > 0 ).astype(bool)
 
            self.config['active_actuator_indicies'] = np.where( abs(np.sum( self.config['M2C'], axis=1 )) > 0 )[0]

        elif  basis_name == 'WFS_Eigenmodes':  
            self.config['basis'] = basis_name
            # treated differently because we need covariance of interaction matrix to build the modes. 
            # number of modes infered by size of IM covariance matrix 
            try:
                IM = self.ctrl_parameters[controller_label]['IM']  #interaction matrix
            except:
                raise TypeError( 'nothing found in self.ctrl_parameters[controller_label]["IM"]' ) 

            M2C_old = self.config['M2C']

            # some checks its a covariance matrix
            if not hasattr(IM, '__len__'):
                raise TypeError('IM needs to be 2D matrix' )
                if len(np.array(IM).shape) != 2:
                    raise TypeError('IM  needs to be 2D matrix')

            #spectral decomposition 
            IM_covariance = np.cov( IM )
            U,S,UT = np.linalg.svd( IM_covariance )


            # project our old modes represented in the M2C matrix to new ones
            M2C_unnormalized  = M2C_old @ U.T

            # normalize 
            M2C = [] 
            for m in np.array(M2C_unnormalized).T:
                M2C.append( 1/np.sqrt( np.sum( m*m ) ) * m  ) # normalize all modes <m|m>=1

            M2C = np.array(M2C).T            

            if len(M2C.T)  !=  self.config['number_of_controlled_modes']:
                print( '\n\n==========\nWARNING: number of mode  self.config["number_of_controlled_modes"] != len( IM )')

            if dm_control_diameter!=None: 
                print( ' cannot update dm control diameter with eigenmodes, it inherits it from the previous basis') 
            if dm_control_center!=None:
                print( ' cannot update dm control center with eignmodes, it inherits it from the previous basis') 

            self.config['M2C'] = M2C 

            self.config['Kp'] = [0 for _ in range( self.config['number_of_controlled_modes'] )] # proportional gains
            self.config['Ki'] = [0 for _ in range( self.config['number_of_controlled_modes'] )] # integral gains
            self.config['active_actuator_filter'] = (abs(np.sum( self.config['M2C'], axis=1 )) > 0 ).astype(bool)
 
            self.config['active_actuator_indicies'] = np.where( abs(np.sum( self.config['M2C'], axis=1 )) > 0 )[0]




    def build_control_model(self, ZWFS, poke_amp = -0.15, label='ctrl_1', debug = True):
        
        
        # remember that ZWFS.get_image automatically crops at corners ZWFS.pupil_crop_region
        ZWFS.states['busy'] = 1
        #update_references = int( input('get new reference intensities (1/0)') )

        imgs_to_median = 10 # how many images do we take the median of to build signal the reference signals        
        if 1 : #update_references : #| ( (self.I0==None) | (self.N0==None) ):


            # check other states match such as source etc
        
            # =========== PHASE MASK OUT 
            hardware.set_phasemask( phasemask = 'out' ) # THIS DOES NOTHING SINCE DONT HAVE MOTORS YET.. for now we just look at the pupil so we can manually move phase mask in and out. 
            _ = input('MANUALLY MOVE PHASE MASK OUT OF BEAM, PRESS ANY KEY TO BEGIN' )
            util.watch_camera(ZWFS, frames_to_watch = 50, time_between_frames=0.05)

            ZWFS.states['fpm'] = 0
    
            N0_list = []
            for _ in range(imgs_to_median):
                N0_list.append( ZWFS.get_image(  ) ) #REFERENCE INTENSITY WITH FPM IN
            N0 = np.median( N0_list, axis = 0 ) 
            #put self.config['fpm'] phasemask on-axis (for now I only have manual adjustment)

            # =========== PHASE MASK IN 
            hardware.set_phasemask( phasemask = 'posX' ) # # THIS DOES NOTHING SINCE DONT HAVE MOTORS YET.. for now we just look at the pupil so we can manually move phase mask in and out. 
            _ = input('MANUALLY MOVE PHASE MASK BACK IN, PRESS ANY KEY TO BEGIN' )
            util.watch_camera(ZWFS, frames_to_watch = 50, time_between_frames=0.05)

            ZWFS.states['fpm'] = self.config['fpm']

            I0_list = []
            for _ in range(imgs_to_median):
                I0_list.append( ZWFS.get_image(  ) ) #REFERENCE INTENSITY WITH FPM IN
            I0 = np.median( I0_list, axis = 0 ) 
        
            # === ADD ATTRIBUTES 
            self.I0 = I0.reshape(-1)[np.array( ZWFS.pupil_pixels )] / np.mean( I0 ) # append reference intensity over defined     pupil with FPM IN 
            self.N0 = N0.reshape(-1)[np.array( ZWFS.pupil_pixels )] / np.mean( N0 ) # append reference intensity over defined pupil with FPM OUT 

            # === also add the unfiltered so we can plot and see them easily on square grid after 
            self.I0_2D = I0 / np.mean( I0 ) # 2D array (not filtered by pupil pixel filter)  
            self.N0_2D = N0 / np.mean( N0 ) # 2D array (not filtered by pupil pixel filter)

        #  also update b attribute in phase controller
        self.update_b( ZWFS, self.I0_2D, self.N0_2D )
        

        modal_basis = self.config['M2C'].copy().T # more readable
        IM=[] # init our raw interaction matrix 

        for i,m in enumerate(modal_basis):

            print(f'executing cmd {i}/{len(modal_basis)}')
            
            ZWFS.dm.send_data( list( ZWFS.dm_shapes['flat_dm'] + poke_amp * m )  )
            time.sleep(0.05)
            img_list = []  # to take median of 
            for _ in range(imgs_to_median):
                img_list.append( ZWFS.get_image(  ).reshape(-1)[np.array( ZWFS.pupil_pixels )] )
                time.sleep(0.01)
            I = np.median( img_list, axis = 0) 

            if (I.shape == self.I0.shape) & (I.shape == self.N0.shape):
                # !NOTE! we take median of pupil reference intensity with FPM out (self.N0)
                # we do this cause we're lazy and do not want to manually adjust FPM every iteration (we dont have motors here) 
                # real system prob does not want to do this and normalize pixel wise. 
 
                # HAVE TO NORMALIZE BEFORE SENDING TO get_img_err
                I *= 1/np.mean( I ) 
                errsig =  self.get_img_err( I ) #np.array( ( (I - self.I0) / np.median( self.N0 ) ) )
            else: 
                raise TypeError(" reference intensity shapes do not match shape of current measured intensity. Check phase_controller.I0 and/or phase_controller.N0 attributes. Workaround would be to retake these. ")
            IM.append( list( errsig.reshape(-1) ) )

        # FLAT DM WHEN DONE
        ZWFS.dm.send_data( list( ZWFS.dm_shapes['flat_dm'] ) )
               
        # SVD
        U,S,Vt = np.linalg.svd( IM , full_matrices=True)

        # filter number of modes in eigenmode space when using zonal control  
        if self.config['basis'] == 'Zonal': # then we filter number of modes in the eigenspace of IM 
            S_filt = S > S[ self.config['number_of_controlled_modes'] ] # we consider the highest eigenvalues/vectors up to the number_of_controlled_modes
            Sigma = np.zeros( np.array(IM).shape, float)
            np.fill_diagonal(Sigma, S[S_filt], wrap=False) #
        else: # else #modes decided by the construction of modal basis. We may change their gains later
            S_filt = S > 0 # S > S[ np.min( np.where( abs(np.diff(S)) < 1e-2 )[0] ) ]
            Sigma = np.zeros( np.array(IM).shape, float)
            np.fill_diagonal(Sigma, S[S_filt], wrap=False) #

        if debug:
            #plotting DM eigenmodes to see which to filter 
            print( ' we can only easily plot eigenmodes if pupil_pixels is square region!' )
            """
            fig,ax = plt.subplots(6,6,figsize=(15,15))
            
            for i,axx in enumerate( ax.reshape(-1) ) :
                axx.set_title(f'eigenmode {i}')
                axx.imshow( util.get_DM_command_in_2D( U[:,i] ) )
            plt.suptitle( 'DM EIGENMODES' ) #f"rec. cutoff at i={np.min( np.where( abs(np.diff(S)) < 1e-2 )[0])}", fontsize=14)
            plt.show()
            """
        # control matrix
        CM =  np.linalg.pinv( U @ Sigma @ Vt ) # C = A @ M #1/abs(poke_amp)
       
        # class specific controller parameters
        ctrl_parameters = {}
       
        ctrl_parameters['active'] = 0 # 0 if unactive, 1 if active (should only have one active phase controller)

        ctrl_parameters['ref_pupil_FPM_out'] = N0

        ctrl_parameters['ref_pupil_FPM_in'] = I0

        ctrl_parameters['IM'] = IM
       
        ctrl_parameters['CM'] = CM
       
        ctrl_parameters['P2C'] = None # pixel to cmd registration (i.e. what region)

        ZWFS.states['busy'] = 0
       
        self.ctrl_parameters[label] = ctrl_parameters
       
        if debug: # plot covariance of interaction matrix 
            plt.title( 'Covariance of Interaciton Matrix' )
            plt.imshow( np.cov( self.ctrl_parameters[label]['IM'] ) )
            plt.colorbar()
            plt.show()


    def update_b( self, ZWFS, I0_2D, N0_2D ):
        #I0 = reference image with FPM in (2D array - CANNOT be flatttened 1D array)
        #N0 = reference image with FPM out (2D array - CANNOT be flatttened 1D array)
        # even though I0_2D, N0_2D is attribute in self we allow user to specify new ones to update b
        image_filter = ZWFS.refpeak_pixel_filter | ZWFS.outside_pixel_filter

        b_pixel_space = util.fit_b_pixel_space(I0_2D, self.theta, image_filter , debug=False)
        # full fit over ZWFS image 
        self.b_2D =  b_pixel_space
        #flattened and filtered in pupil space 
        self.b = b_pixel_space.reshape(-1)[ZWFS.pupil_pixel_filter] 




    def get_img_err( self , img_normalized ):
        # input img has to be normalized by sum of entire raw image in cropped region 
        # flattened and filtered in pupil region 
        #(i.e. get raw img then apply img.reshape(-1)[ ZWFS.pupil_pixels])

        # !NOTE! we take median of pupil reference intensity with FPM out (self.N0)
        # we do this cause we're lazy and do not want to manually adjust FPM every iteration (we dont have motors here) 
        # real system prob does not want to do this and normalize pixel wise. 
        #errsig =  np.array( ( (img - self.I0) / np.median( self.N0 ) ) )
        
        # normalize image 

        errsig =  np.array( ( (img_normalized - self.I0) / (2 * self.b * np.sin( self.theta ) ) ) )

        return(  errsig )


    def update_FPM_OUT_reference(self, ZWFS, N0_2D):
        #N0_2D must be a 2D array of the estimate of the pupil intensity with the FPM mask out.
        #!!!! IMPORTANT !!! WE DO NOT NORMALIZE HERE - SO I0_2D SHOULD BE NORMALIZED BY MEAN OVER FULL PUPIL IMAGE REGION ON INPUT!
        #ZWFS must be provided since this holds the pixel filter for the wavefront sensing 

        # we should do some checking here that the input N0_2D matches number of pixels from ZWFS.get_image() 
        r1, r2, c1, c2 = self.pupil_crop_region
        if N0_2D.shape != [r2-r1, c2-c1]:
            raise TypeError(f'N0_2D needs to have shape {[r2-r1,c2-c1]} to match image size produced from the input ZWFS object.')  

        self.N0 =  N0_2D.reshape(-1)[np.array( ZWFS.pupil_pixels )]
        self.N0_2D =  N0_2D   
        

    def update_FPM_IN_reference(self, ZWFS, I0_2D):
        #i0_2D must be a 2D array of the estimate of the pupil intensity with the FPM mask out.
        #!!!! IMPORTANT !!! WE DO NOT NORMALIZE HERE - SO I0_2D SHOULD BE NORMALIZED BY MEAN OVER FULL PUPIL IMAGE REGION ON INPUT!
        #ZWFS must be provided since this holds the pixel filter for the wavefront sensing 

        # we should do some checking here that the input N0_2D matches number of pixels from ZWFS.get_image() 
        r1, r2, c1, c2 = self.pupil_crop_region
        if np.array(I0_2D).shape != [r2-r1, c2-c1]:
            raise TypeError(f'N0_2D needs to have shape {[r2-r1,c2-c1]} to match image size produced from the input ZWFS object.')  

        self.I0 = I0_2D.reshape(-1)[np.array( ZWFS.pupil_pixels )]
        self.I0_2D =  I0_2D   


    def measure_FPM_OUT_reference(self, ZWFS) : 
        
        imgs_to_median = 10 # how many images do we take the median of to build signal the reference signals 
        # check other states match such as source etc

        # =========== PHASE MASK OUT 
        hardware.set_phasemask( phasemask = 'out' ) # THIS DOES NOTHING SINCE DONT HAVE MOTORS YET.. for now we just look at the pupil so we can manually move phase mask in and out. 
        _ = input('MANUALLY MOVE PHASE MASK OUT OF BEAM, PRESS ANY KEY TO BEGIN' )
        util.watch_camera(ZWFS, frames_to_watch = 50, time_between_frames=0.05)

        ZWFS.states['fpm'] = 0

        N0_list = []
        for _ in range(imgs_to_median):
            N0_list.append( ZWFS.get_image(  ) ) #REFERENCE INTENSITY WITH FPM IN
        N0 = np.median( N0_list, axis = 0 ) 
        #put self.config['fpm'] phasemask on-axis (for now I only have manual adjustment)

        # === ADD ATTRIBUTES 
        self.N0 =  1/np.mean( N0 ) * N0.reshape(-1)[np.array( ZWFS.pupil_pixels )] # append reference intensity over defined pupil with FPM OUT 

        # === also add the unfiltered so we can plot and see them easily on square grid after 
        self.N0_2D = 1/np.mean( N0 ) * N0 # 2D array (not filtered by pupil pixel filter) 

    def measure_FPM_IN_reference(self, ZWFS):

        imgs_to_median = 10 # how many images do we take the median of to build signal the reference signals 
        # =========== PHASE MASK IN 
        hardware.set_phasemask( phasemask = 'posX' ) # # THIS DOES NOTHING SINCE DONT HAVE MOTORS YET.. for now we just look at the pupil so we can manually move phase mask in and out. 
        _ = input('MANUALLY MOVE PHASE MASK BACK IN, PRESS ANY KEY TO BEGIN' )
        util.watch_camera(ZWFS, frames_to_watch = 50, time_between_frames=0.05)

        ZWFS.states['fpm'] = self.config['fpm']

        I0_list = []
        for _ in range(imgs_to_median):
            I0_list.append( ZWFS.get_image(  ) ) #REFERENCE INTENSITY WITH FPM IN
        I0 = np.median( I0_list, axis = 0 ) 
        
        # === ADD ATTRIBUTES 
        self.I0 = 1/np.mean( I0 ) * I0.reshape(-1)[np.array( ZWFS.pupil_pixels )] # append reference intensity over defined pupil with FPM IN 

        # === also add the unfiltered so we can plot and see them easily on square grid after 
        self.I0_2D = 1/np.mean( I0 ) * I0 # 2D array (not filtered by pupil pixel filter)  
  

    def control_phase(self, img, controller_name ):
        # look for active ctrl_parameters, return label
        im_err = self.get_img_err( img )
        cmd = im_err @ self.ctrl_parameters[controller_name]['CM'] 
       
        return( cmd )

























