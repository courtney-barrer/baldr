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
            self.config['active_actuator_indicies'] = range(140) # all of them
            self.config['Kp'] = [0 for _ in range( self.config['number_of_controlled_modes'] )] # proportional gains
            self.config['Ki'] = [0 for _ in range( self.config['number_of_controlled_modes'] )] # integral gains

            self.config['dm_control_diameter'] = 12 # diameter (actuators) of active actuators
            self.config['dm_control_center'] = [0,0] # in-case of mis-alignments we can offset control basis on DM
            self.I0 = None #reference intensity over defined pupil with FPM IN 
            self.N0 = None #reference intensity over defined pupil with FPM OUT 

            # mode to command matrix
            M2C = util.construct_command_basis( basis=self.config['basis'] , \
                    number_of_modes = self.config['number_of_controlled_modes'],\
                        Nx_act_DM = 12,\
                           Nx_act_basis = self.config['dm_control_diameter'],\
                               act_offset= self.config['dm_control_center'], without_piston=True) # cmd = M2C @ mode_vector, i.e. mode to command matrix
           
            self.config['M2C'] = M2C
                #self.config[tel]['control_parameters'] = [] # empty list cause we have none
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





    def build_control_model(self, ZWFS, poke_amp = -0.15, label='ctrl_1', debug = True):
        
        
        # remember that ZWFS.get_image automatically crops at corners ZWFS.pupil_crop_region
        ZWFS.states['busy'] = 1
        
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
        self.I0 = I0.reshape(-1)[np.array( ZWFS.pupil_pixels )] # append reference intensity over defined pupil with FPM IN 
        self.N0 = N0.reshape(-1)[np.array( ZWFS.pupil_pixels )] # append reference intensity over defined pupil with FPM OUT 

        # === also add the unfiltered so we can plot and see them easily on square grid after 
        self.I0_2D = I0 # 2D array (not filtered by pupil pixel filter)  
        self.N0_2D = N0 # 2D array (not filtered by pupil pixel filter)  

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
                errsig =  self.get_img_err( I ) #np.array( ( (I - self.I0) / np.median( self.N0 ) ) )
            else: 
                raise TypeError(" reference intensity shapes do not match shape of current measured intensity. Check phase_controller.I0 and/or phase_controller.N0 attributes. Workaround would be to retake these. ")
            IM.append( list(errsig.reshape(-1)) )

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
        CM = 1/abs(poke_amp) * np.linalg.pinv( U @ Sigma @ Vt ) # C = A @ M
       
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



    def get_img_err( self , img ):
        # input img has to be flattened and filtered in pupil region 
        #(i.e. get raw img then apply img.reshape(-1)[ ZWFS.pupil_pixels])

        # !NOTE! we take median of pupil reference intensity with FPM out (self.N0)
        # we do this cause we're lazy and do not want to manually adjust FPM every iteration (we dont have motors here) 
        # real system prob does not want to do this and normalize pixel wise. 
        errsig =  np.array( ( (img - self.I0) / np.median( self.N0 ) ) )
        return(  errsig )

    def update_FPM_OUT_reference() : 
        print('to do')


    def update_FPM_IN_reference() :
        print('to do')


    def control_phase(self, img, controller_name ):
        # look for active ctrl_parameters, return label
       
        cmd = img @ self.ctrl_parameters[controller_name]['CM'] 
       
        return( cmd )
        # get
























