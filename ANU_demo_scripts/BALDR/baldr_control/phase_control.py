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
            self.config['basis'] = 'Zernike' # either Zernike, Zonal, KL
            self.config['number_of_controlled_modes'] = 20 # number of controlled modes
            self.config['source'] = 1
           
            self.ctrl_parameters = {} # empty dictionary cause we have none
            #for tel in self.config['telescopes']:
            self.config['active_actuator_indicies'] = range(140) # all of them
            self.config['Kp'] = [0 for _ in range( self.config['number_of_controlled_modes'] )] # proportional gains
            self.config['Ki'] = [0 for _ in range( self.config['number_of_controlled_modes'] )] # integral gains

            self.config['dm_control_diameter'] = 8 # diameter (actuators) of active actuators
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



    def change_control_basis_parameters(self, number_of_controlled_modes, basis_name ,dm_control_diameter=None, dm_control_center=None):
        # standize updating of control basis parameters so no inconsistencies
        # warning no error checking here! 

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

    def build_control_model(self, ZWFS, poke_amp = -0.15, label='ctrl_1', debug = True):
        
        
        # remember that ZWFS.get_image automatically crops at corners ZWFS.pupil_crop_region
        ZWFS.states['busy'] = 1
        
        imgs_to_median = 10 # how many images do we take the median of to build signal the reference signals 
        # check other states match such as source etc

        # =========== PHASE MASK OUT 
        hardware.set_phasemask( phasemask = 'out' ) # THIS DOES NOTHING SINCE DONT HAVE MOTORS YET.. for now we just look at the pupil so we can manually move phase mask in and out. 
        _ = input('MANUALLY MOVE PHASE MASK OUT OF BEAM, PRESS ANY KEY TO BEGIN' )
        util.watch_camera(ZWFS, frames_to_watch = 10, time_between_frames=0.05)

        ZWFS.states['fpm'] = 0

        N0_list = []
        for _ in range(imgs_to_median):
            N0_list.append( ZWFS.get_image(  ).reshape(-1)[np.array( ZWFS.pupil_pixels )] ) #REFERENCE INTENSITY WITH FPM IN
        N0 = np.median( N0_list, axis = 0 ) 
        #put self.config['fpm'] phasemask on-axis (for now I only have manual adjustment)

        # =========== PHASE MASK IN 
        hardware.set_phasemask( phasemask = 'posX' ) # # THIS DOES NOTHING SINCE DONT HAVE MOTORS YET.. for now we just look at the pupil so we can manually move phase mask in and out. 
        _ = input('MANUALLY MOVE PHASE MASK BACK IN, PRESS ANY KEY TO BEGIN' )
        util.watch_camera(ZWFS, frames_to_watch = 10, time_between_frames=0.05)

        ZWFS.states['fpm'] = self.config['fpm']

        I0_list = []
        for _ in range(imgs_to_median):
            I0_list.append( ZWFS.get_image(  ).reshape(-1)[np.array( ZWFS.pupil_pixels )] ) #REFERENCE INTENSITY WITH FPM IN
        I0 = np.median( I0_list, axis = 0 ) 
        
        # === ADD ATTRIBUTES 
        self.I0 = I0 # append reference intensity over defined pupil with FPM IN 
        self.N0 = N0 # append reference intensity over defined pupil with FPM OUT 

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
                errsig =  np.array( ( (I - self.I0) / np.median( self.N0 ) ) )
            else: 
                raise TypeError(" reference intensity shapes do not match shape of current measured intensity. Check phase_controller.I0 and/or phase_controller.N0 attributes. Workaround would be to retake these. ")
            IM.append( list(errsig.reshape(-1)) )

  
               
        # SVD
        U,S,Vt = np.linalg.svd( IM , full_matrices=True)

        # filter number of modes in eigenmode space when using zonal control  
        if self.config['basis'] == 'Zonal': # then we filter number of modes in the eigenspace of IM 
            S_filt = S > S[ self.config['number_of_controlled_modes'] ] # we consider the highest eigenvalues/vectors up to the number_of_controlled_modes
            Sigma = np.zeros( np.array(IM).shape, float)
            np.fill_diagonal(Sigma, 1/poke_amp * S[S_filt], wrap=False) #
        else: # else #modes decided by the construction of modal basis. We may change their gains later
            S_filt = S > 0 # S > S[ np.min( np.where( abs(np.diff(S)) < 1e-2 )[0] ) ]
            Sigma = np.zeros( np.array(IM).shape, float)
            np.fill_diagonal(Sigma, 1/poke_amp * S[S_filt], wrap=False) #

        if debug:
            print('plot first 36 DM eigenmodes')

        # control matrix
        CM = np.linalg.pinv( U @ Sigma @ Vt ) # C = A @ M
       
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
            plt.title('Covariance of Interaciton Matrix')
            plt.imshow( np.cov( self.ctrl_parameters[label]['IM'] ) )
            plt.colorbar()
            plt.show()


    def update_FPM_OUT_reference() : 
        print('to do')


    def update_FPM_IN_reference() :
        print('to do')


    def control_phase(self, img, controller_name ):
        # look for active ctrl_parameters, return label
       
        cmd = img @ self.ctrl_parameters[controller_name]['CM'] 
       
        return( cmd )
        # get