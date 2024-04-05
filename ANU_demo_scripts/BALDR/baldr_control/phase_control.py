from . import utilities as utils
import numpy as np

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
            self.config['number_of_controlled_modes'] = 70 # number of controlled modes
            self.config['source'] = 1
           
            self.ctrl_parameters = {} # empty dictionary cause we have none
            #for tel in self.config['telescopes']:
            self.config['active_actuator_indicies'] = range(140) # all of them
            self.config['Kp'] = [0 for _ in range( self.config['number_of_controlled_modes'] )] # proportional gains
            self.config['Ki'] = [0 for _ in range( self.config['number_of_controlled_modes'] )] # integral gains
            # TO DO <==============================
            self.config['pupil_pixels'] = [np.nan] #when building ?

            self.config['dm_control_diameter'] = 8 # diameter (actuators) of active actuators
            self.config['dm_control_center'] = [0,0] # in-case of mis-alignments we can offset control basis on DM
           
            # mode to command matrix
            M2C = utils.construct_command_basis( basis=self.config['basis'] , \
                    number_of_modes = self.config['number_of_controlled_modes'],\
                        Nx_act_DM = 12,\
                           Nx_act_basis = self.config['active_dm_diameter'],\
                               act_offset= self.config['dm_control_center'], without_piston=True) # cmd = M2C @ mode_vector, i.e. mode to command matrix
           
            self.config['M2C'] = M2C
                #self.config[tel]['control_parameters'] = [] # empty list cause we have none
        else:
            raise TypeError( 'config_file type is wrong. It must be None or a string indicating the config file to read in' )


    def build_control_model(self, ZWFS, poke_amp = -0.15, label='ctrl_1'):
       
        ZWFS.states['busy'] = 1
       
        # check other states match such as source etc

        move_fpm(self.states['telescopes'],  pos = 0)
        ZWFS.states['fpm'] = 0
        N0 = ZWFS.get_image( ZWFS.camera ) #REFERENCE INTENSITY WITH FPM OUT
       
        #put self.config['fpm'] phasemask on-axis (for now I only have manual adjustment)

        move_fpm( self.states['telescopes'], pos = self.config['fpm'])
        ZWFS.states['fpm'] = self.config['fpm']
        I0 = ZWFS.get_image( ZWFS.camera ) #REFERENCE INTENSITY WITH FPM IN
       
        image_pupil_filter = self.config['pupil_pixels'].copy()
        modal_basis = self.config['M2C'].copy().T # more readable
        IM=[]

        for i,m in enumerate(modal_basis):

            print(f'executing cmd {i}/{len(modal_basis)}')
            ZWFS.dm.send_data( flat_dm_cmd + poke_amp * m )
            time.sleep(0.005)
            I = ZWFS.get_image( ZWFS.camera )
            errsig =  np.array( ( (I - I0) / N0 ) )[np.array( self.config['pupil_pixels'] )]

            IM.append( list(errsig.reshape(-1)) )
               
        # SVD
        U,S,Vt = np.linalg.svd( IM , full_matrices=True)

        # filter modes (for now no filtering)
        S_filt = S > 0 # S > S[ np.min( np.where( abs(np.diff(S)) < 1e-2 )[0] ) ]
        Sigma = np.zeros( np.array(IM).shape, float)
        np.fill_diagonal(Sigma, 1/poke_amp * S[S_filt], wrap=False) #

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
       
       
    def update_FPM_OUT_reference() : 
        print('to do')


    def update_FPM_IN_reference() :
        print('to do')


    def control_phase(self, img):
        # look for active ctrl_parameters, return label
       
        cmd = self.ctrl_parameters[label]['CM'] @ img[self.config['pupil_pixels']]
       
        return( cmd )
        # get
