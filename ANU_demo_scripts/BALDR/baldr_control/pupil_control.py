import time 
import numpy as np 
import matplotlib.pyplot as plt

figure_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/ANU_data/'

class pupil_controller_1():
    """
    measures and controls the ZWFS pupil
    """
   
    def __init__(self, pupil_crop_region=None, config_file = None):
       
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
            self.config['source'] = 1
            self.config['pupil_control_motor'] = 'XXX'
            if pupil_crop_region == None:
                self.pupil_crop_region = [None, None, None, None] # no cropping of the image
            elif hasattr(pupil_crop_region, '__len__'):
                if len( pupil_crop_region ) == 4:
                    if all(isinstance(x, int) for x in pupil_crop_region):
                        self.pupil_crop_region = pupil_crop_region
                    else:
                        self.pupil_crop_region = [None, None, None, None]
                        print('pupil_crop_region has INVALID entries, it needs to be integers')
                else:
                    self.pupil_crop_region = [None, None, None, None]
                    print('pupil_crop_region has INVALID length, it needs to have length = 4')
            else:
                self.pupil_crop_region = [None, None, None, None]
                print('pupil_crop_region has INVALID type. Needs to be list of integers of length = 4')

            self.ctrl_parameters = {} # empty dictionary cause we have none
            

    def measure_dm_center_offset(self, zwfs, debug=True): 
        
        zwfs.states['busy'] = 1

        #rows, columns to crop
        r1,r2,c1,c2 = self.pupil_crop_region

        # make sure phase mask is in 

        # check if camera is running

        amp = 0.1
        #push DM corner actuators & get image 
        zwfs.send_cmd(zwfs.dm_shapes['flat_dm'] + amp * zwfs.dm_shapes['four_torres'] ) 
        time.sleep(0.003)
        img_push = zwfs.get_image().astype(int) # get_image returns uint16 which cannot be negative
        #pull DM corner actuators & get image 
        zwfs.send_cmd(zwfs.dm_shapes['flat_dm'] - amp * zwfs.dm_shapes['four_torres'] ) 
        time.sleep(0.003)
        img_pull = zwfs.get_image().astype(int) # get_image returns uint16 which cannot be negative
        
        delta_img = abs( img_push - img_pull )[r1:r2,c1:c2]

        #define symetric coordinates in image (i.e. origin is at image center)  
        y = np.arange( -delta_img.shape[0]//2, delta_img.shape[0]//2) # y is row
        x = np.arange( -delta_img.shape[1]//2, delta_img.shape[1]//2) # x is col
        #x,y position weights
        w_x = np.sum( delta_img, axis = 0 ) #sum along columns 
        w_y = np.sum( delta_img, axis = 1 ) #sum along rows 
        #x,y errors
        e_x = np.mean( x * w_x ) / np.mean(w_x) 
        e_y = np.mean( y * w_y ) / np.mean(w_y) 

        if debug:
            plt.figure()
            plt.pcolormesh(y, x, delta_img ) 
            plt.xlabel('x pixels',fontsize=15)
            plt.ylabel('y pixels',fontsize=15)
            plt.gca().tick_params(labelsize=15)
            plt.arrow( 0, 0, e_x, e_y, color='w', head_width=1.2 ,width=0.3, label='error vector')
            plt.legend(fontsize=15) 
            plt.tight_layout()
            #plt.savefig(figure_path + 'process_1.2_center_source_DM.png',dpi=300) 
            plt.show()

        return( e_x, e_y )    


    def set_pupil_reference_pixels(self ):
        print('to do')

    def set_pupil_filter( self ):
        print('to do')  

        
