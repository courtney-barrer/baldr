import time 
import numpy as np 
import matplotlib.pyplot as plt
from . import utilities as util 
from . import hardware 

figure_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/ANU_data/'

class pupil_controller_1():
    """
    measures and controls the ZWFS pupil
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
            self.config['source'] = 1
            self.config['pupil_control_motor'] = 'XXX'
            
            self.ctrl_parameters = {} # empty dictionary cause we have none
            

    def measure_dm_center_offset(self, zwfs, debug=True): 
        
        zwfs.states['busy'] = 1

        #rows, columns to crop
        r1,r2,c1,c2 = zwfs.pupil_crop_region

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

        

def analyse_pupil_openloop( zwfs, debug = True, return_report = True):
    # want to get DM center, also P2C matrix (pixel to command registration matrix), could return teo - one with only one pixel per cmd, another with 9 pixels per command to look at surrounding region. pixel indicies where pupil defined, could check len(pixel indicies)
    
    

    #demodulation of a closer waffle, split image in quadrants, get x,y of four peaks, find intercept between them 

    #P2C_1 
    #P2C_2 

    report = {} # initialize empty dictionary for report

    #rows, columns to crop
    r1, r2, c1, c2 = zwfs.pupil_crop_region


    # check if camera is running

    #================================
    #============= Measure pupil center 
    
    # make sure phase mask is OUT !!!! 
    hardware.set_phasemask( phasemask = 'out' ) # no motors to implement this on yet, so does nothing 

    # simple idea: we are clearly going to have 2 modes in distribution of pixel intensity, one centered around the mean pupil intensity where illuminated, and another centered around the "out of pupil" region which will be detector noise / scattered light. np.histogram in default setup automatically calculates bins that incoorporate the range and resolution of data. Take the median frequency it returns (which is an intensity value) and set this as the pupil intensity threshold filter. This should be roughly between the two distributions.

    img = zwfs.get_image().astype(int)[r1: r2, c1: c2]
    density, intensity_edges  = np.histogram( img.reshape(-1) )
    intensity_threshold =  np.median( intensity_edges )

    pupil_filter = img.reshape(-1) > intensity_threshold
    pupil_pixels =  np.where( pupil_filter )[0]

    X, Y = np.meshgrid( zwfs.col_coords, zwfs.row_coords )
    x_pupil_center, y_pupil_center = np.mean(X.reshape(-1)[pupil_filter]), np.mean(Y.reshape(-1)[pupil_filter])

    #================================
    #============= Add to report card 
    report['pupil_pixel_filter'] = pupil_filter
    report['pupil_pixels'] = pupil_pixels
    report['pupil_center_ref_pixels'] = ( x_pupil_center, y_pupil_center ) 

    if debug: 
        fig,ax = plt.subplots(2,1,figsize=(5,10))
        ax[0].pcolormesh( zwfs.col_coords, zwfs.row_coords,   img) 
        ax[0].set_title('measured pupil')
        ax[1].pcolormesh( zwfs.col_coords, zwfs.row_coords,   pupil_filter.reshape(img.shape) ) 
        ax[1].set_title('derived pupil filter')
        for axx in ax.reshape(-1):
            axx.axvline(x_pupil_center,color='r',label='measured center')
            axx.axhline(y_pupil_center,color='r')
        plt.legend() 

    #================================
    #============= Measure DM center 

    # make sure phase mask is IN !!!! 
    hardware.set_phasemask( phasemask = 'posX' ) # no motors to implement this on yet, so does nothing 

    amp = 0.02
    delta_img_list = [] # hold our images, which we will take median of 
    for _ in range(10): # get median of 10 images 
        #push DM corner actuators & get image 
        zwfs.send_cmd(zwfs.dm_shapes['flat_dm'] + amp * zwfs.dm_shapes['four_torres_2'] ) 
        time.sleep(0.003)
        img_push = zwfs.get_image().astype(int) # get_image returns uint16 which cannot be negative
        #pull DM corner actuators & get image 
        zwfs.send_cmd(zwfs.dm_shapes['flat_dm'] - amp * zwfs.dm_shapes['four_torres_2'] ) 
        time.sleep(0.003)
        img_pull = zwfs.get_image().astype(int) # get_image returns uint16 which cannot be negative
        
        delta_img_list.append( abs( img_push - img_pull )[r1:r2,c1:c2] ) # DEFINE THIS in the crop region

    zwfs.send_cmd(zwfs.dm_shapes['flat_dm']) # flat DM 
    delta_img = np.median( delta_img_list, axis = 0 ) #get median of our modulation images 


    # define our cropped regions coordinates ( maybe this can be done with initiation of crop region attribute - move this to zwfs object since it is ALWAYS asscoiated with images taken from this object)
    y = zwfs.row_coords  #rows
    x = zwfs.col_coords  #columns

    #image quadrants coordinate indicies (row1, row2, col1, col2)
    q_11 = 0,  delta_img.shape[0]//2,  0,  delta_img.shape[1]//2
    q_12 = 0,  delta_img.shape[0]//2, delta_img.shape[1]//2,  None
    q_21 = delta_img.shape[0]//2, None, delta_img.shape[1]//2,  None
    q_22 = delta_img.shape[0]//2,  None,  0, delta_img.shape[1]//2


    ep = [] # to hold x,y of each quadrants peak pixel which will be our line end points
    for q in [q_11, q_12, q_21, q_22]: #[Q11, Q12, Q21, Q22]:
        xq = x[ q[2]:q[3] ] # cols
        yq = y[ q[0]:q[1] ] # rows
        d = delta_img[  q[0]:q[1], q[2]:q[3] ]
        x_peak = xq[ np.argmax( np.sum( d  , axis = 0) ) ] # sum along y (row) axis and find x where peak
        y_peak = yq[ np.argmax( np.sum( d  , axis = 1) ) ] # sum along x (col) axis and find y where peak
        #plt.figure()
        #plt.imshow( d )

        ep.append( (x_peak, y_peak) )  #(col, row)
    
    # define our line end points 
    line1 = (ep[0], ep[2]) # top left, bottom right 
    line2 = (ep[1], ep[3]) # top right, bottom left

    # find intersection to get centerpoint of DM in the defined cropped region coordinates 
    x_dm_center, y_dm_center = util.line_intersection(line1, line2)

    #print('CENTERS=', x_dm_center, y_dm_center) 

    if debug:
        plt.figure()
        plt.pcolormesh( x, y, delta_img)
        plt.colorbar(label='[adu]')
        xx1 = [ep[0] for ep in line1]
        yy1 = [ep[1] for ep in line1]
        xx2 = [ep[0] for ep in line2]
        yy2 = [ep[1] for ep in line2] 
        plt.plot( xx1, yy1 ,linestyle = '-',color='r',lw=3 ) 
        plt.plot( xx2, yy2 ,linestyle = '-',color='r',lw=3 )  
        plt.xlabel('x [pixels]',fontsize=15)
        plt.ylabel('y [pixels]',fontsize=15)
        plt.gca().tick_params(labelsize=15) 
        plt.tight_layout()
        #plt.savefig(figure_path + 'process_1.3_analyse_pupil_DM_center.png',dpi=300) 
        plt.show() 

    #================================
    #============= Add to report card 

    report['dm_center_ref_pixels'] = x_dm_center, y_dm_center


    #================================
    #============= Now some basic quality checks 

    if (np.sum( report['pupil_pixel_filter'] ) > 0) & (np.sum( report['pupil_pixel_filter'] ) < 1e20) : # TO DO put reasonable values here (how many pixels do we expect the pupil to cover? -> this will be mode dependent if we are in 12x12 or 6x6. 
        report['got_expected_illum_pixels'] = 1
    else:
        report['got_expected_illum_pixels'] = 0 # check for vignetting etc. 

    if abs(x_dm_center - x_pupil_center) < 5 : #pixels - TO DO put reasonable values here 
        report['dm_center_pix_x=pupil_center_x'] = 1
    else:
        report['dm_center_pix_x=pupil_center_x'] = 0

    if abs(y_dm_center - y_pupil_center) < 5 : #pixels - TO DO put reasonable values here 
        report['dm_center_pix_y=pupil_center_y'] = 1
    else:
        report['dm_center_pix_y=pupil_center_y'] = 0

    #etc we can do other quality control tests 

    report['pupil_quality_flag'] = report['got_expected_illum_pixels'] & report['dm_center_pix_x=pupil_center_x'] & report['dm_center_pix_y=pupil_center_y']
    
    return( report ) 


    #================================
    #============= Measure P2C <-- this should be in phase controller ..



