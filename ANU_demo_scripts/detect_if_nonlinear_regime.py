import numpy as np
import pandas as pd 
import os 
import time 
import matplotlib.pyplot as plt 
root_path = '/home/baldr/Documents/baldr'
data_path = root_path + '/ANU_demo_scripts/ANU_data/'
fig_path = root_path + '/figures/'

os.chdir(root_path)
from functions import baldr_demo_functions as bdf


def apply_dm_cmd_and_get_im(dm, camera, cmd, number_of_frames = 5, cropping_corners=None, subregion_corners=None): 

    if subregion_corners!=None: 
        cp_x1, cp_x2, cp_y1, cp_y2 = subregion_corners

    dm.send_data( cmd ) 
    time.sleep(0.005) # wait a second
    #record image
    im = np.median( bdf.get_raw_images(camera, number_of_frames=number_of_frames,  cropping_corners=cropping_corners) , axis=0)
    if subregion_corners!=None: 
        S = im[cp_x1:cp_x2, cp_y1:cp_y2] # signal
    else:
        S = im # signal
 
    return(S)

def get_ref_n_poke_images( dm, camera, basis, flat_dm_cmd, mode_amp = -0.03, cropping_corners=None, subregion_corners=None):

    # input flat (P array) and poked images (P x Nact matrix) 
    mode_amp = -0.03
    im_poke_matrix = []
    for i,cmd in enumerate(basis):
        print(f'executing mode {i}/{len(basis)}')
        # relative command 
        delta_c = mode_amp * cmd
        # absolute command 
        cmd = flat_dm_cmd + delta_c
        # get image after applying command 
        im = apply_dm_cmd_and_get_im(dm, camera, cmd, number_of_frames = 5, cropping_corners=cropping_corners, subregion_corners=subregion_corners)
        # append our image
        im_poke_matrix.append( im )

    # get reference image when applying flat DM command
    im_ref = apply_dm_cmd_and_get_im(dm, camera, flat_dm_cmd, number_of_frames = 5, cropping_corners=cropping_corners, subregion_corners=subregion_corners)

    dm.send_data( flat_dm_cmd )

    return( im_ref, im_poke_matrix ) 


def get_DM_influence_region( im_ref, im_poke_matrix , debug=True):
    if debug:
        fig,ax= plt.subplots( 4, 4, figsize=(10,10))
        num_pixels = []
        candidate_thresholds = np.linspace(0, 0.5, 16)
        for axx, thresh in zip(ax.reshape(-1),candidate_thresholds):
            dm_pupil_filt = thresh < np.array( [np.max( abs( (poke_im - im_ref)/im_ref ) ) for poke_im in im_poke_matrix] ) 

            axx.imshow( bdf.get_DM_command_in_2D( dm_pupil_filt ) ) 
            axx.set_title('threshold = {}'.format(thresh ),fontsize=12) 
            axx.axis('off')
            num_pixels.append(sum(dm_pupil_filt)) 
    # we could use this to automate threshold decision.. look for where 
    # d num_pixels/ d threshold ~ 0.. np.argmin( abs( np.diff( num_pixels ) )[:10])
        plt.show()
    

    recommended_threshold = candidate_thresholds[np.argmin( abs( np.diff( num_pixels ) )[:11]) + 1 ]
    print( f'\n\nrecommended threshold ~ {recommended_threshold} \n(check this makes sense with the graph by checking the colored area is stable around changes in threshold about this value)\n\n')

    pupil_filt_threshold = float(input('input threshold of peak differences'))

    dm_cmd_space_filter = pupil_filt_threshold <  np.array( [np.max( abs( (poke_im - im_ref)/im_ref ) ) for poke_im in im_poke_matrix] ) 

    return( dm_cmd_space_filter )



def get_P2C(dm, camera, dm_cmd_space_filter,  im_ref, im_poke_matrix , subwindow_pixels=3, debug = True):

    Sw_x, Sw_y = subwindow_pixels,subwindow_pixels #+- pixels taken around region of peak influence. PICK ODD NUMBERS SO WELL CENTERED!   
    act_img_mask = {} # dictionary to hold 1 at pixel indicies where actuator cmd exerts peak influence, 0 otherwise. Dictionary will be indexed by actuator number

    for act_idx in range(len(im_poke_matrix)):
        delta = im_poke_matrix[act_idx] - im_ref

        mask = np.zeros( im_ref.shape )
   
        if dm_cmd_space_filter[act_idx]:
            i,j = np.unravel_index( np.argmax( abs(delta) ),im_ref.shape )

        #act2pix_idx.append( (i,j) ) 
            mask[i-Sw_x-1: i+Sw_x, j-Sw_y-1:j+Sw_y] = 1 # keep centered, normalize by #pixels in window 
            mask *= 1/np.sum(mask[i-Sw_x-1: i+Sw_x, j-Sw_y-1:j+Sw_y])
            act_img_mask[act_idx] = mask 

        else :
            act_img_mask[act_idx] = mask # zeros, effectively filter these out

    if debug:
        plt.title('masked regions of influence per actuator')
        plt.imshow( np.sum( list(act_img_mask.values()), axis = 0 ) )
        plt.show()

# turn our dictionary to a big camera pixel to DM command (P2C) registration matrix 
    P2C = np.array([list(act_img_mask[act_idx].reshape(-1)) for act_idx in act_img_mask])

    return(P2C) 
 

def detect_if_nonlinear_regime( current_dm_cmd, dither_cmd, P2C , cropping_corners=None, subregion_corners=None):

    # construct DM +/- dither cmds
    cmd_plus = current_dm_cmd + dither_cmd
    cmd_minus = current_dm_cmd - dither_cmd

    # take +/- images
    S_plus = apply_dm_cmd_and_get_im(dm, camera, cmd_plus, number_of_frames=5, cropping_corners=cropping_corners, subregion_corners=subregion_corners)
    
    S_minus = apply_dm_cmd_and_get_im(dm, camera, cmd_minus, number_of_frames=5, cropping_corners=cropping_corners, subregion_corners=subregion_corners)

    # amplitude difference in the measured setpoints 
    delta_S = np.array(S_plus) - S_minus 

    # if the  response gradient in the (P2C aggregated) images was positive or negative from the DM push/pull commands 
    sg = np.sign( np.array( dither_cmd ) * ( P2C @ delta_S.reshape(-1) ) )

    # create our flags    
    nonlinear_flags = sg > 0

    return( nonlinear_flags ) 


# have to filter cmd space to regions that exert influence on pixel space

# create P2C matrix that aggregates around pixels of influence and maps pixel space to command space, effectively filtering out actuators that don't exert influence 

# detect non-linear regions in command space  




if __name__ == "__main__":

    # --- load pre-defined commands 
    
    # flat DM 
    flat_dm_cmd = pd.read_csv(bdf.path_to_dm_flat_map, header=None)[0].values 

    # DM gain
    cmd2nm = 3500  #nm/DM cmd 
    # checkers pattern on DM 
    waffle_dm_cmd = pd.read_csv(root_path + '/DMShapes/dm_checker_pattern.csv', index_col=[0]).values.ravel() 

    # --- setup camera
    fps = 600
    camera = bdf.setup_camera(cameraIndex=0) #connect camera and init camera object
    camera = bdf.set_fsp_dit( camera, fps=fps, tint=None) # set up initial frame rate, tint=None means max integration time for given FPS

    # --- setup DM
    dm, dm_err_code =  bdf.set_up_DM(DM_serial_number='17DW019#053')

    # where to crop full image (row min, row max, col min, col max)
    cropping_corners = [140, 280, 90, 290]

    # pupil cropping coordinates (cropped image has a pupil and PSF in it) 
    pupil_corners = [21, 129, 11, 119]

    # PSF cropping coordinates (cropped image has a pupil and PSF in it)  
    psf_corners = [67, 139, 124, 196] 

    ## ====== Let us get to business 
    
    # reference and interaction images
    im_ref, im_poke_matrix = get_ref_n_poke_images( dm, camera, basis=np.eye(140), flat_dm_cmd = flat_dm_cmd, mode_amp = -0.03, cropping_corners=cropping_corners, subregion_corners = pupil_corners )

    # make command space filter (consider where actuator space that can influence pupil images)
    dm_cmd_space_filter = get_DM_influence_region( im_ref, im_poke_matrix , debug=True)

    # create our pixel to command registration matrix 
    P2C = get_P2C(dm, camera, dm_cmd_space_filter,  im_ref, im_poke_matrix , subwindow_pixels=3, debug = True)
    """ 
    #SINGLE ITERATION FOR EACH ACTUATOR 
    #to detect which ones are non-linear with random initialization of position and waffle amplitde
    
    # create a disturbance DM command 
    delta_c = 0.05*np.random.randn(140) #np.zeros(140) #no aberration 
    #delta_c[62] = 0.1 # add aberration to one actuator
    current_dm_cmd = delta_c + flat_dm_cmd  # create absolute command
     
    # define our +/- dither commands 
    dither_cmd = 0.02 * waffle_dm_cmd 
    

    #estimate our non-linear flags
    nonlinear_flags = detect_if_nonlinear_regime( current_dm_cmd, dither_cmd, P2C , cropping_corners=cropping_corners, subregion_corners=pupil_corners)

    #get the image 
    pupil_image = apply_dm_cmd_and_get_im(dm, camera, current_dm_cmd, number_of_frames = 5, cropping_corners=cropping_corners, subregion_corners=pupil_corners)
   
    # 
    pupil_registration = P2C @ pupil_image.reshape(-1)
    """
    
   
    pupil_registration_list = []
    delta_c_list = []
    nonlinear_flag_list = []
    delta_c = np.zeros(140) # init our delta_c to zeros 
    act_idx = 65
    for it in range(300):
        delta_c[act_idx] = 0.07*np.random.randn()
     
        current_dm_cmd = delta_c + flat_dm_cmd  # create absolute command

        dither_cmd = 0.02 * waffle_dm_cmd 

        nonlinear_flags = detect_if_nonlinear_regime( current_dm_cmd, dither_cmd, P2C , cropping_corners=cropping_corners, subregion_corners=pupil_corners)

        pupil_image = apply_dm_cmd_and_get_im(dm, camera, current_dm_cmd, number_of_frames = 5, cropping_corners=cropping_corners, subregion_corners=pupil_corners)
   
        # 
        pupil_registration = P2C @ pupil_image.reshape(-1)
        
        pupil_registration_list.append( pupil_registration[act_idx] )
        nonlinear_flag_list.append( nonlinear_flags[act_idx] )
        delta_c_list.append( delta_c[act_idx] )
    
    pupil_registration_f = np.array(pupil_registration_list).ravel()

    #normalize between 0-1
    pupil_registration_f = (np.array(pupil_registration_list).ravel() - np.min( np.array(pupil_registration_list).ravel() ) ) / ( np.max( np.array(pupil_registration_list).ravel() ) -np.min( np.array(pupil_registration_list).ravel() ) )

    nonlinear_flags_f = np.array(nonlinear_flag_list).ravel()
    delta_c_f = np.array(delta_c_list).ravel()

    

if 1: 
    # ============= PLOTTING 
    zero_point_intensity=0.15
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)
    # Create the Axes.
    #ax0 = fig.add_subplot(gs[0, 1])

    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)


    #P = 100 * 0.8*np.random.randn(1000)
    #v = 1+1*np.cos(P/100 +2.2)

    #filt = P > 100
    filt_l =  ((nonlinear_flags_f>0) & (pupil_registration_f < zero_point_intensity)) |  (( pupil_registration_f > 0. ) & (nonlinear_flags_f<1)) # linear filt
    filt_nl = ( pupil_registration_f >= zero_point_intensity ) & (nonlinear_flags_f>0) # non-linear filt
    #filt_l = ( pupil_registration > 0 ) & (nonlinear_flags<1) #linear filt 
    ylim = [-0.5 , 1]
    xlim = [cmd2nm*1.2*np.min(delta_c_f), cmd2nm*1.2*np.max(delta_c_f)]
    color='Grey'

    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    sc=ax.plot(cmd2nm*delta_c_f[filt_nl],pupil_registration_f[filt_nl],'.',alpha=0.6,color='orange',label='phase wrap detection')
    sc=ax.plot(cmd2nm*delta_c_f[filt_l],pupil_registration_f[filt_l],'.',alpha=0.6,color=color)

    xbins = np.linspace(xlim[0],  xlim[1], 30)
    ybins = np.linspace(ylim[0] , ylim[1], 30)    

    ax_histx.hist(cmd2nm*delta_c_f[filt_l], bins=xbins, color=color)
    ax_histx.hist(cmd2nm*delta_c_f[filt_nl], bins=xbins, color='orange')

    ax_histy.hist(pupil_registration_f[filt_l], bins=ybins, color=color, orientation='horizontal')
    ax_histy.hist(pupil_registration_f[filt_nl], bins=ybins, color='orange', orientation='horizontal')

    ax_histx.set_yticks([])
    ax_histy.set_xticks([])
    #ax.set_ylim(ylim)
    #ax.set_xlim(xlim)

    ax.axvline(0,color='k',linestyle='--',label='zero aberration')
    ax.axhline(zero_point_intensity,color='k',linestyle=':',label='zero point intensity')
    ax_histx.axvline(0,color='k',linestyle=':')

    ax.set_xlabel('OPD [nm]',fontsize=25)
    ax.set_ylabel('ZWFS Pixel Intensity\n(normalized)' , fontsize=25)
    ax.tick_params(labelsize=25)
    ax.legend(fontsize=18)


    #ax0.imshow( bdf.get_DM_command_in_2D(delta_c) ,cmap='Greys')
    #ax0.set_xticks(np.arange(-.5, 10, 1), minor=True)
    #ax0.set_yticks(np.arange(-.5, 10, 1), minor=True)
    #ax0.grid(which='minor', color='k', linestyle='-', linewidth=1)
    #ax0.tick_params(which='minor', bottom=False, left=False)
    #ax0.axis('off')
    #ax0.set_title('DM',fontsize=18)
    #ax0.set_xticks([])
    #ax0.set_yticks([])


    ax_histx.tick_params(labelsize=25)
    ax_histy.tick_params(labelsize=25)

    #plt.tight_layout()
    fig.savefig( data_path + 'non-linearity_estimator.png',dpi=300 , bbox_inches="tight") 




    """
    ## ====== This is just to get cropping coordinates 
    available_recon_pupil_files = glob.glob( data_path+'BDR_RECON_*.fits' )
    available_recon_pupil_files.sort(key=os.path.getctime) # sort by most recent 
    most_recent_file = available_recon_pupil_files[-1]

    recon_data = fits.open( most_recent_file )
    
    #cp_x1,cp_x2,cp_y1,cp_y2 =  int(recon_data[0].header['cp_x1']),int(recon_data[0].header['cp_x2']),int(recon_data[0].header['cp_y1']),int(recon_data[0].header['cp_y2'])

    """

