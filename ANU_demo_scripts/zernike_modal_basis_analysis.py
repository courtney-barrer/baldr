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


"""
Here we experiment with creating flexible zernike basis on subspace of DM 
creating interaction matrix with this modal basis and reconstructing random phase aberrations
compare results with/without pupil filtering (using so-called P2C (pixel to cmd) matrix) 


"""

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


if __name__ == "__main__"
    # flat DM 
    flat_dm_cmd = pd.read_csv(bdf.path_to_dm_flat_map, header=None)[0].values 

    # DM gain
    cmd2nm = 3500  #nm/DM cmd 

    # checkers pattern on DM 
    waffle_dm_cmd = pd.read_csv(root_path + '/DMShapes/dm_checker_pattern.csv', index_col=[0]).values.ravel() 

if 1:
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

    # create our pixel to command registration matrix (essentially a pupil filter)  
    P2C = get_P2C(dm, camera, dm_cmd_space_filter,  im_ref, im_poke_matrix , subwindow_pixels=3, debug = True)
    # find center of valid command region (where we see significant influence from actuation) 
    x = 6.5 - np.linspace(1,12,12)  # since even number of actuators center lies between 2 actuators
    
    cmd_basis_offset = ( round(np.mean( x[ np.nansum( bdf.get_DM_command_in_2D(dm_cmd_space_filter), axis=0) > 0 ] ) ), round( np.mean( x[ np.nansum( bdf.get_DM_command_in_2D(dm_cmd_space_filter), axis=1) > 0 ] ) ) )

    # look how wide (average x-y) cmd space filter to define number of actuators across our basis 
    Nx_act_basis = 2*  round( np.mean( [np.max( x[ np.nansum( bdf.get_DM_command_in_2D(dm_cmd_space_filter),
     ...: axis=0) > 0 ] ), np.max( x[ np.nansum( bdf.get_DM_command_in_2D(dm_cmd_space_filter),
     ...: axis=1) > 0 ] ) ] ) )

if 1:
    # make modal basis 
    M2C = construct_command_basis( basis='Zernike', number_of_modes = 20, Nx_act_DM = 12, Nx_act_basis = Nx_act_basis, act_offset=cmd_basis_offset)

    M2C_nopiston = M2C[:,1:]

    # reference and interaction images
    mode_amp = -0.2
    im_ref, im_modal_matrix = get_ref_n_poke_images( dm, camera, basis = M2C_nopiston.T, flat_dm_cmd = flat_dm_cmd, mode_amp = mode_amp, cropping_corners=cropping_corners, subregion_corners = pupil_corners )

    # plot 
    plt.figure()
    plt.imshow( im_modal_matrix[5] - im_ref ); plt.show()
   

    IM = (im_modal_matrix - im_ref).reshape(len(im_modal_matrix), im_ref.shape[0]*im_ref.shape[1])
    # also look at filtering pupil through P2C 
    IM_p2c = np.array( [P2C @ (m - im_ref ).reshape(-1) for m in im_modal_matrix] ) 
    print( 'IM condition = ' ,np.linalg.cond( IM ) )
    U,S,Vt = np.linalg.svd( IM  ,full_matrices=True)
    U_p2c,S_p2c,Vt_p2c = np.linalg.svd( IM_p2c  ,full_matrices=True)
    # plotting the eigenmodes in DM command space 
    plt.figure() 
    plt.imshow( bdf.get_DM_command_in_2D( (M2C_nopiston @ U)[:,15] ) ); plt.show()

    #control matrix CM 
    CM = np.linalg.pinv(IM) 
    CM_p2c = np.linalg.pinv(IM_p2c) 

    # ok now apply an aberration in our zernike modal basis and see if we can reconstruct in with CM
    modes2poke = np.array([1,3,6,7]) 
    mode_aberration = np.zeros( M2C_nopiston.shape[1] ) 
    for i in modes2poke:
        mode_aberration[i] = np.random.randn()*0.05
    
    dm.send_data( flat_dm_cmd )
    time.sleep(0.5)
    im = apply_dm_cmd_and_get_im(dm, camera, cmd=flat_dm_cmd + M2C_nopiston @ mode_aberration , number_of_frames = 5, cropping_corners=cropping_corners, subregion_corners=pupil_corners)

    errsig =  im - im_ref
    errsig_p2c = P2C @ (im - im_ref).reshape(-1) 

    plt.figure() 
    plt.plot( mode_amp * errsig.reshape(-1) @ (CM), label='reconstructed mode amplitudes')
    plt.plot( mode_amp * errsig_p2c.reshape(-1) @ (CM_p2c), label='reconstructed mode amplitudes P2C')
    plt.plot( mode_aberration, label='applied mode amplitudes')
    plt.legend()
    plt.xlabel('mode index')
    plt.ylabel('mode amplitude') 

    recon_cmd = mode_amp * errsig.reshape(-1) @ (CM)
    fig,ax = plt.subplots(1,2)
    im0=ax[0].imshow( bdf.get_DM_command_in_2D( M2C_nopiston @ mode_aberration ) )
    ax[0].set_title(r'original command')
    im1=ax[1].imshow( bdf.get_DM_command_in_2D( M2C_nopiston @ recon_cmd ) ) 
    ax[1].set_title(r'reconstructed command')
    plt.colorbar(im0,ax=ax[0])
    plt.colorbar(im1,ax=ax[1])
    plt.show()

