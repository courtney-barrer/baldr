
import numpy as np 
import matplotlib.pyplot as plt 
import pyzelda.utils.zernike as zernike
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy import ndimage

# ============== UTILITY FUNCTIONS
def construct_command_basis( basis='Zernike', number_of_modes = 20, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True):
    """
    returns a change of basis matrix M2C to go from modes to DM commands, where columns are the DM command for a given modal basis. e.g. M2C @ [0,1,0,...] would return the DM command for tip on a Zernike basis. Modes are normalized on command space such that <M>=0, <M|M>=1. Therefore these should be added to a flat DM reference if being applied.    

    basis = string of basis to use
    number_of_modes = int, number of modes to create
    Nx_act_DM = int, number of actuators across DM diameter
    Nx_act_basis = int, number of actuators across the active basis diameter
    act_offset = tuple, (actuator row offset, actuator column offset) to offset the basis on DM (i.e. we can have a non-centered basis)
     
    """

   
    # shorter notations
    #Nx_act = DM.num_actuators_width() # number of actuators across diameter of DM.
    #Nx_act_basis = actuators_across_diam
    c = act_offset
    # DM BMC-3.5 is 12x12 missing corners so 140 actuators , we note down corner indicies of flattened 12x12 array.
    corner_indices = [0, Nx_act_DM-1, Nx_act_DM * (Nx_act_DM-1), -1]

    bmcdm_basis_list = []
    # to deal with
    if basis == 'Zernike':
        if without_piston:
            number_of_modes += 1 # we add one more mode since we dont include piston 

        raw_basis = zernike.zernike_basis(nterms=number_of_modes, npix=Nx_act_basis )
        for i,B in enumerate(raw_basis):
            # normalize <B|B>=1, <B>=0 (so it is an offset from flat DM shape)
            Bnorm = np.sqrt( 1/np.nansum( B**2 ) ) * B
            # pad with zeros to fit DM square shape and shift pixels as required to center
            if np.mod( Nx_act_basis, 2) == 0:
                pad_width = (Nx_act_DM - B.shape[0] )//2
                padded_B = shift( np.pad( Bnorm , pad_width , constant_values=(np.nan,)) , c[0], c[1])
            else:
                pad_width = (Nx_act_DM - B.shape[0] )//2 + 1
                padded_B = shift( np.pad( Bnorm , pad_width , constant_values=(np.nan,)) , c[0], c[1])[:-1,:-1]  # we take off end due to odd numebr

            flat_B = padded_B.reshape(-1) # flatten basis so we can put it in the accepted DM command format
            np.nan_to_num(flat_B,0 ) # convert nan -> 0
            flat_B[corner_indices] = np.nan # convert DM corners to nan (so lenght flat_B = 140 which corresponds to BMC-3.5 DM)

            # now append our basis function removing corners (nan values)
            bmcdm_basis_list.append( flat_B[np.isfinite(flat_B)] )

        # our mode 2 command matrix
        if without_piston:
            M2C = np.array( bmcdm_basis_list )[1:].T #remove piston mode
        else:
            M2C = np.array( bmcdm_basis_list ).T # take transpose to make columns the modes in command space.
    elif basis == 'Zonal': 
        M2C = np.eye(Nx_act_DM) # we just consider this over all actuators (so defaults to 140 modes) 
        # we filter zonal basis in the eigenvectors of the control matrix. 
 
    return(M2C)



def get_DM_command_in_2D(cmd,Nx_act=12):
    # function so we can easily plot the DM shape (since DM grid is not perfectly square raw cmds can not be plotted in 2D immediately )
    #puts nan values in cmd positions that don't correspond to actuator on a square grid until cmd length is square number (12x12 for BMC multi-2.5 DM) so can be reshaped to 2D array to see what the command looks like on the DM.
    corner_indices = [0, Nx_act-1, Nx_act * (Nx_act-1), Nx_act*Nx_act]
    cmd_in_2D = list(cmd.copy())
    for i in corner_indices:
        cmd_in_2D.insert(i,np.nan)
    return( np.array(cmd_in_2D).reshape(Nx_act,Nx_act) )


def shift(xs, n, m, fill_value=np.nan):
    # shifts a 2D array xs by n rows, m columns and fills the new region with fill_value

    e = xs.copy()
    if n!=0:
        if n >= 0:
            e[:n,:] = fill_value
            e[n:,:] =  e[:-n,:]
        else:
            e[n:,:] = fill_value
            e[:n,:] =  e[-n:,:]
   
       
    if m!=0:
        if m >= 0:
            e[:,:m] = fill_value
            e[:,m:] =  e[:,:-m]
        else:
            e[:,m:] = fill_value
            e[:,:m] =  e[:,-m:]
    return e


def line_intersection(line1, line2):
    """
    find intersection of lines given by their endpoints, 
       line1 = (A,B)
       line2 = (C,D)
       where A=[x1_1, y1_1], B=[x1_2,y1_2], are end points of line1 
             C=[x2_1, y2_1], D=[x2_2, y2_2], are end points of line2
        
    """
 
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return( x, y )


def move_fpm( tel, pos = 0):
    # in real life this will command a motor - but for now i do this manually
    """
    # SETS FOCAL PLANE MASK - TO IMPLIMENT ONCE WE HAVE FPM MOTORS
    if pos == 0:    
        print( 'move FPM to position X')
    elif pos == 1:  
        print( 'move FPM to position Y') #etc
    """
    return(None)


def watch_camera(zwfs, frames_to_watch = 10, time_between_frames=0.01,cropping_corners=None) :
  
    print( f'{frames_to_watch} frames to watch with ~{time_between_frames}s wait between frames = ~{5*time_between_frames*frames_to_watch}s watch time' )

    #t0= datetime.datetime.now() 
    plt.figure(figsize=(15,15))
    plt.ion() # turn on interactive mode 
    #FliSdk_V2.Start(camera)     
    seconds_passed = 0
    if type(cropping_corners)==list: 
        x1,x2,y1,y2 = cropping_corners #[row min, row max, col min, col max]

    for i in range(int(frames_to_watch)): 
        
        a = zwfs.get_image()
        if type(cropping_corners)==list: 
            plt.imshow(a[x1:x2,y1:y2])
        else: 
            plt.imshow(a)
        plt.pause( time_between_frames )
        #time.sleep( time_between_frames )
        plt.clf() 
    """
    while seconds_passed < seconds_to_watch:
        a=FliSdk_V2.GetRawImageAsNumpyArray(camera,-1)
        plt.imshow(a)
        plt.pause( time_between_frames )
        time.sleep( time_between_frames )
        plt.clf() 
        t1 = datetime.datetime.now() 
        seconds_passed = (t1 - t0).seconds"""

    #FliSdk_V2.Stop(camera) 
    plt.ioff()# turn off interactive mode 
    plt.close()







def create_phase_screen_cmd_for_DM(scrn, flat_reference, scaling_factor=0.1, drop_indicies = None, plot_cmd=False):
    """
    aggregate a scrn (aotools.infinitephasescreen object) onto a DM command space. phase screen is normalized by
    between +-0.5 and then scaled by scaling_factor and offset by flat_reference command. Final DM command values should
    always be between 0-1. phase screens are usually a NxN matrix, while DM is MxM with some missing pixels (e.g. 
    corners). drop_indicies is a list of indicies in the flat MxM DM array that should not be included in the command space. 
    """

    #print('----------\ncheck phase screen size is multiple of DM\n--------')
    
    Nx_act = 12 #number of actuators across DM diameter
    
    scrn_array = ( scrn.scrn - np.min(scrn.scrn) ) / (np.max(scrn.scrn) - np.min(scrn.scrn)) - 0.5 # normalize phase screen between -0.5 - 0.5 
    
    size_factor = int(scrn_array.shape[0] / Nx_act) # how much bigger phase screen is to DM shape in x axis. Note this should be an integer!!
    
    # reshape screen so that axis 1,3 correspond to values that should be aggregated 
    scrn_to_aggregate = scrn_array.reshape(scrn_array.shape[0]//size_factor, size_factor, scrn_array.shape[1]//size_factor, size_factor)
    
    # now aggreagate and apply the scaling factor 
    scrn_on_DM = scaling_factor * np.mean( scrn_to_aggregate, axis=(1,3) ).reshape(-1) 

    #If DM is missing corners etc we set these to nan and drop them before sending the DM command vector
    #dm_cmd =  scrn_on_DM.to_list()
    if drop_indicies != None:
        for i in drop_indicies:
            scrn_on_DM[i]=np.nan
             
    if plot_cmd: #can be used as a check that the command looks right!
        fig,ax = plt.subplots(1,2,figsize=(12,6))
        im0 = ax[0].imshow(np.mean(flat_reference) + scrn_on_DM.reshape([Nx_act,Nx_act]) )
        ax[0].set_title('DM command (averaging offset)')
        im1 = ax[1].imshow(scrn.scrn)
        ax[1].set_title('original phase screen')
        plt.colorbar(im0, ax=ax[0])
        plt.colorbar(im1, ax=ax[1]) 
        plt.show() 

    dm_cmd =  list( scrn_on_DM[np.isfinite(scrn_on_DM)] ) #drop non-finite values which should be nan values created from drop_indicies array
    return(dm_cmd) 






def block_sum(ar, fact): # sums over subwindows of a 2D array
    # ar is the 2D array, fact is the factor to reduce it by 
    # obviously  fact should be factor of ar.shape 
    assert isinstance(fact, int), type(fact)
    sx, sy = ar.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    regions = sy//fact * (X//fact) + Y//fact
    res = ndimage.sum(ar, labels=regions, index=np.arange(regions.max() + 1))
    res.shape = (sx//fact, sy//fact)
    return res




def nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list,cbar_label_list, fontsize=15, cbar_orientation = 'bottom', axis_off=True, savefig=None):

    n = len(im_list)
    fs = fontsize
    fig = plt.figure(figsize=(5*n, 5))

    for a in range(n) :
        ax1 = fig.add_subplot(int(f'1{n}{a+1}'))
        ax1.set_title(title_list[a] ,fontsize=fs)

        im1 = ax1.imshow(  im_list[a] )
        ax1.set_title( title_list[a] ,fontsize=fs)
        ax1.set_xlabel( xlabel_list[a] ,fontsize=fs) 
        ax1.set_ylabel( ylabel_list[a] ,fontsize=fs) 
        ax1.tick_params( labelsize=fs ) 

        if axis_off:
            ax1.axis('off')
        divider = make_axes_locatable(ax1)
        if cbar_orientation == 'bottom':
            cax = divider.append_axes('bottom', size='5%', pad=0.05)
            cbar = fig.colorbar( im1, cax=cax, orientation='horizontal')
        elif cbar_orientation == 'top':
            cax = divider.append_axes('top', size='5%', pad=0.05)
            cbar = fig.colorbar( im1, cax=cax, orientation='horizontal')
        else: # we put it on the right 
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar( im1, cax=cax, orientation='vertical')           
        cbar.set_label( cbar_label_list[a], rotation=0,fontsize=fs)
        cbar.ax.tick_params(labelsize=fs)
    if savefig!=None:
        plt.savefig( savefig , bbox_inches='tight', dpi=300) 

    plt.show() 



def test_controller_in_cmd_space( zwfs, phase_controller, Vw =None , D=None , AO_lag=None  ):
    #outputs fits file with telemetry  
    # applies open loop for X iterations, Closed loop for Y iterations
    # ability to ramp number of controlled modes , how to deal with gains? gain = 1/number controlled modes for Zernike basis

    #open loop 
    for i in range(X) : 

        for i in range( 3 ) :
            raw_img_list.append( zwfs.get_image() ) # @D, remember for control_phase method this needs to be flattened and filtered for pupil region

        raw_img.append( np.median( raw_img_list, axis = 0 ) ) 

        # ALSO MANUALLY GET PSF IMAGE 
        
        err_img.append(  phase_controller.get_img_err( raw_img[-1].reshape(-1)[zwfs.pupil_pixels]  ) )

        reco_modes.append( phase_controller.control_phase( err_img[-1]  , controller_name = ctrl_method_label) )

        reco_dm_cmds.append( phase_controller.config['M2C'] @ mode_reco[-1] )

        delta_cmd.append( Ki * delta_cmd[-1] - Kp * reco_dm_cmds[-1] )

        # propagate_phase_screen one step 

        # don't yet apply the correction cmd while in open loop
        zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] + dist[-1] ) 

        time.sleep(0.1)


    #close loop 
    for i in range(X) : 

        for i in range( 3 ) :
            raw_img_list.append( zwfs.get_image() ) # @D, remember for control_phase method this needs to be flattened and filtered for pupil region

        raw_img.append( np.median( raw_img_list, axis = 0 ) ) 

        err_img.append(  phase_controller.get_img_err( raw_img[-1].reshape(-1)[zwfs.pupil_pixels]  ) )

        reco_modes.append( phase_controller.control_phase( err_img[-1]  , controller_name = ctrl_method_label) )

        reco_dm_cmds.append( phase_controller.config['M2C'] @ mode_reco[-1] )

        delta_cmd.append( Ki * delta_cmd[-1] - Kp * reco_dm_cmds[-1] )

        # propagate_phase_screen one step 

        # then apply the correction 
        zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] + delta_cmd[-1] + dist[-1] )

        time.sleep(0.1)




