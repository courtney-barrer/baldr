import sys

sys.path.insert(1, '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/')

from baldr_control import ZWFS
from baldr_control import phase_control
from baldr_control import pupil_control
from baldr_control import utilities as util

import pickle
import numpy as np
import matplotlib.pyplot as plt 
import time 
import datetime
from astropy.io import fits
import pandas as pd 

fig_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/figures/' 
data_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/data/' 

# THIS FUNCTION SHOULD BE WRITEN IN UTILS...
def err_signal(I, I0, bias, norm_flux=None):
    #I0 should already have bias subtracted and be normalized.
    if norm_flux == None:
        e = (I-bias) / np.sum( (I-bias) ) - I0
    else:
        e =  (I-bias) / norm_flux - I0
    return( e )

# timestamp
tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

# define filenames for the pupil classification file and pokeramp file to write and read in when constructing reconstructor.
pupil_classification_filename = f'pupil_classification_{tstamp}.pickle'
pokeramp_filename = f'pokeramp_data_{tstamp}.fits'

debug = True # plot some intermediate results 
fps = 400 
DIT = 2e-3 #s integration time 

#sw = 8 # 8 for 12x12, 16 for 6x6 
#pupil_crop_region = [157-sw, 269+sw, 98-sw, 210+sw ] #[165-sw, 261+sw, 106-sw, 202+sw ] #one pixel each side of pupil.  #tight->[165, 261, 106, 202 ]  #crop region around ZWFS pupil [row min, row max, col min, col max] 

#readout_mode = '12x12' # '6x6'
#pupil_crop_region = pd.read_csv('/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/' + f'T1_pupil_region_{readout_mode}.csv',index_col=[0])['0'].values.astype(int)


#  
pupil_crop_region = [None,None,None,None]
#pupil_crop_region = [157, 350, 0, 400] # bullshit region 

#init our ZWFS (object that interacts with camera and DM)
zwfs = ZWFS.ZWFS(DM_serial_number='17DW019#053', cameraIndex=0, DMshapes_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/DMShapes/', pupil_crop_region=pupil_crop_region ) 

# SOMETIMES IF CROPPING YOU WILL NEED TO RESET BEFORE HAND 
# USE zwfs.restore_default_settings()

#restore default settings of camera
#zwfs.restore_default_settings()

# ,------------------ AVERAGE OVER 8X8 SUBWIDOWS SO 12X12 PIXELS IN PUPIL
#zwfs.pixelation_factor = sw #8 # sum over 8x8 pixel subwindows in image
# HAVE TO PROPAGATE THIS TO PUPIL COORDINATES 
#zwfs._update_image_coordinates( )

zwfs.set_camera_fps(fps) # set the FPS 
zwfs.set_camera_dit(DIT) # set the DIT 

# cropped columns must be multiple of 32 - multiple of 32 minus 1
# cropped rows must be multiple of 4 - multiple of 4 minus 1
#zwfs.set_camera_cropping(r1=152, r2=267, c1=96, c2=255) # 
zwfs.set_camera_cropping(r1=152, r2=267, c1=96, c2=223)
zwfs.enable_frame_tag(tag = False) # first 1-3 pixels count frame number etc

##
##    START CAMERA 
zwfs.start_camera()
# ----------------------
# look at the image for a second
util.watch_camera(zwfs)


#init our pupil controller (object that processes ZWFS images and outputs VCM commands)
pupil_ctrl = pupil_control.pupil_controller_1(config_file = None)

# 1.2) analyse pupil and decide if it is ok
pupil_report = pupil_control.analyse_pupil_openloop( zwfs, debug = True, return_report = True,symmetric_pupil=True)

# issue with this ^^ when using camera cropping modes...??

## I NEED TO UPDATE THIS TO ZWFS OBJECT!!! 
zwfs.update_reference_regions_in_img( pupil_report ) # 

# write to a pickle file 
with open(data_path + pupil_classification_filename , 'wb') as handle:
    pickle.dump(pupil_report, handle, protocol=pickle.HIGHEST_PROTOCOL)


#ensure flat DM 
zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] )
# get our poke ramp data and write to fits 
recon_data = util.GET_BDR_RECON_DATA_INTERNAL(zwfs, number_amp_samples = 18, amp_max = 0.2, number_images_recorded_per_cmd = 2, save_fits = data_path + pokeramp_filename) 

# ====================================================================
# ================== read in and build RECOCONSTRUCTORS - save to fits  
# stand alone script found in BALDR/baldr_control/build_reconstructors

debug= True #False #True 
#========= USER INPUTS 

usr_label = input('give a descriptive name for reconstructor fits file')

# desired DM amplitude (normalized between 0-1) full DM pitch ~3.5um
desired_amp = -0.03
reconstruction_method = 'act_poke'

light_fits = False # do we want to discard intermediate results and assumptions in our reconstructor fits file? 

# ============== READ IN RAMP DATA (TO BUILD RECONSTRUCTORS )
ramp_data = fits.open( data_path + pokeramp_filename )

#poke_imgs = recon_data['SEQUENCE_IMGS'].data[1:].reshape(No_ramps, 140, I0.shape[0], I0.shape[1])

cp_x1, cp_x2, cp_y1, cp_y2 = ramp_data[0].header['CP_X1'],ramp_data[0].header['CP_X2'],ramp_data[0].header['CP_Y1'],ramp_data[0].header['CP_Y2']
# psf corners 

ampMax =  ramp_data['SEQUENCE_IMGS'].header['HIERARCH in-poke max amp']
Nsamp = ramp_data['SEQUENCE_IMGS'].header['HIERARCH #ramp steps']
poke_amps = np.linspace(-ampMax,ampMax,Nsamp )
if ramp_data['SEQUENCE_IMGS'].header['HIERARCH take_median_of_images']:
    # if we took median of a bunch of images per cmd there is only one array here
    im_per_cmd = 1
else: # we kept a bunch of images per DM command 
    im_per_cmd =  ramp_data['SEQUENCE_IMGS'].header['HIERARCH #images per DM command']  
Nxpix = ramp_data['SEQUENCE_IMGS'].data[0,0].shape[0] #number of x pixels 
Nypix = ramp_data['SEQUENCE_IMGS'].data[0,0].shape[1] #number of y pixels
Nact = ramp_data['DM_CMD_SEQUENCE'].shape[1] # should be 140 for Baldr! 

# reference image with flat DM and FPM in the beam 
I0 = ramp_data['FPM_IN'].data # could also use first image in sequence: np.median( ramp_data['SEQUENCE_IMGS'].data[0], axis=0) #
# reference image with flat DM and FPM out of the beam 
N0 =  ramp_data['FPM_OUT'].data 
# bias 
bias = np.median( ramp_data['BIAS'].data ,axis=0 )

# WFS response from pushing and pulling each actuator over given range
# shape = [push_value, Nact, Nx, Ny]
mode_ramp = np.median( ramp_data['SEQUENCE_IMGS'].data[1:].reshape(len(poke_amps), Nact, im_per_cmd, Nxpix, Nypix ) ,axis=2)

# some debuggin checking plots 
if debug:
    fig,ax = plt.subplots( 1,3,figsize=(15,5))
    ax[0].imshow( I0-bias )
    ax[0].set_title('flat DM, FPM IN')

    ax[1].imshow( N0-bias )
    ax[1].set_title('flat DM, FPM OUT')

    actN = 65 # look at DM actuator 65
    # find where we get peak influence in WFS for small amplitude push
    peak_idx = np.unravel_index( np.argmax( abs(mode_ramp[Nsamp//2-1][actN][1:,1:] - I0[1:,1:] )) , I0[1:,1:].shape) # we skip first row since first and second index are often used as frame counters 
    ax[2].set_title(f'WFS response from act{actN} poke on pixel {peak_idx}')
    ax[2].plot(poke_amps,(mode_ramp[:,actN,:,:] - I0 )[:,*peak_idx] )
    ax[2].set_ylabel( r'$frac{I[x,y] - I0[x,y]}{N0[x,y]}$' )
    ax[2].set_xlabel('Normalized command amplitude')
    plt.show()


# ============== READ IN PUPIL CLASSIFICATION DATA

with open(data_path + pupil_classification_filename, 'rb') as handle:
    pup_classification = pickle.load(handle)


# see pupil region classifications 
if debug:
    # If you want to check
    fig,ax = plt.subplots(1,5,figsize=(20,4))

    ax[0].imshow( pup_classification['pupil_pixel_filter'].reshape(I0.shape))# cp_x2-cp_x1, cp_y2-cp_y1) )
    ax[1].imshow( pup_classification['outside_pupil_pixel_filter'].reshape(I0.shape))#( cp_x2-cp_x1, cp_y2-cp_y1) )
    ax[2].imshow( pup_classification['secondary_pupil_pixel_filter'].reshape(I0.shape))#( cp_x2-cp_x1, cp_y2-cp_y1) )
    ax[3].imshow( I0-bias )
    ax[4].imshow( N0-bias )
    
    for axx,l in zip(ax, ['inside pupil','outside pupil','secondary','I0','N0']):
        axx.set_title(l)
    plt.show()
# ============== START TO BUILD RECONSTRUCTORS 

# get the index of our poke amplitudes that
# is closest to the desired amplitude for building 
# interaction matrix (we could consider other methods)
amp_idx = np.argmin(abs( desired_amp -  poke_amps) )

# construct a Zernike basis on DM to isolate tip/tilt 
# G is mode 
G = util.construct_command_basis( basis='Zernike', number_of_modes = 2000, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True)
Ginv = np.linalg.pinv(G)

"""
plt.figure();plt.imshow( util.get_DM_command_in_2D( G.T[-1] )  );plt.show()
"""



# define a filter for where our (circular) pupil is
#pupil_filter =(I0>-100).reshape(-1) #pup_classification['pupil_pixel_filter']
pupil_filter = pup_classification['pupil_pixel_filter']
# If we just read the entire image (does matrix condition improve?)
#pupil_filter = np.array([True for _ in range(len( pup_classification['pupil_pixel_filter'] ))])
# build interaction matrix 
IM = [] 
for act_idx in range(mode_ramp.shape[1]):
    # each row is flattened (I-I0) / N0 signal
    # intensity from a given actuator push
    I = mode_ramp[amp_idx,act_idx ].reshape(-1)[pupil_filter]
    i0 = I0.reshape(-1)[pupil_filter] / np.sum( I0 )
    n0 = N0.reshape(-1)[pupil_filter]
    bi = bias.reshape(-1)[pupil_filter]
    # our signal is (I-I0)/sum(N0) defined in err_signal function  
    signal =  list( err_signal(I, I0=i0,  bias=bi))
    IM.append(signal )
    #IM.append( ( (mode_ramp[amp_idx,act_idx ][cp_x1: cp_x2 , cp_y1: cp_y2] - ref_im[cp_x1: cp_x2 , cp_y1: cp_y2])/N0 ).reshape(-1)  ) 

#plt.figure();plt.imshow( np.cov( IM ) ) ; plt.show()
print( 'raw IM condition = ', np.linalg.cond( IM ) )

U,S,Vt = np.linalg.svd( IM , full_matrices=True)


# estimate how many actuators are well registered 

if 1:# debug : 
    #singular values
    plt.figure() 
    plt.semilogy(S/np.max(S))
    plt.axvline( np.pi * (10/2)**2, color='k', ls=':', label='number of actuators in pupil')
    plt.legend() 
    plt.xlabel('mode index')
    plt.ylabel('singular values')
    #plt.savefig(fig_path + f'singularvalues_{tstamp}.png',bbox_inches='tight',dpi=300)
    plt.show()
    
    # THE IMAGE MODES 

    fig,ax = plt.subplots(8,8,figsize=(30,30))
    plt.subplots_adjust(hspace=0.1,wspace=0.1)
    for i,axx in enumerate(ax.reshape(-1)):
        # we filtered circle on grid, so need to put back in grid
        tmp = pupil_filter.copy()
        vtgrid = np.zeros(tmp.shape)
        vtgrid[tmp] = Vt[i]
        axx.imshow( vtgrid.reshape(I0.shape ) #cp_x2-cp_x1,cp_y2-cp_y1) )
        #axx.set_title(f'\n\n\nmode {i}, S={round(S[i]/np.max(S),3)}',fontsize=5)
        axx.text( 10,10,f'{i}',color='w',fontsize=4)
        axx.text( 10,20,f'S={round(S[i]/np.max(S),3)}',color='w',fontsize=4)
        axx.axis('off')
        #plt.legend(ax=axx)
    plt.tight_layout()
    #plt.savefig(fig_path + f'det_eignmodes_{tstamp}.png',bbox_inches='tight',dpi=300)
    plt.show()
    
    # THE DM MODES 
    fig,ax = plt.subplots(8,8,figsize=(30,30))
    plt.subplots_adjust(hspace=0.1,wspace=0.1)
    for i,axx in enumerate(ax.reshape(-1)):
        axx.imshow( util.get_DM_command_in_2D( U.T[i] ) )
        #axx.set_title(f'mode {i}, S={round(S[i]/np.max(S),3)}')
        axx.text( 1,2,f'{i}',color='w',fontsize=6)
        axx.text( 1,3,f'S={round(S[i]/np.max(S),3)}',color='w',fontsize=6)
        axx.axis('off')
        #plt.legend(ax=axx)
    plt.tight_layout()
    #plt.savefig(fig_path + f'dm_eignmodes_{tstamp}.png',bbox_inches='tight',dpi=300)
    plt.show()

minMode_i = int(input('what is the minimum (int) eigenmode you want to keep in control matrix (enter integer, hint 0 or 1)'))
maxMode_i = int(input('up to what eigenmode do you want to keep in control matrix (enter integer, hint 20-50)'))

"""#proper
Sigma = np.zeros( np.array(IM).shape, float) #np.zeros((U.shape[0], Vt.shape[0]), dtype=float)
S_filt = np.array( [s if ((i < maxMode_i) & (i>minMode_i)) else np.min(S) for i,s in enumerate(S)])
# putting zero at first indices seems to make the condition go crazy!
Sigma[:len(S),:len(S)] = np.diag(S_filt)
"""

Sigma = np.zeros( np.array(IM).shape, float) #np.zeros((U.shape[0], Vt.shape[0]), dtype=float)
S_filt = np.array( [((i < maxMode_i) & (i>minMode_i)) for i,_ in enumerate(S)])
# NOTE THIS JUST SHIFTS IT... SO FIRST INDEX IS STILL not filtered.. but seems to work.
np.fill_diagonal(Sigma, S[S_filt], wrap=False)
# putting zero at first indices seems to make the condition go crazy!

IM_filt = U @ Sigma @ Vt


#plt.figure();plt.imshow( np.cov( IM ) ) ; plt.show()
print( 'filtered IM condition = ', np.linalg.cond( IM_filt ) )

CM = 2*poke_amps[amp_idx] * np.linalg.pinv( IM_filt )

# tip/tilt index in the DM mode space basis
TT_idx = [0,1]
#G : cmd -> mode G
Ginv_TT = Ginv.copy()
for i in range(len(Ginv_TT)):
    if (i!=TT_idx[0] ) & (i!=TT_idx[1]):
        Ginv_TT[i] = 0 * Ginv_TT[i]

Ginv_HO = Ginv.copy()
for i in range(len(Ginv_HO)):
    if (i==TT_idx[0] ) | (i==TT_idx[1]):
        Ginv_HO[i] = 0 * Ginv_HO[i]


R_TT = G @ (Ginv_TT @ CM.T )

R_HO = G @ (Ginv_HO @ CM.T )


if debug:
    act_idx = 65
    #check the reconstructions in CM and R_HO
    im_list = [util.get_DM_command_in_2D( ramp_data[1].data[act_idx+1] - ramp_data[1].data[0] ), util.get_DM_command_in_2D( CM.T @ IM[act_idx] ), util.get_DM_command_in_2D( R_HO @ IM[act_idx]  ) ]
    xlabel_list = ['','','']
    ylabel_list = ['','','']
    title_list = ['DM aberration','reconstruction with CM', 'reconstruction with R_HO']
    cbar_label_list = ['normalized cmds','normalized cmds','normalized cmds' ]
    util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list,cbar_label_list, fontsize=15, cbar_orientation = 'bottom', axis_off=True, savefig=None)
    plt.show()





#%% WRITE TO FITS 

# headers to copy from ramp_data 
# NOTE cropping_corners relates to full frame readout but and cropping post imaging
#  cropping_rows, cropping_columns relates to actual camera cropping !
headers2copy = ['CAMERA','camera_fps','camera_tint','camera_gain',\
'cropping_corners_r1','cropping_corners_r2','cropping_corners_c1','cropping_corners_c2',\
'cropping_rows','cropping_columns']

info_fits = fits.PrimaryHDU( [] ) #[pupil_classification_file,WFS_response_file] )

info_fits.header.set('EXTNAME','INFO')
info_fits.header.set('IM_construction_method',f'{reconstruction_method}')
info_fits.header.set('DM poke amplitude[normalized]',f'{poke_amps[amp_idx]}')
info_fits.header.set('what is?','names of input files for building IM')
#info_fits.header.set('pupil_classification_file',f'{pupil_classification_file }')
#info_fits.header.set('WFS_response_file',f'{WFS_response_file}')
for h in headers2copy:
    info_fits.header.set(h, ramp_data['SEQUENCE_IMGS'].header[h])


# CHECK WHAT MIKE WROTE IN DOCUMENT FOR WHAT WE WANT HERE
pupil_fits = fits.PrimaryHDU( pup_classification['pupil_pixels']  )
# I0.reshape(-1)[pup_classification['pupil_pixels'] ]
pupil_fits.header.set('what is?','pixels_inside_pupil')
pupil_fits.header.set('EXTNAME','pupil_pixels')

# secondary 
secondary_fits = fits.PrimaryHDU( pup_classification['secondary_pupil_pixels']  )
# I0.reshape(-1)[pup_classification['pupil_pixels'] ]
secondary_fits.header.set('what is?','pixels_inside_secondary obstruction')
secondary_fits.header.set('EXTNAME','secondary_pixels')


outside_fits = fits.PrimaryHDU( pup_classification['outside_pupil_pixels'] )
outside_fits.header.set('what is?','pixels_outside_pupil')
outside_fits.header.set('EXTNAME','outside_pixels')


bias_fits = fits.PrimaryHDU( bias )
bias_fits.header.set('what is?','bias used in IM')
bias_fits.header.set('EXTNAME','bias')


U_fits = fits.PrimaryHDU( U )
U_fits.header.set('what is?','U in SVD of IM')
U_fits.header.set('EXTNAME','U')

S_fits = fits.PrimaryHDU( S )
S_fits.header.set('what is?','singular values in SVD of IM')
S_fits.header.set('EXTNAME','S')

Vt_fits = fits.PrimaryHDU( Vt )
Vt_fits.header.set('what is?','Vt in SVD of IM')
Vt_fits.header.set('EXTNAME','Vt')

Sfilt_fits = fits.PrimaryHDU( S_filt.astype(float))
Sfilt_fits.header.set('what is?','filtered singular values used for CM')
Sfilt_fits.header.set('EXTNAME','Sfilt')

CM_fits = fits.PrimaryHDU( CM )
CM_fits.header.set('what is?','filtered control matrix (CM)')
CM_fits.header.set('EXTNAME','CM')

RTT_fits = fits.PrimaryHDU( R_TT )
RTT_fits.header.set('what is?','tip-tilt reconstructor')
RTT_fits.header.set('EXTNAME','R_TT')

RHO_fits = fits.PrimaryHDU( R_HO )
RHO_fits.header.set('what is?','higher-oder reconstructor')
RHO_fits.header.set('EXTNAME','R_HO')

fits_list = [info_fits, bias_fits, U_fits, S_fits, Vt_fits, Sfilt_fits,\
               CM_fits, RTT_fits, RHO_fits, ramp_data['FPM_IN'],\
               ramp_data['FPM_OUT'],pupil_fits, secondary_fits, outside_fits]


# 
reconstructor_fits = fits.HDUList( [] )
for f in fits_list:
    reconstructor_fits.append( f )


if not light_fits:
    G_fits = fits.PrimaryHDU( G )
    G_fits.header.set('what is?','DM modal basis used in TT and HO projection')
    G_fits.header.set('EXTNAME','G')
    reconstructor_fits.append( G_fits )


reconstructor_fits.writeto( data_path + f'RECONSTRUCTORS_{usr_label}_DIT-{round(float(info_fits.header["camera_tint"]),6)}_gain_{info_fits.header["camera_gain"]}_{tstamp}.fits',overwrite=True )  #data_path + 'ZWFS_internal_calibration.fits'






