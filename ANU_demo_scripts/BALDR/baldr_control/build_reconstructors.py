#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 17:09:25 2024

@author: bencb
"""


import numpy as np
import matplotlib.pyplot as plt 
import os
import glob
from astropy.io import fits 
import pandas as pd 
import aotools
import pyzelda.utils.zernike as zernike
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime 

# THIS NEEDS TO BE CONSISTENT EVERY WHERE !!
# IT IS HOW IM IS BUILT - AND HOW WE PROCESS NEW SIGNALS 
"""def err_signal(I, I0, N0, bias):
    e = ( (I-bias) - (I0-bias) ) / np.sum( (N0-bias) )
    return( e )
"""
def err_signal(I, I0, bias, norm_flux=None):
    #I0 should already have bias subtracted and be normalized.
    if norm_flux == None:
        e = (I-bias) / np.sum( (I-bias) ) - I0
    else:
        e =  (I-bias) / norm_flux - I0
    return( e )

# below functions are also in the baldr_control module but 
# pulled them out here so this can be stand-alone 
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
    
def get_DM_command_in_2D(cmd, Nx_act=12):
    #puts nan values in cmd positions that don't correspond to actuator on a square grid until cmd length is square number (12x12 for BMC multi-2.5 DM) so can be reshaped to 2D array to see what the command looks like on the DM. 
    corner_indices = [0, Nx_act-1, Nx_act * (Nx_act-1), Nx_act*Nx_act]
    cmd_in_2D = list(cmd.copy())
    for i in corner_indices:
        cmd_in_2D.insert(i,np.nan)
    return( np.array(cmd_in_2D).reshape(12,12) )


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


def construct_command_basis( basis='Zernike', number_of_modes = 20, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True):
    """
    returns a change of basis matrix M2C to go from modes to DM commands, where columns are the DM command for a given modal basis. e.g. M2C @ [0,1,0,...] would return the DM command for tip on a Zernike basis. Modes are normalized on command space such that <M>=0, <M|M>=1. Therefore these should be added to a flat DM reference if being applied.    

    basis = string of basis to use
    number_of_modes = int, number of modes to create
    Nx_act_DM = int, number of actuators across DM diameter
    Nx_act_basis = int, number of actuators across the active basis diameter
    act_offset = tuple, (actuator row offset, actuator column offset) to offset the basis on DM (i.e. we can have a non-centered basis)
    IM_covariance = None or an interaction matrix from command to measurement space. This only needs to be provided if you want KL modes, for this the number of modes is infered by the shape of the IM matrix. 
     
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
            # we also shift the basis center with respect to DM if required
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


    elif basis == 'KL':         
        if without_piston:
            number_of_modes += 1 # we add one more mode since we dont include piston 

        raw_basis = zernike.zernike_basis(nterms=number_of_modes, npix=Nx_act_basis )
        b0 = np.array( [np.nan_to_num(b) for b in raw_basis] )
        cov0 = np.cov( b0.reshape(len(b0),-1) )
        U , S, UT = np.linalg.svd( cov0 )
        KL_raw_basis = ( b0.T @ U ).T # KL modes that diagonalize Zernike covariance matrix 
        for i,B in enumerate(KL_raw_basis):
            # normalize <B|B>=1, <B>=0 (so it is an offset from flat DM shape)
            Bnorm = np.sqrt( 1/np.nansum( B**2 ) ) * B
            # pad with zeros to fit DM square shape and shift pixels as required to center
            # we also shift the basis center with respect to DM if required
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
 
    #elif basis == 'Sensor_Eigenmodes': this is done specifically in a phase_control.py function - as it needs a interaction matrix covariance first 

    return(M2C)


#%%
# timestamp
tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
debug= False #True 
#========= USER INPUTS 

usr_label = input('give a descriptive name for reconstructor fits file')

# desired DM amplitude (normalized between 0-1) full DM pitch ~3.5um
desired_amp = -0.05
reconstruction_method = 'act_poke'


fig_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/figures/' 
data_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/data/' 

# input file that holds pupil classification data
pupil_classification_file = 'pupil_classification_31-05-2024T15.26.52.pickle' #'pupil_classification_30-05-2024T15.41.34.pickle'

# input file that holds poke images of DM -> ZWFS
WFS_response_file =  'recon_data_UT_SECONDARY_31-05-2024T15.36.42.fits' # 'recon_data_30-05-2024T14.07.10.fits'

light_fits = False # do we want to discard intermediate results and assumptions in our reconstructor fits file? 

# ============== READ IN RAMP DATA (TO BUILD RECONSTRUCTORS )
ramp_data = fits.open( data_path + f'{WFS_response_file}'  )

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
I0 =ramp_data['FPM_IN'].data # could also use first image in sequence: np.median( ramp_data['SEQUENCE_IMGS'].data[0], axis=0) #
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
    peak_idx = np.unravel_index( np.argmax( abs(mode_ramp[Nsamp//2-1][actN] - I0 )) , I0.shape)
    ax[2].set_title(f'WFS response from act{actN} poke on pixel {peak_idx}')
    ax[2].plot(poke_amps,(mode_ramp[:,actN,:,:] - I0 )[:,*peak_idx] )
    ax[2].set_ylabel( r'$frac{I[x,y] - I0[x,y]}{N0[x,y]}$' )
    ax[2].set_xlabel('Normalized command amplitude')
    plt.show()


# ============== READ IN PUPIL CLASSIFICATION DATA


with open(data_path + f'{pupil_classification_file}', 'rb') as handle:
    pup_classification = pickle.load(handle)

# ============== START TO BUILD RECONSTRUCTORS 

# get the index of our poke amplitudes that
# is closest to the desired amplitude for building 
# interaction matrix (we could consider other methods)
amp_idx = np.argmin(abs( desired_amp -  poke_amps) )

# construct a Zernike basis on DM to isolate tip/tilt 
# G is mode 
G = construct_command_basis( basis='Zernike', number_of_modes = 2000, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True)
Ginv = np.linalg.pinv(G)

"""
plt.figure();plt.imshow( get_DM_command_in_2D( G.T[-1] )  );plt.show()
"""



# define a filter for where our (circular) pupil is
pupil_filter = pup_classification['pupil_pixel_filter']

# build interaction matrix 
IM = [] 
for act_idx in range(mode_ramp.shape[1]):
    # each row is flattened (I-I0) / N0 signal
    # intensity from a given actuator push
    I = mode_ramp[amp_idx,act_idx ].reshape(-1)[pupil_filter]
    i0 = I0.reshape(-1)[pupil_filter]
    n0 = N0.reshape(-1)[pupil_filter]
    bi = bias.reshape(-1)[pupil_filter]
    # our signal is (I-I0)/sum(N0) defined in err_signal function  
    signal =  list( err_signal(I, I0=i0, N0=n0, bias=bi))
    IM.append(signal )
    #IM.append( ( (mode_ramp[amp_idx,act_idx ][cp_x1: cp_x2 , cp_y1: cp_y2] - ref_im[cp_x1: cp_x2 , cp_y1: cp_y2])/N0 ).reshape(-1)  ) 

#plt.figure();plt.imshow( np.cov( IM ) ) ; plt.show()

U,S,Vt = np.linalg.svd( IM )


# estimate how many actuators are well registered 

if 1:# debug : 
    #singular values
    plt.figure() 
    plt.loglog(S/np.max(S))
    plt.axvline( np.pi * (10/2)**2, color='k', ls=':', label='number of actuators in pupil')
    plt.legend() 
    plt.xlabel('mode index')
    plt.ylabel('singular values')
    plt.savefig(fig_path + f'singularvalues_{tstamp}.png',bbox_inches='tight',dpi=300)
    plt.show()
    
    # THE IMAGE MODES 

    fig,ax = plt.subplots(8,8,figsize=(30,30))
    plt.subplots_adjust(hspace=0.1,wspace=0.1)
    for i,axx in enumerate(ax.reshape(-1)):
        # we filtered circle on grid, so need to put back in grid
        tmp = pup_classification['pupil_pixel_filter'].copy()
        vtgrid = np.zeros(tmp.shape)
        vtgrid[tmp] = Vt[i]
        axx.imshow( vtgrid.reshape( cp_x2-cp_x1,cp_y2-cp_y1) )
        #axx.set_title(f'\n\n\nmode {i}, S={round(S[i]/np.max(S),3)}',fontsize=5)
        axx.text( 10,10,f'{i}',color='w',fontsize=4)
        axx.text( 10,20,f'S={round(S[i]/np.max(S),3)}',color='w',fontsize=4)
        axx.axis('off')
        #plt.legend(ax=axx)
    plt.tight_layout()
    plt.savefig(fig_path + f'det_eignmodes_{tstamp}.png',bbox_inches='tight',dpi=300)
    plt.show()
    
    # THE DM MODES 
    fig,ax = plt.subplots(8,8,figsize=(30,30))
    plt.subplots_adjust(hspace=0.1,wspace=0.1)
    for i,axx in enumerate(ax.reshape(-1)):
        axx.imshow( get_DM_command_in_2D( U.T[i] ) )
        #axx.set_title(f'mode {i}, S={round(S[i]/np.max(S),3)}')
        axx.text( 1,2,f'{i}',color='w',fontsize=6)
        axx.text( 1,3,f'S={round(S[i]/np.max(S),3)}',color='w',fontsize=6)
        axx.axis('off')
        #plt.legend(ax=axx)
    plt.tight_layout()
    plt.savefig(fig_path + f'dm_eignmodes_{tstamp}.png',bbox_inches='tight',dpi=300)
    plt.show()

minMode_i = int(input('what is the minimum (int) eigenmode you want to keep in control matrix (enter integer, hint 0 or 1)'))
maxMode_i = int(input('up to what eigenmode do you want to keep in control matrix (enter integer, hint 20-50)'))


smat = np.zeros((U.shape[0], Vt.shape[0]), dtype=float)
S_filt = [s if ((i < maxMode_i) & (i>minMode_i)) else 0 for i,s in enumerate(S)]
smat[:len(S),:len(S)] = np.diag(S_filt)

IM_filt = U @ smat @ Vt

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
    im_list = [get_DM_command_in_2D( ramp_data[1].data[act_idx+1] - ramp_data[1].data[0] ), get_DM_command_in_2D( CM.T @ IM[act_idx] ), get_DM_command_in_2D( R_HO @ IM[act_idx]  ) ]
    xlabel_list = ['','','']
    ylabel_list = ['','','']
    title_list = ['DM aberration','reconstruction with CM', 'reconstruction with R_HO']
    cbar_label_list = ['normalized cmds','normalized cmds','normalized cmds' ]
    nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list,cbar_label_list, fontsize=15, cbar_orientation = 'bottom', axis_off=True, savefig=None)
    plt.show()


# see pupil region classifications 
if debug:
    # If you want to check
    fig,ax = plt.subplots(1,3)
    ax[0].imshow( pup_classification['pupil_pixel_filter'].reshape( cp_x2-cp_x1, cp_y2-cp_y1) )
    ax[1].imshow( pup_classification['outside_pupil_pixel_filter'].reshape( cp_x2-cp_x1, cp_y2-cp_y1) )
    ax[2].imshow( pup_classification['secondary_pupil_pixel_filter'].reshape( cp_x2-cp_x1, cp_y2-cp_y1) )
    for axx,l in zip(ax, ['inside pupil','outside pupil','secondary']):
        axx.set_title(l)
    plt.show()


#%% WRITE TO FITS 

# headers to copy from ramp_data 
headers2copy = ['CAMERA','camera_fps','camera_tint','camera_gain',\
'cropping_corners_r1','cropping_corners_r2','cropping_corners_c1','cropping_corners_c2']

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

Sfilt_fits = fits.PrimaryHDU( S_filt )
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


