"""
find modal sensitivity 
sigma(hat a_i) * sqrt(Nph) . sigma(a_i) ~ sqrt(Nph)
put 1rad amplitude on the mode 
12.5 rad per cmd => poke IM with 0.8 amplitude cmds normalized in cmd space <M|M>=1.
"""

from baldr_control import ZWFS
from baldr_control import phase_control
from baldr_control import pupil_control
from baldr_control import utilities as util

import numpy as np
import matplotlib.pyplot as plt 
import time 
import datetime
from astropy.io import fits

fig_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/figures/' 
data_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/data/' 

# timestamp
tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

debug = True # plot some intermediate results 
fps = 400 
DIT = 2e-3 #s integration time 

sw = 8 # 8 for 12x12, 16 for 6x6 
pupil_crop_region = [157-sw, 269+sw, 98-sw, 210+sw ] #[165-sw, 261+sw, 106-sw, 202+sw ] #one pixel each side of pupil.  #tight->[165, 261, 106, 202 ]  #crop region around ZWFS pupil [row min, row max, col min, col max] 

#init our ZWFS (object that interacts with camera and DM)
zwfs = ZWFS.ZWFS(DM_serial_number='17DW019#053', cameraIndex=0, DMshapes_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/DMShapes/', pupil_crop_region=pupil_crop_region ) 

# ,------------------ AVERAGE OVER 8X8 SUBWIDOWS SO 12X12 PIXELS IN PUPIL
zwfs.pixelation_factor = sw #8 # sum over 8x8 pixel subwindows in image
# HAVE TO PROPAGATE THIS TO PUPIL COORDINATES 
zwfs._update_image_coordinates( )

zwfs.set_camera_fps(fps) # set the FPS 
zwfs.set_camera_dit(DIT) # set the DIT 

##
##    START CAMERA 
zwfs.start_camera()
# ----------------------
# look at the image for a second
util.watch_camera(zwfs)

#init our phase controller (object that processes ZWFS images and outputs DM commands)
phase_ctrl = phase_control.phase_controller_1(config_file = None) 
#phase_ctrl.change_control_basis_parameters(number_of_controlled_modes=20, basis_name='Zernike' , dm_control_diameter = 10 ) 

#plt.figure(); plt.imshow( util.get_DM_command_in_2D( phase_ctrl.config['M2C'].T[0] ) )
#init our pupil controller (object that processes ZWFS images and outputs VCM commands)

pupil_ctrl = pupil_control.pupil_controller_1(config_file = None)


# 1.2) analyse pupil and decide if it is ok
pupil_report = pupil_control.analyse_pupil_openloop( zwfs, debug = True, return_report = True)

zwfs.update_reference_regions_in_img( pupil_report )

# 1.3) builds our control model with the zwfs
#control_model_report
ctrl_method_label = 'ctrl_1'
phase_ctrl.build_control_model( zwfs , poke_amp = -0.08, label=ctrl_method_label, debug = True)  


# ==========================================================
#%% USING VARIANCE PHOTON ESTIMATE , Nph ~ P < \sigma^2(I_pixel) >


M2C = phase_ctrl.config['M2C'] # readability 
CM = phase_ctrl.ctrl_parameters[ctrl_method_label]['CM'] # readability 

# put a mode on DM and reconstruct it with our CM #(12.5 rad/cmd * 0.08 cmd ~ 1 rad)
amp = -0.08


# estimate number of photons 

raw_img_list = []
for i in range( 1000 ) :

    img = zwfs.get_image()   
    raw_img_list.append( img.reshape(-1)[zwfs.pupil_pixel_filter] )
    time.sleep(0.01)

# estimate number of photons = #pixels in pupil 
Nph = np.sum(zwfs.pupil_pixel_filter) * np.mean( np.var( raw_img_list ,axis=0) ) 
number_samples_per_mode = 500
mode_amp_dict_method2 = {} 
for mode_indx in range(0,70) :  
    print( mode_indx )
    #mode_indx = 11
    mode_aberration = M2C.T[mode_indx]
    #plt.imshow( util.get_DM_command_in_2D(amp*mode_aberration));plt.colorbar();plt.show()
    
    dm_cmd_aber = zwfs.dm_shapes['flat_dm'] + amp * mode_aberration 

    Nph_list = []
    mode_amp_est = []
    for _ in range(number_samples_per_mode):
        zwfs.dm.send_data( dm_cmd_aber )
        time.sleep(0.005)
        raw_img_list = []
        for i in range( 10 ) :
            raw_img_list.append( zwfs.get_image() ) # @D, remember for control_phase method this needs to be flattened and filtered for pupil region
        raw_img = np.median( raw_img_list, axis = 0) 
     
        err_img = phase_ctrl.get_img_err( 1/np.mean(raw_img) *  raw_img.reshape(-1)[zwfs.pupil_pixel_filter]  ) 

        mode_res = CM.T @ err_img 

        mode_amp_est.append( mode_res[mode_indx] )

    mode_amp_dict_method2[mode_indx] = mode_amp_est 


# do the same for zero aberration to get baseline 
mode_amp_dict_baseline = {}
zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] )
time.sleep(0.1)
for mode_indx in range(0,70) :  
    print( mode_indx )
    mode_aberration = M2C.T[mode_indx]

    Nph_list = []
    mode_amp_est = []
    for _ in range(number_samples_per_mode):

        time.sleep(0.005)
        raw_img_list = []
        for i in range( 10 ) :
            raw_img_list.append( zwfs.get_image() ) # @D, remember for control_phase method this needs to be flattened and filtered for pupil region
        raw_img = np.median( raw_img_list, axis = 0) 
     
        err_img = phase_ctrl.get_img_err( 1/np.mean(raw_img) *  raw_img.reshape(-1)[zwfs.pupil_pixel_filter]  ) 

        mode_res = CM.T @ err_img 

        mode_amp_est.append( mode_res[mode_indx] )

    mode_amp_dict_baseline[mode_indx] = mode_amp_est 


# statistics from modal signals 
a_std = np.array( [ np.std(mode_amp_dict_method2[m]) for m in mode_amp_dict_method2 ] )
a_mean = np.array( [ np.mean(mode_amp_dict_method2[m]) for m in mode_amp_dict_method2 ] )

# and our modal noise floor (applying zero aberrations and measuring reconstructor) 
a0_std = np.array( [ np.std(mode_amp_dict_baseline[m]) for m in mode_amp_dict_method2 ] )
a0_mean = np.array( [ np.mean(mode_amp_dict_baseline[m]) for m in mode_amp_dict_method2 ] )

readout_mode = #'12x12'

#-- mean signal 
plt.figure()
plt.errorbar( range(1,len(a_std)+1), a_mean, yerr = a_std / np.sqrt(number_samples_per_mode) ,label='measured modal amplitude',fmt='.', capsize=3)
#plt.errorbar( range(1,len(a_std)+1), a0_mean, yerr = a0_std / np.sqrt(number_samples_per_mode), color='k',)
plt.axhline(1,ls=':',color='k',label='applied modal amplitude')
plt.ylabel(r'$<\hat{ a_i}>$' ,fontsize=15)
plt.xlabel('mode index (i)',fontsize=15)
plt.gca().tick_params(labelsize=15)
plt.legend(fontsize=12)
plt.savefig( data_path + f'Zernike_modal_sensitivity_mean_reconstructed_amp_readout_mode-{readout_mode}_{tstamp}.png' , bbox_inches='tight', dpi=300) 
plt.show()


#-- std signal 
plt.figure()
plt.errorbar( range(1,len(a_std)+1), a_std, yerr = a_std / np.sqrt(2*number_samples_per_mode-2) ,label=r'$\sigma(\hat{a}_i)$',fmt='.', capsize=1)

plt.errorbar( range(1,len(a_std)+1), a0_std, yerr = a_std / np.sqrt(2*number_samples_per_mode-2) ,label='$\sigma(\hat{a}_i)\ |\ Z_i=0$',fmt='.', capsize=1)

plt.ylabel(r'$\sigma(\hat{ a_i})$' ,fontsize=15)
plt.xlabel('mode index (i)',fontsize=15)
plt.gca().tick_params(labelsize=15)
plt.legend(fontsize=12)
plt.savefig( data_path + f'Zernike_modal_sensitivity_std_reconstructed_amp_readout_mode-{readout_mode}_{tstamp}.png' , bbox_inches='tight', dpi=300) 
plt.show()


#-- sigma(ai) * sqrt(Nph/pixel) 
plt.figure()
plt.errorbar( range(1,len(a_std)+1), a_std * np.sqrt(Nph)/np.sum(zwfs.pupil_pixel_filter)**0.5 , yerr = a_std / (np.sqrt(2*number_samples_per_mode-2) ) ,label='$\sigma(\hat{a}_i)$',fmt='.', capsize=2)
#plt.plot( range(1,len(a_std)+1), a0_std * np.sqrt(Nph)/np.sum(zwfs.pupil_pixel_filter)**0.5 , color='k', ls='--',label='$\sigma(\hat{a}_i)\ |\ Z_i=0$')
plt.ylabel(r'$\sigma(\hat{ a_i})\sqrt{\frac{N_{ph}}{pixel}}$',fontsize=15)
plt.xlabel('mode index (i)',fontsize=15)
plt.legend(fontsize=12)
plt.gca().tick_params(labelsize=15)
plt.savefig( data_path + f'Zernike_modal_sensitivity_Nph-per-pixel_readout_mode-{readout_mode}_{tstamp}.png' , bbox_inches='tight', dpi=300) 
#plt.savefig( data_path + f'Zernike_modal_sensitivity_Nph-per-pixel_WITH_BASELINE_readout_mode-{readout_mode}_{tstamp}.png' , bbox_inches='tight', dpi=300) 

plt.show()

#-- sigma(ai) * sqrt(Nph) 
plt.figure()
plt.errorbar( range(1,len(a_std)+1), a_std * np.sqrt(Nph) , yerr = a_std / (np.sqrt(2*number_samples_per_mode-2) ) ,label='$\sigma(\hat{a}_i)$', fmt='.', capsize=2)

plt.plot( range(1,len(a_std)+1), a0_std * np.sqrt(Nph) , color='k', ls='--',label='$\sigma(\hat{a}_i)\ |\ Z_i=0$')
plt.ylabel(r'$\sigma(\hat{ a_i})\sqrt{N_{ph}}$',fontsize=15)
plt.xlabel('mode index (i)',fontsize=15)
plt.gca().tick_params(labelsize=15)
#plt.legend(fontsize=12)
plt.savefig( data_path + f'Zernike_modal_sensitivity_Nph_readout_mode-{readout_mode}_{tstamp}.png' , bbox_inches='tight', dpi=300) 
#plt.savefig( data_path + f'Zernike_modal_sensitivity_Nph-per-pixel_WITH_BASELINE_readout_mode-{readout_mode}_{tstamp}.png' , bbox_inches='tight', dpi=300) 
plt.show()






# ==========================================================
#%% USING BACKGROUND SUBTRACTION METHOD , Nph ~ I - I_bkg 

_ = input('block beam and press enter to get a background measurement')
bkg_img_list = []
for i in range( 10 ) :
    bkg_img_list.append( zwfs.get_image() ) # @D, remember for control_phase method this needs to be flattened and filtered for pupil region
bkg_img = np.median( bkg_img_list, axis = 0) 
plt.figure();plt.imshow( bkg_img );plt.colorbar(label='adu');plt.show()


M2C = phase_ctrl.config['M2C'] # readability 
CM = phase_ctrl.ctrl_parameters[ctrl_method_label]['CM'] # readability 

# put a mode on DM and reconstruct it with our CM #(12.5 rad/cmd * 0.08 cmd ~ 1 rad)
amp = -0.08

Nph_dict = {}
mode_amp_dict = {}
for mode_indx in range(0,70) :  
    print( mode_indx )
    #mode_indx = 11
    mode_aberration = M2C.T[mode_indx]
    #plt.imshow( util.get_DM_command_in_2D(amp*mode_aberration));plt.colorbar();plt.show()
    
    dm_cmd_aber = zwfs.dm_shapes['flat_dm'] + amp * mode_aberration 

    Nph_list = []
    mode_amp_est = []
    for _ in range(20):
        zwfs.dm.send_data( dm_cmd_aber )
        time.sleep(0.1)
        raw_img_list = []
        for i in range( 10 ) :
            raw_img_list.append( zwfs.get_image() ) # @D, remember for control_phase method this needs to be flattened and filtered for pupil region
        raw_img = np.median( raw_img_list, axis = 0) 
     
        err_img = phase_ctrl.get_img_err( 1/np.mean(raw_img) *  raw_img.reshape(-1)[zwfs.pupil_pixel_filter]  ) 

        Nph = (raw_img.reshape(-1) - bkg_img.reshape(-1) )[zwfs.pupil_pixel_filter] 
     
        mode_res = CM.T @ err_img 

        Nph_list.append( Nph ) 
        mode_amp_est.append( mode_res[mode_indx] )

    Nph_dict[mode_indx] = np.mean( Nph_list )  #SHOULD THIS BE SUM!!!!
    mode_amp_dict[mode_indx] = np.std( mode_amp_est )

beta = [np.sqrt(n) * std_ai for n, std_ai in zip(Nph_dict.values(),mode_amp_dict.values())]

plt.figure()
plt.errorbar( range(0,70), beta, yerr=beta / np.sqrt(20))
plt.ylabel(r'$\sigma(\hat{ a_i})\sqrt{N_{ph}}$')
plt.xlabel('mode index (i)')

plt.savefig( data_path + f'Zernike_modal_sensitivity_readout_mode-FULL_{tstamp}.png' , bbox_inches='tight', dpi=300) 


"""
    #plt.figure(); plt.plot( mode_res ); plt.show()

    cmd_res = M2C @ mode_res
    
    time.sleep(0.01) 
    #reco.append( util.get_DM_command_in_2D( cmd_res ) )

    # =============== plotting 
    if mode_indx < 5:
    
        im_list = [util.get_DM_command_in_2D( mode_aberration ), raw_img.T/np.max(raw_img), util.get_DM_command_in_2D( cmd_res)  ]
        xlabel_list = [None, None, None]
        ylabel_list = [None, None, None]
        title_list = ['Aberration on DM', 'ZWFS Pupil', 'reconstructed DM cmd']
        cbar_label_list = ['DM command', 'Normalized intensity' , 'DM command' ] 
        savefig = None #fig_path + f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'

        util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)

"""





"""
plt.figure()
plt.errorbar( range(1,len(a_std)+1), (a_std**2 - a0_std**2)**0.5 * np.sqrt(Nph)   , yerr = a_std / (np.sqrt(2*number_samples_per_mode-2) ) )
plt.ylabel(r'$\sigma(\hat{ a_i})\sqrt{\frac{N_{ph}}{pixel}}$')
plt.xlabel('mode index (i)')
#plt.savefig( data_path + f'Zernike_modal_sensitivity_Nph-per-pixel_readout_mode-FULL_{tstamp}.png' , bbox_inches='tight', dpi=300) 
plt.show()


plt.figure()
plt.errorbar( range(1,len(a_std)+1), a0_std * np.sqrt(Nph) , yerr = a_std / (np.sqrt(2*number_samples_per_mode-2) ) )
plt.ylabel(r'$\sigma(\hat{ a_i})\sqrt{N_{ph}}$')
plt.xlabel('mode index (i)')
#plt.savefig( data_path + f'Zernike_modal_sensitivity_a0_stdxsqrtNph_readout_mode-FULL_{tstamp}.png' , bbox_inches='tight', dpi=300) 
plt.show()
"""


# build Fourier basis 

P=2
Nact=12
x=np.linspace(-1,1,Nact) #normalized DM coordinates
y=np.linspace(-1,1,Nact) #normalized DM coordinates
X,Y = np.meshgrid(x,y) 




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













