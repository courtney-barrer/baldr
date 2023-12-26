"""
functions to help organise and structure configuration files and objects for Baldr simulation 

configuration files are dictionaries that are typically saved as JSON files. They include

    base configuration files include 
    -tel_config_file
    -phasemask_config_file
    -DM_config_file
    -detector_config_file

    these are combined to make a ZWFS mode configuration file which defines the hardware setup of the ZWFS 

    a calibration source(s) can be added to a mode to create an interaction matrix - this creates an ZWFS control mode configuration file

    once a control mode is defined we can do open or closed loop simulation by inputing fields to the objects created from 
    the ZWFS control mode configuration file.

"""

import numpy as np

def init_telescope_config_dict(use_default_values = True) : 
    tel_config_dict = {}
    tel_config_dict['type'] = 'telescope'
    
    if use_default_values:
    
        tel_config_dict['pupil_nx_pixels'] = 12*20  # pixels
        tel_config_dict['telescope_diameter'] = 1.8 # m
        tel_config_dict['telescope_diameter_pixels'] = 12*20 #pixels
        tel_config_dict['pup_geometry'] = 'disk' # 'disk', 'AT', or 'UT'
        tel_config_dict['airmass'] = 1 # unitless
        tel_config_dict['extinction'] = 0.18 #  unitless

    else:
        tel_config_dict['pupil_nx_pixels'] = np.nan     # pixels
        tel_config_dict['telescope_diameter'] = np.nan # m
        tel_config_dict['telescope_diameter_pixels'] = np.nan #pixels
        tel_config_dict['pup_geometry'] = np.nan # 'disk', 'AT', or 'UT'
        tel_config_dict['airmass'] = np.nan # unitless
        tel_config_dict['extinction'] = np.nan #  unitless
     
    return( tel_config_dict )
    
def init_phasemask_config_dict(use_default_values = True) : 
    phasemask_config_dict = {}
    phasemask_config_dict['type'] = 'phasemask'
    
    if use_default_values:

        phasemask_config_dict['off-axis_transparency'] = 1 # unitless (between 0-1).. sometimes represented by parameter 'A'
        phasemask_config_dict['on-axis_transparency'] = 1 # unitless (between 0-1). sometimes represented by parameter 'B'
        phasemask_config_dict['on-axis_glass'] = 'sio2' #material type (e.g. 'sio2', 'su8')
        phasemask_config_dict['off-axis_glass'] = 'sio2' #material type
        phasemask_config_dict['on-axis phasemask depth'] = 1e-6 * 21 # m 
        phasemask_config_dict['off-axis phasemask depth'] = 1e-6 * (21-1.6/4) # m  (default causes 90 deg phase shift at 1.6um)
        phasemask_config_dict['fratio'] = 21 # unitless
        phasemask_config_dict['phasemask_diameter'] = 1.06 * phasemask_config_dict['fratio'] * 1.6e-6 # meters , defults to 1.06resolution elements at 1.6um (remember diameter = number_of_resolution_elements * f_ratio * lambda )...diffraction limit microscope (m) d = lambda/(2*NA) = lambda * F where F=focal length/D 
        phasemask_config_dict['N_samples_across_phase_shift_region'] = 10 # number of pixels across phase shift region in focal plane
        phasemask_config_dict['nx_size_focal_plane'] = 12*20 # number of pixels in x in focal plane
        phasemask_config_dict['cold_stop_diameter'] = 10 * phasemask_config_dict['fratio'] * 1.6e-6  #[m] default 10 \lambda/D @1.6um 
    else:
        phasemask_config_dict['off-axis_transparency'] = np.nan # unitless (between 0-1)
        phasemask_config_dict['on-axis_transparency'] = np.nan # unitless (between 0-1)
        phasemask_config_dict['on-axis_glass'] = np.nan #material type (e.g. 'sio2', 'su8')
        phasemask_config_dict['off-axis_glass'] = np.nan #material type
        phasemask_config_dict['on-axis phasemask depth'] = np.nan # um 
        phasemask_config_dict['off-axis phasemask depth'] = np.nan # um 
        phasemask_config_dict['fratio'] = np.nan # unitless
        phasemask_config_dict['phasemask_diameter'] = np.nan # lambda/D
        phasemask_config_dict['N_samples_across_phase_shift_region'] = np.nan # number of pixels across phase shift region in focal plane
        phasemask_config_dict['nx_size_focal_plane'] = np.nan # number of pixels in x in focal plane
        phasemask_config_dict['cold_stop_diameter'] = np.nan
    
    
    return( phasemask_config_dict )

def init_DM_config_dict(use_default_values = True) : 
    DM_config_dict = {}
    DM_config_dict['type'] = 'DM'
    
    if use_default_values:
        DM_config_dict['N_act'] = 12 # number of actuators across DM diameter (default is DM is square)
        DM_config_dict['m/V'] = 1 # meters of displacement per volt applied to DM (commands sent in DM update shape are in volts)
        DM_config_dict['angle'] = 0 # angle between DM surface normal and input beam (rad)
        DM_config_dict['surface_type'] = 'continuous' # options are 'continuous','segmented'
    else:
        DM_config_dict['N_act'] = np.nan # number of actuators across DM diameter (default is DM is square)
        DM_config_dict['m/V'] = np.nan # meters of displacement per volt applied to DM (commands sent in DM update shape are in volts)
        DM_config_dict['angle'] = np.nan # angle between DM surface normal and input beam (rad)
        DM_config_dict['surface_type'] = np.nan # options are  'continuous','segmented'
        
    return( DM_config_dict )

"""
npix_det = D_pix//pw # number of pixels across detector 
pix_scale_det = dx * pw # m/pix
"""

def init_detector_config_dict(use_default_values = True) : 
    detector_config_dict = {}
    detector_config_dict['type'] = 'detector'
    
    if use_default_values:
        detector_config_dict['detector_npix'] = 12 #pixels
        detector_config_dict['pix_scale_det'] =  1.8/12 # m/pixels
        detector_config_dict['DIT'] = 0.001 #s
        detector_config_dict['ron'] = 1 #e-
        detector_config_dict['quantum_efficiency'] = 1 #
        detector_config_dict['det_wvl_min'] = 1.4 # um
        detector_config_dict['det_wvl_max'] = 1.8 # um
        detector_config_dict['number_wvl_bins'] = 10 #unitless 
    else:
        detector_config_dict['detector_npix'] = np.nan #pixels
        detector_config_dict['pix_scale_det'] =  np.nan # m/pixels
        detector_config_dict['DIT'] = np.nan #s
        detector_config_dict['ron'] = np.nan #e-
        detector_config_dict['quantum_efficiency'] = np.nan #
        detector_config_dict['det_wvl_min'] = np.nan # um
        detector_config_dict['det_wvl_max'] = np.nan # um
        detector_config_dict['number_wvl_bins'] = np.nan #unitless 
    
    return( detector_config_dict )



def init_calibration_source_config_dict(use_default_values = True):

    calibration_source_config_dict = {} 
    calibration_source_config_dict['type'] = 'calibration_source'
    if use_default_values:
        calibration_source_config_dict['calsource_pup_geometry'] = 'disk' # 'disk', 'AT', or 'UT'
        calibration_source_config_dict['temperature'] = 8000 # Kelvin
        calibration_source_config_dict['flux'] = 1e-20 # Watts / m^2
    else:
        calibration_source_config_dict['calsource_pup_geometry'] = np.nan # 'disk', 'AT', or 'UT'
        calibration_source_config_dict['temperature'] = np.nan #Kelvin
        calibration_source_config_dict['flux'] = np.nan # Watts / second / m^2    
        
    return( calibration_source_config_dict )
    

                
        

def create_mode_config_dict( tel_config_file, phasemask_config_file, DM_config_file, detector_config_file):

    mode_dict = {}
    if tel_config_file['type'] == 'telescope':
        mode_dict['telescope'] = tel_config_file
    else:
        raise TypeError('telescope configuration file does not have type "telescope" - revise that the configuration dictionary is correct')

    if phasemask_config_file['type'] == 'phasemask':
        mode_dict['phasemask'] = phasemask_config_file
    else:
        raise TypeError('phasemask configuration file does not have type "phasemask" - revise that the configuration dictionary is correct')

    if  DM_config_file['type'] == 'DM':
        mode_dict['DM'] =  DM_config_file
    else:
        raise TypeError('DM configuration file does not have type "DM" - revise that the configuration dictionary is correct')

    if detector_config_file['type'] == 'detector':
        mode_dict['detector'] =  detector_config_file
    else:
        raise TypeError('detector configuration file does not have type "detector" - revise that the configuration dictionary is correct')

    # NOTE : ZWFS class in baldr functions is initialized with mode dictionary
    return( mode_dict )
    




    
""" To be done in baldr_functions_2.py
def create_ZWFS(mode_dic):
    # eventually I should make a ZWFS object that holds the configuration files etc 
    pup = baldr.pick_pupil(pupil_geometry=mode_dict['telescope']['pup_geometry'] , dim=mode_dict['telescope']['pupil_nx_pixels'], diameter = mode_dict['telescope']['telescope_diameter_pixels'])
    
    dm = DM(surface=np.zeros([mode_dict['DM']['N_act'],mode_dict['DM']['N_act']]), gain=mode_dict['DM']['m/V'] ,\
        angle=mode_dict['DM']['angle'],surface_type = mode_dict['DM']['surface_type']) 
    
    FPM = zernike_phase_mask(A=mode_dict['phasemask']['off-axis_transparency'],B=mode_dict['phasemask']['on-axis_transparency'],\
        phase_shift_diameter=mode_dict['phasemask']['phasemask_diameter'],f_ratio=mode_dict['phasemask']['fratio'],\
        d_on=mode_dict['phasemask']['on-axis phasemask depth'],d_off=mode_dict['phasemask']['off-axis phasemask depth'],\
        glass_on=mode_dict['phasemask']['on-axis_glass'],glass_off=mode_dict['phasemask']['off-axis_glass'])
    
    # -------- NOTE WE USE BY DEFAULT 10 WAVELENGTH BINS ! --------------
    wvls = np.linspace( mode_dict['detector']['det_wvl_min'] ,  mode_dict['detector']['det_wvl_max'], 10 ) 
    QE = mode_dict['detector']['quantum_efficiency']
    det = detector(npix=mode_dict['detector']['detector_npix'], pix_scale = mode_dict['detector']['pix_scale_det'] , DIT= mode_dict['detector']['DIT'], ron=mode_dict['detector']['ron'], QE={w:QE for w in wvls})
    
    return(pup, dm, FPM, det)"""


"""
# put as method in ZWFS object, each ZWFS object can hold multiple control parameters
def setup_control_parameters( self, calibration_source_config_dict, N_controlled_modes, modal_basis='zernike', pokeAmp = 50e-9 , label='control_1')

    # create calibration field and pupil
    
    ZWFS.control[label] = {}
    
    cmd = np.zeros( dm.surface.reshape(-1).shape ) 
    dm.update_shape(cmd) #zero dm first
    
    
    # get the reference signal from calibration field with phase mask in
    sig_on_ref = baldr.detection_chain(calibration_field, dm, FPM, det)
    sig_on_ref.signal = np.mean( [baldr.detection_chain(calibration_field, dm, FPM, det).signal for _ in range(10)]  , axis=0) # average over a few 
    
    # estimate #photons of in calibration field by removing phase mask (zero phase shift)   
    sig_off_ref = baldr.detection_chain(calibration_field, dm, FPM_cal, det)
    sig_off_ref.signal = np.mean( [baldr.detection_chain(calibration_field, dm, FPM_cal, det).signal for _ in range(10)]  , axis=0) # average over a few 
    Nph_cal = np.sum(sig_off_ref.signal)
    
    # Put modes on DM and measure signals from calibration field
    pokeAmp = 50e-9
    
    # CREATE THE CONTROL BASIS FOR OUR DM
    control_basis = baldr.create_control_basis(dm=dm, N_controlled_modes=N_controlled_modes, basis_modes='zernike')
    
    # BUILD OUR INTERACTION AND CONTROL MATRICESFROM THE CALIBRATION SOURCE AND OUR ZWFS SETUP
    IM_modal, pinv_IM_modal = baldr.build_IM(calibration_field=calibration_field, dm=dm, FPM=FPM, det=det, control_basis=control_basis, pokeAmp=pokeAmp)
    
    ZWFS.control[label]['calsource_config_dict'] = calibration_source_config_dict
    
    ZWFS.control[label]['IM']= IM_modal
    ZWFS.control[label]['CM'] = pinv_IM_modal
    ZWFS.control[label]['control_basis'] = control_basis
    ZWFS.control[label]['pokeAmp'] = pokeAmp
    ZWFS.control[label]['N_controlled_modes'] = N_controlled_modes
    ZWFS.control[label]['Nph_cal'] = Nph_cal
    ZWFS.control[label]['sig_on_ref'] = sig_on_ref.signal
    ZWFS.control[label]['sig_off_ref'] = sig_off_ref.signal
"""     