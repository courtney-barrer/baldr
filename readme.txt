



data_structure_functions have a series of functions to help organise and structure objects for Baldr simulation 

configuration files are dictionaries that are typically saved as JSON files. They include
base configuration files that relate to either :
	-telescope and grid geometry 
	-focal plane phasemask
	-deformable mirror 
	-detector
Dictionaries for these configurations can be initialized with default parameters in the data_structure_functions module using:

	tel_config = init_telescope_config_dict(use_default_values = True)
	phasemask_config = init_phasemask_config_dict(use_default_values = True) 
	DM_config = init_DM_config_dict(use_default_values = True) 
	detector_config = init_detector_config_dict(use_default_values = True)

Note the wavelength grid of the simulation is defined in detector_config under keys: 'det_wvl_min', 'det_wvl_max', 'number_wvl_bins'. Linear spacing of wavelength grid is always assumed. All the configuration dictionaries can then be combined to make a ZWFS mode configuration file which defines the full hardware setup of the ZWFS. This can be done in the data_structure_functions module using

	mode_dict = create_mode_config_dict( tel_config, phasemask_config, DM_config, detector_config)


From here a ZWFS object used in the simulation can be created from the baldr_functions_2 module using 

	my_zwfs = ZWFS(mode_dict) 
	
Which has attributes of :
	my_zwfs.pup # the telescope pupil
	my_zwfs.dm # the DM object used for the ZWFS
	my_zwfs.FPM # the focal-plane phase mask object used for the ZWFS
	my_zwfs.det # the detector object used for the ZWFS
	my_zwfs.wvls # the wavelength considered in the ZWFS (for simulations) 
	my_zwfs.mode # the mode dictionary used to create to the current ZWFS object
	my_zwfs.control_variables # a dictionary (initialized empty) to hold any future control variables of the system
		
In-order to do closed loop control from the hardware we need to set-up a calibration source and interaction matrix or control laws etc. This is initially began done from the data_structure_functions module using:


	calib_config = init_calibration_source_config_dict(use_default_values = True, detector_config_dict=None)

Calibration source needs a temperature to calculate spectral shape, power to calculate total photon flux, pupil geometry (which may be different to telescope pupil), detector configuration file for if the detector has different settings to the on sky measurement, and level of internal phase aberrations (rms).  This then gets used to create a field object with flux corresponding to #photons and phase aberrations zero mean normal with variance corresponding to specified internal phase aberration level. Note field flux units are usually assumed to be ph_m2_s_nm! 

	calibration_field = baldr.create_field_from_calibration_source_config( calib_config ) 

From this 

 	create_control_basis(dm, N_controlled_modes, basis_modes='zernike')

And then we build the IM 

	build_IM(calibration_field, dm, FPM, det, control_basis, pokeAmp=50e-9)

Then we need to append this information to my_zwfs.control_variables including the control basis, the IM, and control matrix, the number of photons detected on the calibration source when building IM etc.. everything we need to know to then take sky measurements and reconstruct phase.  


	a calibration source(s) can be added to a mode to create an interaction matrix - this creates an ZWFS control mode configuration file

	once a control mode is defined we can do open or closed loop simulation by inputing fields to the objects created from 
	the ZWFS control mode configuration file.



A few additional notes
- Coordinates are nearly always referenced in the input pupil plane for the field, DM, and detector
- Coordinates of detectors are always centered at detector center (0,0). Therefore input field coordinates need to defined relative to detector coorindates when detecting field
- Coordinates of the DM are initialized using DM.define_coordinates(x,y). 
- If you apply DM to a field via field.applyDM( DM ) any field points that are outside the DM coordinates will automatically make field amp=0, field phase = np.nan.  


"""