README
==========

T1_pupil_region.csv -> pupil_control.analyse_pupil_openloop -> pickle -> util.GET_BDR_RECON_DATA_INTERNAL -> pokeramp.fits -> build_reconstructors.py -> RECONSTRUCTOR.fits.

use T1_pupil_region_12x12.csv or T1_pupil_region_6x6.csv to readin pupil crop region (this should be common for all calibrations!) 


for input into RTC use "1-1_internal_acqusition.py". For pupil region classification and saving pickle file use 

	# 1.2) analyse pupil and decide if it is ok
	pupil_report = pupil_control.analyse_pupil_openloop( zwfs, debug = True, return_report = True)

	with open(data_path + f'pupil_classification_{tstamp}.pickle', 'wb') as handle:
	    pickle.dump(pupil_report, handle, protocol=pickle.HIGHEST_PROTOCOL)


and for ramping the actuators and saving the fits

	# 1.22) fit data internally (influence functions, b etc) 
	# use baldr.
	recon_data = util.GET_BDR_RECON_DATA_INTERNAL(zwfs, number_amp_samples = 18, amp_max = 0.2, number_images_recorded_per_cmd = 2, save_fits = data_path+f'recon_data_UT_SECONDARY_{tstamp}.fits') 


Then in baldr_control folder there is a script "build_reconstructors.py". Run this. This will output a fits file that can be read in to RTC with all info regarding TT/HO reconstructors, pupil phase control region in the detector. etc. 

 

TO DO
======
Now read in RECONSTRUCTOR fits files and test reconstructing static modes! BALDR/static_mode_reconstruction_tests.py 


