import streamlit as st
import numpy as np 
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pyzelda.utils.zernike as zernike
import pyzelda.utils.mft as mft
import scipy
import os 
import aotools
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json 

os.chdir('/Users/bcourtne/Documents/ANU_PHD2/baldr')
from functions import baldr_functions_2 as baldr

col1, col2, col3, col4, col5, col6 = st.columns(6)

if 'key' not in st.session_state:
    st.session_state['key'] = 'value'
    
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False
    st.session_state.placeholder = 'write here'
    
# philosphy is that only first principle parameters are included in config file - no derived parameters 
tel_config_dict={}
field_config_dict={}
phasemask_config_dict={}
DM_config_dict={}
detector_config_dict={}
calibration_source_config_dict = {}
# hardcoded parameters that are not user inputs :

pw = 20 # ratio of number of pixels in input field to number of pixels on detector
npix_det = 12 # number of pixels on detector  
dim = npix_det * pw # number of pixels in input field 
D_pix = dim # number of pixels across pupil diameter 
N_samples_across_phase_shift_region = 10 # number of pixels across phase shift region in focal plane
nx_size_focal_plane = dim # number of pixels in x in focal plane

with col1:
	st.markdown('grid & pupil setup')

	#dim = st.text_input("Nx pixels", value=12*20, label_visibility=st.session_state.visibility,\
	#disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key='OL_dim_input')

	#try:
	#	dim = int(dim)
	#except:
	#	st.markdown(':red[Nx pixels needs to be input as an integer]')

	D = st.text_input("telescope diameter (m)", value=1.8, label_visibility=st.session_state.visibility,\
	disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key='OL_D_input')

	try:
		D = float(D)
	except:
		st.markdown(':red[telescope diameter needs to be input as a float]')

	
	#D_pix = st.text_input("pixels across pupil", value=dim, label_visibility=st.session_state.visibility,\
	#disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key='OL_Dpix_input')

	#try:
	#	D_pix = int(D_pix)
	#except:
	#	st.markdown(':red[pixels across pupil needs to be input as an int > 0]')
	
	
	pup_geometry = st.selectbox('Pupil geometry',('disk', 'AT', 'UT'), index=0, key='OL_geompupil_input')

	
	airmass = st.text_input("airmass", value=1.3, label_visibility=st.session_state.visibility,\
	disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key='OL_airmas_input')

	try:
		airmass = float(airmass)
	except:
		st.markdown(':red[airmass needs to be input as a float]')
		
	tel_config_dict['pupil_nx_pixels'] = dim		# pixels
	tel_config_dict['telescope_diameter'] = D # m
	tel_config_dict['telescope_diameter_pixels'] = D_pix #pixels
	tel_config_dict['pup_geometry'] = pup_geometry # name of pupil geometry
	tel_config_dict['airmass'] = airmass # 
	tel_config_dict['extinction'] = 0.18 # 
	
	
with col2:
	st.markdown('input field setup')

	Jmag = st.text_input("Jmag", value=2, label_visibility=st.session_state.visibility,\
	disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key='OL_Jmag_input')

	try:
		Hmag = float(Jmag)
	except:
		st.markdown(':red[Jmag needs to be input as a float]')

	
	Hmag = st.text_input("Hmag", value=2, label_visibility=st.session_state.visibility,\
	disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key='OL_Hmag_input')

	try:
		Hmag = float(Hmag)
	except:
		st.markdown(':red[Hmag needs to be input as a float]')


	Kmag = st.text_input("Kmag", value=2, label_visibility=st.session_state.visibility,\
	disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key='OL_Kmag_input')

	try:
		Kmag = float(Kmag)
	except:
		st.markdown(':red[Kmag needs to be input as a float]')

	Nwvl_bins = st.text_input("wavelength bins", value=10, label_visibility=st.session_state.visibility,\
	disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key='OL_wbins_input')

	try:
		field_Nwvl_bins = int(Nwvl_bins)
	except:
		st.markdown(':red[wavelength bins needs to be input as an int > 0]')

	field_config_dict['Jmag'] = Hmag #stellar magnitude 
	field_config_dict['Hmag'] = Hmag #stellar magnitude 
	field_config_dict['Kmag'] = Kmag #stellar magnitude 
	field_config_dict['field_Nwvl_bins'] = field_Nwvl_bins # 
	
	

	

with col3:
	st.markdown('phase mask setup')

	A = st.text_input("off-axis transparency (0-1)", value=1, label_visibility=st.session_state.visibility,\
	disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key='OL_A_input')

	try:
		A = float(A)
	except:
		st.markdown(':red[off-axis transparency needs to be input as a float between 0-1]')

	
	
	B = st.text_input("on-axis transparency (0-1)", value=1, label_visibility=st.session_state.visibility,\
	disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key='OL_B_input')

	try:
		B = float(B)
	except:
		st.markdown(':red[on-axis transparency needs to be input as a float between 0-1]')

	phasemask_config_dict['on-axis_transparency'] = A
	
	glass_on = st.selectbox('glass on-axis',('sio2', 'su8'), index=0 ,key='OL_glasson_input')

	glass_off = st.selectbox('glass off-axis',('sio2', 'su8'), index=0 ,key='OL_glassoff_input')

	
	#desired_phase_shift = st.text_input("desired phase shift (deg)", value=90, label_visibility=st.session_state.visibility,\
	#disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key='OL_phaseshift_input')

	#try:
	#	desired_phase_shift = float(desired_phase_shift)
	#except:
	#	st.markdown(':red[desired phase shift needs to be input as a float]')
	
	z_on = st.text_input("on-axis phasemask depth (um)", value=21, label_visibility=st.session_state.visibility,\
	disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key='OL_zon')

	try:
		z_on = float(z_on)
	except:
		st.markdown(':red[z_on needs to be input as a float]')
		
	z_off = st.text_input("off-axis phasemask depth (um)", value=21-1.6/4, label_visibility=st.session_state.visibility,\
	disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key='OL_zoff')

	try:
		z_off = float(z_off)
	except:
		st.markdown(':red[z_off needs to be input as a float]')

	f_ratio = st.text_input("f-ratio", value=21, label_visibility=st.session_state.visibility,\
	disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key='OL_fratio_input')

	try:
		f_ratio = float(f_ratio)
	except:
		st.markdown(':red[f_ratio needs to be input as a float]')


	diam_lam_o_D = st.text_input(r'phase mask diameter ($\lambda/D$)', value=1, label_visibility=st.session_state.visibility,\
	disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key='OL_FPMdiam_input')

	try:
		diam_lam_o_D = float(diam_lam_o_D)
	except:
		st.markdown(':red[phase mask diameter needs to be input as a float]')
		
	
	phasemask_config_dict['off-axis_transparency'] = A #
	phasemask_config_dict['on-axis_transparency'] = B #
	phasemask_config_dict['on-axis_glass'] = glass_on #material type
	phasemask_config_dict['off-axis_glass'] = glass_off #material type
	phasemask_config_dict['on-axis phasemask depth'] = z_on # um 
	phasemask_config_dict['off-axis phasemask depth'] = z_off # um 
	#phasemask_config_dict['desired_phase_shift'] = desired_phase_shift
	phasemask_config_dict['fratio'] = f_ratio # 
	phasemask_config_dict['phasemask_diameter'] = diam_lam_o_D # lambda/D
	
	# hard coded 
	phasemask_config_dict['N_samples_across_phase_shift_region'] = N_samples_across_phase_shift_region # pixels
	phasemask_config_dict['nx_size_focal_plane'] = nx_size_focal_plane

with col4:
	st.markdown('DM setup')

	N_act = st.text_input("Number of actuators (x)", value=12, label_visibility=st.session_state.visibility,\
	disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key='OL_Nact_input')   

	try:
		N_act = int(N_act)
	except:
		st.markdown(':red[number of actuators in x needs to be input as an int > 0]')
	
	surface_type = st.selectbox('DM surface type',['continuous','segmented'], index=0, key='OL_dmsurfaceType_input')

	DM_config_dict['N_act'] = N_act 
	DM_config_dict['surface_type'] = surface_type
   
with col5:
	st.markdown('detector setup')

	#npix_det = st.text_input("Nx pixels on detector", value=12, label_visibility=st.session_state.visibility,\
	#disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key='OL_npixdet_input')

	#try:
	#	npix_det = int(npix_det)
	#except:
	#	st.markdown(':red[Nx pixels on detector needs to be input as an int > 0]')
	
	DIT = st.text_input("detector integration time (s)", value=0.001, label_visibility=st.session_state.visibility,\
	disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key='OL_dit_input')

	try:
		DIT = float(DIT)
	except:
		st.markdown(':red[detector integration time needs to be input as a float > 0]')

	ron = st.text_input("read out noise (e-)", value=1, label_visibility=st.session_state.visibility,\
	disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key='OL_ron_input')

	try:
		ron = float(ron)
	except:
		st.markdown(':red[read out noise needs to be input as a float > 0]')

	qe = st.text_input("quantum efficiency", value=1, label_visibility=st.session_state.visibility,\
	disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key='OL_qe_input')

	try:
		qe = float(qe)
	except:
		st.markdown(':red[quantum efficiency needs to be input as a float between 0-1]')
		
	
	wvl_min =  st.text_input("min wavelength (um)", value=1, label_visibility=st.session_state.visibility,\
	disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key='OL_wvlmin_input')

	try:
		wvl_min = float(wvl_min)
	except:
		st.markdown(':red[min wavelength needs to be input as a float in micrometers]')

	
	wvl_max = st.text_input("max wavelength (um)", value=2, label_visibility=st.session_state.visibility,\
	disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key='OL_wvlmax_input')

	try:
		wvl_max = float(wvl_max)
	except:
		st.markdown(':red[max wavelength needs to be input as a float in micrometers]')


	detector_config_dict['detector_npix'] = npix_det #pixels
	detector_config_dict['DIT'] = DIT #s
	detector_config_dict['ron'] = ron #e-
	detector_config_dict['quantum_efficiency'] = qe #
	
	detector_config_dict['det_wvl_min'] = wvl_min # um
	detector_config_dict['det_wvl_max'] = wvl_max # um
	
	
with col6:
	st.markdown('calibration source')
	cal_source_type = st.selectbox('calibration source type',['blackbody','flat'], index=0, key='OL_calibrationtype')
	
	cal_source_temp = st.text_input("source temperature (K)", value=6000, label_visibility=st.session_state.visibility,\
	disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key='OL_calsourcetemp')

	try:
		cal_source_temp = float(cal_source_temp)
	except:
		st.markdown(':red[source temperature needs to be input as a float in degrees Kelvin]')

	
	calibration_source_config_dict['type'] = cal_source_type 
	calibration_source_config_dict['temperature'] = cal_source_temp
	

config_dict_list = [tel_config_dict, field_config_dict, phasemask_config_dict, DM_config_dict, detector_config_dict,calibration_source_config_dict]
config_dict_namelist = ['telescope', 'field', 'phasemask', 'DM', 'detector', 'calibration_source'] 

for config_dict, label in zip(config_dict_list,  config_dict_namelist):
	config_file_name = st.text_input(f"{label} configuration file name", value=f'my_{label}_config_X.json', label_visibility=st.session_state.visibility,\
		disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key=f'{label}_config_file_name')

	json_string = json.dumps(config_dict)
	st.json(json_string, expanded=True)
	st.download_button(
		label=f"Download {label} configuration file",
		file_name= config_file_name,
		mime="application/json",
		data=json_string,
	)

