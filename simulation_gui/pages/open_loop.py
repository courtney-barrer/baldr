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

os.chdir('/Users/bcourtne/Documents/ANU_PHD2/baldr')
from functions import baldr_functions_2 as baldr


col1, col2, col3, col4, col5 = st.columns(5)

if 'key' not in st.session_state:
    st.session_state['key'] = 'value'
    
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False
    st.session_state.placeholder = 'write here'
    



with col1:
	st.markdown('grid & pupil setup')

	dim = st.text_input("Nx pixels", value=12*20, label_visibility=st.session_state.visibility,\
	disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key='OL_dim_input')

	try:
		dim = int(dim)
	except:
		st.markdown(':red[Nx pixels needs to be input as an integer]')

	D = st.text_input("telescope diameter (m)", value=1.8, label_visibility=st.session_state.visibility,\
	disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key='OL_D_input')

	try:
		D = float(D)
	except:
		st.markdown(':red[telescope diameter needs to be input as a float]')

	D_pix = st.text_input("pixels across pupil", value=12*20, label_visibility=st.session_state.visibility,\
	disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key='OL_Dpix_input')

	try:
		D_pix = int(D_pix)
	except:
		st.markdown(':red[pixels across pupil needs to be input as an int > 0]')

	pup_geometry = st.selectbox('Pupil geometry',('disk', 'AT', 'UT'), index=0, key='OL_geompupil_input')


with col2:
	st.markdown('input field setup')

	Nwvl_bins = st.text_input("wavelength bins", value=10, label_visibility=st.session_state.visibility,\
	disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key='OL_wbins_input')

	try:
		Nwvl_bins = int(Nwvl_bins)
	except:
		st.markdown(':red[wavelength bins needs to be input as an int > 0]')

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


	mode = st.selectbox('input phase mode',['Kolmogorov'] + [zernike.zern_name(i) for i in range(1,20)], index=0, key='OL_phase_input')


	Hmag = st.text_input("Hmag", value=2, label_visibility=st.session_state.visibility,\
	disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key='OL_Hmag_input')

	try:
		Hmag = float(Hmag)
	except:
		st.markdown(':red[Hmag needs to be input as a float]')

	airmass = st.text_input("airmass", value=1.3, label_visibility=st.session_state.visibility,\
	disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key='OL_airmas_input')

	try:
		airmass = float(airmass)
	except:
		st.markdown(':red[airmass needs to be input as a float]')



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

	glass_on = st.selectbox('glass on-axis',('sio2', 'su8'), index=0 ,key='OL_glasson_input')

	glass_off = st.selectbox('glass off-axis',('sio2', 'su8'), index=0 ,key='OL_glassoff_input')

	f_ratio = st.text_input("f-ratio", value=21, label_visibility=st.session_state.visibility,\
	disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key='OL_fratio_input')

	try:
		f_ratio = float(f_ratio)
	except:
		st.markdown(':red[f_ratio needs to be input as a float]')

	desired_phase_shift = st.text_input("desired phase shift (deg)", value=90, label_visibility=st.session_state.visibility,\
	disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key='OL_phaseshift_input')

	try:
		desired_phase_shift = float(desired_phase_shift)
	except:
		st.markdown(':red[desired phase shift needs to be input as a float]')

	diam_lam_o_D = st.text_input(r'phase mask diameter ($\lambda/D$)', value=1, label_visibility=st.session_state.visibility,\
	disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key='OL_FPMdiam_input')

	try:
		diam_lam_o_D = float(diam_lam_o_D)
	except:
		st.markdown(':red[phase mask diameter needs to be input as a float]')

with col4:
	st.markdown('DM setup')

	N_act = st.text_input("Number of actuators (x)", value=12, label_visibility=st.session_state.visibility,\
	disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key='OL_Nact_input')   

	try:
		N_act = int(N_act)
	except:
		st.markdown(':red[number of actuators in x needs to be input as an int > 0]')
		
	surface_type = st.selectbox('DM surface type',['continuous','segmented'], index=0, key='OL_dmsurfaceType_input')

	   
with col5:
	st.markdown('detector setup')

	npix_det = st.text_input("Nx pixels on detector", value=12, label_visibility=st.session_state.visibility,\
	disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key='OL_npixdet_input')

	try:
		npix_det = int(npix_det)
	except:
		st.markdown(':red[Nx pixels on detector needs to be input as an int > 0]')
		
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


epsilon = st.slider('phase scaling', 0.0, 10.0, 0.1, key='OL_epsilon_input')

#some hard coded values 
wvls = 1e-6 * np.linspace( wvl_min, wvl_max , Nwvl_bins )
wvl0 = wvls[len(wvls)//2] #center wavelength roughly 

dx = D/float(D_pix) # grid spatial element (m/pix)

pw = 20
npix_det = D_pix//pw # number of pixels across detector 
pix_scale_det = dx * pw # m/pix

extinction =  0.18
N_samples_across_phase_shift_region = 10
nx_size_focal_plane = dim

basis = zernike.zernike_basis(nterms=20, npix=D_pix)

basis_name2i = {zernike.zern_name(i):i for i in range(1,20)}
basis_i2name = {v:k for k,v in basis_name2i.items()}

# ============ set up input field  
ph_flux_H = baldr.star2photons('H', Hmag, airmass=airmass, k = extinction, ph_m2_s_nm = True) # convert stellar mag to ph/m2/s/nm 

pup = baldr.pick_pupil(pupil_geometry=pup_geometry, dim=D_pix, diameter=D_pix) # baldr.AT_pupil(dim=D_pix, diameter=D_pix) #telescope pupil
pup_nan = pup.copy()
#np.num2nan( ) pup_nan[pup<0.5] = np.nan

if mode == 'Kolmogorov':
    
    r0 = 0.1 # Fried parameter (m)
    L0 = 25 # turbulent outerscale (m)
    
    @st.cache_data
    def tmp_get_phase(D_pix,dx,r0,L0): # cached function to hold phase screen data 
        screens = aotools.turbulence.infinitephasescreen.PhaseScreenVonKarman(D_pix, pixel_scale=dx,\
            r0=r0 , L0=L0, n_columns=2,random_seed = 1) # phase screen (radians)
        phase = screens.scrn
        return(phase)
    
    phase = tmp_get_phase(D_pix,dx,r0,L0) 
    #epsilon = 0.8
    input_phases = epsilon * np.array( [ np.nan_to_num( phase ) * (500e-9/w)**(6/5) for w in wvls] )
   
else:
    phase = basis[basis_name2i[mode]-1]
    
    #epsilon = 5
    input_phases = epsilon * np.array( [ np.nan_to_num( phase ) * (500e-9/w)**(6/5) for w in wvls] )

input_fluxes = [ph_flux_H * pup  for _ in wvls] # ph_m2_s_nm

input_field = baldr.field( phases =  input_phases  , fluxes = input_fluxes  , wvls=wvls )
input_field.define_pupil_grid(dx=dx, D_pix=D_pix)



# ============ set up DM
dm = baldr.DM(surface=np.zeros([N_act,N_act]), gain=1, angle=0,surface_type = surface_type) 



# ============ WFS phase mask
# dot diameter (m)
phase_shift_diameter = diam_lam_o_D * f_ratio * wvl0   ##  f_ratio * wvls[0] = lambda/D  given f_ratio

# init depths before optimization based on desired_phase_shift
d_on = 26.5e-6
d_off = 26e-6

FPM = baldr.zernike_phase_mask(A,B,phase_shift_diameter,f_ratio, d_on, d_off, glass_on, glass_off)
FPM.optimise_depths(desired_phase_shift=desired_phase_shift, across_desired_wvls=wvls ,verbose=False)

# calibrator mask (zero phase shift)
FPM_cal = baldr.zernike_phase_mask(A,B,phase_shift_diameter,f_ratio,d_on,d_off,glass_on,glass_off)
FPM_cal.d_off = FPM_cal.d_on


# ============ set up detector object
det = baldr.detector(npix=npix_det, pix_scale = pix_scale_det , DIT= DIT, ron=ron, QE={w:qe for w in wvls})
#mask = baldr.pick_pupil(pupil_geometry='disk', dim=det.npix, diameter=det.npix) 


sig = baldr.detection_chain(input_field, dm, FPM, det)

sig_cal = baldr.detection_chain(input_field, dm, FPM_cal, det)

b = input_field.calculate_b(FPM)

phi_reco = baldr.reco(input_field, sig, sig_cal, FPM, det, b, order='first')
phi_reco[~np.isfinite(phi_reco)] = 0 # set non-finite values to zero in cmd 

cmd =  wvl0/(2*np.pi) * (phi_reco - np.mean(phi_reco)).reshape(-1)

dm.update_shape(cmd = cmd)
input_field_dm = input_field.applyDM(dm)


fig_1_OL = plt.figure(figsize=(24, 6))
    
ax1 = fig_1_OL.add_subplot(141)
ax1.set_title('input phase',fontsize=20)
ax1.axis('off')
im1 = ax1.imshow( input_field.phase[wvl0])
ax1.text(dim/3,dim/3, f'strehl = {round(np.exp(-np.var( input_field.phase[wvl0][pup>0.5] )), 2)}',fontsize=20, color='white')    
divider = make_axes_locatable(ax1)
cax = divider.append_axes('bottom', size='5%', pad=0.05)
cbar = fig_1_OL.colorbar( im1, cax=cax, orientation='horizontal')
cbar.set_label( 'phase [rad]', rotation=0)
    
ax2 = fig_1_OL.add_subplot(142)
ax2.set_title('detector',fontsize=20)
ax2.axis('off')
im2 = ax2.imshow(sig.signal)
    
divider = make_axes_locatable(ax2)
cax = divider.append_axes('bottom', size='5%', pad=0.05)
cbar = fig_1_OL.colorbar(im2, cax=cax, orientation='horizontal')
cbar.set_label( 'Intensity [adu]', rotation=0)

ax3 = fig_1_OL.add_subplot(143)
ax3.set_title('DM',fontsize=20)
ax3.axis('off')
im3 = ax3.imshow(dm.surface)
    
divider = make_axes_locatable(ax3)
cax = divider.append_axes('bottom', size='5%', pad=0.05)
cbar = fig_1_OL.colorbar(im3, cax=cax, orientation='horizontal')
cbar.set_label( 'OPL [m]', rotation=0)

ax4 = fig_1_OL.add_subplot(144)
ax4.set_title('corrected phase',fontsize=20)
ax4.axis('off')
im4 = ax4.imshow(input_field_dm.phase[wvl0])
ax4.text(dim/3, dim/3, f'strehl = {round(np.exp(-np.var( input_field_dm.phase[wvl0][pup>0.5] )), 2)}',fontsize=20, color='white')    
divider = make_axes_locatable(ax4)
cax = divider.append_axes('bottom', size='5%', pad=0.05)
cbar = fig_1_OL.colorbar(im4, cax=cax, orientation='horizontal')
cbar.set_label( 'phase [rad]', rotation=0)


st.pyplot(fig_1_OL)


#st.markdown( phi_reco )


st.title('Input Field')

#fig_1_OL, ax_1_OL = plt.subplots(1,2,sharey=False)
#ax_1_OL[0].pcolormesh(input_field.x, input_field.y,  pup * input_field.phase[wvl0])
#ax_1_OL[1].pcolormesh(input_field.x, input_field.y,  pup * input_field.flux[wvl0])

fig_2_OL = plt.figure(figsize=(12, 6))
    
ax1 = fig_2_OL.add_subplot(121)
ax1.set_title('input phase',fontsize=20)
ax1.axis('off')
im1 = ax1.pcolormesh(input_field.x, input_field.y,  pup_nan * input_field.phase[wvl0])
    
divider = make_axes_locatable(ax1)
cax = divider.append_axes('bottom', size='5%', pad=0.05)
cbar = fig_2_OL.colorbar( im1, cax=cax, orientation='horizontal')
cbar.set_label( 'phase [rad]', rotation=0)
    
ax2 = fig_2_OL.add_subplot(122)
ax2.set_title('input flux',fontsize=20)
ax2.axis('off')
im2 = ax2.pcolormesh(input_field.x, input_field.y,  pup_nan * input_field.flux[wvl0])
    
divider = make_axes_locatable(ax2)
cax = divider.append_axes('bottom', size='5%', pad=0.05)
cbar = fig_2_OL.colorbar(im2, cax=cax, orientation='horizontal')
cbar.set_label( 'flux [ph/s/m^2/nm]', rotation=0)

st.pyplot(fig_2_OL)



st.title('Phase Mask')

#plot results
thetas = np.array( [FPM.phase_mask_phase_shift( w ) for w in wvls] ) #degrees

fig_2_OL = plt.figure()
plt.plot(1e6 * wvls,  thetas, color='k')
plt.ylabel('phase shift [deg] ',fontsize=15)
plt.xlabel('wavelength '+r'[$\mu$m]',fontsize=15)
plt.grid()
plt.gca().tick_params(labelsize=15)
st.pyplot(fig_2_OL)

fig_3_OL = plt.figure()
z = np.ones(FPM.x_focal_plane.shape) * FPM.d_off
z[abs(FPM.x_focal_plane) < FPM.phase_shift_diameter/2] = FPM.d_on
plt.plot( 1e6 * FPM.x_focal_plane, 1e6 * z, color='b',lw=2)
plt.fill_between(x=1e6 * FPM.x_focal_plane, y1=0, y2=1e6 * z , color='b',alpha=0.4)
plt.ylim( [0 , max(1e6 * z)*1.2] )
plt.xlim( [5e6 * -FPM.phase_shift_diameter , 5e6 * FPM.phase_shift_diameter] )
plt.axvline( 1e6 * -FPM.phase_shift_diameter/2, color='k', linestyle=':' ,lw=0.5)
plt.axvline( 1e6 * FPM.phase_shift_diameter/2, color='k', linestyle=':' ,lw=0.5)
diam_lam_o_D = 1 #!!!!!!
plt.text(0,1e6 * FPM.d_on * 1.1, f'core diameter={ round( 1e6 * FPM.phase_shift_diameter, 2) }um ({diam_lam_o_D}'+r'$\lambda$/D)')

plt.text(1e6 * -FPM.phase_shift_diameter/8 ,1e6 * FPM.d_on / 2, r'$z_{on}$'+f'={ round( 1e6 * FPM.d_on, 2) }um' ,rotation=90)

plt.text(1e6 * FPM.phase_shift_diameter/1.5 , 1e6 * FPM.d_on / 2, r'$z_{off}$'+f'={ round( 1e6 * FPM.d_off, 2) }um' ,rotation=90 )

plt.title(f'f-ratio = {f_ratio}')
plt.xlabel(r'x [$\mu m$]')
plt.ylabel(r'z [$\mu m$]')
st.pyplot(fig_3_OL)

""" 
dim = 12*20
stellar_dict = { 'Hmag':2,'r0':0.1,'L0':25,'V':50,'airmass':1.3,'extinction': 0.18 }
tel_dict = { 'dim':dim,'D':1.8,'D_pix':12*20,'pup_geometry':'AT' }
naomi_dict = { 'naomi_lag':4.5e-3, 'naomi_n_modes':14, 'naomi_lambda0':0.6e-6,\
              'naomi_Kp':1.1, 'naomi_Ki':0.93 }
FPM_dict = {'A':1, 'B':1, 'f_ratio':21, 'd_on':26.5e-6, 'd_off':26e-6,\
            'glass_on':'sio2', 'glass_off':'sio2','desired_phase_shift':90,\
                'rad_lam_o_D':1.2 ,'N_samples_across_phase_shift_region':10,\
                    'nx_size_focal_plane':dim}
det_dict={'DIT' : 1, 'ron' : 0.5, 'pw' : 20, 'QE' : 1,'npix_det':12}
"""

"""
fig_1_OL = make_subplots(rows=1, cols=2)

fig_1_OL.add_trace(
   go.Heatmap(x=input_field.x, y = input_field.y, z=  pup * input_field.phase[wvl0]  ),
    row=1, col=1
)

fig_1_OL.add_trace(
   go.Heatmap( x=input_field.x, y = input_field.y, z=  pup * input_field.flux[wvl0]  ),
    row=1, col=2
)

# Update xaxis properties
fig_1_OL.update_xaxes(title_text="x [m]", row=1, col=1)
fig_1_OL.update_yaxes(title_text="y [m]", row=1, col=1)
# Update yaxis properties
fig_1_OL.update_xaxes(title_text="x [m]", row=1, col=2)
fig_1_OL.update_yaxes(title_text='y [m] ', row=1, col=2)

fig_1_OL.update_coloraxes(colorbar_orientation='h',row=1, col=1)  
fig_1_OL.update_coloraxes(colorbar_title_text='phase [rad]', row=1, col=1)   

fig_1_OL.update_coloraxes(colorbar_orientation='h',row=1, col=2)  
fig_1_OL.update_coloraxes(colorbar_title_text='flux [ph/m^2/s/nm]', row=1, col=2)   

#flux [ph/m^2/s/nm]
#fig_1_OL.update_traces(colorbar_orientation='h', selector=dict(type='heatmap'))
#fig_1_OL.layout.coloraxis.colorbar.title = 'input phase [rad]'

st.plotly_chart(fig_1_OL)
"""