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
    
    Nwvl_bins = st.text_input("wavelength bins", value=3, label_visibility=st.session_state.visibility,\
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
    
    glass_off = st.selectbox('glass on-axis',('sio2', 'su8'), index=0 ,key='OL_glassoff_input')
    
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
    st.markdown('detector setup')

    npix_det = st.text_input("Nx pixels on detector", value=12, label_visibility=st.session_state.visibility,\
    disabled=st.session_state.disabled, placeholder=st.session_state.placeholder,key='OL_npixdet_input')
    
    try:
        npix_det = int(npix_det)
    except:
        st.markdown(':red[Nx pixels on detector needs to be input as an int > 0]')
            
    DIT = st.text_input("detector integration time (s)", value=1, label_visibility=st.session_state.visibility,\
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
        
        
#some hard coded values 
wvls = 1e-6 * np.linspace( wvl_min, wvl_max , Nwvl_bins )
dx = D/float(D_pix) # grid spatial element (m/pix)

pw = 20
npix_det = D_pix//pw # number of pixels across detector 
pix_scale_det = dx * pw # m/pix

extinction =  0.18
N_samples_across_phase_shift_region = 10
nx_size_focal_plane = dim

basis = zernike.zernike_basis(nterms=20, npix=D_pix)

basis_i2name = {zernike.zern_name(i):i for i in range(1,20)}
basis_name2i = {v:k for k,v in basis_i2name.items()}

# calculating parameters 
ph_flux_H = baldr.star2photons('H', Hmag, airmass=airmass, k = extinction, ph_m2_s_nm = True) # convert stellar mag to ph/m2/s/nm 

pup = baldr.pick_pupil(pupil_geometry=pup_geometry, dim=D_pix, diameter=D_pix) # baldr.AT_pupil(dim=D_pix, diameter=D_pix) #telescope pupil


if mode == 'Kolmogorov':
    
    r0 = 0.1 # Fried parameter (m)
    L0 = 25 # turbulent outerscale (m)
    
    screens = aotools.turbulence.infinitephasescreen.PhaseScreenVonKarman(D_pix, pixel_scale=dx,\
          r0=r0 , L0=L0, n_columns=2,random_seed = 1) # phase screen (radians)
    phase = screens.scrn
    
else:
    phase = basis[basis_name2i[mode]]
    
epsilon = 0.5
input_phases = np.array( [ np.nan_to_num( phase ) * (500e-9/w)**(6/5) for w in wvls] )

input_fluxes = [ph_flux_H * pup  for _ in wvls] # ph_m2_s_nm

input_field = baldr.field( phases = epsilon * input_phases  , fluxes = input_fluxes  , wvls=wvls )
input_field.define_pupil_grid(dx=dx, D_pix=D_pix)

st.title('Input Field')

fig_1_OL     = go.FigureWidget()
trace   = go.Heatmap()

trace.x = input_field.x
trace.y = input_field.y
trace.z = np.real( phase )

trace.colorbar.ticks     = 'outside'
fig_1_OL.add_trace(trace)
fig_1_OL.layout.height = 640
fig_1_OL.layout.yaxis.title = 'y'
fig_1_OL.layout.xaxis.title = 'x'
fig_1_OL.layout.xaxis.zeroline = False
fig_1_OL.layout.xaxis.showgrid = False
fig_1_OL.layout.yaxis.zeroline = False
fig_1_OL.layout.yaxis.showgrid = False

st.plotly_chart(fig_1_OL)



     
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