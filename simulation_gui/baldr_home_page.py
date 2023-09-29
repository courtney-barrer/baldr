import streamlit as st
import numpy as np 
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pyzelda.utils.zernike as zernike
import os 
os.chdir('/Users/bcourtne/Documents/ANU_PHD2/baldr')
from functions import baldr_functions_2 as baldr


#B_over_A = st.slider('B/A', min_value=0, max_value=100, value=1, step=1, format=None, key='B_over_A_for_interactive_plot')

#theta = st.slider('Phase shift (deg)', min_value=0, max_value=360, value=90, step=1, format=None, key='theta_for_interactive_plot')

dim = 12*20
wvls = np.array( [1.65e-6] )
stellar_dict = { 'Hmag':6,'r0':0.1,'L0':25,'V':50,'airmass':1.3,'extinction': 0.18 }
tel_dict = { 'dim':dim,'D':1.8,'D_pix':12*20,'pup_geometry':'AT' }
#P0 = baldr.pick_pupil('disk',dim=dim+2*pad, diameter=diam)
#psf_diffraction_limit = np.fft.fftshift(np.fft.fft2(P0))

locals().update(stellar_dict)
locals().update(tel_dict)

dx = D/D_pix # grid spatial element (m/pix)
ph_flux_H = baldr.star2photons('H', Hmag, airmass=airmass, k = extinction, ph_m2_s_nm = True) # convert stellar mag to ph/m2/s/nm 

pup = baldr.pick_pupil(pupil_geometry='disk', dim=D_pix, diameter=D_pix) # baldr.AT_pupil(dim=D_pix, diameter=D_pix) #telescope pupil

basis = zernike.zernike_basis(nterms=20, npix=D_pix)


input_phases = [np.nan_to_num(basis[7]) * (500e-9/w)**(6/5) for w in wvls]
calibration_phases = [np.nan_to_num(basis[0])  for w in wvls]

input_fluxes = [ph_flux_H * pup  for _ in wvls] # ph_m2_s_nm

input_field = baldr.field( phases = input_phases  , fluxes = input_fluxes  , wvls=wvls )

input_field.define_pupil_grid(dx=dx, D_pix=D_pix)



#Psi_A = input_field.flux[wvls[0]] * np.exp(1j * input_field.phase[wvls[0]])



FPM = baldr.zernike_phase_mask(A=1, B=1, phase_shift_diameter=1e-6, f_ratio=21, d_on=26.5e-6, d_off=26e-6, glass_on='sio2', glass_off='sio2')
desired_phase_shift = 90
FPM.optimise_depths( desired_phase_shift=desired_phase_shift, across_desired_wvls=wvls , fine_search = False, verbose=True)     
thetas = np.deg2rad( np.array([FPM.phase_mask_phase_shift( w ) for w in wvls]) ) 
H = {w: FPM.get_filter_design_parameter( w ) for w in wvls }       

nx_size_focal_plane = input_field.nx_size

dx_focal_plane = FPM.phase_shift_diameter/10

FPM.sample_phase_shift_region( nx_pix=nx_size_focal_plane, dx=dx_focal_plane, verbose=False) 
#Fourier trnsofrm field and then 

FPM.get_output_field(input_field , wvl_lims=[-np.inf, np.inf] , nx_size_focal_plane = None, dx_focal_plane=None, cold_stop_diam = None, keep_intermediate_products=True)

#plt.imshow(np.real( FPM.combined_filter_parameter * FPM.b[0] ))

#plt.imshow(np.imag( FPM.combined_filter_parameter * FPM.b[0] ))



#fig,ax = plt.subplots(

fig     = go.FigureWidget()
trace   = go.Heatmap()

trace.x = input_field.x
trace.y = input_field.y
trace.z = np.real( FPM.combined_filter_parameter * FPM.b[0] )

trace.colorbar.ticks     = 'outside'
fig.add_trace(trace)
fig.layout.height = 640
fig.layout.yaxis.title = 'frequency [Hz]'
fig.layout.xaxis.title = 'date'
fig.layout.xaxis.zeroline = False
fig.layout.xaxis.showgrid = False
fig.layout.yaxis.zeroline = False
fig.layout.yaxis.showgrid = False

st.plotly_chart(fig)


fig     = go.FigureWidget()
trace   = go.Heatmap()

trace.x = input_field.x
trace.y = input_field.y
trace.z = np.imag( FPM.combined_filter_parameter * FPM.b[0] )

trace.colorbar.ticks     = 'outside'
fig.add_trace(trace)
fig.layout.height = 640
fig.layout.yaxis.title = 'frequency [Hz]'
fig.layout.xaxis.title = 'date'
fig.layout.xaxis.zeroline = False
fig.layout.xaxis.showgrid = False
fig.layout.yaxis.zeroline = False
fig.layout.yaxis.showgrid = False

st.plotly_chart(fig)
    
"""
import streamlit as st
import pyvista as pv
import numpy as np
from stpyvista import stpyvista

"# 🧱 Structured grid"

## Create coordinate data
x = np.arange(-10, 10, 0.25)
y = np.arange(-10, 10, 0.25)
x, y = np.meshgrid(x, y)
z = np.sin(np.sqrt(x**2 + y**2))

## Set up plotter
plotter = pv.Plotter(window_size=[600,600])
surface = pv.StructuredGrid(x, y, z)
plotter.add_mesh(surface, color='teal', show_edges=True)

## Pass the plotter (not the mesh) to stpyvista
stpyvista(plotter, key="surface")
"""