from baldr_control import ZWFS
from baldr_control import phase_control
from baldr_control import pupil_control
from baldr_control import utilities as util

import numpy as np
import matplotlib.pyplot as plt 
import time 
import datetime
from astropy.io import fits

data_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/data/' 

# timestamp
tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

sw = 8 # 8 for 12x12, 16 for 6x6 
pupil_crop_region = [157-sw, 269+sw, 98-sw, 210+sw ]

zwfs = ZWFS.ZWFS(DM_serial_number='17DW019#053', cameraIndex=0, DMshapes_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/DMShapes/', pupil_crop_region=pupil_crop_region ) 

data = util.scan_detector_framerates(zwfs, frame_rates=[100,200,300,400,450,500,550,600], number_images_recorded_per_cmd = 1000, cropping_corners=pupil_crop_region, save_fits = None)#data_path +  f'background_detector_statistics_lights_on_{tstamp}.fits')

# no source , only background light 

fps = np.array( [d.header['camera_fps'] for d in data] ) # frames per second
dit = 1e3 * np.array( [float(d.header['camera_tint']) for d in data] ) #ms

expected_var = np.array( [np.mean( np.var( d.data ,axis=0) ) for d in data] ) 
expected_var_uncert = np.array( [np.var( np.var( d.data ,axis=0) ) for d in data] )

expected_signal = np.array( [np.mean( np.mean( d.data ,axis=0) ) for d in data] ) 


fig,ax = plt.subplots(1,1,figsize=(8,5));
ax=['',ax]

ax[1].errorbar( dit, expected_var**0.5, yerr=expected_var_uncert**0.5/np.sqrt(data[0].data.shape[0] *  data[0].data.shape[1])  , label=r'$<\sigma(I_{pixel})>$') 
ax[1].plot( dit, np.sqrt(expected_signal), label=r'$\sqrt{<I_{pixel}>}$' ) 

ax[1].set_xlabel('DIT [ms]',fontsize=15)
ax[1].set_ylabel(r'$<\sigma(I_{pixel})>$ [adu]',fontsize=15)
ax[1].legend(fontsize=12)
ax[1].tick_params(labelsize=15)
#plt.savefig(data_path + f'background_detector_statistics_lights_off_{tstamp}.png', dpi=300, bbox_inches='tight' )
plt.show()

# variance structure 
i,j = 5,0
im_list = [ np.var( data[i].data ,axis=0)/expected_signal[i], np.var( data[j].data ,axis=0)/expected_signal[j]  ]
xlabel_list = ['col pixels', 'col pixels']
ylabel_list = ['row pixels','row pixels']
title_list = [f'DIT={round(dit[i],2)}ms', f'DIT={round(dit[j],2)}ms']
cbar_label_list = [r'$<\sigma^2(I_{pixel})>/<I_{pixel}>$',r'$<\sigma^2(I_{pixel})>/<I_{pixel}>$' ] 
savefig = None # data_path + f'structure_on_background_detector_statistics_lights_off_{tstamp}.png'

util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)




