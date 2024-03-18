import os 
import datetime 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import glob
from scipy import interpolate
from astropy.io import fits
import aotools
os.chdir('/opt/FirstLightImaging/FliSdk/Python/demo/')
import FliSdk_V2 

root_path = '/home/baldr/Documents/baldr'
data_path = root_path + '/ANU_demo_scripts/ANU_data/' 
fig_path = root_path + '/figures/' 

os.chdir(root_path)
from functions import baldr_demo_functions as bdf


filenames_list = glob.glob(data_path + 'closed_loop_on_static*.fits')
filename = max(filenames_list, key=os.path.getctime)

#filename = 'closed_loop_on_static_aberration_disturb-kolmogorov_amp-0.1_PID-[1, 0, 0]_t-26-02-2024T15.37.53'#'closed_loop_on_static_aberration_disturb-kolmogorov_amp-0.1_PID-[1, 0, 0]_t-26-02-2024T15.23.30.fits' #'closed_loop_on_static_aberration_disturb-kolmogorov_amp-0.1_PID-[1, 0, 0]_t-26-02-2024T13.22.01.fits' #'closed_loop_on_static_aberration_disturb-kolmogorov_amp-0.1_PID-[1, 0, 0]_t-26-02-2024T13.22.01.fits'
tstamp = filename.split('_t-')[-1] #tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

data = fits.open( filename )

# plot the images 
plt.figure()
plt.imshow( np.rot90( bdf.get_DM_command_in_2D(data['DISTURBANCE'].data).T,2) )
#plt.savefig(fig_path + f'static_aberration_{tstamp}.png') 
#plt.savefig(fig_path + f'dynamic_aberration_{tstamp}.png') 
plt.show()
fig,ax = plt.subplots( data['IMAGES'].data.shape[0],3,figsize=(5,1.5*data['IMAGES'].data.shape[0]) )
plt.subplots_adjust(hspace=0.5,wspace=0.5)
for i, (im, err, cmd) in enumerate( zip(data['IMAGES'].data, data['ERRS'].data, data['CMDS'].data[1:])):
    imA = ax[i,0].imshow( im[160:260, 100:200] )
    errA = ax[i,1].imshow(np.rot90( bdf.get_DM_command_in_2D(err).T,2))
    cmdA = ax[i,2].imshow(np.rot90( bdf.get_DM_command_in_2D(cmd).T,2))
    for A,axx in zip([imA,errA,cmdA], [ax[i,0],ax[i,1],ax[i,2]]): 
        plt.colorbar(A, ax=axx)
        axx.axis('off')
    ax[i,0].set_title(f'image {i}')
    ax[i,1].set_title(f'err {i}') 
    ax[i,2].set_title(f'cmd {i}') 
plt.savefig(fig_path + f'closed_loop_iterations_{tstamp}.png') 
plt.show()



