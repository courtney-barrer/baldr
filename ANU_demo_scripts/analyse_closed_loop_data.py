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

filenames_list = glob.glob(data_path + 'closed_loop_on_*.fits')
filename = max(filenames_list, key=os.path.getctime)

#filename = 'closed_loop_on_static_aberration_disturb-kolmogorov_amp-0.1_PID-[1, 0, 0]_t-26-02-2024T15.37.53'#'closed_loop_on_static_aberration_disturb-kolmogorov_amp-0.1_PID-[1, 0, 0]_t-26-02-2024T15.23.30.fits' #'closed_loop_on_static_aberration_disturb-kolmogorov_amp-0.1_PID-[1, 0, 0]_t-26-02-2024T13.22.01.fits' #'closed_loop_on_static_aberration_disturb-kolmogorov_amp-0.1_PID-[1, 0, 0]_t-26-02-2024T13.22.01.fits'
tstamp = filename.split('_t-')[-1] #tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

data = fits.open( filename )

# pupil cropping coordinates
cp_x1,cp_x2,cp_y1,cp_y2 = data['IMAGES'].header['cp_x1'],data['IMAGES'].header['cp_x2'],data['IMAGES'].header['cp_y1'],data['IMAGES'].header['cp_y2']
# PSF cropping coordinates 
ci_x1,ci_x2,ci_y1,ci_y2 = data['IMAGES'].header['ci_x1'],data['IMAGES'].header['ci_x2'],data['IMAGES'].header['ci_y1'],data['IMAGES'].header['ci_y2']
 
psf_corners=[ci_x1,ci_x2,ci_y1,ci_y2]
pupil_corners = [cp_x1,cp_x2,cp_y1,cp_y2]

open_loop_iterations = int( data['IMAGES'].header['open_loop_iter'] ) #  how long we kept in open loop before closing 


RMSE = data['RMS'].data
CMD_RMS = np.std(data['CMDS'].data, axis=1)
psf_max = np.array( [ np.max( psf[ci_x1:ci_x2,ci_y1:ci_y2] )  for psf in data['IMAGES'].data[1:] ] ) # 1st image is reference with no disturbance
psf_max_ref = np.max( data['IMAGES'].data[0,ci_x1:ci_x2,ci_y1:ci_y2]  )

fig, ax = plt.subplots(3,1, sharex=True)
ax[0].plot( RMSE ) 
ax[0].set_ylabel('RMSE in CMD SPACE')

ax[1].plot( CMD_RMS ) 
ax[1].set_ylabel('DM CMD RMS')
 
ax[2].plot( psf_max/psf_max_ref )
ax[2].set_ylabel('max(I) / max(I_ref)')
ax[2].set_xlabel('iteration')

for axx in ax:
   axx.axvline( open_loop_iterations, color='k',linestyle=':', label='close loop')  
   axx.legend()

plt.figure()
plt.imshow(data['CMDS'].data[1:] - data['DISTURBANCE'].data[1:] )
plt.colorbar()
plt.xlabel('actuator')
plt.ylabel('iteration')




# how far before and after we close the loop should we plot?
iii = int( input('how many iterations do you want to before loop was closed?') )
jjj = int( input('how many iterations do you want to after loop was closed?') )
if open_loop_iterations - iii < 0:
    iii=0

# plot the images 
plt.figure()
plt.imshow( np.rot90( bdf.get_DM_command_in_2D(data['DISTURBANCE'].data[0]).T,2) )
#plt.savefig(fig_path + f'static_aberration_{tstamp}.png') 
##plt.savefig(fig_path + f'dynamic_aberration_{tstamp}.png') 
plt.show()
fig,ax = plt.subplots( iii+jjj,3,figsize=(5,iii+jjj) )
plt.subplots_adjust(hspace=0.5,wspace=0.5)
for i, (im, err, cmd) in enumerate( zip(data['IMAGES'].data[open_loop_iterations-iii:open_loop_iterations + jjj], data['ERRS'].data[open_loop_iterations-iii:open_loop_iterations + jjj], data['CMDS'].data[open_loop_iterations-iii:open_loop_iterations + jjj])):
    imA = ax[i,0].imshow( im[160:260, 100:200] )
    errA = ax[i,1].imshow(np.rot90( bdf.get_DM_command_in_2D(err).T,2))
    cmdA = ax[i,2].imshow(np.rot90( bdf.get_DM_command_in_2D(cmd).T,2))
    for A,axx in zip([imA,errA,cmdA], [ax[i,0],ax[i,1],ax[i,2]]): 
        plt.colorbar(A, ax=axx)
        axx.axis('off')
    ax[i,0].set_title(f'image {i}')
    ax[i,1].set_title(f'err {i}') 
    ax[i,2].set_title(f'cmd {i}') 
#plt.savefig(fig_path + f'closed_loop_iterations_{tstamp}.png') 
plt.show()



