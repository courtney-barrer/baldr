"""
# questions 

1 - how to crop camera 

2 - how to set gain



"""
import numpy as np
import matplotlib.pyplot as plt
import time 
import sys 
import pickle 
sys.path.insert(1, '/opt/FirstLightImaging/FliSdk/Python/demo/')
sys.path.insert(1,'/opt/Boston Micromachines/lib/Python3/site-packages/')
import FliSdk_V2 as fli
import bmc

data_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/data/' 
fig_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/figures/' 

"""
class CroppingData(ctypes.Structure):
    _fields_ = [("col1", ctypes.c_int16),
                ("col2", ctypes.c_int16),
                ("row1", ctypes.c_int16),
                ("row2", ctypes.c_int16),
                ("cred1Cols", ctypes.c_bool * 10),
                ("cred1Rows", ctypes.c_bool * 256)]

CROP
========
To set the cropping window’s columns use the command:
“set cropping columns xxx-www” where xxx is the starting column and www the ending column.
To set its rows, use the command:
“set cropping rows yyy-zzz” where yyy is the starting row and zzz the ending row.
The defined cropped area is active only when the cropping is enabled. To switch it on or off, use the commands
“set cropping on” or “set cropping off” respectively.
As for any other commands, you can get the current value by repeating it without the “set” keyword.


GAIN
========
The signal can be integrated in low, medium or high gain, corresponding to high, medium and small integration
capacity respectively. Modifying the integration capacity impacts the dynamic of the signal and thus implies a
change of the noise level.
It is possible to modify the integration capacity using “set sensitivity low”, “set sensitivity
medium” or “set sensitivity high” commands in the command line interpreter.
"""



camera = fli.Init() # init camera object
listOfGrabbers = fli.DetectGrabbers(camera)
listOfCameras = fli.DetectCameras(camera)
# print some info and exit if nothing detected
if len(listOfGrabbers) == 0:
    print("No grabber detected, exit.")
    exit()
if len(listOfCameras) == 0:
    print("No camera detected, exit.")
    exit()
for i,s in enumerate(listOfCameras):
    print("- index:" + str(i) + " -> " + s)

cameraIndex = 0
print(f'--->using cameraIndex={cameraIndex}')
# set the camera
camera_err_flag = fli.SetCamera(camera, listOfCameras[cameraIndex])
if not camera_err_flag:
    print("Error while setting camera.")
    exit()
print("Setting mode full.")
fli.SetMode(camera, fli.Mode.Full)
print("Updating...")
camera_err_flag = fli.Update(camera)
if not camera_err_flag:
    print("Error while updating SDK.")
    exit()

fli.SetMode(camera, fli.Mode.Full)
fli.Update(camera)
fli.Start(camera)

#get image to test 
test = fli.GetRawImageAsNumpyArray( camera , -1)

fli.FliSerialCamera.SendCommand(camera, "set cropping off")
# cropped columns must be multiple of 32 - multiple of 32 minus 1
#fli.FliSerialCamera.SendCommand(camera, "set cropping columns 64-287")
fli.FliSerialCamera.SendCommand(camera, "set cropping columns 96-255")
# cropped rows must be multiple of 4 - multiple of 4 minus 1
#fli.FliSerialCamera.SendCommand(camera, "set cropping rows 120-299")
fli.FliSerialCamera.SendCommand(camera, "set cropping rows 152-267")
fli.FliSerialCamera.SendCommand(camera, "set cropping on")

camera_err_flag, maxfps = fli.FliSerialCamera.SendCommand(camera, "maxfps")

print( f'min fps {maxfps}') 

# set it
# 2800Hz frame rate 
fli.FliSerialCamera.SetFps(camera,float(maxfps.split(': ')[1]) )

print( 'current DIT = ', fli.FliSerialCamera.SendCommand(camera, "tint raw") )

camera_err_flag, minDIT = fli.FliSerialCamera.SendCommand(camera, "mintint raw")
camera_err_flag, maxDIT = fli.FliSerialCamera.SendCommand(camera, "maxtint raw")

print( f'max int {1e3 * float(maxDIT)}ms') 
print( f'min int {1e3 * float(minDIT)}ms') 

_,fps = fli.FliSerialCamera.GetFps(camera) #Hz 
_, DIT = fli.FliSerialCamera.SendCommand(camera, "tint raw") #s

# 3ms integration time (3.3kHz)
#fli.FliSerialCamera.SendCommand(camera, "set tint " + str(0.0003))
#fli.FliSerialCamera.SendCommand(camera, "set tint " + str(0.0001))

print( 'fps = ', fps, 'DIT=', DIT  ) 
# 4864Hz frame rate
 
# tag images to count frames
#   This the first and second pixels of the images are used to store a frame counter 
#   that increments by one for each frame acquired from the sensor. These two pixels 
#   are treated together as a 32 bits number, which will represent each specific frame’s number. 
#   The third pixel value depends on the current readout mode of the camera.
fli.FliSerialCamera.SendCommand(camera, "set imagetags on")
# lets see how many iterations it takes to get new frame 
frame_tag_test_list = []
for i in range(30):
    frame_tag_test_list.append( fli.GetRawImageAsNumpyArray( camera , -1) ) 

print( 'frame per iteration of taking image:', [f[0,0] for f in frame_tag_test_list] )

#new cropped image with 3.3kHz frame rate  
test = fli.GetRawImageAsNumpyArray( camera , -1)

# set the gain (sensitivity) 
fli.FliSerialCamera.SendCommand(camera, "set sensitivity high") #options: high, medium, low
img_high_sens = fli.GetRawImageAsNumpyArray( camera , -1)

# create and apply bias 
nb = 256 # number of frames for building bias 
fli.FliSerialCamera.SendCommand(camera, f"buildnuc bias {nb}")
fli.FliSerialCamera.SendCommand(camera, "set bias on")

time.sleep(0.1) 
img_bias_corrected = fli.GetRawImageAsNumpyArray( camera , -1)

fig,ax = plt.subplots(1,2) 
ax[0].imshow(img_high_sens) 
ax[0].set_title('pre bias corr.')
ax[1].imshow( img_bias_corrected ) 
ax[1].set_title('post bias corr.')
plt.show() 





"""
fli.FliSerialCamera.SendCommand(camera, "set sensitivity medium")
img_med_sens = fli.GetRawImageAsNumpyArray( camera , -1)

fli.FliSerialCamera.SendCommand(camera, "set sensitivity low")
img_low_sens = fli.GetRawImageAsNumpyArray( camera , -1)

plt.imshow( img_high_sens ) ; plt.show()
plt.imshow( img_low_sens ) ; plt.show()
"""

"""
testcrop = [("col1", 70),
 ("col2", 291),
 ("row1", 132),
 ("row2", 300),
 ("cred1Cols", ctypes.c_bool * 10),
 ("cred1Rows", ctypes.c_bool * 256)]
"""


### INIT DM 

dm = bmc.BmcDm() # init DM object
err_code = dm.open_dm('17DW019#053') # open DM
if err_code:
    print('Error initializing DM')
    raise Exception(dm.error_string(err_code))


### LATANCY TEST 

#actuator to poke
act_idx = 65
#differential amplitude to apply with poke
damp = 0.15
#DM cmds to swap between 
cmd_1 = 0.5 * np.ones(140)
cmd_2 = 0.5 * np.ones(140)
cmd_2[act_idx] -= damp
cmd_list = [cmd_1,cmd_2]

# get reference images with the DM in each state 
dm.send_data(cmd_list[0])
time.sleep(1)
I0 = fli.GetRawImageAsNumpyArray( camera , -1) 
time.sleep(1)
ref_img_1 = fli.GetRawImageAsNumpyArray( camera , -1) 
dm.send_data(cmd_list[1])
time.sleep(1)
ref_img_2 = fli.GetRawImageAsNumpyArray( camera , -1) 

# check reference images 
fig, ax = plt.subplots(1,2)
ax[0].imshow( ref_img_1 - I0  )
ax[0].set_title('flat')
ax[1].imshow( ref_img_2 - I0 )
ax[1].set_title('poke')
plt.show()

img_list = [] #list to hold images
t0_im_list=[] #begin to get last image
t1_im_list=[] #finished getting last frame
t0_dm_list=[] # begin to actuate DM
t1_dm_list=[] # finish actuating DM 
flag_list=[] #if DM is poked (1) or flat (0)

j=0 # used to calculate flag
flag=0 #jumps between 1, 0 for on, off pokes
change_every = 100 # update dm every X iterations 
for i in range(1000):
    t0_im_list.append( time.time() ) 
    img_list.append( fli.GetRawImageAsNumpyArray( camera , -1) ) 
    t1_im_list.append( time.time() )
    flag_list.append(flag) 
    #print(flag_list[-1])
    if np.mod(i,change_every)==0: # change DM state every 10 iterations
        #if img_list[-1][0,0] - img_list[-2][0,0] > 0: # 
        j+=1
        flag = np.mod(j,2)  
        #print(i, j, flag)
        t0_dm_list.append( time.time() )
        dm.send_data(cmd_list[flag])
        t1_dm_list.append( time.time() )

# frame number (first pixel in image encodes the frame number)
frame_no = [i[0,0] for i in img_list]

out_dict = {'fps':fps, 'DIT':DIT, 'frame_no':np.array(frame_no)-frame_no[0], 't0_img':list(np.array(t0_im_list)-t0_im_list[0]), 't1_img':list(np.array(t1_im_list)-t0_im_list[0]),'t0_dm':list(np.array(t0_dm_list)-t0_im_list[0]),'t1_dm':list(np.array(t1_dm_list)-t0_im_list[0]), 'I0':I0, 'img_list':img_list, 'flag_list':flag_list} 

with open(data_path + f'latency_test_FPS-{round(float(fps),1)}_DIT-{round(1e3*float(DIT),3)}ms_pokeDM_every_{change_every}iters.pickle','wb') as handle:
    pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

#%% =========================== read it in 
with open(data_path + f'latency_test_FPS-{round(float(fps),1)}_DIT-{round(1e3*float(DIT),3)}ms_pokeDM_every_{change_every}iters.pickle', 'rb') as handle:
    out_dict = pickle.load(handle)


fps = out_dict['fps']
DIT = out_dict['DIT']

frame_no = np.array(out_dict['frame_no'])

t0_img = np.array(out_dict['t0_img']) # begining of getting latest frame from camera 
t1_img = np.array(out_dict['t1_img']) # end of getting latest frame from camera 

t0_dm = np.array(out_dict['t0_dm']) # begining of poking dm
t1_dm = np.array(out_dict['t1_dm']) # end of poking dm

I0 = np.array(out_dict['I0']) # reference intensity with "flat" dm 
img_list = np.array(out_dict['img_list']) # list of saved images each iteration 

flag_list = np.array(out_dict['flag_list']) # flag to indicate if DM is poked or flat (jumps between 1, 0 for on, off pokes)


# check how long to update frame (=fps?)
plt.plot( t0_img[:40]*1e6, frame_no[:40] ,'.')
plt.xlabel('time (us)')
plt.ylabel('frame #')
plt.show()

# peak pixel change for act 65 around row 77,  col 97
#pv = np.array([i[77,97] - I0[77,97] for i in img_list])
pv = np.array([i[45,65] - I0[45,65] for i in img_list])
pv[pv > 60000] = 0

i0 = 100
i1 = 460 

plt.plot(t0_img[i0:i1], pv[i0:i1]/np.max(pv[i0:i1]) , '.',label='norm. pixel value')

new_frame_times = t1_img[np.where( np.diff( frame_no ) )] # at end of grabbing 
for i,trig in enumerate(new_frame_times):
    if i!=0:
        plt.axvline( trig , color='k',linestyle='-',alpha=0.9) 
    else:
        plt.axvline( trig , color='k',linestyle='-',alpha=0.9, label='new frame') 
"""
for i,trig in enumerate(t0_img[i0:i1]):
    if i!=0:
        plt.axvline( trig , color='blue',linestyle=':',alpha=0.1) 
    else:
        plt.axvline( trig , color='blue',linestyle=':',alpha=0.1, label='get frame') 
for i,trig in enumerate(t1_img[i0:i1]):
    if i!=0:
        plt.axvline( trig , color='black',linestyle=':',alpha=0.1) 
    else:
        plt.axvline( trig , color='black',linestyle=':',alpha=0.1, label='got frame') 
"""
for i,trig in enumerate(t0_dm[(t0_dm >= t0_img[i0]) & (t0_dm <= t0_img[i1])]):
    if i!=0:
        plt.axvline( trig , color='green') 
    else:
        plt.axvline( trig , color='green', label='send DM command') 
for i,trig in enumerate(t1_dm[(t1_dm >= t0_img[i0]) & (t1_dm <= t0_img[i1])]):
    if i!=0:
        plt.axvline( trig , color='red') 
    else:
        plt.axvline( trig , color='red', label='DM command sent') 


plt.xlim([t0_img[i0],t0_img[i1]])
plt.ylim([-0.2,1.2])
plt.xlabel('time [s]') 
plt.ylabel('normalized intensity')
plt.title(f'FPS={round(float(fps),1)}Hz, DIT={round(1e3*float(DIT),3)}ms')
plt.legend() #bbox_to_anchor=(1,1))
plt.savefig(fig_path + f'latency_test_FPS-{round(float(fps),1)}_DIT-{round(1e3*float(DIT),3)}ms_pokeDM_every_{change_every}iters.png',bbox_inches='tight',dpi=300)
plt.show()


print( f'latency < {np.mean( t1_dm - t0_dm )*1e6}us' )


print( 'mean delta t = ' , np.mean(np.diff(out_dict['t0'])) )
print( 'median delta t = ' , np.median(np.diff(out_dict['t0'])) )

img_std = np.array([np.std(i) for i in img_list])
img_std /= (np.max(img_std)-np.min(img_std))

plt.figure()
plt.plot( frame_no, (img_std-np.min(img_std) )/ (np.max(img_std)-np.min(img_std)) ,'.',label='norm std in img')
plt.plot( frame_no, flag_list ,'.',label='poked DM?')
plt.legend()
plt.show()


#testcrop = fli.GetCroppingState(camera)[-1]._fields_
