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
fli.FliSerialCamera.SendCommand(camera, "set cropping columns 64-287")
# cropped rows must be multiple of 4 - multiple of 4 minus 1
fli.FliSerialCamera.SendCommand(camera, "set cropping rows 120-299")
fli.FliSerialCamera.SendCommand(camera, "set cropping on")

camera_err_flag, minDIT = fli.FliSerialCamera.SendCommand(camera, "mintint raw")
camera_err_flag, maxDIT = fli.FliSerialCamera.SendCommand(camera, "maxtint raw")

print( f'max int {1e3 * float(maxDIT)}ms') 
print( f'max int {1e3 * float(minDIT)}ms') 

# 3ms integration time (3.3kHz)
fli.FliSerialCamera.SendCommand(camera, "set tint " + str(0.0003))

# 2800Hz frame rate 
fli.FliSerialCamera.SetFps(camera, 2800)

print( 'fps = ', fli.FliSerialCamera.GetFps(camera) ) 

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
time.sleep(.5)
I0 = fli.GetRawImageAsNumpyArray( camera , -1) 
time.sleep(.5)
ref_img_1 = fli.GetRawImageAsNumpyArray( camera , -1) 
dm.send_data(cmd_list[1])
time.sleep(.5)
ref_img_2 = fli.GetRawImageAsNumpyArray( camera , -1) 

# check reference images 
fig, ax = plt.subplots(1,2)
ax[0].imshow( ref_img_1 - I0  )
ax[0].set_title('flat')
ax[1].imshow( ref_img_2 - I0 )
ax[1].set_title('poke')
plt.show()

img_list = [] #list to hold images
t0_list=[]
t1_list=[]
flag_list=[]

j=0 # used to calculate flag
flag=0 #jumps between 1, 0 for on, off pokes

for i in range(1000):
    t0_list.append( time.time() ) 
    img_list.append( fli.GetRawImageAsNumpyArray( camera , -1) ) 
    flag_list.append(flag) 
    #print(flag_list[-1])
    if np.mod(i,10)==0: # change DM state every 10 iterations
        j+=1
        flag = np.mod(j,2)  
        #print(i, j, flag) 
        dm.send_data(cmd_list[flag])
    
    t1_list.append( time.time() )



# peak pixel change for act 65 around row 77,  col 97
pv = np.array([i[77,97] - I0[77,97] for i in img_list])

i0 = 700
i1 = 760 
t0 = np.array(t0_list[i0:i1 ])-t0_list[0]
t1 = np.array(t1_list[i0:i1])-t0_list[0]
plt.plot(t0, pv[i0:i1]/np.max(pv[:100]) , label='pixel val')
plt.plot( t0, flag_list[i0:i1] ,'.', label='flag begin')
plt.plot( t1, flag_list[i0:i1] ,'.', label='flag end')
start_triggers = t0[np.where( np.diff( flag_list[i0:i1] ) )[0]]
end_triggers = t1[np.where( np.diff( flag_list[i0:i1] ) )[0]]
for trig in start_triggers:
    plt.axvline( trig ,color='green') 
for trig in end_triggers:
    plt.axvline( trig , color='red') 
plt.ylim([-0.2,1.2])
plt.legend()
plt.show()


out_dict = {'t0':list(np.array(t0_list)-t0_list[0]), 't1':list(np.array(t1_list)-t1_list[0]),'I0':I0, 'img_list':img_list, 'flag_list':flag_list} 

with open(data_path + 'latency_test_2.pickle','wb') as handle:
    pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print( 'mean delta t = ' , np.mean(np.diff(out_dict['t0'])) )
print( 'median delta t = ' , np.median(np.diff(out_dict['t0'])) )


#testcrop = fli.GetCroppingState(camera)[-1]._fields_
