"""
# questions 

1 - how to crop camera 

2 - how to set gain



"""
import numpy as np
import matplotlib.pyplot as plt
import time 
import sys 
sys.path.insert(1, '/opt/FirstLightImaging/FliSdk/Python/demo/')
import FliSdk_V2 as fli


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

# 2000Hz frame rate 
fli.FliSerialCamera.SetFps(camera, 2800)

print( 'fps = ', fli.FliSerialCamera.GetFps(camera) ) 
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





#testcrop = fli.GetCroppingState(camera)[-1]._fields_
