"""
# questions 

1 - how to crop camera 

2 - how to set gain



"""


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
"""



camera = FliSdk_V2.Init() # init camera object
listOfGrabbers = FliSdk_V2.DetectGrabbers(camera)
listOfCameras = FliSdk_V2.DetectCameras(camera)
# print some info and exit if nothing detected
if len(listOfGrabbers) == 0:
    print("No grabber detected, exit.")
    exit()
if len(listOfCameras) == 0:
    print("No camera detected, exit.")
    exit()
for i,s in enumerate(listOfCameras):
    print("- index:" + str(i) + " -> " + s)

print(f'--->using cameraIndex={cameraIndex}')
# set the camera
camera_err_flag = FliSdk_V2.SetCamera(camera, listOfCameras[cameraIndex])
if not camera_err_flag:
    print("Error while setting camera.")
    exit()
print("Setting mode full.")
FliSdk_V2.SetMode(camera, FliSdk_V2.Mode.Full)
print("Updating...")
camera_err_flag = FliSdk_V2.Update(camera)
if not camera_err_flag:
    print("Error while updating SDK.")
    exit()




fli.SetImageDimension(camera, width=100, height=100)

fli.FliSdk_isStarted_V2(camera)

fli.GetCurrentImageDimension(camera)

fli.GetCroppingState(camera)[-1]._fields_

CroppingData(ctypes.Structure)

testcrop = [("col1", 0),
 ("col2", 50),
 ("row1", 0),
 ("row2", 30),
 ("cred1Cols", ctypes.c_bool * 10),
 ("cred1Rows", ctypes.c_bool * 256)]
 
 CroppingData(testcrop )

 fli.CroppingData(testcrop ) # doesnt work 


testcrop = fli.GetCroppingState(camera)[-1]._fields_