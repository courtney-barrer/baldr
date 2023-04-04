#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 02:52:39 2023

@author: bcourtne
"""

import numpy as np
import matplotlib.pyplot as plt 


class lens:
    def __init__(self,x,f):
        self.f = f
        self.x = x
        
    def get_image(self,_object):
        #returns a new object in the image plane
        do = (self.x - _object.x)
        if do != self.f:
            di = do * self.f / (do - self.f)
        else:
            di = np.inf
        
        M = -di/do
        
        i = obj(x = self.x + di, h = M * _object.h)
        
        return( i )
    
class obj:
    def __init__(self,x,h):
        self.x = x
        self.h = h
        
        
def propagate_object(lens_list, _object, plot=True):
    
    object_list = [ _object ]
    
    if plot:
        plt.figure()
        plt.axhline(0,color='k',lw=2)
        plt.plot( _object.x, _object.h ,'x' ,color='g',label='input object')

    for i,l in  enumerate(lens_list):
        next_object = l.get_image(object_list[-1])
        object_list.append( next_object )
        
        if plot:
            plt.axvline(l.x,linestyle=':',color='k',label=f'lens {i+1}, f={l.f}')
        
    if plot:
        plt.plot( object_list[-1].x, object_list[-1].h ,'x' ,color='b',label='object image')
        plt.legend()
        plt.xlabel( 'x' )
        plt.ylabel( 'y' )
        
    return( object_list )
        

def plot_sys(sys, object_list, plot_initial_object=False):
    plt.figure()
    plt.axhline(0,color='k',lw=2)
    if plot_initial_object:
        plt.plot( object_list[0].x, object_list[0].h ,'x' ,color='g',label='input object')
    for i,l in enumerate(sys):
        plt.axvline(l.x,linestyle=':',color='k',label=f'lens {i+1}, f={l.f}')
    plt.plot( object_list[-1].x, object_list[-1].h ,'x' ,color='b',label='object image')
    plt.legend()
    plt.xlabel( 'x' )
    plt.ylabel( 'y' )


#%% sanity check following results from https://www.youtube.com/watch?v=aHHa0cK_3as

obj1= obj(x=-50, h=4e-3 ) 
l1 = lens(x= 0,        f= 30 )
l2 = lens(x= 10,        f= 20)

sys=[l1,l2]
test0 = propagate_object(sys, obj1, plot=True)

#image formed at 25.3cm 


#%% simulating Baldr optical design described in system described in baldr_calc_8

o0 = obj(x=0,h=6) # pupil 

#lenses
l1 = lens(x= 2110, f = 254.016)
l2 = lens(x= 2110 + 254.016 + 30.747, f= 30.747)
l3 = lens(x= 2110 + 254.016 + 30.747 + 1200, f= 204.996 )

#combined system
sys = [l1,l2,l3]

#propagate object through each lens
objs = propagate_object(sys, o0, plot=True)

for i,o in enumerate( objs[1:] ) :
    print( f'object image position after lens {i+1} = {round(o.x)}mm')
    

#%% # constraint 1 - check l1 virtual image DM edge (4mm) is at x=0 and h= 12mm/2 = 6mm
# ++++++++++ relative distances between things (mm)
x1 = 2000  # pupil to DM (Constraint: lens 1 needs to image DM (virtually))
x2 = 1000  # DM to lens 1 

# ++++++++++ lens focal lens (mm) 
f1 = 1500
f2 = 254
f3 = 30
f4 = 200

# ++++++++++ define lens 
l1 = lens(x= x1 + x2,        f= f1 )

DM_rad = 3.6 #mm
oDM = obj(x=x1, h=DM_rad )  # DM

# check l1 virtual image DM edge (4mm) is at x=0 and h= 12mm/2 = 6mm
test1 = propagate_object([l1], oDM, plot=True)

#%%  constraint 2 - lens 2 to FPM : star needs to be imaged on FPM 
# ++++++++++ relative distances between things (mm)
x1 = 2000  # pupil to DM (Constraint: lens 1 needs to image DM (virtually))
x2 = 1000  # DM to lens 1 
x3 = 20  # lens 1 to OAP (lens 2) 

# ++++++++++ lens focal lens (mm) 
f1 = 1500
f2 = 254
f3 = 30
f4 = 200

# ++++++++++ define lens 
l1 = lens(x= x1 + x2,        f= f1 )
l2 = lens(x= l1.x + x3,      f= f2 )

Sta_rad = 2 #mm
oSta = obj(x=-1e20, h=Sta_rad )  # star

# lens 2 to FPM (Constraint: star needs to be imaged on FPM )
test2 = propagate_object([l1,l2], oSta, plot=True)
print( f'l2.x  + x4 = {test2[-1].x} (Constraint: star needs to be imaged on FPM )\ntherefore x4 = {test2[-1].x-l2.x}')

#%% # test 3 , given pupil edge image height propagated after lens3 set x6,x7 such that it matches our desired # pixels
# we need to set x6,x7 such that lens 4 images h1 to h2 (constrained by det 
# pitch and how many pixels we want to image across)

# ++++++++++ relative distances between things (mm)
x1 = 2000  # pupil to DM (Constraint: lens 1 needs to image DM (virtually))
x2 = 1000  # DM to lens 1 
x3 = 20  # lens 1 to OAP (lens 2) 
x4 = 216.794  # lens 2 to FPM (Constraint: star needs to be imaged on FPM )
x5 = 30  # FPM to lens 3

# ++++++++++ lens focal lens (mm) 
f1 = 1500
f2 = 254
f3 = 30
f4 = 200

# ++++++++++ define lens 
l1 = lens(x= x1 + x2,        f= f1 )
l2 = lens(x= l1.x + x3,      f= f2 )
l3 = lens(x= l2.x + x4 + x5, f= f3 )
l4 = lens(x= np.nan,     f= f4 )
Pup_rad = 2 #mm
oPup = obj(x=0, h=Pup_rad ) #pupil
test3 = propagate_object([l1,l2,l3], oPup, plot=True)

# Detector 640x512 pixels with 15um pitch (image across 12 pixels = 180um, h = 90um)
desired_h2 = -np.sign(test3[-1].h) * 90e-3 # mm ()



def get_x6_x7(objs_, l3, l4, desired_h2):
    
    """
       obj_.x
    l3  |         l4 (lens)
    |   | h       |
 ---|---.---------|-------| DET
    |             |
     x'    x_o      x_i
     
    |-  - x6- -  -|- x7  -|
    
    M=-x_i/x_o,  
    x_i = (x_o * f) / (x_o-f)
    
    wolfram solves x_o = f(M-1)/M
    """
    
    M = desired_h2 / objs_[-1].h
    
    x_o = l4.f * (M-1)/M
    
    x7 = -M * x_o

    x6 =  (objs_[-1].x  - l3.x) + x_o
    
    return( (x6,x7) )

x6, x7 = get_x6_x7( test3, l3,l4, desired_h2 )
print( f'x6={x6}, x7={x7} calculated such that pupil edge images to h2={desired_h2}   ')


#
#%% Lets see the complete system
# define pupil at x=0 !

# DM  BMC 492-1.5 aperture = 6.90mm, pitch (300um) , 24 actuators across pupil ,=> r = 3.6mm
# Detector 640x512 pixels with 15um pitch (image across 12 pixels = 180um, h =90um)

Sta_rad = 2 #mm
Pup_rad = 2 #mm
DM_rad = 3.6 #mm


# ++++++++++ relative distances between things (mm)
x1 = 2000  # pupil to DM (Constraint: lens 1 needs to image DM (virtually))
x2 = 1000  # DM to lens 1 
x3 = 20  # lens 1 to OAP (lens 2) 
x4 = 216.794  # lens 2 to FPM (Constraint: star needs to be imaged on FPM )
x5 = 30  # FPM to lens 3
x6 = 784.686 #1614.05 #1.200  # lens 3 to lens 4
x7 = 265.915 #250  # lens 4 to detector (Constraint: needs to image pupil )

x_s = [x1,x2,x3,x4,x5,x6,x7] #relative positions
z_s = np.cumsum( x_s ) #absolute positions 

# ++++++++++ lens focal lens (mm) 
f1 = 1500 #mm
f2 = 254
f3 = 30
f4 = 200

# ++++++++++ define lens 
l1 = lens(x= x1 + x2,        f= f1 )
l2 = lens(x= l1.x + x3,      f= f2 )
l3 = lens(x= l2.x + x4 + x5, f= f3 )
l4 = lens(x= l3.x + x6 ,     f= f4 )

# ++++++++++ define of objects to study (propagate)
oPup = obj(x=0, h=Pup_rad  ) #pupil
oSta = obj(x=-1e15, h=Sta_rad  )   # star
oDM = obj(x=x1, h=DM_rad )  # DM
oFPM = obj(x=l2.x + x4, h=0 ) #FPM

sys = [l1, l2, l3, l4]

# test
pup_ims = propagate_object(sys, oPup, plot=True)


print( pup_ims[-1].h ) #check it matches expectations 


