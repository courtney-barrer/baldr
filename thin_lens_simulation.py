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
        
#%% simulating Baldr optical design described in system described in baldr_calc_8

o0 = obj(x=0,h=6) # pupil 

#lenses
l1 = lens(x= 2110, f= 254.016)
l2 = lens(x= 2110 + 254.016 + 30.747, f= 30.747)
l3 = lens(x= 2110 + 254.016 + 30.747 + 1200, f= 204.996 )

#combined system
sys = [l1,l2,l3]

#propagate object through each lens
objs = propagate_object(sys, o0, plot=True)

for i,o in enumerate( objs[1:] ) :
    print( f'object image position after lens {i+1} = {round(o.x)}mm')