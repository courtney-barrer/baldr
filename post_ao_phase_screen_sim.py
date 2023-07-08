#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 19:17:09 2023

@author: bcourtne


Generalized method for generating AO residual phase screens 
"""

from scipy.special import kv
from scipy.special import gamma
import numpy as np 
import matplotlib.pyplot as plt 

from numpy import linalg as la
from scipy.stats import multivariate_normal


def D_phi(r,L0,r0, kolmogorov = False, finite_aperture=None,filter_order=None ): # atmospheric phase structure function 
    
    if r==0:
        Dphi=0
    else:
        if  kolmogorov :
            Dphi = 6.88 * (r/r0)**(5/3)
        else:
            if finite_aperture is not None: #we apply telescope filtering 
                if r < finite_aperture:
                    Dphi = (L0/r0)**(5/3) * 2**(1/6) * gamma(11/6)/np.pi**(8/3) * ((24/5)*gamma(6/5))**(5/6) * (gamma(5/6)/2**(1/6)-(2*np.pi*r/L0)**(5/6)*kv(5/6,2*np.pi*r/L0))
                    # saturates at (L0/r0)**(5/3) * 2**(1/6) * gamma(11/6)/np.pi**(8/3) * ((24/5)*gamma(6/5))**(5/6) * gamma(5/6)/2**(1/6)
                else:
                    if filter_order is not None:
                        #power law decay (power law index = filter_order )
                        Dphi = (L0/r0)**(5/3) * 2**(1/6) * gamma(11/6)/np.pi**(8/3) * ((24/5)*gamma(6/5))**(5/6) * (gamma(5/6)/2**(1/6)-(2*np.pi*finite_aperture/L0)**(5/6)*kv(5/6,2*np.pi*finite_aperture/L0)) / (1+r-finite_aperture)**filter_order
                        #exponetial decay 
                        #(L0/r0)**(5/3) * 2**(1/6) * gamma(11/6)/np.pi**(8/3) * ((24/5)*gamma(6/5))**(5/6) * (gamma(5/6)/2**(1/6)-(2*np.pi*finite_aperture/L0)**(5/6)*kv(5/6,2*np.pi*finite_aperture/L0)) * np.exp( -filter_order * (r-finite_aperture) )
                    else:
                        Dphi = 0
            else:
                Dphi = (L0/r0)**(5/3) * 2**(1/6) * gamma(11/6)/np.pi**(8/3) * ((24/5)*gamma(6/5))**(5/6) * (gamma(5/6)/2**(1/6)-(2*np.pi*r/L0)**(5/6)*kv(5/6,2*np.pi*r/L0))
                
    return(Dphi)




def Dr_phi(r,sigma2_ao, D, N_act, L0, r0, kolmogorov = False, finite_aperture=None,filter_order=None ): # ao residual phase structure function 

    if r==0:
        Dr=0
    else:
        #interactuator distance in telescope pupil space 
        d_act = np.sqrt( (D/2)**2 /  N_act ) # pi*r_act^2 = np.pi(D/2)^2 / N_act
    
        if finite_aperture is not None: #we apply telescope filtering 
            if r<=finite_aperture: # if the distance is less then the DM inter-actuator spacing then we have no AO correction and turbulence structure is from atmosphere
                Dr =  sigma2_ao * (1 - (2*np.pi*r/d_act)**(5/6)*kv(5/6,2*np.pi*r/d_act) )   # saturates to sigma2_ao at r=d_act    
            else: # outside aperture, so structure function fgoes to zero
                if filter_order is not None:
                    #power law decay (power law index = filter_order )
                    Dr = sigma2_ao * (1 - (2*np.pi*finite_aperture/d_act)**(5/6)*kv(5/6,2*np.pi*finite_aperture/d_act) ) / (1+r-finite_aperture)**filter_order #0 #sigma2_ao + D_phi(r-D/2, L0, r0, kolmogorov) 
                    #exponetial decay 
                    #Dr = sigma2_ao * (1 - (2*np.pi*finite_aperture/d_act)**(5/6)*kv(5/6,2*np.pi*finite_aperture/d_act) ) * np.exp( -filter_order * (r-finite_aperture) )
                else:
                    Dr=0
        else: 
            Dr =  sigma2_ao * (1 - (2*np.pi*r/d_act)**(5/6)*kv(5/6,2*np.pi*r/d_act) ) 
            
    return(Dr)



def cov_function(r , sigma2 ,L0, r0, kolmogorov = False,  finite_aperture=None, filter_order=None):
    
    
    
    if finite_aperture is not None: 
        if r<=finite_aperture:
            B = sigma2 - D_phi(r,L0,r0, kolmogorov, finite_aperture,filter_order) / 2 #covariance 
        else:
            if filter_order is not None:
                #power law decay (power law index = filter_order )
                B = (sigma2 - D_phi(finite_aperture,L0,r0, kolmogorov, finite_aperture,filter_order) / 2) / (1+r-finite_aperture)**filter_order #0 #sigma2_ao + D_phi(r-D/2, L0, r0, kolmogorov) 
                #exponetial decay 
                #B = (sigma2 - D_phi(finite_aperture,L0,r0, kolmogorov, finite_aperture,filter_order) / 2) * np.exp( -filter_order * (r-finite_aperture) )#0 #sigma2_ao + D_phi(r-D/2, L0, r0, kolmogorov) 
            else:
                B=0
    else:
        B = sigma2 - D_phi(r,L0,r0, kolmogorov, finite_aperture,filter_order) / 2 #covariance 
        
    #BB=[b if b>0 else 0 for b in B]
    return(B) 
    
    
def ao_cov_function(r,sigma2_ao, D, N_act, L0, r0, kolmogorov = False , finite_aperture=None,filter_order=None):
    
    B = sigma2_ao - Dr_phi(r,sigma2_ao, D, N_act, L0, r0, kolmogorov, finite_aperture, filter_order) / 2 #covariance (finite_aperture=D implies telescope filtering of over aperature size D)

    return(B) 



def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    
    python code from here https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite 
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False




#%% 


r0=0.1 #m
L0=25 #m
D=1.8
sigma2 = 1.0299 * (D/r0)**(5/3) #rad
N_act=1200 # number of actuaors on DM
sigma2_ao=0.01 #rad 
filter_order=10 # None
"""
#- if finite_aperture = float (not None) we apply a cut off at finite_aperture in the structure function, 
#   we can either do a hard cutoff where D-phi(r>finite_aperture)=0 (filter_order=None) 
#    or a roll off at r^(filter_order) for r> finite_aperture.
"""
r =np.linspace(1e-3,200,100000) #radius #m


for finite_aperture in [None,D]:
    D_atm = [D_phi(rtmp, L0, r0, kolmogorov = False, finite_aperture=finite_aperture,filter_order=filter_order)  for rtmp in r]  # put D=None to turn off telescope filtering effects
    D_ao = [Dr_phi(rtmp, sigma2_ao, D, N_act, L0, r0, kolmogorov = False,finite_aperture=finite_aperture,filter_order=filter_order) for rtmp in r]
    D_kol = [D_phi(rtmp, L0, r0,kolmogorov = True,finite_aperture=None) for rtmp in r] 
    # Structure Function 
    plt.figure()
    plt.title(f'pupil filter = {finite_aperture}')
    plt.loglog( r, D_atm ,linestyle='-' ,label=r'atmosphere')
    plt.loglog( r, D_ao ,linestyle='-' ,label='AO residuals')
    plt.loglog( r, D_kol,color='k',linestyle=':' ,lw=3,label=r'$L_0->\infty$')
    plt.axvline(D,color='grey',linestyle='--',label=f'D={D}m')
    plt.legend()
    plt.xlabel(r'$r$ [m]',fontsize=14)
    plt.ylabel(r'$D_\phi(r)$ [rad]',fontsize=14)
    plt.gca().tick_params(labelsize=14)
    plt.grid()
    plt.ylim([1e-4,1e4])

    # Covariance 
    # ITS IMPORTANT TO REFLECT AROUNDORIGIN TO MAKE SYMETRIC - OTHERWISE FFT OF ASYMETRIC FUNCTION IS IMAGINARY!
    #rr = np.concatenate([ -r[::-1], r] )

    B_atm =  np.array( [cov_function(rtmp , sigma2 , L0, r0, kolmogorov = False, finite_aperture=finite_aperture,filter_order=filter_order) for rtmp in r] )
    B_atm = np.concatenate([ B_atm[::-1], B_atm] )
    
    B_ao =   np.array([ao_cov_function(rtmp,sigma2_ao, D, N_act, L0, r0, kolmogorov = False, finite_aperture=finite_aperture,filter_order=filter_order) for rtmp in r] )
    B_ao = np.concatenate([ B_ao[::-1], B_ao] )
    
    
    plt.figure()
    plt.title(f'pupil filter = {finite_aperture}')
    plt.plot(np.concatenate([ -r[::-1], r] ), B_atm, label='atmosphere')
    plt.plot(np.concatenate([ -r[::-1], r] ), B_ao, label=r'AO residual')
    plt.axvline(D,color='grey',linestyle='--',label=f'D={D}m')
    plt.yscale('symlog')
    plt.xscale('symlog')
    plt.legend()
    plt.xlabel('r [m]',fontsize=14)
    plt.ylabel('B(r) [rad$^2$]',fontsize=14)
    plt.gca().tick_params(labelsize=14)
    plt.grid()


    # PSD (spatial)
    k_atm =  np.fft.fftfreq(  len(B_atm), d=np.diff(r)[0] )[:len(B_atm)//2]
    psd_atm = np.diff(r)[0] *  np.fft.fft( B_atm )[:len(B_atm)//2] 
    
    k_ao  =  np.fft.fftfreq(len(B_ao), d=np.median(np.diff(r)) )[:len(B_ao)//2]
    psd_ao = np.diff(r)[0] * np.fft.fft(B_ao)[:len(B_ao)//2]
    
    psd_theory = 7.2e-3 * (D/r0)**(5/3) * abs(k_ao)**(-11/3) # Tatarski 1961
    
    plt.figure()
    plt.title(f'pupil filter = {finite_aperture}')
    plt.loglog(k_ao, 1/k_ao * np.abs(psd_ao), label='AO residual')  # 1/k factor because we are in 2D and have to integrate (in Fourier space) across 1 dimension!!
    plt.loglog(k_atm,1/k_atm * np.abs(psd_atm),label='atmosphere')  
    plt.axvline(1/L0, label=r'k=1/$L_0$',linestyle=':',color='k')
    plt.axvline(1/D, label=r'k=1/D',linestyle='--',color='k')
    plt.loglog(k_ao, psd_theory,color='k',linestyle=':' ,lw=3,label=r'$L_0->\infty$') 
    plt.legend()
    plt.xlabel(r'k [$m^{-1}$]',fontsize=14)
    plt.ylabel(r'PSD $[rad^2/m^{-1}]$',fontsize=14)
    plt.gca().tick_params(labelsize=14)
    plt.grid()






#%% Generate phase screens 


r0=0.1 #m
L0=25 #m
D=1.8 #m

finite_aperture = None
filter_order = None
"""
#- if finite_aperture = float (not None) we apply a cut off at finite_aperture in the structure function, 
#   we can either do a hard cutoff where D-phi(r>finite_aperture)=0 (filter_order=None) 
#    or a roll off at r^(filter_order) for r> finite_aperture.
"""
#r =np.linspace(1e-3,200,100000) #radius #m

#B_atm =  np.array( [cov_function(rtmp , sigma2 , L0, r0, kolmogorov = False, finite_aperture=finite_aperture,filter_order=filter_order) for rtmp in r] )


Npix = 2**5 # number of pixels in row
dx = D/Npix #m
#P = np.zeros([Npix,Npix]) # init pupil 
rho = {} # Npix x Npix x Npix**2

for i0 in range(Npix):
    for j0 in range(Npix):
        rho[(i0,j0)] = np.zeros([Npix,Npix])
        for i in range(rho[(i0,j0)].shape[0]):
            for j in range(rho[(i0,j0)].shape[1]):
                r = dx * np.sqrt( (i0-i)**2 + (j0-j)**2 ) 
                rho[(i0,j0)][i,j] = cov_function(r , sigma2 ,L0, r0, kolmogorov = False,  finite_aperture=None, filter_order=None) 
            

Sigma = np.array( [rho[x] for x in rho] ) # cov_matrix



cov_matrix = Sigma.reshape(Npix**2,Npix**2) # have to make #D covariance matrix 2D 

# cov_matrix as a few (small) negative or non-zero  imaginary eigenvalues meaning not semi positive definite.. 
# so we find the nearest positive semi-definite matrix 

cov_matrix_nearestPD = nearestPD( cov_matrix ) 

# lets look at differences 
fig,ax = plt.subplots(1,2,figsize=(10,5))
ax[0].imshow(cov_matrix ) 
ax[0].set_title(r'$\Sigma$ ',fontsize=20)
ax[1].imshow( cov_matrix_nearestPD  )
ax[1].set_title('nearest semi-definite \n'+r'matrix to $\Sigma$',fontsize=20)

plt.figure() # eigenvalues 
plt.plot( np.real( la.eigvals(cov_matrix)) ,color='b',linestyle='-',label=r'real eigenvals of $\Sigma$')
plt.plot( np.imag( la.eigvals(cov_matrix)) ,color='b',linestyle=':',label=r'imaginary eigenvals of $\Sigma$')

plt.plot( np.real( la.eigvals(cov_matrix_nearestPD)) ,color='g',linestyle='-',label=r'real eigenvals of nearest PD to $\Sigma$')
plt.plot( np.imag( la.eigvals(cov_matrix_nearestPD)) ,color='g',linestyle=':',label=r'imaginary eigenvals of nearest PD to $\Sigma$')
plt.xlabel('eigenvalue index ',fontsize=14)
plt.ylabel('eigenvalue',fontsize=14)
plt.yscale('symlog')
plt.xscale('log')
plt.gca().tick_params(labelsize=14)
plt.legend()

# Matrix is singular! USE singular value decomposition 
cov_svd = la.svd(cov_matrix_nearestPD) # U, S, V

# check for zero (or very small values) in S
plt.figure()
plt.loglog( cov_svd[1] )
plt.ylabel('S in SVD')
plt.ylabel('index')
# ok seems like just the last element is bad.. delete it from SVD 

Ndrop = 1
cov_mat_n =cov_mat_n = cov_svd[0][:-Ndrop ,:-Ndrop ] @ np.diag(cov_svd[1])[:-Ndrop ,:] @ cov_svd[2][:,:-Ndrop] 
print( cov_mat_n.shape )
plt.figure()
plt.imshow( cov_mat_n )
plt.title('covariance matrix after removing singular values')

# create our multivariant Guassian to sample screens from 
mu = np.zeros(cov_mat_n.shape[0]) # zero mean 
rv = multivariate_normal( mean=mu, cov=cov_mat_n  )

# generate signular phase screen 
screen0 = rv.rvs(1)
# append nan values to fill pixels that were dropped in SVD analysis so it can be reshaped to initial array size 
screen = np.append(screen0, np.nan* np.ones(Ndrop)).reshape(Npix,Npix)

plt.figure()
plt.title('simulated phase screen')
plt.imshow(screen)







# SOME other analysis
#%%
# Plotting atmospheric phase structure function 
r0=0.1 #m
r= np.logspace(-3,3,1000)
plt.figure()
for L0 in np.logspace(-1,3,5):
    plt.loglog( r, D_phi(r, L0, r0,kolmogorov = False),linestyle='-' ,label=r'$L_0$'+f'={L0}m')
plt.loglog( r, D_phi(r, L0, r0,kolmogorov = True),color='k',linestyle=':' ,lw=3,label=r'$L_0->\infty$')
plt.legend()
plt.xlabel(r'$r$ [m]',fontsize=14)
plt.ylabel(r'$D_\phi(r)$ [rad]',fontsize=14)
plt.gca().tick_params(labelsize=14)
plt.grid()


# Plotting AO residual structure function 
r0=0.1 #m
D=1.8 #m 
N_act=1200 # number of actuaors on DM
sigma2_ao=0.01 #rad 

r= np.logspace(-3,3,1000)
plt.figure()
for L0 in np.logspace(-1,3,5):
    plt.loglog( r, [Dr_phi(rr, sigma2_ao, D, N_act, L0, r0, kolmogorov = False) for rr in r],linestyle='-' ,label=r'$L_0$'+f'={L0}m')
plt.loglog( r, D_phi(r, L0, r0,kolmogorov = True),color='k',linestyle=':' ,lw=3,label=r'$L_0->\infty$')
plt.legend()
plt.xlabel(r'$r$',fontsize=14)
plt.ylabel(r'$D_\phi(r)$',fontsize=14)
plt.gca().tick_params(labelsize=14)
plt.grid()


#%% Compare covariance 

r0=0.1 #m
L0=25 #m
D=1.8
N_act=1200 # number of actuaors on DM
sigma2_ao=0.01 #rad 
sigma2 = 1.0299 * (D/r0)**(5/3) #rad

#covariances
Bao =   np.array([ao_cov_function(rr,sigma2_ao, D, N_act, L0, r0, kolmogorov = False ) for rr in r])
B = cov_function(r , sigma2 , L0, r0, kolmogorov = False)

plt.loglog(r, abs(B), label='atmosphere')
plt.loglog(r, abs(Bao), label=r'AO residual')
plt.legend()
plt.xlabel('r [m]',fontsize=14)
plt.ylabel('|B(r)| [rad$^2$]',fontsize=14)
plt.gca().tick_params(labelsize=14)
plt.grid()

#%%

# Wiener–Khinchin theorem ( PSD = fft[covariance] ), lets plot PSDs 
r0=0.1 #m
L0=25 #m
D=1.8
sigma2 = 1.0299 * (D/r0)**(5/3) #rad
r =np.linspace(1e-3,200,100000) #radius #m

B =  cov_function(r , sigma2 , L0, r0, kolmogorov = False) 

#k , psd =  np.fft.fftfreq(len(B), d=np.median(np.diff(r)) ), np.diff(r)[0] * np.fft.fft(B)

# ITS IMPORTANT TO REFLECT AROUNDORIGIN TO MAKE SYMETRIC - OTHERWISE FFT OF ASYMETRIC FUNCTION IS IMAGINARY!
k,psd =  np.fft.fftfreq( len(np.concatenate([ B[::-1], B] )), d=np.diff(r)[0] )[:len(B)//2], np.diff(r)[0] *  np.fft.fft( np.concatenate([ B[::-1], B] ) )[:len(B)//2] 

psd_theory = 7.2e-3 * (D/r0)**(5/3) * abs(k)**(-11/3) # Tatarski 1961

plt.figure()
#plt.loglog(k,abs(np.real(psd)),label='|Re[fft(B(r))]|')
#plt.loglog(k,abs(np.imag(psd)),label='|Im[fft(B(r))]|')
#plt.loglog(k, abs(np.real(psd)) * abs(np.imag(psd)),label='|Im[fft(B(r))]|')
plt.loglog(k,1/k*np.abs(psd),label='1/k|fft[B(r)]|')  # WHY THE 1/k ??????
#plt.loglog(k[:len(k)//2],k[:len(k)//2]**(-5/6),label=r'$f^{-5/6}$')
plt.axvline(1/L0, label='f=1/L0',linestyle=':',color='k')
plt.loglog(k, psd_theory, label='Kolmogorov theory') 
plt.legend()
plt.xlabel(r'k [$m^{-1}$]',fontsize=14)
plt.ylabel(r'PSD $[rad^2/m^{-1}]$',fontsize=14)


#where is zero crossing in covariance
#plt.figure()
#plt.semilogx(r,abs(B))









