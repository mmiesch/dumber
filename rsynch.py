"""
Explore problems and solutions with synchronization of data sets with nearly equal time sampling
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.optimize as opt

from scipy.interpolate import interpolate as interp

from scipy.fft import fft

#-----------------------------------------------------------------------------

# default titles - change only if desired
xtitle = 'time (arbitrary units)'
ytitle = 'signal (arbitrary units)'

#----------------------------------------------------------------------------- 

sam = 4

if sam == 1:
    dt1 = 1.01
    offset = 0.31
    ns=1000
    k=400
elif sam == 2:
    dt1 = 1.01
    offset = 0.31
    ns=500
    k=100
elif sam == 3:
    dt1 = 1.01
    offset = 0.0
    ns=2000
    k=200
elif sam == 4:
    dt1 = 0.99
    offset = 0.0
    ns=2000
    k=200
else:
    dt1 = 1.01
    offset = 0.31
    ns=1000
    k=200

time1 = offset + dt1*np.arange(ns, dtype='float')

tmin = np.min(time1)
tmax = np.max(time1)

t1 = int(tmin) + 1
t2 = int(tmax) + 1
time2 = np.arange(t1,t2,dtype='float')

t0 = time2[0]

time1 -= t0
time2 -= t0
tmin -= t0
tmax -= t0

b1 = np.cos(2*np.pi*k*time1/tmax)

#-----------------------------------------------------------------------------
# linear interpolation
f = interp.interp1d(time1,b1,kind='linear')
b2 = f(time2)

#-----------------------------------------------------------------------------
# Bartlett filter

b3 = np.zeros(len(time2))

for j in np.arange(2,len(time2)-2):
    tj = time2[j]
    mm = (tj - time1[0])/dt1
    m = int(np.rint(mm))

    #print(f"x- {im-1} {tj - time1[im-1]}")
    #print(f"xm {im}   {tj - time1[im]  }")
    #print(f"x+ {im+1} {tj - time1[im+1]}")

    x = tj - time1[m]
    assert np.abs(x) <= 0.5

    ell = m - int(2 - x)
    n = m + int(2+x) 

    nw = n - ell + 1
    w = np.zeros(nw)

    kk = 0
    for k in np.arange(1,m-ell+1):
        w[kk] = 1.0 - 0.5*(k+x)
        kk += 1

    w[kk] = 1.0 - 0.5 * x
    kk += 1

    for k in np.arange(1,n-m+1):
        w[kk] = 1.0 - 0.5*(k-x)
        kk += 1

    w = w / np.sum(w)

    b3[j] = np.sum(w*b1[ell:n+1])

time3 = time2[2:len(time2)-2]
b3 = b3[2:len(time2)-2]

#-----------------------------------------------------------------------------
# this plots the time series

#plt.figure(figsize=(30,6))
#
#plt.plot(time1,b1,'k-',linewidth=6)
##plt.plot(time2,b2,'y-',linewidth=3)
#
#plt.plot(time3,b3,'y-',linewidth=3)
#
#
#plt.xlabel(xtitle)
#plt.ylabel(ytitle)
#

#-----------------------------------------------------------------------------
# this plots the power spectra and phase

bhat1 = fft(b1)
bhat2 = fft(b2)
bhat3 = fft(b3)

n1 = int(len(bhat1)/2)
n2 = int(len(bhat2)/2)
n3 = int(len(bhat3)/2)

ps1 = np.absolute(bhat1[:n1])**2
ps2 = np.absolute(bhat2[:n2])**2
ps3 = np.absolute(bhat3[:n3])**2

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,6))

ax[0].set_yscale("log")
ax[0].plot(ps1,color='k')
ax[0].plot(ps2,color='r')
ax[0].plot(ps3,color='b')

phase1 = np.angle(bhat1[:n1],deg=True)
phase2 = np.angle(bhat2[:n2],deg=True)
phase3 = np.angle(bhat3[:n3],deg=True)

ax[1].plot(phase1,color='k')
#ax[1].plot(phase2,color='r')
ax[1].plot(phase3,color='b')

#-----------------------------------------------------------------------------
# phase of the peak mode

imax1 = np.argmax(ps1)
imax2 = np.argmax(ps2)
imax3 = np.argmax(ps3)

print(f"1: {imax1} {ps1[imax1]} {phase1[imax1]}")
print(f"2: {imax2} {ps2[imax2]} {phase2[imax2]}")
print(f"3: {imax3} {ps3[imax3]} {phase3[imax3]}")

#-----------------------------------------------------------------------------

plt.show()
