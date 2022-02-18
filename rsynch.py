"""
Explore problems and solutions with synchronization of data sets with nearly equal time sampling
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.optimize as opt

from scipy.interpolate import interpolate as interp

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

plt.figure(figsize=(30,6))

plt.plot(time1,b1,'k-',linewidth=6)
#plt.plot(time2,b2,'y-',linewidth=3)

plt.plot(time3,b3,'y-',linewidth=3)


plt.xlabel(xtitle)
plt.ylabel(ytitle)

plt.show()

#-----------------------------------------------------------------------------
