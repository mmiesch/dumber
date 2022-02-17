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

sam = 3

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
    dt1 = 1.01
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

b2check = np.cos(2*np.pi*k*time2/tmax)

#-----------------------------------------------------------------------------

plt.figure(figsize=(30,6))

#plt.plot(time1,b1,'k-',linewidth=6)
#plt.plot(time2,b2,color='#B1FB17',linewidth=4)

#diff = b2check - b2
#plt.plot(time2,diff,'b-',linewidth=4)

#s = b2check + b2
#plt.plot(time2,s,'b-',linewidth=4)

plt.plot(time1,b1,'k-',linewidth=6)
plt.plot(time2,b2,'y-',linewidth=3)


plt.xlabel(xtitle)
plt.ylabel(ytitle)

plt.show()

#-----------------------------------------------------------------------------
