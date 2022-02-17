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

ns = 100

# resolution of input data not exactly 1 minute
dt1 = 1.1
offset = 0.31

time1 = offset + dt1*np.arange(ns, dtype='float')

tmin = np.min(time1)
tmax = np.max(time1)

t1 = int(tmin) + 1
t2 = int(tmax) + 1
time2 = np.arange(t1,t2,dtype='float')

b1 = np.cos(2*np.pi*time1/tmax)

#-----------------------------------------------------------------------------
# linear interpolation
f = interp.interp1d(time1,b1,kind='linear')
b2 = f(time2)

#-----------------------------------------------------------------------------

plt.figure(figsize=(12,6))

plt.plot(time1,b1,'k-')
plt.plot(time2,b2,'bo')

plt.xlabel(xtitle)
plt.ylabel(ytitle)

plt.show()

#-----------------------------------------------------------------------------
