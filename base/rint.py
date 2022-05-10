"""
Explore robust methods for interpolation, for use with Level 3 data products for 
the SWiPS instrument on SWFO-L1.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp

from scipy.stats import norm, cauchy

#------------------------------------------------------------------------------
# define an input time grid that has a cadence of approximately 
# but not exactly 1 minute

# size of data sample
N = 50

tmax = float(N)

t2 = np.linspace(0, tmax, num = N+1, endpoint = True, dtype = 'float')

# define random seed for reproducibility
rseed = 52134
np.random.seed(rseed)

eps = norm.rvs(loc = 0.0, scale = 0.1, size = N+1)

# input time grid is regular grid plus random fluctuations and an offset.
t1 = t2 + eps + 0.021
t1 -= t1[0]

# remove a few data points
mask = np.ones(len(t1), dtype=bool)
mask[[7, 12, 43]] = False
t1 = t1[mask]
N = len(t1)

# only use data between 0 and tmax
t2 = t2[np.logical_and(t2 > t1[0], t2 < t1[N-1])]

print(t1)

#------------------------------------------------------------------------------
# define a function on t1 with lots of outliers

# smooth profile
#b1 = np.cos(2*np.pi*t1/tmax)

# here is a shock-like profile
beta = 2.0
b1 = 2*np.arctan(beta*(t1-0.5*tmax))/np.pi

# add outliers
#b1 += cauchy.rvs(loc = 0.0, scale = 0.1, size = N)
b1 += cauchy.rvs(loc = 0.0, scale = 0.01, size = N)

#------------------------------------------------------------------------------
# interpolate

f = interp.interp1d(t1,b1,kind='linear')
b2 = f(t2)

f = interp.interp1d(t1,b1,kind='slinear')
b3 = f(t2)

#------------------------------------------------------------------------------
# Inverse distance weighing

b4 = np.zeros(len(t2))

dt = 2.0

nw = 4

j1 = 0
j2 = 4
for i in np.arange(len(t2)):
    t = t2[i]

    while t1[j1] < t-dt:
        j1 += 1
    j2 = j1 + nw

    dd = np.abs(t1[j1:j2] - t)
    
    w = np.where(dd > 0.0, 1.0/dd, 1.0)

    b4[i] = np.sum(w*b1[j1:j2])/np.sum(w)


#------------------------------------------------------------------------------

plt.figure(figsize=(12,6))

plt.plot(t1,b1,'ko')
plt.plot(t2,b2,'k-',linewidth=4)
plt.plot(t2,b4,'b-',linewidth=2)

plt.show()

