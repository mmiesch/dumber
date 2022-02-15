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

t2 = np.linspace(0, tmax, num = N, endpoint = True, dtype = 'float')

# define random seed for reproducibility
rseed = 52134
np.random.seed(rseed)

eps = norm.rvs(loc = 0.0, scale = 0.1, size = N)

# input time grid is regular grid plus random fluctuations and an offset.
t1 = t2 + eps + 0.021
t1 -= t1[0]

# only use data between 0 and tmax
t2 = t2[np.logical_and(t2 > t1[0], t2 < t1[N-1])]

print(t1)

#------------------------------------------------------------------------------
# define a function on t1 with lots of outliers

b1 = np.cos(2*np.pi*t1/tmax)
b1 += cauchy.rvs(loc = 0.0, scale = 0.1, size = N)

#------------------------------------------------------------------------------
# interpolate

f = interp.interp1d(t1,b1,kind='linear')

b2 = f(t2)

#------------------------------------------------------------------------------

plt.figure(figsize=(12,6))

plt.plot(t1,b1,'bo')
plt.plot(t2,b2,'k-')

plt.show()

