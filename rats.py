"""
Assess different averaging techniques
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

from netCDF4 import Dataset

#-----------------------------------------------------------------------------
# open file for reading

dir = '/home/mark.miesch/data/DSCOVR/MAG/L1/'

# Feb 5
file = dir+'oe_mg1_dscovr_s20220205000000_e20220205235959_p20220206013755_pub.nc'

#-----------------------------------------------------------------------------
#  get a data segment to work with

i1 = 2000
i2 = i1 + 41

rootgrp = Dataset(file, "r", format="NETCDF3")

tvar = rootgrp.variables['time']
time = tvar[i1:i2] - tvar[0]

bzvar = rootgrp.variables['bz_gsm']
bz = bzvar[i1:i2].copy()

#-----------------------------------------------------------------------------
# define bins

# ns is the length of the sample
ns = len(time)

# nw is the length of the averaging window
nw = 4

# na is the length of the averaged variable
na = int(ns / 4 + 1)

print(f"ns, na = {ns} {na}")

# define time at the center of each window
tbox = np.empty(na, dtype = float)

for i in np.arange(na):
    i1 = i*nw
    i2 = np.min([i1+nw,ns])
    print(f"{i1} {i2-1}")
    mw = i2 - i1
    tbox[i] = 0.5*(time[i1] + time[i2-1])

#-----------------------------------------------------------------------------
# resampling with boxcar average

bzbox = np.empty(na, dtype = float)

for i in np.arange(na):
    i1 = i*nw
    i2 = np.min([i1+nw,ns])
    mw = i2 - i1
    bzbox[i] = np.sum(bz[i1:i2]) / mw


#-----------------------------------------------------------------------------

plt.plot(time,bz,'bo')
plt.plot(tbox,bzbox,'k-')

plt.show()

#-----------------------------------------------------------------------------

rootgrp.close()
