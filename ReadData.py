"""
Simple example of how to read and plot L1 MAG data from DSCOVR
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.io import netcdf

#-----------------------------------------------------------------------------
# open file for reading

dir = '/home/mark.miesch/data/DSCOVR/MAG/L1/'

# Feb 5
file = dir+'oe_mg1_dscovr_s20220205000000_e20220205235959_p20220206013755_pub.nc'

# old school netcdf3 approach seems to work:

dfile = netcdf.NetCDFFile(file,'r')

tdata = dfile.variables['time']
time = tdata[:].copy().astype('float')

bzdata = dfile.variables['bz_gsm']
bz = bzdata[:].copy().astype('float')

dfile.close()

#-----------------------------------------------------------------------------

i1 = 2000
i2 = i1 + 400

plt.plot(time[i1:i2],bz[i1:i2])

plt.show()

#-----------------------------------------------------------------------------

