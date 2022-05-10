"""
Simple example of how to read and plot L1 MAG data from DSCOVR
"""

import numpy as np
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# open file for reading

dir = '/home/mark.miesch/data/DSCOVR/MAG/L1/'

# Feb 5
file = dir+'oe_mg1_dscovr_s20220205000000_e20220205235959_p20220206013755_pub.nc'

#-----------------------------------------------------------------------------
# old school netcdf3 approach seems to work:

from scipy.io import netcdf

dfile = netcdf.NetCDFFile(file,'r')

tdata = dfile.variables['time']
time = tdata[:].copy().astype('float') - tdata[0]

bzdata = dfile.variables['bz_gsm']
bz = bzdata[:].copy().astype('float')

dfile.close()

#-----------------------------------------------------------------------------
# an alternative, using the netcdf4 package

from netCDF4 import Dataset

rootgrp = Dataset(file, "r", format="NETCDF3")

tvar = rootgrp.variables['time']
time2 = tvar[:] - tvar[0]

bzvar = rootgrp.variables['bz_gsm']
bz2 = bzvar[:].copy()

#-----------------------------------------------------------------------------

i1 = 2000
i2 = i1 + 50

plt.plot(time[i1:i2],bz[i1:i2])
plt.plot(time2[i1:i2],bz2[i1:i2],'bo')

plt.show()

#-----------------------------------------------------------------------------

rootgrp.close()
