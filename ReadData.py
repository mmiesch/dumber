"""
Simple example of how to read and plot L1 MAG data from DSCOVR
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.io import netcdf

#-----------------------------------------------------------------------------
# open file for reading

dir = '/home/mark.miesch/data/DSCOVR/MAG/L1/'
file = dir+'oe_vc1_dscovr_s20220131000000_e20220131235959_p20220201000011_pub.nc'

# old school netcdf3 approach seems to work:


xfile = netcdf.NetCDFFile(file,'r')
data = xfile.variables['vc0Frames']
x = data[:].copy().astype('float')

print(type(x))

print(f"min, max = {np.min(x)} {np.max(x)}")

xfile.close()

#-----------------------------------------------------------------------------

j = 500

for i in np.arange(100):
    print(x[i,j])

plt.plot(x[:100,500])

plt.show()

#-----------------------------------------------------------------------------

