"""
Assess different averaging techniques
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import statsmodels.api as sm
import scipy.optimize as opt

from netCDF4 import Dataset
from time import perf_counter

#-----------------------------------------------------------------------------

def HL(x,w,n):
    """
    Hodges-Lehmann locaion estimator
    To avoid the computational expense of re-allocating it each time, pass the
    work array and size as arguments

    x = input array of size n
    w = work array of size n(n+1)/2
    n = length of x
    """

    idx = 0
    for j in np.arange(n):
        for i in np.arange(j+1):
            w[idx] = 0.5*(x[i]+x[j])
            idx += 1

    return np.median(w)

def Huber_psi(mu,x,sigma,k=1.345):
    r = (x - mu)/sigma
    z = np.where(np.abs(r) <= k, r, np.sign(r)*k)
    return np.sum(z)

def Huber_psi_prime(mu,x,sigma,k=1.345):
    r = (x - mu)/sigma
    dr = -1.0/sigma
    z = np.where(np.abs(r) <= k, dr, 0.0)
    return np.sum(z)

#-----------------------------------------------------------------------------
# open file for reading

dir = '/home/mark.miesch/data/DSCOVR/MAG/L1/'

# Feb 5
file = dir+'oe_mg1_dscovr_s20220205000000_e20220205235959_p20220206013755_pub.nc'

#-----------------------------------------------------------------------------
#  get a data segment to work with

rootgrp = Dataset(file, "r", format="NETCDF3")
tvar = rootgrp.variables['time']
bzvar = rootgrp.variables['bz_gsm']

sam = 3

if sam == 1:
    # this is a pretty good time range for figures
    label="DSCOVR_MAG"
    i1 = 2000
    i2 = i1 + 200
    doplot = True
elif sam == 2:
    # full range of data: good for efficiency runs
    label="DSCOVR_MAG"
    i1 = 0
    i2 = len(tvar)
    doplot = False
if sam == 3:
    # this is a pretty good time range for figures
    label="DSCOVR_MAG"
    i1 = 2000
    i2 = i1 + 300
    doplot = True
else:
    i1 = 2000; i2 = i1 + 200 # sam1

time = tvar[i1:i2] - tvar[0]
bz = bzvar[i1:i2].copy()

#-----------------------------------------------------------------------------
# define bins

# ns is the length of the sample
ns = len(time)

# nw is the length of the averaging window
#nw = 4
nw = 8

# na is the length of the averaged variable
if np.mod(ns,nw) == 0:
    na = int(ns / nw)
else:
    na = int(ns / nw) + 1

print(f"ns, na = {ns} {na}")

# define time at the center of each window
tbox = np.empty(na, dtype = float)

for i in np.arange(na):
    i1 = i*nw
    i2 = np.min([i1+nw,ns])
    #print(f"{i1} {i2-1}")
    mw = i2 - i1
    tbox[i] = 0.5*(time[i1] + time[i2-1])

#-----------------------------------------------------------------------------
# resampling with boxcar average

bzbox = np.empty(na, dtype = float)

tbox_start = perf_counter()

for i in np.arange(na):
    i1 = i*nw
    i2 = np.min([i1+nw,ns])
    mw = i2 - i1
    bzbox[i] = np.sum(bz[i1:i2]) / mw

tbox_stop = perf_counter()
dtbox = tbox_stop - tbox_start


#-----------------------------------------------------------------------------
# Hodges-Lehmann estimator

bzhl = np.empty(na, dtype = float)

thl_start = perf_counter()

# allocate work array
nhl = int((nw*(nw+1))/2)
work = np.zeros(nhl)

for i in np.arange(na-1):
    i1 = i*nw
    i2 = i1+nw
    bzhl[i] = HL(bz[i1:i2],work,nw)

# do the last bin seperately because it may not be the same size
i1 = (na-1)*nw
i2 = ns
mw = i2 - i1
nhl2 = int((mw*(mw+1))/2)
work2 = np.zeros(nhl2)
bzhl[na-1] = HL(bz[i1:i2],work2,mw)

thl_stop = perf_counter()
dthl = thl_stop - thl_start

#-----------------------------------------------------------------------------
# M-estimator

bzm = np.empty(na, dtype = float)

tm_start = perf_counter()

#huber = sm.robust.scale.Huber(tol=1e-6, maxiter=1000)
#try:
#    for i in np.arange(na):
#        i1 = i*nw
#        i2 = i1+nw
#    
#        loc, scale = huber(bz[i1:i2])
#    
#        bzm[i] = loc
#except:
#
#    print("STATSMOD FAILED: Using custom estimator")

for i in np.arange(na):
    i1 = i*nw
    i2 = i1+nw

    x = bz[i1:i2]

    scale = sm.robust.scale.mad(x)
    mu0 = np.median(x)

    loc = opt.newton(Huber_psi, mu0, fprime=Huber_psi_prime, args = (x, scale), tol=1.e-6)

    bzm[i] = loc

tm_stop = perf_counter()
dtm = tm_stop - tm_start

#-----------------------------------------------------------------------------
# print timings to a csv file


print(80*"-"+"\nTimings")
print("boxcar".center(21)+"HL".center(21)+"M-estimator".center(21))
print("{0:18.6e}, {1:18.6e}, {2:18.6e}".format(dtbox, dthl, dtm))
print(80*"-")

# also write to a csv file
outfilename = "timings/"+label+"_"+str(nw)+"_"+str(na)+".csv"

outfile = open(outfilename,"a")
outfile.write("{0:18.6e}, {1:18.6e}, {2:18.6e}\n".format(dtbox, dthl, dtm))
outfile.close()

#-----------------------------------------------------------------------------

if doplot:

    plt.plot(time,bz,'k-')
    plt.plot(tbox,bzbox,linewidth=4,color='#808080')
    plt.plot(tbox,bzhl,linewidth=4,color='#00FFFF')
    plt.plot(tbox,bzm,linewidth=4,color='#B1FB17')
    plt.show()

#-----------------------------------------------------------------------------

rootgrp.close()
