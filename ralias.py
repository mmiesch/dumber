"""
Assess different averaging techniques
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.optimize as opt

from netCDF4 import Dataset
from time import perf_counter
from scipy.stats import cauchy

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

def MAD(x, c = 1.4826):
    """
    Median Absolute Deviation of a sample 
    """

    x0 = np.median(x)
    y = np.abs(x - x0)
    return c * np.median(y)

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
#  get a data segment to work with

sam = 1

# default titles - change only if desired
xtitle = 'time (arbitrary units)'
ytitle = 'signal (arbitrary units)'

if sam == 1:
    # atrificial data with a wider window
    ns = 160
    nw = 8
    nb = nw
    #nb = 2*nw
    #nb = 3*nw
    k = 12.0
else:
    print("pick a sam")
    exit()
 
t2 = float(ns)
time = np.linspace(0, t2, num = ns, endpoint = False, dtype='float')

bz = np.cos(2*np.pi*k*time/t2)

#-----------------------------------------------------------------------------
# define bins

# ns is the length of the sample
ns = np.int64(len(time))

# na is the length of the averaged variable
if np.mod(ns,nw) == 0:
    na = np.int64(ns / nw)
else:
    na = np.int64(ns / nw) + 1

print(f"ns, na = {ns} {na}")

# define time at the center of each window
tbox = np.empty(na, dtype = float)

for i in np.arange(na,dtype=np.int64):
    i1 = i*nw
    i2 = np.min([i1+nw,ns])
    #print(f"{i1} {i2-1}")
    mw = i2 - i1
    tbox[i] = 0.5*(time[i1] + time[i2-1])

#-----------------------------------------------------------------------------
# resampling with boxcar average

bzbox = np.empty(na, dtype = float)

# bin overlap
nov = int((nb - nw)/2)

for i in np.arange(na,dtype=np.int64):
    i1 = i*nw - nov
    i2 = i1 + nb
    i1 = np.max([i1,0])
    i2 = np.min([i2,ns])
    print(f"{i1} {i2}")
    mw = i2 - i1
    bzbox[i] = np.sum(bz[i1:i2]) / mw


#-----------------------------------------------------------------------------
# Hodges-Lehmann estimator

bzhl = np.empty(na, dtype = float)

# allocate work array
nhl = int((nb*(nb+1))/2)
work = np.zeros(nhl)

for i in np.arange(1,na-1,dtype=np.int64):
    i1 = i*nw - nov
    i2 = i1 + nb

    bzhl[i] = HL(bz[i1:i2],work,nb)

# do the first and last bins seperately because they may not be the same size
i1 = 0
i2 = nw + nov
mw = i2 - i1
nhl2 = int((mw*(mw+1))/2)
work2 = np.zeros(nhl2)
bzhl[na-1] = HL(bz[i1:i2],work2,mw)

i1 = (na-1)*nw
i2 = ns
mw = i2 - i1
nhl2 = int((mw*(mw+1))/2)
work2 = np.zeros(nhl2)
bzhl[na-1] = HL(bz[i1:i2],work2,mw)

#-----------------------------------------------------------------------------
# M-estimator

bzm = np.empty(na, dtype = float)

tm_start = perf_counter()

for i in np.arange(na,dtype=np.int64):
    i1 = i*nw - nov
    i2 = i1 + nb
    i1 = np.max([i1,0])
    i2 = np.min([i2,ns])

    x = bz[i1:i2]

    scale = MAD(x)
    mu0 = np.median(x)

    try:
        loc = opt.newton(Huber_psi, mu0, fprime=Huber_psi_prime, args = (x, scale), tol=1.e-6)
    except:
        print("M-ESTIMATOR FAILED TO CONVERGE: DEFAULTING TO HL")
        mw = i2 - i1
        nhl = int(mw*(mw+1)/2)
        work = np.empty(nhl)
        loc = HL(x,work,mw)

    bzm[i] = loc

tm_stop = perf_counter()
dtm = tm_stop - tm_start

#-----------------------------------------------------------------------------

plt.figure(figsize=(12,6))

ii = nw

yy = np.array([-2,2])

while ii < ns:
    t = time[ii] * np.array([1.0,1.0])
    plt.plot(t,yy,'k:')
    ii += nw

xx = np.array([0,np.max(time)])
yy = np.array([0,0])
plt.plot(xx,yy,'k:')

plt.plot(time,bz,'k-')
plt.plot(tbox,bzbox,linewidth=4,color='red')

#plt.plot(tbox,bzhl,linewidth=6,color='blue')
#plt.plot(tbox,bzm,linewidth=4,color='#B1FB17')

plt.xlim([0,np.max(time)])
plt.ylim([-2,2])

plt.xlabel(xtitle)
plt.ylabel(ytitle)

plt.show()

#-----------------------------------------------------------------------------
