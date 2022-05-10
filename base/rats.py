"""
Assess different averaging techniques
"""

from email.errors import NoBoundaryInMultipartDefect
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

def binrange(i, nw, nb, nov, ns):
    i1 = np.max([i*nw-nov,0])
    i2 = np.min([i1+nb,ns])
    return i1, i2

#-----------------------------------------------------------------------------
# DSCOVR MAG file

dir = '/home/mark.miesch/data/DSCOVR/MAG/L1/'

# Feb 5
file = dir+'oe_mg1_dscovr_s20220205000000_e20220205235959_p20220206013755_pub.nc'

#-----------------------------------------------------------------------------
#  get a data segment to work with

sam = 9

# default titles - change only if desired
xtitle = 'time (arbitrary units)'
ytitle = 'signal (arbitrary units)'

kk = 1.0

if sam == 1:
    # this is a pretty good time range for figures
    label="DSCOVR_MAG"
    i1 = 2000
    i2 = i1 + 200
    doplot = True
    nw = 4
    xtitle = 'time (seconds)'
    ytitle = 'Bz (nT)'
elif sam == 2:
    # full range of data: good for efficiency runs
    label="DSCOVR_MAG"
    i1 = 0
    i2 = -1
    doplot = False
    nw = 4
elif sam == 3:
    # For playing around 
    label="DSCOVR_MAG"
    i1 = 2000
    i2 = i1 + 300
    doplot = True
    nw = 4
elif sam == 4:
    # a long run (equivalent to 6 hrs at 8 points/sec) 
    # with a long window to make sure it's doing the Right Thing
    label="DSCOVR_MAG"
    i1 = 2000
    i2 = i1 + 172800
    doplot = False
    nw = 480
elif sam == 5:
    # a first experiment with artificial data
    label="ART_Cauchy"
    rseed = 584303
    ns = 200
    nw = 4
    doplot = True
elif sam == 6:
    # artificial data with a wider window
    label="ART_Cauchy"
    rseed = 73947652
    ns = 800
    nw = 8
    doplot = True
elif sam == 7:
    # long run with artificial data to test timing, convergence
    label="ART_Cauchy"
    rseed = 73947652
    ns = 1000000
    nw = 480
    doplot = False
elif sam == 8:
    # Check smoothing with a sharp profile
    label="ART_Shock"
    rseed = 73947652
    ns = 800
    nw = 8
    doplot = True
elif sam == 9:
    # Accuracy runs
    label="ART_Cauchy"
    rseed = 73947652
    ns = 80000
    kk = 10
    nw = 480
    doplot = False
else:
    i1 = 2000; i2 = i1 + 200 # sam1

if label == "DSCOVR_MAG":

    rootgrp = Dataset(file, "r", format="NETCDF3")
    tvar = rootgrp.variables['time']
    bzvar = rootgrp.variables['bz_gsm']

    if i2 < 0:
       i2 = np.int64(len(tvar))

    time = (tvar[i1:i2] - tvar[i1])*1.e-3
    bz = bzvar[i1:i2].copy()

elif label == "ART_Shock":
    # Artificial data with a Cauchy distribution
    t2 = float(ns)
    time = np.linspace(0, t2, num = ns, endpoint = True, dtype='float')

    beta = 2.0
    tmax = np.max(time)
    bz = 2*np.arctan(beta*(time-0.5*tmax))/np.pi
    
    #np.random.seed(rseed)
    #noise = cauchy.rvs(loc = 0.0, scale = 0.1, size = ns) 
    #bz += noise

else:

    # Artificial data with a Cauchy distribution
    t2 = float(ns)
    time = np.linspace(0, t2, num = ns, endpoint = True, dtype='float')

    bz = np.cos(2*np.pi*kk*time/t2)
    np.random.seed(rseed)
    noise = cauchy.rvs(loc = 0.0, scale = 0.1, size = ns) 
    bz += noise

#-----------------------------------------------------------------------------
# define bins

# ns is the length of the sample
ns = np.int64(len(time))

# nb is the size if the averaging window
# nov is the overlap with neighboring bins
nb = 2*nw
#nb = nw
nov = int((nb - nw)/2)

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

tbox_start = perf_counter()

for i in np.arange(na,dtype=np.int64):
    i1, i2 = binrange(i, nw, nb, nov, ns)
    mw = i2 - i1
    bzbox[i] = np.sum(bz[i1:i2]) / mw

tbox_stop = perf_counter()
dtbox = tbox_stop - tbox_start


#-----------------------------------------------------------------------------
# Hodges-Lehmann estimator

bzhl = np.empty(na, dtype = float)

thl_start = perf_counter()

# allocate work array
nhl = int((nb*(nb+1))/2)
work = np.zeros(nhl)

for i in np.arange(1,na-2,dtype=np.int64):
    i1, i2 = binrange(i, nw, nb, nov, ns)
    bzhl[i] = HL(bz[i1:i2],work,nb)

# do end bins seperately because they may not be the same size
i1 = 0
i2 = nw + nov
mw = i2 - i1
nhl2 = int((mw*(mw+1))/2)
work2 = np.zeros(nhl2)
bzhl[0] = HL(bz[i1:i2],work2,mw)

i = na-2
i1, i2 = binrange(i, nw, nb, nov, ns)
mw = i2 - i1
nhl2 = int((mw*(mw+1))/2)
work2 = np.zeros(nhl2)
bzhl[na-2] = HL(bz[i1:i2],work2,mw)

i1 = (na-1)*nw - nov
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

#
#  This was having problems converging for nw greater than 4
#
#huber = sm.robust.scale.Huber(tol=1e-6, maxiter=1000)
#    for i in np.arange(na):
#        i1 = i*nw
#        i2 = i1+nw
#    
#        loc, scale = huber(bz[i1:i2])
#    
#        bzm[i] = loc

for i in np.arange(na,dtype=np.int64):
    i1, i2 = binrange(i, nw, nb, nov, ns)
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
# print timings to a csv file

print(80*"-"+"\nTimings")
print("boxcar".center(21)+"HL".center(21)+"M-estimator".center(21))
print("{0:18.6e}, {1:18.6e}, {2:18.6e}".format(dtbox, dthl, dtm))
print(80*"-")

# also write to a csv file
outfilename = "timings/"+label+"_"+str(nw)+"-"+str(nb)+"_"+str(ns)+".csv"

outfile = open(outfilename,"a")
outfile.write("{0:18.6e}, {1:18.6e}, {2:18.6e}\n".format(dtbox, dthl, dtm))
outfile.close()

#-----------------------------------------------------------------------------
# Compute accuracy

if label == "ART_Cauchy":

    tmax = max(tbox)

    # correct answer on the tbox grid
    bzcheck = np.cos(2*np.pi*kk*tbox/tmax)

    # sanity check
    #plt.figure(figsize=(12,6))
    #plt.plot(tbox,bzcheck)
    ##plt.plot(tbox,bzbox)
    ##plt.plot(tbox,bzhl)
    #plt.plot(tbox,bzm)
    #plt.ylim(-4,4)
    #plt.show()

    # make sure the lengths are the same
    print(f"data lengths: {len(bzcheck)} {len(bzbox)} {len(bzhl)} {len(bzm)}")

    dbox = np.abs(bzbox - bzcheck)
    dhl = np.abs(bzhl - bzcheck)
    dm = np.abs(bzm - bzcheck)

    ebox = np.sum(dbox)/len(dbox)
    ehl = np.sum(dhl)/len(dhl)
    edm = np.sum(dm)/len(dm)

    print(f"Accuracy: {ebox} {ehl} {edm}")

#-----------------------------------------------------------------------------

if doplot:

    plt.figure(figsize=(12,6))

    if label == "ART_Shock":
        
        plt.plot(time,bz,'k-')
        plt.plot(tbox,bzbox,linewidth=4,color='red')
        plt.plot(tbox,bzhl,linewidth=8,color='blue')
        plt.plot(tbox,bzm,linewidth=4,color='#B1FB17')
    
        plt.xlim(380,420)
        plt.plot(time,bz,'o',color='black')
        plt.plot(tbox,bzbox,'o',color='red')
        plt.plot(tbox,bzhl,'o',color='blue')
        plt.plot(tbox,bzm,'o',color='#B1FB17')

    else:
        plt.plot(time,bz,'k-')
        plt.plot(tbox,bzbox,linewidth=4,color='red')
        plt.plot(tbox,bzhl,linewidth=6,color='blue')
        plt.plot(tbox,bzm,linewidth=4,color='#B1FB17')
    
        if label == "ART_Cauchy":
            plt.ylim(-4,4)

    plt.xlabel(xtitle)
    plt.ylabel(ytitle) 

    plt.show()

#-----------------------------------------------------------------------------


if label == "DSCOVR_MAG":
    rootgrp.close()
