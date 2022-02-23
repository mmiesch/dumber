"""
A simple figure highlighting aliasing
"""

import numpy as np

import matplotlib.pyplot as plt

n = 1000

tmax = float(n)

time = np.linspace(0, tmax, num = n, endpoint = False, dtype='float')

k = 30
b = np.cos(2*np.pi*k*time/tmax)

n2 = 28
time2 = np.linspace(0, tmax, num = n2, endpoint = False, dtype='float')

k2 = 2
b2 = np.cos(2*np.pi*k2*time2/tmax)
b2b = np.cos(2*np.pi*k2*time/tmax)

plt.figure(figsize=(12,6))

plt.plot(time,b,'k-')
plt.plot(time,b2b,'b:')
plt.plot(time2,b2,'bo')

plt.xlabel("time (arbitrary units)")

plt.show()