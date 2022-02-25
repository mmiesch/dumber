"""
Code to summarize the timings for writing the ATBD
"""

import numpy as np
import pandas as pd

sam = 3

if sam == 1:
    file1 = 'DSCOVR_MAG_4_4317541.csv'
    file2 = 'DSCOVR_MAG_4-8_4317541.csv'
    ns = 4317541
    nw = 4
elif sam == 2:
    file1 = 'DSCOVR_MAG_8_4317541.csv'
    file2 = 'DSCOVR_MAG_8-16_4317541.csv'
    ns = 4317541
    nw = 8
elif sam == 3:
    file1 = 'DSCOVR_MAG_480_172800.csv'
    file2 = 'DSCOVR_MAG_480-960_172800.csv'
    ns = 172800
    nw = 480


# na is the length of the averaged variable
if np.mod(ns,nw) == 0:
    na = np.int64(ns / nw)
else:
    na = np.int64(ns / nw) + 1


file1 = 'timings/'+file1
file2 = 'timings/'+file2

# time in seconds to compute a 6-hr internal with 1 min windows 
# (360 window)
norm = 360.0 / (float(na))

#-------------------------

df1 = pd.read_csv(file1, header=None)

df1[3] = df1[1] * norm
df1[4] = df1[2] * norm

df1[5] = df1[1]/df1[0]
df1[6] = df1[2]/df1[0]

print(80*'*')
print(file1)
print(df1)

#-------------------------

df2 = pd.read_csv(file2, header=None)

df2[3] = df2[1] * norm
df2[4] = df2[2] * norm

df2[5] = df2[1]/df2[0]
df2[6] = df2[2]/df2[0]

print(80*'*')
print(file2)
print(df2)

#-------------------------
# overhead of overlapping windwo

print(80*'*')

print(df2.iloc[-1,1]/df1.iloc[-1,1])
print(df2.iloc[-1,2]/df1.iloc[-1,2])

#-------------------------
print(80*'*')
print("HL: {ts:18.2f} {tb:18.2f}".format(ts=df1.iloc[-1,3],tb=df1.iloc[-1,5]))
print(" M: {ts:18.2f} {tb:18.2f}".format(ts=df1.iloc[-1,4],tb=df1.iloc[-1,6]))
print("HL: {ts:18.2f} {tb:18.2f}".format(ts=df2.iloc[-1,3],tb=df2.iloc[-1,5]))
print(" M: {ts:18.2f} {tb:18.2f}".format(ts=df2.iloc[-1,4],tb=df2.iloc[-1,6]))
print(80*'*')




