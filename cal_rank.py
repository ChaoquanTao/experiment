# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 10:01:34 2019

@author: Arrow
"""
import matplotlib.mlab as mlab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as la

def plot_image(A):
    plt.imshow(A.T)
    plt.show()

data_path = "../data/DlRsrpSinrStats.txt"

data_set = pd.read_table(data_path,delimiter='\t')
print(data_set.shape)

df = pd.DataFrame(data_set)
print(df.size)

orsrp = np.array((df.loc[0:3199].sort_values(by='IMSI'))['rsrp'])

#rsrp=orsrp[0:1600]
#sky_rsrp = orsrp[1600:]
num = 499

for offset in range(num):
    batch_x = np.array((df.loc[offset * 3200:(offset+1)*3200-1].sort_values(by='IMSI'))['rsrp'])
#    batch_y = np.array((df.loc[offset * 3200+1600:(offset+1)*3200-1].sort_values(by='IMSI'))['rsrp']).reshape(1,1600)
    orsrp = np.c_[orsrp,batch_x]

orsrp = 10*np.log10(orsrp) + 30   # turn the data into dBm
mrsrp = orsrp[0:1600,:]
mrsrp_sky = orsrp[1600:,:]
mrsrp = np.floor(mrsrp)
mrsrp = np.transpose(mrsrp)
mrsrp_sky = np.transpose(mrsrp_sky)


head = np.reshape(mrsrp[1,:],[40,40])
tail = np.reshape(mrsrp[-1,:],[40,40])
head_sky=np.reshape(mrsrp_sky[1,:],[40,40])
res = head/head_sky
res1 = res * 2500
re1 = np.reshape(res1,[1,1600])[0]

U,sigma,VT=la.svd(tail)
print(U.shape,sigma.shape,VT.shape)
near = U.dot(np.diag(sigma)).dot(VT)
print(np.linalg.matrix_rank(tail))
print(np.linalg.matrix_rank(near))
# plot_image(head)
# plot_image(tail)
# plot_image(near)

all=[]
thanone=[]
for rsrp in mrsrp:
    U,sigma,VT=la.svd(np.reshape(rsrp,[40,40]))
    all.extend(sigma)
    cnt = sum(i > 20 for i in sigma)
    thanone.append(cnt)
 

#n, bins, patches = plt.hist(all)
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 18,
}
plt.xlabel('Singular value')
plt.ylabel('Amount')

plt.show()

sub = [n for n in all if n<20]
hist, bin_edges = np.histogram(sub)

plt.hist(sub)
plt.xlabel('Singular value',font2)
plt.ylabel('Amount',font2)

plt.show()

#plt.plot(hist)
#plt.show()
cdf = np.cumsum(hist/sum(hist))
#plt.hist(all)


#plt.hist(sub)
#plt.plot(bin_edges[:len(bin_edges)-1],cdf)
#plt.show()
