# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 15:13:55 2019

@author: Arrow
"""

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as la
from matrix_completion import gen_mask
from matrix_completion import *
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler
from sklearn.metrics import mean_absolute_error #平方绝对误差

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
num = 9

for offset in range(num):
    batch_x = np.array((df.loc[offset * 3200:(offset+1)*3200-1].sort_values(by='IMSI'))['rsrp'])
#    batch_y = np.array((df.loc[offset * 3200+1600:(offset+1)*3200-1].sort_values(by='IMSI'))['rsrp']).reshape(1,1600)
    orsrp = np.c_[orsrp,batch_x]

orsrp = 10*np.log10(orsrp) + 30   # turn the data into dBm
mrsrp = orsrp[0:1600,:]
mrsrp_sky = orsrp[1600:,:]
#mrsrp = np.floor(mrsrp)
mrsrp = np.transpose(mrsrp)
mrsrp_sky = np.transpose(mrsrp_sky)

head = np.reshape(mrsrp[1,:],[40,40])
tail = np.reshape(mrsrp[-1,:],[40,40])
head_sky=np.reshape(mrsrp_sky[1,:],[40,40])
plot_image(head)
#%%
height=[50]
const=2025
#进行映射的函数
def mapping(sky,m,n):
    powers=[]
    weights=[]
    pad=[-1,0,1]
    for i in pad:
        for j in pad:
            if(m+i>=0 and m+i<40 and n+j>=0 and n+j<40):
                #计算映射信号
                if sky[m+i][n+j]!=0:
                    d2= np.square(i*50)+np.square(j*50)+np.square(height)
                    power=sky[m+i][n+j]
                    weight= 1/d2
                    powers.append(power)
                    weights.append(weight)
                
#    for i in range(40):
#        for j in range(40):
#            if(sky[i][j]!=0):
#                d2=np.square((i-m)*50)+np.square((j-n)*50)+np.square(height)
##                power=const*sky[i][j]/d2
#                power=sky[i][j]*1.01
#                weight= 1/d2
#                powers.append(power)
#                weights.append(weight)
    res=0
    for k in range(len(powers)):
        res += powers[k]*(weights[k]/sum(weights))
        
    return res,sum(weights)       
    

sampling_rate=[0.1,0.2,0.3,0.4,0.5]

for sr in sampling_rate:
    mask = gen_mask(40,40,prob_masked=1-sr)
    rsrp=head*mask
    rsrp[rsrp==0]=np.NaN
    X_filled_nnm = NuclearNormMinimization().fit_transform(rsrp)
    print('sampling rate ',sr)
    plot_image(X_filled_nnm)
    error = mean_absolute_error(head,X_filled_nnm)
    print(error)

#%%
for sr in sampling_rate:
    step=sr/10
    mae=[]
    for rsrp in mrsrp:
        rsrp=np.reshape(rsrp,[40,40])
        errors=np.linspace(0,0,10)
        for i in range(10):
    #        i=1
            MASK1 = gen_mask(40,40,prob_masked=1-(sr-i*step))
            MASK2 = gen_mask(40,40,prob_masked=1-i*step)
    
            ground = MASK1* rsrp
            print('origin sample ground')
            plot_image(ground)
            plot_image(rsrp)
            
            
            sky=MASK2*head_sky
         
            weights=np.zeros((40,40))
            #遍历地面矩阵元素
            for m in range(40):
                if sky.all==0:
                    break
                for n in range(40):
                    if(ground[m][n]==0):
                  
                        ground[m][n],weights[m][n] = mapping(sky,m,n)
                    else:
                        weights[m][n]=1
         
            print('mapping ground')
            plot_image(ground)
            plot_image(weights)
            print(MASK1)
            print(head)
            print(ground)
    #%%        
    #        tw= np.reshape(weights,(1,1600))
    #        print(tw[0].size)
    #        order = np.argsort(tw)
    #        ground=np.reshape(ground,(1,1600))
    #    
    #        for k in range(1600-480):
    #   
    #            ground[0][order[0][k]]=0
    #        ground=np.reshape(ground,[40,40])
    #        plot_image(ground)
    #        print(ground)
    #%%     
            ground[ground==0]=np.NaN
    #         X_filled_nnm = SoftImpute().fit_transform(ground)
            X_filled_nnm = NuclearNormMinimization().fit_transform(ground)
            print('NuclearNormMinimization')
            plot_image(X_filled_nnm)
            error = mean_absolute_error(rsrp,X_filled_nnm)
            
            errors[i]+=error
        
    errors/=(num+1)   
    out = open('data/mcplus2.csv','a',newline='')
    writer=csv.writer(out,dialect='excel')
    writer.writerow(errors)

out.close()






