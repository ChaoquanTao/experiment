# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 15:56:10 2019

@author: Arrow
"""

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from matrix_completion import *
from matrix_completion import gen_mask
import matplotlib.pyplot as plt
from impyute.imputation.cs import fast_knn
from impyute.imputation.cs import mice
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score#R square
from sklearn import preprocessing
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler
import tensorflow as tf
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
import time

def gen_mask2(mr):
    mat = np.ones((40,40),dtype=int)
    for i in range(mr):
        for j in range(mr):
            mat[i+5][j+5]=0
    return np.reshape(mat,(1,-1))

def plot_image(A):
    plt.imshow(A.T)
    plt.show()
def show():
    time_slots = 100 

    data_path = "../data/DlRsrpSinrStats.txt"
    
    #data_set = np.loadtxt(data_path,delimiter='	',skiprows=1);
    data_set = pd.read_table(data_path,delimiter='\t')
    print(data_set.shape)
    
    df = pd.DataFrame(data_set)
    
    
    
    rsrp=[]
    #construct matrix with location and timeslot
    bias = 3200*149
    for i in range(time_slots):
        temp = np.array((df.loc[bias+i*3200:bias+(i+1)*3200-1].sort_values('IMSI'))['rsrp'])
        temp=temp[0:1600]

        row = np.array(temp)
      
        rsrp.append(row)
    #rsrp = np.array((df.loc[0:time_slots*total_num-1].sort_values('IMSI'))['rsrp'])
    #print('rsrp shape:',rsrp.shape)
    #print(rsrp.shape)

    mrsrp=[]
    for power in rsrp:
        temp_power = 10*np.log10(power) + 30
        mrsrp.append(temp_power)
    
    
    mrsrp = np.array(mrsrp)
    
#    mrsrp = scale(mrsrp)
    
    
    print('mrsrp',mrsrp.shape)

   
    # generate a maskmatrix
    m = 40
    n = 40
    
    missing_rates = [10,15,20,25,30]
    #[0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
    #random_mat = np.random.uniform(size=[])
    xaxis=[]
 
    knnxaxis=[]
    knnyaxis=[]

#    iiyaxis=[]
    nnmxaxis=[]
    nnmyaxis=[]
    micexaxis=[]
    miceyaxis=[]
    
    knntime=[]
    nnmtime=[]
    micetime=[]
    
    for missing_rate in missing_rates:
        knny=[]
#        iiy=[]
        nny=[]
        micey=[]
        nnmy=[]
        
        ktime=0
        ntime=0
        mtime=0
        
        for mat_rsrp in mrsrp:
            mat_rsrp = np.array(mat_rsrp).reshape(40,40)
            #mask = gen_mask(m, n, missing_rate)
            mask = gen_mask2(missing_rate).reshape(40,40)
            print(mask)
            sample_data = mat_rsrp* mask  
            
            print('origin_data')
            plot_image(mat_rsrp)
            
            print('the sample_data')
            plot_image(sample_data)
            
            try:
                t1 = time.time()
                sample_data[sample_data==0] = np.nan
                knn_recover = fast_knn(sample_data,k=3)
                print('knn')
                plot_image(knn_recover)
                error_knn = mean_absolute_error(mat_rsrp,knn_recover)
                knny.append(error_knn)
                t2 = time.time()
                ktime = ktime + (t2-t1)
            except ValueError:
                knny.append(2)
                t2 = time.time()
                ktime = ktime + 600*(1+missing_rate)
                
            try:   
                t1 = time.time()
                mice_data = mice(sample_data)
                print('mice')
                plot_image(mice_data)
                error_mice = mean_absolute_error(mat_rsrp,mice_data)
                micey.append(error_mice)
                
                t2 = time.time()
                mtime = mtime + (t2-t1)
            except ValueError:
                micey.append(2)
                
                t2 = time.time()
                mtime = mtime + 600*(1+missing_rate)
                
            try:
                t1 = time.time()
                
                X_filled_nnm = SoftImpute().fit_transform(sample_data)
                print('NuclearNormMinimization')
                plot_image(X_filled_nnm)
                error_nuclear = mean_absolute_error(mat_rsrp,X_filled_nnm)
                nnmy.append(error_nuclear)
                
                t2 = time.time()
                ntime = ntime + (t2-t1)
            except:
                nnmy.append(2)
           
                t2 = time.time()
                ntime = ntime + 600*(1+missing_rate)
            
            break
#            print("\tknn:",error_knn,"\tmice:",error_mice,"\titer:",error_iter,"\tnuclear",error_nuclear)
        knntime.append(ktime)
        nnmtime.append(ntime)
        micetime.append(mtime)
        
        knnyaxis.append(np.mean(np.array(knny)))  
        miceyaxis.append(np.mean(np.array(micey)))
#        iiyaxis.append(np.mean(np.array(iiy)))
        nnmyaxis.append(np.mean(np.array(nnmy)))
        xaxis.append(missing_rate)
    
    
#    plt.plot(xaxis,iiy,c='red',label='iter')
#    plt.plot(xaxis,knny,c='blue',label='knn')
#    plt.plot(xaxis,nnmy,c='orange',label='nnm')
#    plt.plot(xaxis,micey,c='black',label='mice')
#    
#    plt.xlabel("missing rate")
#    plt.ylabel("mae")
#    plt.legend()
#    plt.show()
        res = np.array([xaxis,knnyaxis,nnmyaxis,miceyaxis, knntime,nnmtime,micetime])
    return res ;

res = show()
print(res)
out = open('data/block_loc_loc.csv','a', newline='')
csv_write = csv.writer(out,dialect='excel')
for d in res:
    print(d)
    csv_write.writerow(d)
out.close()