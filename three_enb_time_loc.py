#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 14:41:34 2019

@author: tao
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from matrix_completion import *
from impyute.imputation.cs import fast_knn
from impyute.imputation.cs import mice
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score#R square
from sklearn import preprocessing
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler,IterativeSVD
import csv
import tensorflow as tf
from sklearn.preprocessing import scale
import time

def plot_image(A):
    plt.imshow(A.T)
    plt.show()

def show():
    data_path = "../data/DlRsrpSinrStats.txt"
    grid_width = 40 
    time_slots = 100 
    ue_num = grid_width*grid_width
    uav_num = grid_width*grid_width
    total_num = ue_num+uav_num
    
    #data_set = np.loadtxt(data_path,delimiter='	',skiprows=1);
    data_set = pd.read_table(data_path,delimiter='\t')
    print(data_set.shape)
    
    df = pd.DataFrame(data_set)
    
#    out = open('/home/tao/dataset/out2.txt','w')
#    csv_write = csv.writer(out)
            
    
    rsrp=[]
    #construct matrix with location and timeslot
    bias = 3200*149
    for i in range(time_slots):
        temp = np.array((df.loc[bias+i*total_num:bias+(i+1)*total_num-1].sort_values('IMSI'))['rsrp'])
        temp=temp[0:1600]
     
        row = np.array(temp)
#        csv_write.writerow(temp)
        rsrp.append(row)
    #rsrp = np.array((df.loc[0:time_slots*total_num-1].sort_values('IMSI'))['rsrp'])
    #print('rsrp shape:',rsrp.shape)
    #print(rsrp.shape)
    mrsrp=[]
    for power in rsrp:
        temp_power = 10*np.log10(power) + 30
        mrsrp.append(temp_power)
    
    
    mrsrp = np.array(mrsrp)
    mrsrp = scale(mrsrp)
    
    
    print('mrsrp',mrsrp.shape)
    mat_rsrp = mrsrp.T
    print('fdfa')
    
    
    m = mat_rsrp.shape[0]
    n = mat_rsrp.shape[1]
    
    missing_rates = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]
#    [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
    
    # add some random noise to the data
    def add_noise(origin_data,noise_level):
        np.random.seed(42)
        mu = 0
        sigma = noise_level
        noise = np.random.normal(mu,sigma,origin_data.shape)
        # vonvert the noise the dBm
        print(noise)
        return origin_data + noise
    #random_mat = np.random.uniform(size=[])
    #mat_rsrp = add_noise(mat_rsrp,3)
    xaxis=[]
    yaxis=[]
    knny=[]

    nnmy=[]
    micey=[]
    
    knntime=[]
    nnmtime=[]
    micetime=[]

    for missing_rate in missing_rates:
        mask = gen_mask(m, n, missing_rate)
        sample_data = mat_rsrp * mask
        print(missing_rate)
        print('origin_data')
        plot_image(mat_rsrp)
        print(mat_rsrp)
        
        
        
        print('the sample_data')
        plot_image(sample_data)
        print(sample_data)
        
     
        try:
            t1 = time.time()
            
            sample_data[sample_data==0] = np.nan
            knn_recover = fast_knn(sample_data,k=3)
            print('knn')
            plot_image(knn_recover)
            error_knn = mean_absolute_error(mat_rsrp,knn_recover)
            knny.append(error_knn)
            
            t2 = time.time()
            ktime = (t2-t1)
            
            knntime.append(ktime)
        except ValueError:
            knny.append(2)
            
            t2 = time.time()
            knntime.append(1200*(1+missing_rate))
                
        try:   
            t1 = time.time()
            
            mice_data = mice(sample_data)
            print('mice')
            plot_image(mice_data)
            error_mice = mean_absolute_error(mat_rsrp,mice_data)
            micey.append(error_mice)
            
            t2 = time.time()
            micetime.append(t2-t1)
        except ValueError:
            micey.append(2)
            
            t2 = time.time()
            micetime.append(1200*(1+missing_rate))
        
            
        try:
            t1 = time.time()
            
            X_filled_nnm = SoftImpute().fit_transform(sample_data)
            print('SoftImpute')
            plot_image(X_filled_nnm)
            error_nuclear = mean_absolute_error(mat_rsrp,X_filled_nnm)
            nnmy.append(error_nuclear)
            
            t2 = time.time()
            nnmtime.append(t2-t1)
            
        except:
            nnmy.append(2)
        
            t2 = time.time()
            nnmtime.append(1200*(1+missing_rate))
        

        xaxis.append(missing_rate)
     
        
#    plt.plot(xaxis,yaxis)
#    plt.xlabel("missing_rate")
#    plt.ylabel("mae")
#    plt.show()
    return np.array([xaxis,knny,micey,nnmy, knntime, micetime, nnmtime]) ;

res = show()
print(res)
out = open('test2.csv','a', newline='')
#csv_write = csv.writer(out,dialect='excel')
for d in res:
    print(d)
    csv_write.writerow(d)
out.close()






    