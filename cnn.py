#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 19:51:45 2019

@author: tao
"""

from __future__ import print_function
import tensorflow as tf

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
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
import time

#import three_enb_loc_loc
#import three_enb_time_loc
#xaxis,knny,nnmy,micey = three_enb_loc_loc.show()
#
#filename = './data/figure.txt'
#with open(filename,'w') as file_object:
#    file_object.write(xaxis+'\n'+knny+'\n'+nnmy+'\n'+micey)

#_,knny2,nnmy2,micey2 = three_enb_time_loc.show()

tf.reset_default_graph()
def plot_image(A):
    plt.imshow(A.T)
    plt.show()

data_path = "../data/DlRsrpSinrStats.txt"

#data_set = np.loadtxt(data_path,delimiter='	',skiprows=1);
data_set = pd.read_table(data_path,delimiter='\t')
print(data_set.shape)

df = pd.DataFrame(data_set)
print(df.size)

orsrp = np.array((df.loc[0:3199].sort_values(by='IMSI'))['rsrp'])
#rsrp=orsrp[0:1600]
#sky_rsrp = orsrp[1600:]

num = 99

for offset in range(num):
    batch_x = np.array((df.loc[offset * 3200:(offset+1)*3200-1].sort_values(by='IMSI'))['rsrp'])
#    batch_y = np.array((df.loc[offset * 3200+1600:(offset+1)*3200-1].sort_values(by='IMSI'))['rsrp']).reshape(1,1600)
    orsrp = np.c_[orsrp,batch_x]
    
#orsrp = np.array((df.loc[0:3199].sort_values(by='IMSI'))['rsrp'])
rsrp=orsrp[0:1600,:]
sky_rsrp = orsrp[1600:,:]
# %%
print(rsrp.size)

bias = 150*3200
trsrp=np.array((df.loc[149*3200:150*3200-1].sort_values(by='IMSI'))['rsrp'])
for offset in range(num):
    test = np.array((df.loc[bias+offset*3200:bias + (offset+1)*3200-1].sort_values(by='IMSI'))['rsrp'])
    trsrp= np.c_[trsrp,test]
trsrp_g = trsrp[0:1600,:]
trsrp_s = trsrp[1600:3200,:]

mrsrp = []
mrsrp_sky = []
test_rsrp=[]
test_sky=[]
for i in range(num+1):
    for power in rsrp[:,i]:
        temp_power = 10*np.log10(power) + 30.0
        mrsrp.append(temp_power)    
        
    for sky_power in sky_rsrp[:,i]:
        temp_power = 10*np.log10(sky_power) + 30
        mrsrp_sky.append(temp_power)
    
    tpower= 10*np.log10(trsrp_g[:,i]) + 30
    test_rsrp.append(tpower)
    
    tpower = 10*np.log10(trsrp_s[:,i]) + 30
    test_sky.append(tpower)
    
mrsrp = np.array(mrsrp).reshape(1600,num+1)
mrsrp_sky = np.array(mrsrp_sky).reshape(1600,num+1)
test_rsrp = np.array(test_rsrp).reshape(1600,num+1)
test_sky = np.array(test_sky).reshape(1600,num+1)

# %%
# data scale
#mrsrp = scale(mrsrp)
#mrsrp_sky = scale(mrsrp_sky)

scaler = StandardScaler().fit(mrsrp)
sky_scaler =StandardScaler().fit(mrsrp_sky)
mrsrp = scaler.transform(mrsrp)
mrsrp_sky = sky_scaler.transform(mrsrp_sky)


print("origin data image")
origin_i = mrsrp[:,2]
origin_i = np.reshape(origin_i,(40,40))
plot_image(origin_i)
# %%
trsrp=test_rsrp
trsrp_sky=test_sky

#for power in test_rsrp:
#    temp_power = 10*np.log10(power) + 30.0
#    trsrp.append(temp_power)
#
#for power in test_sky:
#    temp_power = 10*np.log10(power) + 30.0
#    trsrp_sky.append(temp_power)

# data scale 
#trsrp = scale(trsrp)
#trsrp_sky = scale(trsrp_sky)
tscaler=StandardScaler()
tscaler_sky = StandardScaler()
trsrp = tscaler.fit_transform(trsrp)
trsrp_sky = tscaler_sky.fit_transform(trsrp_sky)

mrsrp = np.array(mrsrp).reshape(-1,1600)
mrsrp_sky = np.array(mrsrp_sky).reshape(-1,1600)
trsrp = np.array(trsrp).reshape(-1,1600)
trsrp_sky=np.array(trsrp_sky).reshape(-1,1600)



print(mrsrp)
#out = open('/home/tao/dataset/out.csv','w')
#csv_write = csv.writer(out)
#for i in mrsrp:
#    csv_write.writerow(i)    
mat_rsrp = mrsrp
matsky_rsrp = mrsrp_sky
print('fdfa')

# preprocessing
#stand_means = preprocessing.MinMaxScaler()
#mat_rsrp = stand_means.fit_transform(mat_rsrp)
#matsky_rsrp = stand_means.fit_transform(matsky_rsrp)
#trsrp = stand_means.fit_transform(trsrp)
#trsrp_sky = stand_means.fit_transform(trsrp_sky)

# generate a maskmatrix
m = mat_rsrp.shape[0]
n = mat_rsrp.shape[1]

#missing_rate = 0.75
#missing_sky = 0.8
##random_mat = np.random.uniform(size=[])
#mask = gen_mask(m, n, missing_rate)
#mask_sky = gen_mask(m, n, missing_sky)
#sample_data = mat_rsrp* mask
#sample_sky = matsky_rsrp * mask_sky # sample data in the sky
#idx = np.argwhere(sample_sky != 0)
#sample_sky = np.array(sample_sky)
#sample_sky = stand_means.fit_transform(sample_sky)

def sample_z():
    z_gen =  np.random.normal(0.0, 1.0, size=[1,1600])
    return z_gen

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
#xs = tf.placeholder(tf.float32, [None, 1600])/255.   # 28x28
xs = tf.placeholder(tf.float32, [None, 1600])   # 28x28
ys = tf.placeholder(tf.float32, [None, 1600])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 40, 40, 1])
# print(x_image.shape)  # [n_samples, 28,28,1]

## conv1 layer ##
W_conv1 = weight_variable([5,5, 1,32]) # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.tanh(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)                                         # output size 14x14x32

## conv2 layer ##
W_conv2 = weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)                                         # output size 7x7x64

## fc1 layer ##
W_fc1 = weight_variable([10*10*64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 10*10*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([1024, 1600])
b_fc2 = bias_variable([1600])
prediction = (tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
print('prediction:',tf.reshape(prediction,[40,40]))

# the error between prediction and real data
#cross_entropy = tf.reduce_mean((ys-prediction)**2)       # loss
cross_entropy = tf.reduce_mean(tf.abs(ys-prediction))
#cross_entropy = mean_absolute_error(ys,prediction)
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

#sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
#sess.run(init)

xaxis=[]
yaxis=[]
ctime=[]
accu=[]
missing_rates = [0.7]
#[0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
#[0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]

    
    
    
for ms in missing_rates:
    with tf.Session() as sess:
        sess.run(init)
        for mr in missing_rates:
            for i in range(2000):
            #    batch_xs, batch_ys = mnist.train.next_batch(100)
                offset = i % 100
                batch_ys = np.array(mrsrp[offset,:]).reshape(1,1600)  # 地面的数据，目标值
                MASK = gen_mask(1,1600,prob_masked=ms)
                batch_xs = np.array(mrsrp_sky[offset,:]).reshape(1,1600) # 空中的数据
                z  = sample_z()
                batch_xs = batch_ys * MASK + (1-MASK)*z
                _, accuracy,predict = sess.run([train_step,cross_entropy,prediction], feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
                if i % 400 == 0:
                    
                    print("epoch:", i," mean square error", accuracy)
                    print("the output is :", predict)
                    plot_image(np.reshape(predict,[40,40]))
                    
                # test
            
            t1 = time.time()
            
            i=0
            print("test")
            test_=[]
            for t in trsrp_sky:
                MASK = gen_mask(1,1600,prob_masked=mr)
                t = MASK * t
                test_.append(t)
            test_ = np.array(test_).reshape(100,1600)
            print("origin sky")
            plot_image(np.reshape(trsrp_sky[0],[40,40]))
            print("origin sky missing")
            plot_image(np.reshape(test_[0],[40,40]))
            accuracy,predict = sess.run([cross_entropy,prediction], feed_dict={xs: test_, ys: trsrp, keep_prob: 0.5})
            
            print("origin ground")
            plot_image(np.reshape(trsrp[0],[40,40]))
            
            print("predict")
            plot_image(np.reshape(predict[0],[40,40]))
            print("test mean square:", accuracy)
            yaxis.append(accuracy)
            xaxis.append(mr)
            
            t2 = time.time()
            ctime.append(t2-t1)
            
            accu.append(accuracy)
        
        print(missing_rates)
        print(accu)
        print(ctime)
        #讲数据写文件
        out = open('data/cnnmae2.csv','a', newline='')
        csv_write = csv.writer(out,dialect='excel')
    #    csv_write.writerow(missing_rates)
        csv_write.writerow(accu)
    #    csv_write.writerow(ctime)  
        
        out2 = open('data/cnntime2.csv','a', newline='')
        csv_write = csv.writer(out2,dialect='excel')
    #    csv_write.writerow(missing_rates)
    #    csv_write.writerow(accu)
        csv_write.writerow(ctime)  

out.close()
out2.close()

#plt.plot(xaxis,yaxis,color='r',label='ours',marker='o',ls='-')
#plt.plot(xaxis,knny,c='blue',label='knn',marker='s',ls=':')
#plt.plot(xaxis,nnmy,c='orange',label='mc',marker='*',ls='--')
#plt.plot(xaxis,micey,c='black',label='mice',marker='h',ls='-.')
#plt.legend()
#plt.xlabel('missing rate')
#plt.ylabel('mae')
#plt.show()

#plt.plot(xaxis,yaxis,color='r',label='ours',marker='o',ls='-')
#plt.plot(xaxis,knny2,c='blue',label='knn',marker='s',ls=':')
#plt.plot(xaxis,nnmy2,c='orange',label='mc',marker='*',ls='--')
#plt.plot(xaxis,micey2,c='black',label='mice',marker='h',ls='-.')
#plt.legend()
#plt.xlabel('missing rate')
#plt.ylabel('mae')
#plt.show()
    