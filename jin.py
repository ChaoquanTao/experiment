#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 20:53:18 2019

@author: tao
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 14:33:06 2019

@author: tao
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
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, IterativeImputer, BiScaler
import tensorflow as tf

def plot_image(A):
    plt.imshow(A.T)
    plt.show()

data_path = "/home/tao/tarballs/ns-allinone-3.29/ns-3.29/DlRsrpSinrStats.txt"

#data_set = np.loadtxt(data_path,delimiter='	',skiprows=1);
data_set = pd.read_table(data_path,delimiter='\t')
print(data_set.shape)

df = pd.DataFrame(data_set)


orsrp = np.array((df.loc[0:3199].sort_values(by='IMSI'))['rsrp'])
rsrp=orsrp[0:1600]
sky_rsrp = orsrp[1600:]

print(rsrp.shape)
print(orsrp[0:1600])
mrsrp = []
mrsrp_sky = []
#for i in range(50):
#
#    mrsrp_o.append(rsrp[i*2*100:(i*2+1)*100])
#
#for power in mrsrp_o:
#    temp_power = 10*np.log10(power) + 30
#    mrsrp.append(temp_power)

for power in rsrp:
    temp_power = 10*np.log10(power) + 30.0
    mrsrp.append(temp_power)
    
for sky_power in sky_rsrp:
    temp_power = 10*np.log10(sky_power) + 30
    mrsrp_sky.append(temp_power)


mrsrp = np.array(mrsrp).reshape(40,40)
mrsrp_sky = np.array(mrsrp_sky).reshape(40,40)

print(mrsrp)
out = open('/home/tao/dataset/out.csv','w')
csv_write = csv.writer(out)
for i in mrsrp:
    csv_write.writerow(i)    
mat_rsrp = mrsrp
matsky_rsrp = mrsrp_sky
print('fdfa')

# preprocessing
stand_means = preprocessing.MinMaxScaler()
mat_rsrp = stand_means.fit_transform(mat_rsrp)
stand_means = preprocessing.MinMaxScaler()
matsky_rsrp = stand_means.fit_transform(matsky_rsrp)


# generate a maskmatrix
m = mat_rsrp.shape[0]
n = mat_rsrp.shape[1]

missing_rate = 0.75
missing_sky = 0.8
#random_mat = np.random.uniform(size=[])
mask = gen_mask(m, n, missing_rate)
mask_sky = gen_mask(m, n, missing_sky)
sample_data = mat_rsrp* mask
sample_sky = matsky_rsrp * mask_sky # sample data in the sky
idx = np.argwhere(sample_sky != 0)
sample_sky = np.array(sample_sky)
sample_sky = stand_means.fit_transform(sample_sky)
# def neural_network():
size = 40
X = tf.placeholder(tf.float32,shape=[None,size])
Y = tf.placeholder(tf.float32,shape=[None,size])

kernel = tf.random_normal(shape=[2,2,3,1])#正向卷积的kernel的模样
output_shape=[1,5,5,3]
#Sample_X = tf.placeholder(tf.float32,shape=[None,size])
def neural_network():
    w1 = tf.Variable(tf.truncated_normal(shape=[40,128],mean=0,stddev=0.01))
    b1 = tf.Variable(tf.zeros(shape=[128]))
    w2 = tf.Variable(tf.truncated_normal(shape=[128,64],mean=0,stddev=0.01))
    b2 = tf.Variable(tf.zeros(shape=[64]))
    w3 = tf.Variable(tf.truncated_normal(shape=[64,40],mean=0,stddev=0.01))
    b3 = tf.Variable(tf.zeros(shape=[40]))
    
    inputs_conc = tf.nn.conv2d_transpose(Y,kernel,strides=[1,1,1,1] ,padding = "SAME",output_shape=[40,40])
    inputs = tf.nn.tanh(tf.add(tf.matmul(inputs_conc,w1),b1))
    layer1 = tf.nn.tanh(tf.add(tf.matmul(inputs,w2),b2))
    output = tf.nn.tanh(tf.add(tf.matmul(layer1,w3),b3))
    return output

predict = neural_network()
loss = tf.reduce_mean(tf.square(predict-X)) / 40
optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
max_iter = 1000
for i in range(max_iter):
        _,loss_r = sess.run([optimizer,loss],feed_dict={X:mat_rsrp,Y:matsky_rsrp})
        
        if i % 10 == 0:
            print("loss ",loss_r)
        
_,final_data,final_socre = sess.run([optimizer,predict,loss],feed_dict={X:mat_rsrp,Y:sample_sky})
print("final score", final_socre)
plot_image(final_data)

plot_image(mat_rsrp)


#print('origin_data')
#plot_image(mat_rsrp)
#
#print('the sample_data')
#plot_image(sample_data)
#
#R_hat = svt_solve(mat_rsrp,mask,delta = 1e-2)
#U,S,V  = np.linalg.svd(mat_rsrp)
#print('the R_hat ')
#plot_image(R_hat)
#
#sample_data[sample_data==0] = np.nan
#knn_recover = fast_knn(sample_data,k=3)
#print('knn')
#plot_image(knn_recover)
#
#mice_data = mice(sample_data)
#print('mice')
#plot_image(mice_data)
#
#X_filled_ii = IterativeImputer().fit_transform(sample_data)
#print('IterativeImputer')
#plot_image(X_filled_ii)
#
#X_filled_nnm = NuclearNormMinimization().fit_transform(sample_data)
#print('NuclearNormMinimization')
#plot_image(X_filled_nnm)
#
## compute the error
#error_mat = mean_absolute_error(mat_rsrp,R_hat)
#error_knn = mean_absolute_error(mat_rsrp,knn_recover)
#error_mice = mean_absolute_error(mat_rsrp,mice_data)
#error_iter = mean_absolute_error(mat_rsrp,X_filled_ii)
#error_nuclear = mean_absolute_error(mat_rsrp,X_filled_nnm)
#print("mat:",error_mat,"\tknn:",error_knn,"\tmice:",error_mice,"\titer:",error_iter,"\tnuclear",error_nuclear)
#
#r2_mat = r2_score(mat_rsrp,R_hat)
#r2_knn = r2_score(mat_rsrp,knn_recover)
#r2_mice = r2_score(mat_rsrp,mice_data)
#r2_iter = r2_score(mat_rsrp,X_filled_ii)
#r2_nuclear = r2_score(mat_rsrp,X_filled_nnm)
#print("\nmat:",r2_mat,"\tknn:",r2_knn,"\tmice:",r2_mice,"\titer:",r2_iter,"\tnuclear",r2_nuclear)
#

