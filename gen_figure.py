# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:43:32 2019

@author: Arrow
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_loc_loc(filename):
    data = pd.read_csv(filename,header=-1)
    df = pd.DataFrame(data)
    print(df)
    xaxis= df[0:1].values.tolist()[0]
    knny= df[1:2].values.tolist()[0]
    nnmy=df[2:3].values.tolist()[0]
    micey=df[3:4].values.tolist()[0]
    knntime=df[4:5].values.tolist()[0]
    nnmtime=df[5:6].values.tolist()[0]
    micetime=df[6:7].values.tolist()[0]
    return xaxis,knny,nnmy,micey, knntime,nnmtime,micetime

def get_cnn():
    data = pd.read_csv('./data/cnnmae2.csv', header=-1)
    df = pd.DataFrame(data)
    # xaxis = df[0:1].values.tolist()[0]
    accuracy = df[4:5].values.tolist()[0]

    data = pd.read_csv('./data/cnntime2.csv', header=-1)
    df = pd.DataFrame(data)
    # xaxis = df[0:1].values.tolist()[0]
    cnntime = df[4:5].values.tolist()[0]
    return accuracy,cnntime

def get_time_loc(filename):
    data = pd.read_csv(filename, header=-1)
    df = pd.DataFrame(data)
    xaxis2 = df[0:1].values.tolist()[0]
    knny2 = df[1:2].values.tolist()[0]
    micey2= df[2:3].values.tolist()[0]
    nnmy2 = df[0:1].values.tolist()[0]
    knntime2 = df[1:2].values.tolist()[0]
    micetime2 = df[2:3].values.tolist()[0]
    nnmtime2 = df[3:4].values.tolist()[0]
    return xaxis2,knny2,micey2,nnmy2,knntime2,micetime2,nnmtime2

def get_our_mc():
    data = pd.read_csv('./data/mcmae.csv', header=-1)
    df = pd.DataFrame(data)
    # xaxis = df[0:1].values.tolist()[0]
    mcaccuracy = df[0:1].values.tolist()[0]

    data = pd.read_csv('./data/mctime.csv', header=-1)
    df = pd.DataFrame(data)
    # xaxis = df[0:1].values.tolist()[0]
    mctime = df[0:1].values.tolist()[0]
    return mcaccuracy, mctime

xaxis,knnyaxis,nnmyaxis,miceyaxis, knntime,nnmtime,micetime = get_loc_loc('test.csv')
accuracy,cnntime = get_cnn()
xaxis2,knny2,micey2,nnmy2,knntime2,micetime2,nnmtime2 = get_time_loc('test2.csv')
mcaccu,mctime=get_our_mc()

font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 18,
}

#显示时间对比
plt.plot(xaxis,cnntime,color='r',label='Our cnn',marker='o',ls=':')
plt.plot(xaxis,knntime, c='blue',label='KNN',marker='s',ls=':')
plt.plot(xaxis,nnmtime,c='orange',label='Soft MC',marker='*',ls='--')
plt.plot(xaxis,micetime,c='black',label='MICE',marker='h',ls='-.')
plt.plot(xaxis,mctime,c='g',label='Our MC',marker='>',ls=':')
plt.legend(fontsize='x-large')
plt.xlabel('Missing rate',font2)
plt.ylabel('Execution time/s',font2)
plt.show()

# #显示0.85以前的时间
print('time before 80')
xaxis=xaxis[0:7]
plt.plot(xaxis,cnntime[0:7],color='r',label='Our cnn',marker='o',ls=':')
plt.plot(xaxis,knntime[0:7], c='blue',label='KNN',marker='s',ls=':')
plt.plot(xaxis,nnmtime[0:7],c='orange',label='MC',marker='*',ls='--')
plt.plot(xaxis,micetime[0:7],c='black',label='MICE',marker='h',ls='-.')
plt.plot(xaxis,mctime[0:7],c='g',label='Our MC',marker='>',ls=':')
plt.legend(fontsize='x-large')
plt.xlabel('Missing rate',font2)
plt.ylabel('Execution time/s',font2)
plt.show()

#显示loc time的时间
print('loc time time')
xaxis2=xaxis2[0:10]
plt.plot(xaxis2,cnntime[0:9],color='r',label='Our cnn',marker='o',ls=':')
plt.plot(xaxis2,knntime2, c='blue',label='KNN',marker='s',ls=':')
plt.plot(xaxis2,nnmtime2,c='orange',label='MC',marker='*',ls='--')
plt.plot(xaxis2,micetime2,c='black',label='MICE',marker='h',ls='-.')
plt.plot(xaxis2,mctime[0:9],c='g',label='Our MC',marker='>',ls=':')
plt.legend()
plt.xlabel('missing rate')
plt.ylabel('execution time/s')
plt.show()

# loc loc mae
# plt.plot(xaxis,accuracy,color='r',label='Our cnn',marker='o',ls='-.')
# plt.plot(xaxis,knnyaxis, c='blue',label='KNN',marker='s',ls=':')
# # plt.plot(xaxis,nnmyaxis,c='orange',label='Soft MC',marker='*',ls='--')
# plt.plot(xaxis,miceyaxis,c='black',label='MICE',marker='h',ls='-.')
# plt.plot(xaxis,mcaccu,c='g',label='Our MC',marker='>',ls=':')
# plt.legend()
# plt.xlabel('Missing rate')
# plt.ylabel('MAE')
# plt.show()

##这段代码显示loc loc 0.7以前的子图
# xaxis=xaxis[0:5]
# accuracy=accuracy[0:5]
# knnyaxis=knnyaxis[0:5]
# nnmyaxis=nnmyaxis[0:5]
# miceyaxis=miceyaxis[0:5]
#
# plt.plot(xaxis,accuracy,color='r',label='our cnn',marker='o',ls='-')
# plt.plot(xaxis,knnyaxis, c='blue',label='knn',marker='s',ls=':')
# plt.plot(xaxis,nnmyaxis,c='orange',label='mc',marker='*',ls='--')
# plt.plot(xaxis,miceyaxis,c='black',label='mice',marker='h',ls='-.')
# plt.legend()
# plt.xlabel('missing rate')
# plt.ylabel('mae')
# plt.show()


# xaxis2=xaxis2[0:10]
# accuracy=accuracy[0:9]
#
# print(accuracy)
# print(xaxis2)

#显示loc time的mae  
print('loc time mae')
plt.plot(xaxis2,accuracy[0:9],color='r',label='Our cnn',marker='o',ls=':')
plt.plot(xaxis2,knny2, c='blue',label='KNN',marker='s',ls=':')
plt.plot(xaxis2,nnmy2,c='orange',label='Soft MC',marker='*',ls='--')
plt.plot(xaxis2,micey2,c='black',label='MICE',marker='h',ls='-.')
plt.plot(xaxis2,mcaccu[0:9],c='g',label='Our MC',marker='>',ls=':')
plt.legend()
plt.xlabel('Missing rate')
plt.ylabel('MAE')
plt.show()
