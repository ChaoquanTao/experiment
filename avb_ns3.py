# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:32:46 2019
 使用 adversarial vb 建模ns3的数据，进行缺失数据填充. 
 更新： 将代码中的train vae的loss 加入了prior loss
 更新： 在vae和dis的input 部分加入了额外的噪声数据 \epsilon
@author: Allen
"""


import tensorflow as tf   # http://blog.topspeedsnail.com/archives/10399
from sklearn.preprocessing import scale  # 使用scikit-learn进行数据预处理
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
#from tqdm import tqdm

tf.reset_default_graph()

mb_size = 1
p_miss = 0.95
train_rate = 0.8
alpha = 1e-4 # 1e-3 垃圾

data_path = "../data/DlRsrpSinrStats.txt"
data_set = pd.read_table(data_path, delimiter="\t")

df = pd.DataFrame(data_set)

orsrp = np.array((df.loc[0:3199].sort_values(by='IMSI'))['rsrp'])
num = 99

for offset in range(num):
    batch_x = np.array((df.loc[offset * 3200:(offset+1)*3200-1].sort_values(by='IMSI'))['rsrp'])
#    batch_y = np.array((df.loc[offset * 3200+1600:(offset+1)*3200-1].sort_values(by='IMSI'))['rsrp']).reshape(1,1600)
    orsrp = np.c_[orsrp,batch_x]
    
orsrp = 10*np.log10(orsrp) + 30   # turn the data into dBm
rsrp=orsrp[0:1600,:]
sky_rsrp = orsrp[1600:,:]

# 数据预处理
data = np.array(rsrp)

scaler = MinMaxScaler()
scaler.fit(data)
data =scaler.transform(data)

Data =  np.transpose(data)


Row = Data.shape[0]
Col = Data.shape[1]


# 定义missing matrix
Missing = np.zeros((Row, Col))
p_miss_vec = p_miss * np.ones((Col, 1))

for i in range(Col):
    A = np.random.uniform(0., 1., size=[Row])
    B = A > p_miss_vec[i]
    Missing[:, i] = 1. * B

idx = np.random.permutation(Row)

# training data and test data
Train_Num = int(Row * train_rate)
Test_Num = Row - Train_Num

trainX = Data[idx[:Train_Num], :]
testX = Data[idx[Train_Num:], :]

# Train / Test Missing Indicators
trainM = Missing[idx[:Train_Num], :]
testM = Missing[idx[Train_Num:], :]

def sample_idx(m,minibatch):
    A = np.random.permutation(m)
    idx = A[:minibatch]
    return idx

# 这个目前只能进行40*40矩阵的缺失
def sample_block(m=mb_size,n=Col):
    block = np.ones(shape=[40,40])
#    block.reshape([40,40])
    block[10:26,10:25] = 0
    block = np.reshape(block,[mb_size,Col])
    
    return block

def sample_z(minibatch,n):
    return np.random.normal(size=[minibatch,n])

def plot_image(A):
    plt.imshow(A.T)
    plt.colorbar()

    plt.show()
  
def compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)

print("the original data structure is: \n")
inData = Data[1,:]
inData = np.reshape(inData,(40,40))
plot_image(inData)
    
X = tf.placeholder(tf.float32, shape=[None, Col])  # 网络输入
Mat = tf.placeholder(tf.float32, shape=[None,Col])  # 定义这一次的缺失位置

Y = tf.placeholder(tf.float32,[None, Col]) # 网络输出

real = tf.placeholder(tf.float32,[None,Col])

# 定义神经网络
with tf.variable_scope('autoencoder'):
	# --------------------- Encoder -------------------- #
      
#	epsilon = tf.random_normal([mb_size,Col],0, 1, dtype=tf.float32)
#	data_in = tf.concat([X,epsilon],axis=1)
	layer_1 = tf.layers.dense(X, 256, tf.nn.tanh, name = 'layer1')
	layer_2 = tf.layers.dense(layer_1, 128, tf.nn.tanh, name = 'layer2')
	z_mean = tf.layers.dense(layer_2, 32, tf.nn.tanh, name = 'z_mean')
	z_stddev = tf.layers.dense(layer_2, 32, tf.nn.tanh, name = 'z_std')
	samples = tf.random_normal([mb_size,32],0,1,dtype=tf.float32)
	encoded = z_mean + (z_stddev * samples)
    
	layer_4 = tf.layers.dense(encoded, 128, tf.nn.tanh, name = 'layer_4')
#	layer_5 = tf.layers.dense(layer_4, 128, tf.nn.tanh, name = 'layer_5')
	layer_6 = tf.layers.dense(layer_4, 256, tf.nn.tanh, name = 'layer_6')
	decoded = tf.layers.dense(layer_6, Col, tf.nn.sigmoid, name = 'decoded')
    # adverarial network

with tf.variable_scope('discriminator'):
    
    epsilon = tf.random_normal([mb_size,Col],0, 1, dtype=tf.float32)
    r_in = tf.concat([real, epsilon],axis=1)
    D_00 = tf.layers.dense(r_in, 256, tf.nn.elu,name = 'latent')
    D_01 =  tf.layers.dense(D_00, 128, tf.nn.elu,name = 'latent_1')
    D_02 =  tf.layers.dense(D_01, 64, tf.nn.elu,name = 'latent_2')
    prob_real = tf.layers.dense(D_02, 1, tf.nn.sigmoid, name = 'out')
    
    # reuse the layers for generator
    f_in = tf.concat([decoded, epsilon], axis = 1)
    D_l1 = tf.layers.dense(f_in,256,tf.nn.elu,name='latent',reuse=True)
    D_l2 = tf.layers.dense(D_l1,128,tf.nn.elu,name='latent_1',reuse=True)
    D_l3 = tf.layers.dense(D_l2,64,tf.nn.elu,name='latent_2',reuse=True)
    prob_false = tf.layers.dense(D_l3,1,tf.nn.sigmoid,name= 'out', reuse = True)
    
# 训练神经网络 
def train_neural_networks():
#    decoded,z_mean,z_stddev = neural_networks() 

    us_cost_function =  0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1,1) + 0.5* tf.reduce_mean(tf.pow(X*Mat -Mat*decoded, 2)) +alpha * tf.reduce_mean(tf.log(1-prob_false +1e-8)) 
#    loss_mmd = compute_mmd(X, encoded)
#    us_cost_function = us_cost_function+loss_mmd
    
    s_cost_function = tf.reduce_mean(tf.square(X - decoded)) 
    D_loss = -tf.reduce_mean(tf.log(prob_real) + tf.log(1-prob_false))
    
    us_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(us_cost_function, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='autoencoder'))
    
    d_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))

#    progress = tqdm(range(8000), ncols=10)
    progress = range(5000)
    d_loss_r = []
    train_loss_r = []
    
    with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
#                with progress as t:
            for epoch in progress:
            
                         mb_idx = sample_idx(Train_Num,minibatch=mb_size)   
                         mb_data = trainX[mb_idx,:] # minibatch data 
                         mb_mat = trainM[mb_idx,:]  # minibatch mat matrix
#                         if epoch % 10 == 0:   
#                             mb_mat = sample_block(mb_size,Col)
                         mb_z = sample_z(mb_size,Col)
                         inputs = mb_data * mb_mat  + 1e-1*(1-mb_mat) * mb_z  
#                             if epoch == 2:
#                                 print('the sampled data')
#                                 final = np.reshape(inputs,(40,40))
#                                 plot_image(final) 
#                                 final = np.reshape(mb_mat,(40,40))
#                                 plot_image(final) 
#                                 W = np.ones([4,4]) + 1e-5
#                                 W = np.reshape(W,[4,4,1,1])
#                                 final = final + 1e-5
#                                 final = final.reshape([-1,40,40,1])
#                                 weight =  tf.nn.conv2d(final, W, strides=[1, 1, 1, 1], padding='SAME')
#                                 final = sess.run([weight])
##                                 final = tf.reshape(final,[-1,40])
#                                 final= np.array(final)
#                                 final = final.reshape([40,40])
#                                 plot_image(final)                   
                         _, latent_loss = sess.run([us_optimizer,us_cost_function],feed_dict={X:inputs,Mat:mb_mat,real:mb_data})
                             
                         _,d_loss = sess.run([d_optimizer,D_loss],feed_dict={real:mb_data,X:inputs,Mat:mb_mat})
                         _,d_loss = sess.run([d_optimizer,D_loss],feed_dict={real:mb_data,X:inputs,Mat:mb_mat})
                  
#                             progress.set_description('%d steps, train loss is: %f, d loss is%f' % (epoch, np.mean(latent_loss), np.mean(d_loss)))
                         
                         if epoch % 500 == 0:
                             print(('%d steps, train loss is: %f, d loss is%f' % (epoch, np.mean(latent_loss), np.mean(d_loss))))
                             train_loss_r.append(np.mean(latent_loss))
                             d_loss_r.append(np.mean(d_loss))

#            except KeyboardInterrupt:
#                t.close()
#                raise
#            t.close()

            mb_idx = sample_idx(Test_Num,minibatch=mb_size)
            mb_data = testX[mb_idx,:] # minibatch data 
            mb_mat = testM[mb_idx,:]  # minibatch mat matrix
#            mb_mat = sample_block()
            mb_z = sample_z(mb_size,Col)
#            inputs =  mb_z 
            inputs = mb_data * mb_mat + 1e-1*(1-mb_mat) * mb_z                      
            cost,result = sess.run([s_cost_function,decoded],feed_dict={X:inputs, Mat:mb_mat})
#            print("test cost is :" ,np.mean(cost))
            real_cost =np.power(mb_data-result,2)
            real_cost = np.sum(real_cost)
            print("avb_ns3_new test cost is :" ,real_cost)
#            final = result[5,:]
            final = result
            final = np.reshape(final,(40,40))
            plot_image(final)
            
            plt.figure()
            plt.plot(train_loss_r,label='train loss')
            plt.legend()
            plt.show()
            plt.figure
            plt.plot(d_loss_r,label = 'd loss')
            plt.legend()
            plt.show()
            
train_neural_networks()