#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 20:18:45 2019

@author: tao
"""

import tensorflow as tf
import numpy as np
 
def max_pool(inp, k=2):
    return tf.nn.max_pool(inp, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")
 
def max_unpool(inp, argmax, argmax_mask, k=2):
    return tf.nn.max_unpool(inp, argmax, argmax_mask, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")
 
def conv2d(inp, name):
    w = weights[name]
    b = biases[name]
    var = tf.nn.conv2d(inp, w, [1, 1, 1, 1], padding='SAME')
    var = tf.nn.bias_add(var, b)
    var = tf.nn.relu(var)
    return var
 
def conv2d_transpose(inp, name, dropout_prob):
    w = weights[name]
    b = biases[name]
 
    dims = inp.get_shape().dims[:3]
    dims.append(w.get_shape()[-2]) # adpot channels from weights (weight definition for deconv has switched input and output channel!)
    out_shape = tf.TensorShape(dims)
 
    var = tf.nn.conv2d_transpose(inp, w, out_shape, strides=[1, 1, 1, 1], padding="SAME")
    var = tf.nn.bias_add(var, b)
    if not dropout_prob is None:
        var = tf.nn.relu(var)
        var = tf.nn.dropout(var, dropout_prob)
    return var
 
 
weights = {
    "conv1":    tf.Variable(tf.random_normal([3, 3,  3, 16])),
    "conv2":    tf.Variable(tf.random_normal([3, 3, 16, 32])),
    "conv3":    tf.Variable(tf.random_normal([3, 3, 32, 32])),
    "deconv2":  tf.Variable(tf.random_normal([3, 3, 16, 32])),
    "deconv1":  tf.Variable(tf.random_normal([3, 3,  1, 16])) }
 
biases = {
    "conv1":    tf.Variable(tf.random_normal([16])),
    "conv2":    tf.Variable(tf.random_normal([32])),
    "conv3":    tf.Variable(tf.random_normal([32])),
    "deconv2":  tf.Variable(tf.random_normal([16])),
    "deconv1":  tf.Variable(tf.random_normal([ 1])) }
 
 
## Build Miniature CEDN
x = tf.placeholder(tf.float32, [12, 20, 20, 3])
y = tf.placeholder(tf.float32, [12, 20, 20, 1])
p = tf.placeholder(tf.float32)
 
conv1                                   = conv2d(x, "conv1")
maxp1, maxp1_argmax, maxp1_argmax_mask  = max_pool(conv1)
 
conv2                                   = conv2d(maxp1, "conv2")
maxp2, maxp2_argmax, maxp2_argmax_mask  = max_pool(conv2)
 
conv3                                   = conv2d(maxp2, "conv3")
 
maxup2                                  = max_unpool(conv3, maxp2_argmax, maxp2_argmax_mask)
deconv2                                 = conv2d_transpose(maxup2, "deconv2", p)
 
maxup1                                  = max_unpool(deconv2, maxp1_argmax, maxp1_argmax_mask)
deconv1                                 = conv2d_transpose(maxup1, "deconv1", None)
 
 
## Optimizing Stuff
loss        = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(deconv1, y))
optimizer   = tf.train.AdamOptimizer(learning_rate=1).minimize(loss)
 
 
## Test Data
np.random.seed(123)
batch_x = np.where(np.random.rand(12, 20, 20, 3) > 0.5, 1.0, -1.0)
batch_y = np.where(np.random.rand(12, 20, 20, 1) > 0.5, 1.0,  0.0)
prob    = 0.5
 
 
with tf.Session() as session:
    tf.set_random_seed(123)
    session.run(tf.initialize_all_variables())
 
    print ("\n\n")
    for i in range(10):
        session.run(optimizer, feed_dict={x: batch_x, y: batch_y, p: prob})
        print ("step", i + 1)
        print ("loss",  session.run(loss, feed_dict={x: batch_x, y: batch_y, p: 1.0}))
