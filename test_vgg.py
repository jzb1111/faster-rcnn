# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:19:15 2019

@author: ADMIN
"""

from VGG import vgg16
from support.read_data import generate_fasthead_train_sample,file_name,load_train_data
import tensorflow as tf

pd,ocs,ors,no,rn,gld=load_train_data(0)

xs=tf.placeholder(tf.float32,[1,224,224,3],name='input_xs')

vgg=vgg16(xs)
vggout=vgg._build_model_()

init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)  
    syvo=sess.run(vggout,feed_dict={xs:pd})