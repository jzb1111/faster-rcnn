# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 17:16:03 2019

@author: ADMIN
"""

import tensorflow as tf

feature_map=tf.placeholder(tf.float32,[1,14,14,512],name='feature_map')
coord=tf.placeholder(tf.int32,[5,5,2],name='coord')

coord=tf.reshape(coord,[-1,2])

fm_reshape=tf.reshape(feature_map,[-1,512])
coor_h=coord[:,0]
coor_w=coord[:,1]

coor_re=coor_h*14+coor_w

feature_vector=tf.gather(fm_reshape,coor_re)

fv_out=tf.reshape(feature_vector,[5*5,512],name='fv_out')
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    output_graph_def=tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,output_node_names=['fv_out'])
    with tf.gfile.FastGFile('./get_fv.pb', mode = 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    