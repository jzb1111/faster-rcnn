# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 10:31:11 2019

@author: ADMIN
"""

import tensorflow as tf
from nms_for_train import nms

clsv=tf.placeholder(tf.float32,[None,21],name='clsv')
regv=tf.placeholder(tf.float32,[None,84],name='regv')
boxes=tf.placeholder(tf.float32,[None,4],name='boxes')

fasthead_nms=nms(clsv,regv,boxes)
clsout,boxout=fasthead_nms._build_model_()

clsout=tf.to_int32(tf.reshape(clsout,[-1]),name='clsout')
boxout=tf.to_int32(tf.reshape(boxout,[-1,4]),name='boxout')
init=tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    output_graph_def=tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,output_node_names=['clsout','boxout'])#sess.graph_def
    with tf.gfile.FastGFile('./fasthead_nms.pb', mode = 'wb') as f:
        f.write(output_graph_def.SerializeToString())