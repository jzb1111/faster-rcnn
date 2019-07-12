# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 19:52:49 2019

@author: ADMIN
"""

import tensorflow as tf

def run_GFV(feature_map,coord):
    with tf.Graph().as_default():
        output_graph_def=tf.GraphDef()
        
        with open('./get_fv.pb',"rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ =tf.import_graph_def(output_graph_def,name='')
            
        with tf.Session() as sess:
            sess.graph.as_default()
            init=tf.global_variables_initializer()
            sess.run(init)
            
            featuremap=sess.graph.get_tensor_by_name("feature_map:0")
            coord_=sess.graph.get_tensor_by_name("coord:0")
            fvout=sess.graph.get_tensor_by_name("fv_out:0")
            
            fv_output=sess.run(fvout,feed_dict={featuremap:feature_map,coord_:coord})   
    return fv_output