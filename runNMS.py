# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:11:31 2019

@author: ADMIN
"""

import tensorflow as tf

def run_nms(boxlist,conflist,maxoutput,iouthreshold):
    with tf.Graph().as_default():
        output_graph_def=tf.GraphDef()
        
        with open('./nms.pb',"rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ =tf.import_graph_def(output_graph_def,name='')
            
        with tf.Session() as sess:
            sess.graph.as_default()
            init=tf.global_variables_initializer()
            sess.run(init)
            
            box_list=sess.graph.get_tensor_by_name("boxlist:0")
            conf_list=sess.graph.get_tensor_by_name("conflist:0")
            max_output=sess.graph.get_tensor_by_name("maxoutput:0")
            iou_threshold=sess.graph.get_tensor_by_name("iouthreshold:0")
            
            nms=sess.graph.get_tensor_by_name("nms:0")
            nms=tf.to_int32(nms)
            nms_point=sess.run(nms,feed_dict={box_list:boxlist,conf_list:conflist,max_output:maxoutput,iou_threshold:iouthreshold})   
    return nms_point
        
        