# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:50:23 2019

@author: ADMIN
"""

import tensorflow as tf

def gfv(feature_map,coord):
    #fm[1,14,14,512]
    #coordinate[N,5,5,2]
    
    coord=tf.reshape(coord,[-1,2])
    
    fm_reshape=tf.reshape(feature_map,[-1,512])
    coor_h=coord[:,0]
    coor_w=coord[:,1]
    
    coor_re=coor_h*14+coor_w
    
    feature_vector=tf.gather(fm_reshape,coor_re)
    
    fv_out=tf.reshape(feature_vector,[-1,5*5,512])
    fv_out=tf.to_float(fv_out,name='fv_out')
    return fv_out