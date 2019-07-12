# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 19:29:43 2019

@author: ADMIN
"""

import tensorflow as tf

def rpn_cls_loss(rpn_cls_map,gt_cls_map,rpn_no):#看一下softmax加了没
    rpn_cls_map=tf.reshape(rpn_cls_map,[-1,2])
    gt_cls_map=tf.reshape(gt_cls_map,[-1,2])
    rpn_no=tf.reshape(rpn_no,[-1])
    
    t_select=tf.where(tf.not_equal(rpn_no,-1))
    rpn_cls_map=tf.reshape(tf.gather(rpn_cls_map,t_select),[-1,2])
    gt_cls_map=tf.reshape(tf.gather(gt_cls_map,t_select),[-1,2])
    
    #cls_sq=tf.reduce_mean(tf.reduce_sum(tf.square(rpn_cls_score-rpn_cls_label),reduction_indices=[1]))
    cls_ce=tf.reduce_mean(-tf.reduce_sum(gt_cls_map*tf.log(rpn_cls_map),reduction_indices=[1]))
    rpn_cls_loss=cls_ce
    
    rpn_loss=rpn_cls_loss
    return rpn_loss

def rpn_reg_loss(rpn_reg_map,gt_reg_map,reg_no):
    rpn_reg_map=tf.reshape(rpn_reg_map,[-1,4])
    gt_reg_map=tf.reshape(gt_reg_map,[-1,4])
    reg_no=tf.reshape(reg_no,[-1])
    
    t_select=tf.where(tf.not_equal(reg_no,-1))
    rpn_reg_map=tf.reshape(tf.gather(rpn_reg_map,t_select),[-1,4])
    gt_reg_map=tf.reshape(tf.gather(gt_reg_map,t_select),[-1,4])
    
    loss=tf.reduce_mean(tf.reduce_sum(smooth_l1_loss(rpn_reg_map,gt_reg_map),reduction_indices=1))
    #loss=tf.reduce_mean(tf.sqrt(tf.square(rpn_reg[:,0]-(tf.cast((matches[:,0]-rpn_cls_[:,0])<25,tf.float32)*(matches[:,0]-rpn_cls_[:,0])))+tf.square(rpn_reg[:,1]-(tf.cast((matches[:,0]-rpn_cls_[:,0])<25,tf.float32)*(matches[:,1]-rpn_cls_[:,1])))))
    return loss

def smooth_l1_loss(x1,x2):
    abs_value=tf.abs(x1-x2)
    loss=tf.cast(abs_value<1,dtype=tf.float32)*(0.5*tf.pow(abs_value,2))+tf.cast(abs_value>=1,dtype=tf.float32)*(abs_value-0.5)
    return loss

def fasthead_cls_loss(fasthead_clsv,gt_clsv,gt_clsvno):#fasthead_clsv[N,21];gt_clsv[N,21]
    t_select=tf.where(tf.not_equal(gt_clsvno,-1))
    fasthead_clsv=tf.reshape(tf.gather(fasthead_clsv,t_select),[-1,21])
    gt_clsv=tf.reshape(tf.gather(gt_clsv,t_select),[-1,21])
    loss=tf.reduce_mean(-tf.reduce_sum(gt_clsv*tf.log(fasthead_clsv),reduction_indices=[1]))
    return loss

def fasthead_reg_loss(fasthead_regv,gt_regv,reg_no):#fasthead_regv[N,84];gt_regv[N,84]
    t_select=tf.where(tf.not_equal(reg_no,-1))
    
    fasthead_regv=tf.reshape(tf.gather(fasthead_regv,t_select),[-1,4])
    gt_regv=tf.reshape(tf.gather(gt_regv,t_select),[-1,4])

    loss=tf.reduce_mean(tf.reduce_sum(smooth_l1_loss(fasthead_regv,gt_regv),reduction_indices=1))
    return loss

