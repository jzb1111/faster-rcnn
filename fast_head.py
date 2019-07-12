# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:10:31 2019

@author: ADMIN
"""

import tensorflow as tf
import tensorflow.contrib as tc

class fasthead():
    def __init__(self,rois):
        #self.xs=xs
        self.rois=rois
        #self.boxes=boxes
        
    def _build_model_(self):
        self.rois=tf.reshape(self.rois,[-1,25*512])
        #self.rois=tf.reshape(self.rois,[1,-1])
        bfc=self.base_fc(self.rois)
        clsv=self.cls_vector(bfc)
        regv=self.reg_vector(bfc)
        return clsv,regv
    
    def base_fc(self,rois):
        out=tc.layers.fully_connected(rois,2048)
        out=tc.layers.fully_connected(out,2048)
        return out
    
    def cls_vector(self,vector):
        out=tc.layers.fully_connected(vector,21)
        out=tf.nn.softmax(out)
        #output=tf.reshape(tf.nn.softmax(tf.reshape(output,(-1,2))),(-1,28,28,2))
        return out
    
    def reg_vector(self,vector):
        out=tc.layers.fully_connected(vector,84)
        return out