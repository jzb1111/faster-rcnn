# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:37:29 2019

@author: ADMIN
"""

import tensorflow as tf
import tensorflow.contrib as tc

class vgg16():
    def __init__(self,xs):
        self.xs=xs
        self.convblok1_klist=[3,3]#112
        self.convblok2_klist=[3,3]#56
        self.convblok3_klist=[3,3,1]#28
        self.convblok4_klist=[3,3,1]#14
        
        self.convblok1_clist=[64,64]
        self.convblok2_clist=[128,128]
        self.convblok3_clist=[256,256,256]
        self.convblok4_clist=[512,512,512]
    def _build_model_(self):
        out=self.conv_block(self.xs,self.convblok1_klist,self.convblok1_clist)
        out=self.conv_block(out,self.convblok2_klist,self.convblok2_clist)
        out=self.conv_block(out,self.convblok3_klist,self.convblok3_clist)
        out=self.conv_block(out,self.convblok4_klist,self.convblok4_clist)
        return out
    
    def conv_block(self,out,kernellist,channellist):
        for i in range(len(kernellist)):
            out=tc.layers.conv2d(out,channellist[i],kernellist[i])
        out=tf.nn.max_pool(out,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        return out