# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:27:50 2019

@author: ADMIN
"""

import tensorflow as tf
import tensorflow.contrib as tc

class R_P_N():
    def __init__(self,xs):
        self.xs=xs
    def _build_model_(self):
        bc=self.base_conv(self.xs)
        clsmap=self.cls_conv(bc)
        regmap=self.reg_conv(bc)
        return clsmap,regmap
    def base_conv(self,fm):
        cha=fm.shape[3]
        out=tc.layers.conv2d(fm,cha,3)
        return out
    def cls_conv(self,bc):
        out=tc.layers.conv2d(bc,18,1)
        out=tf.reshape(tf.nn.softmax(tf.reshape(out,(-1,2))),(-1,14,14,18))
        return out
    def reg_conv(self,bc):
        out=tc.layers.conv2d(bc,36,1)
        return out