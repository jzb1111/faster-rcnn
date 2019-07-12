# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:19:50 2019

@author: ADMIN
"""

import tensorflow as tf
import numpy as np

class nms():
    def __init__(self,clsv,regv,boxes):
        self.clsv=clsv#[-1,21]
        self.regv=regv#[-1,84]
        self.boxes=boxes#[-1,4]
    def _build_model_(self):
        self.regv=tf.reshape(self.regv,[-1,21,4])
        cls_n=self.calc_cls(self.clsv)#vector to num
        reg_n=self.gather_reg(cls_n,self.regv)#[-1,4]
        c_boxes=self.calc_box(reg_n,self.boxes)
        clsout,boxout=self.select_bbox(cls_n,c_boxes)
        
        return clsout,boxout
    def calc_cls(self,clsv):
        out=tf.arg_max(clsv,1)
        return out
    def gather_reg(self,clsn,regv):#[-1],[-1,21,4]
        regv=tf.reshape(regv,[-1,4])
        clsn=tf.reshape(clsn,[-1])
        clsone=tf.ones_like(clsn)
        cls_num=tf.where(tf.equal(clsone,1))
        cls_select=clsn+cls_num*21
        #clsl=clsn.shape[0]
        #clsnp=tf.lin_space(tf.to_float(0),tf.to_float(clsl-1),clsl)
        #总之就是要产生0~N
        
        clsgather=cls_select
        out=tf.gather(regv,clsgather)
        out=tf.reshape(out,[-1,4])
        return out
    def select_bbox(self,cls_n,c_boxes):
        cls_n=tf.reshape(cls_n,[-1])
        t_select=tf.where(tf.not_equal(cls_n,20))
        clsout=tf.gather(cls_n,t_select)
        boxout=tf.gather(c_boxes,t_select)
        return clsout,boxout
    def calc_box(self,reggv,boxes):#去掉第21类
        #out=[]
        #for i in range(len(boxes)):
        
        p_lefttop_h=boxes[:,0]
        p_lefttop_w=boxes[:,1]
        p_rightbottom_h=boxes[:,2]
        p_rightbottom_w=boxes[:,3]
        ty=reggv[:,0]
        tx=reggv[:,1]
        th=reggv[:,2]
        tw=reggv[:,3]
        ya=(p_lefttop_h+p_rightbottom_h)/2
        xa=(p_lefttop_w+p_rightbottom_w)/2
        ha=p_rightbottom_h-p_lefttop_h
        wa=p_rightbottom_w-p_rightbottom_w
        y=ty*tf.to_float(ha)+tf.to_float(ya)
        x=tx*tf.to_float(wa)+tf.to_float(xa)
        h=tf.pow(np.e,th)*tf.to_float(ha)
        w=tf.pow(np.e,tw)*tf.to_float(wa)
        lefttop_h=y-h/2
        lefttop_w=x-w/2
        rightbottom_h=y+h/2
        rightbottom_w=x+w/2
        out=tf.stack([lefttop_h,lefttop_w,rightbottom_h,rightbottom_w],axis=1)
        return out
        
        