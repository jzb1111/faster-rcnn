# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 21:02:05 2019

@author: ADMIN
"""

from ROI_pool import ROIs
import tensorflow as tf
import numpy as np

a=tf.convert_to_tensor(np.array([[0,0,10,10],[20,20,40,40],[112,112,223,223]]))
xsnp=np.zeros((1,14,14,512))
for h in range(14):
    for w in range(14):
        for c in range(510):
            xsnp[0][h][w][0]=h
            xsnp[0][h][w][1]=w
            xsnp[0][h][w][c+2]=c+2
xs=tf.convert_to_tensor(xsnp)
roig=ROIs(xs,a)
out=roig._build_model()
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)  
    syo=sess.run(out)
syo0=syo[0]
syo1=syo[1]
syo2=syo[2]    