# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:32:37 2019

@author: ADMIN
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

def check_clsv(pd,s_box,clsv,clsvno):
    im=pd[0]
    for i in range(len(s_box)):
        if clsvno[i]==0:
            tmpbox=s_box[i]
            lth=tmpbox[0]
            ltw=tmpbox[1]
            rbh=tmpbox[2]
            rbw=tmpbox[3]
            clsvn=np.argmax(clsv[i])
            #print(clsvn)
            #print(clsv)
            print(i)
            cv2.putText(im,str(clsvn),(ltw,lth),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)
            cv2.rectangle(im,(ltw,lth),(rbw,rbh),(0,255,0),1)#画出框
    return im