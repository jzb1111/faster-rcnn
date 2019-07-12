# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 18:29:14 2019

@author: ADMIN
"""

from run_model import run_rpn,run_fasthead,run_fasthead_nms
from support.read_data import get_batch_by_num,generate_box,correct_box
from ROI_pool import roi_box
from runNMS import run_nms
import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_box(pd,boxes):
    im=pd[0]
    for i in range(len(boxes)):                
        cv2.rectangle(im,(boxes[i][1],boxes[i][0]),(boxes[i][3],boxes[i][2]),(0,255,0),1)#画出框
    cv2.imshow("Image", im/255)    
    cv2.waitKey (0)
    cv2.destroyAllWindows()
    im=pd[0]
    return 0

def draw_rpn_box(pd,clsmap,regmap):
    im=pd[0]
    boxlis=[]
    for h in range(14):
        for w in range(14):
            for c in range(9):
                conf=clsmap[0][h][w][c][0]-clsmap[0][h][w][c][1]
                boxlis.append([conf,[h,w,c]])
    sortlis=sorted(boxlis,reverse=True)
    for i in range(5):
        #print(sortlis[i][1])
        tmpbox=generate_box(sortlis[i][1][0],sortlis[i][1][1],sortlis[i][1][2])
        lth=tmpbox[0]
        ltw=tmpbox[1]
        rbh=tmpbox[2]
        rbw=tmpbox[3]
        cv2.rectangle(im,(ltw,lth),(rbw,rbh),(0,255,0),1)#画出框
    return im        

def draw_s_box(pd,s_box):
    im=pd[0]
    for i in range(5):
        tmpbox=s_box[i]
        lth=tmpbox[0]
        ltw=tmpbox[1]
        rbh=tmpbox[2]
        rbw=tmpbox[3]
        cv2.rectangle(im,(ltw,lth),(rbw,rbh),(0,255,0),1)#画出框
    return im   

def draw_v_box(pd,s_box,clsv,regv):#还缺少一个nms
    im=pd[0]
    regv=np.reshape(regv,[-1,21,4])
    cls_lis={}
    cls_box={}
    for i in range(20):
        cls_lis[i]=[]
        cls_box[i]=[]
    for i in range(len(clsv)):
        if np.argmax(clsv[i])!=20:
            clsvn=np.argmax(clsv[i])
            tmpbox=s_box[i]
            regs=regv[i][clsvn]
            cor_box=correct_box(tmpbox,regs)
            #lth=cor_box[0]
            #ltw=cor_box[1]
            #rbh=cor_box[2]
            #rbw=cor_box[3]
            cls_lis[clsvn].append(clsv[i][clsvn])
            cls_box[clsvn].append(cor_box)
    #print(cls_lis)
    for i in range(len(cls_lis)):
        if len(cls_lis[i])>0:
            boxlistmp=cls_box[i]
            conflist=cls_lis[i]
            #print(boxlistmp)
            #print(conflist)
            maxoutput=10
            iouthreshold=0.1
            cor_boxlis=run_nms(boxlistmp,conflist,maxoutput,iouthreshold)
            for j in range(len(cor_boxlis)):
                cor_box=cor_boxlis[j]
                lth=cor_box[0]
                ltw=cor_box[1]
                rbh=cor_box[2]
                rbw=cor_box[3]
                cv2.putText(im,str(i),(ltw,lth),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)
                cv2.rectangle(im,(ltw,lth),(rbw,rbh),(0,255,0),2)
    return im     
num=200

ld,pd=get_batch_by_num(num)
pd=pd.astype(np.float32)
clsmap,regmap=run_rpn(pd)

roib=roi_box(clsmap,regmap)
selected_boxes=roib._build_model()
selected_boxes=np.array(selected_boxes).astype(np.int32)

#drb=draw_rpn_box(pd,clsmap,regmap)
#plt.figure(0)
#plt.imshow(drb/255)

#ld,pd=get_batch_by_num(num)
#pd=pd.astype(np.float32)

#dsb=draw_s_box(pd,selected_boxes)
#plt.figure(1)
#plt.imshow(dsb/255)


clsv,regv=run_fasthead(pd,selected_boxes)

dvb=draw_v_box(pd,selected_boxes,clsv,regv)
plt.imshow(dvb/255)
#clsn,box=run_fasthead_nms(clsv,regv,selected_boxes)

#draw_box(pd,boxes)