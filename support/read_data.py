# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:41:54 2019

@author: ADMIN
"""

import re
import os
import numpy as np
import cv2
import random
from xml.dom.minidom import parse
import xml.dom.minidom

def file_name(file_dir):
    root_=[]
    dirs_=[]
    files_=[]
    for root,dirs,files in os.walk(file_dir):
        root_.append(root)
        dirs_.append(dirs)
        files_.append(files)
    return root_,dirs_,files

def clsname2num(name):
    name_dict={'aeroplane':0,'bicycle':1,'bird':2,'boat':3,'bottle':4,'bus':5,'car':6,
               'cat':7,'chair':8,'cow':9,'diningtable':10,'dog':11,'horse':12,'motorbike':13,
               'person':14,'pottedplant':15,'sheep':16,'sofa':17,'train':18,'tvmonitor':19}
    return name_dict[name]

def get_xml(f_n):
    path='./VOC2007/Annotations/'
    DOMTree = xml.dom.minidom.parse(path+f_n)
    annotation = DOMTree.documentElement

    file_name = annotation.getElementsByTagName("filename")[0].firstChild.data
    size = annotation.getElementsByTagName("size")
    width=int(size[0].getElementsByTagName("width")[0].firstChild.data)
    height=int(size[0].getElementsByTagName("height")[0].firstChild.data)
    depth=int(size[0].getElementsByTagName("depth")[0].firstChild.data)
    
    #h_ratio=512/height
    #w_ratio=512/width
    
    objects=annotation.getElementsByTagName("object")
    ob_dict={}
    for i in range(len(objects)):
        tmp_dict={}
        tmp_dict['name']=clsname2num(objects[i].getElementsByTagName("name")[0].firstChild.data)
        bndbox=objects[i].getElementsByTagName("bndbox")[0]
        tmp_dict['xmin']=int(bndbox.getElementsByTagName("xmin")[0].firstChild.data)
        tmp_dict['xmax']=int(bndbox.getElementsByTagName("xmax")[0].firstChild.data)
        tmp_dict['ymin']=int(bndbox.getElementsByTagName("ymin")[0].firstChild.data)
        tmp_dict['ymax']=int(bndbox.getElementsByTagName("ymax")[0].firstChild.data)
        ob_dict[i]=tmp_dict
    xml_dict={}
    xml_dict['f_name']=file_name
    xml_dict['width']=width
    xml_dict['height']=height
    xml_dict['depth']=depth
    xml_dict['obj']=ob_dict
    return xml_dict

def load_image(imageurl):#加载vgg模型必须这样加载图像
    im = cv2.resize(cv2.imread(imageurl),(224,224)).astype(np.float32)
    return im

def get_locdata(xmldata):#没写完，等下接着写,这个函数要写成将xmldata输出为list的函数
    out=[]
    for i in range(len(xmldata)):
        objs=xmldata[i]['obj']
        width=xmldata[i]['width']
        height=xmldata[i]['height']
        w_ratio=224/width
        h_ratio=224/height
        onepic=[]
        for j in range(len(objs)):
            tmp=[]
            tmp.append(round(objs[j]['xmin']*w_ratio))
            tmp.append(round(objs[j]['ymin']*h_ratio))
            tmp.append(round(objs[j]['xmax']*w_ratio))
            tmp.append(round(objs[j]['ymax']*h_ratio))
            tmp.append(objs[j]['name'])
            onepic.append(tmp)
        out.append(onepic)
    return out

def get_picdata(f_n):
    pic_path='./VOC2007/JPEGImages/'
    pic_f=pic_path+f_n
    #pic=np.zeros((224,224,3))
    pic=load_image(pic_f)
    return pic

def get_mini_original_batch(size):
    path1='./VOC2007/Annotations/'
    path2='./VOC2007/JPEGImages/'
    r1,d1,f1=file_name(path1)
    r2,d2,f2=file_name(path2)
    floc=f1[0]
    fpic=f2[0]
    sjs=np.random.randint(0,len(floc),size)
    loc_dict={}
    pic_data=np.zeros((size,224,224,3))
    for i in range(len(sjs)):
        loc_dict[i]=get_xml(floc[sjs[i]])
        pic_data[i]=get_picdata(fpic[sjs[i]])
    return loc_dict,pic_data

def get_batch_by_num(num):
    path1='./VOC2007/Annotations/'
    path2='./VOC2007/JPEGImages/'
    r1,d1,f1=file_name(path1)
    r2,d2,f2=file_name(path2)
    floc=f1
    #print(floc)
    fpic=f2
    #sjs=np.random.randint(0,len(floc),size)
    sjs=[num]
    loc_dict={}
    pic_data=np.zeros((1,224,224,3))
    for i in range(len(sjs)):
        loc_dict[i]=get_xml(floc[sjs[i]])
        pic_data[i]=get_picdata(fpic[sjs[i]])
    return loc_dict,pic_data

def generate_box(hei,wid,cha):#height的范围在0~32
    #h_ratio=224/14
    #w_ratio=224/14
    scalelist=[40,79,158]
    scale=scalelist[int(cha/3)]
    #print(scale)
    ratiolist=[1,1/1.4,1.4]
    ratio=ratiolist[cha%3]
    #print(ratio)
    lefttop_h=(hei+0.5)*16-(scale*ratio)/2
    #print(lefttop_h)
    lefttop_w=(wid+0.5)*16-(scale/ratio)/2
    #print(lefttop_w)
    rightbottom_h=(hei+0.5)*16+(scale*ratio)/2
    #print(rightbottom_h)
    rightbottom_w=(wid+0.5)*16+(scale/ratio)/2
    #lefttop=[lefttop_h,lefttop_w]
    #rightbottom=[rightbottom_h,rightbottom_w]
    out=[int(lefttop_h),int(lefttop_w),int(rightbottom_h),int(rightbottom_w)]
    return out

def IoU(boxA,boxB):
        #boxA=[A的左上角x坐标left，A的左上角y坐标top，A的右下角x坐标right，A的右下角y坐标bottom]
    yA=max(boxA[0],boxB[0])
    xA=max(boxA[1],boxB[1])
    yB=min(boxA[2],boxB[2])
    xB=min(boxA[3],boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    #print(float(boxAArea + boxBArea - interArea))
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

'''def generate_rpn_train_data(loc_data):#loc_data[[[x1,y1,x2,y2,cls][obj2]]]
    outclssample=np.zeros((len(loc_data),14,14,9,2))
    outregsample=np.zeros((len(loc_data),14,14,9,4))
    ocs_no=np.full((len(loc_data),14,14,9),-1)
    reg_no=np.full((len(loc_data),14,14,9),-1)
    ioulist=[]
    iouclslist={}
    for i in range(len(loc_data[0])):
        #for j in range(len(loc_data[0][i])):
        iouclslist[i]=[]
        gt_lefttop_h=loc_data[0][i][1]
        gt_lefttop_w=loc_data[0][i][0]
        gt_rightbottom_h=loc_data[0][i][3]
        gt_rightbottom_w=loc_data[0][i][2]
        gt_cls=loc_data[0][i][4]
        gt_box=[gt_lefttop_h,gt_lefttop_w,gt_rightbottom_h,gt_rightbottom_w]
        for h in range(14):
            for w in range(14):
                for s in range(9):
                    tmp_box=generate_box(h,w,s)
                    if tmp_box[0]>0 and tmp_box[1]>0 and tmp_box[2]<224 and tmp_box[3]<224:
                        ioulist.append([IoU(tmp_box,gt_box),[h,w,s]])
                        iouclslist[i].append([IoU(tmp_box,gt_box),[h,w,s]])
        iouclslist[i]=sorted(iouclslist[i],reverse=True)
        
    right_sample_num=len(loc_data[0])*30#每个object30个框
    for c_num in range(len(loc_data[0])):
        k=0
        right_n=0
        while right_n<=round(right_sample_num/len(loc_data[i])):
            boxtmp=generate_box(iouclslist[c_num][k][1][0],iouclslist[c_num][k][1][1],iouclslist[c_num][k][1][2])
            if boxtmp[0]>0 and boxtmp[1]>0 and boxtmp[2]<224 and boxtmp[3]<224:
                outclssample[0][iouclslist[c_num][k][1][0]][iouclslist[c_num][k][1][1]][iouclslist[c_num][k][1][2]][0]=1
                right_n=right_n+1
                boxtmp=generate_box(iouclslist[c_num][k][1][0],iouclslist[c_num][k][1][1],iouclslist[c_num][k][1][2])
                
                anchor_lefttop_h=boxtmp[0]
                anchor_lefttop_w=boxtmp[1]
                anchor_rightbottom_h=boxtmp[2]
                anchor_rightbottom_w=boxtmp[3]
                
                ya=(anchor_lefttop_h+anchor_rightbottom_h)/2
                xa=(anchor_lefttop_w+anchor_rightbottom_w)/2
                ha=anchor_rightbottom_h-anchor_lefttop_h
                wa=anchor_rightbottom_w-anchor_rightbottom_w
                
                gt_lefttop_h=loc_data[i][c_num][1]
                gt_lefttop_w=loc_data[i][c_num][0]
                gt_rightbottom_h=loc_data[i][c_num][3]
                gt_rightbottom_w=loc_data[i][c_num][2]
                
                gt_y=(gt_rightbottom_h+gt_lefttop_h)/2
                gt_x=(gt_rightbottom_w+gt_lefttop_w)/2
                gt_h=gt_rightbottom_h-gt_lefttop_h
                gt_w=gt_rightbottom_w-gt_lefttop_w
                ty=(gt_y-ya)/ha
                tx=(gt_x-xa)/wa
                tw=np.log(gt_w/wa)
                th=np.log(gt_h/ha)
                outregsample[i][iouclslist[c_num][k][1][0]][iouclslist[c_num][k][1][1]][iouclslist[c_num][k][1][2]][0]=ty
                outregsample[i][iouclslist[c_num][k][1][0]][iouclslist[c_num][k][1][1]][iouclslist[c_num][k][1][2]][1]=tx
                outregsample[i][iouclslist[c_num][k][1][0]][iouclslist[c_num][k][1][1]][iouclslist[c_num][k][1][2]][2]=th
                outregsample[i][iouclslist[c_num][k][1][0]][iouclslist[c_num][k][1][1]][iouclslist[c_num][k][1][2]][3]=tw
            k=k+1
            neg_sample_num=0
            while neg_sample_num<right_sample_num:
                sjs=np.random.randint(0,len(ioulist)/len(loc_data[i]),1)[0]
                ifneg=0
                boxtmp=generate_box(ioulist[sjs][1][0],ioulist[sjs][1][1],ioulist[sjs][1][2])
                for ifn in range(len(loc_data[i])):
                    if ioulist[sjs+ifn*14*14*9][0]>=0.1:
                        ifneg=ifneg+1
                if ifneg==0 and boxtmp[0]>0 and boxtmp[1]>0 and boxtmp[2]<224 and boxtmp[3]<224:
                    outclssample[0][ioulist[sjs][1][0]][ioulist[sjs][1][1]][ioulist[sjs][1][2]][1]=1
                    neg_sample_num=neg_sample_num+1
            for he in range(14):
                for wi in range(14):
                    for st in range(9):
                        if outclssample[0][he][wi][st][0]==1 or outclssample[i][he][wi][st][1]==1:
                            ocs_no[0][he][wi][st]=0
                        if outclssample[0][he][wi][st][0]==1:
                            reg_no[0][he][wi][st]=0 
    return outclssample,outregsample,ocs_no,reg_no'''


def generate_rpn_train_data_v2(loc_data):#loc_data[[[x1,y1,x2,y2,cls][obj2]]]
      
    outclssample=np.zeros((len(loc_data),14,14,9,2))
    outregsample=np.zeros((len(loc_data),14,14,9,4))
    ocs_no=np.full((len(loc_data),14,14,9),-1)
    reg_no=np.full((len(loc_data),14,14,9),-1)
    
    loc_data=loc_data[0]
    for i in range(len(loc_data)):
        obj=loc_data[i]
        lefttop_h=obj[1]
        lefttop_w=obj[0]
        rightbottom_h=obj[3]
        rightbottom_w=obj[2]
        #clsobj=obj[4]
        
        gtbox=[lefttop_h,lefttop_w,rightbottom_h,rightbottom_w]
        ioulist=[]
        for h in range(14):
            for w in range(14):
                for s in range(9):
                    boxtmp=generate_box(h,w,s)
                    #print(gtbox,boxtmp)
                    #print(h,w,s)
                    ioulist.append([IoU(gtbox,boxtmp),[h,w,s]])
        sortlist=sorted(ioulist,reverse=True)
        
        pos_num=25
        reglist=[]
        
        j=0
        for j in range(len(sortlist)):
        #while j<pos_num: 
            #print(sortlist[j])
            hei=sortlist[j][1][0]
            wid=sortlist[j][1][1]
            cha=sortlist[j][1][2]
            
            boxtmp=generate_box(hei,wid,cha)
            anchor_lefttop_h=boxtmp[0]
            anchor_lefttop_w=boxtmp[1]
            anchor_rightbottom_h=boxtmp[2]
            anchor_rightbottom_w=boxtmp[3]
                
            ya=(anchor_lefttop_h+anchor_rightbottom_h)/2
            xa=(anchor_lefttop_w+anchor_rightbottom_w)/2
            ha=anchor_rightbottom_h-anchor_lefttop_h
            wa=anchor_rightbottom_w-anchor_lefttop_w
                
            gt_lefttop_h=lefttop_h
            gt_lefttop_w=lefttop_w
            gt_rightbottom_h=rightbottom_h
            gt_rightbottom_w=rightbottom_w
                
            gt_y=(gt_rightbottom_h+gt_lefttop_h)/2
            gt_x=(gt_rightbottom_w+gt_lefttop_w)/2
            gt_h=gt_rightbottom_h-gt_lefttop_h
            gt_w=gt_rightbottom_w-gt_lefttop_w
            ty=(gt_y-ya)/ha
            tx=(gt_x-xa)/wa
            #print(gt_w,wa)
            tw=np.log(gt_w/wa)
            th=np.log(gt_h/ha)
            #cor_box=correct_box(ty,tx,th,tw)
            
            reglist.append([ty,tx,th,tw])
        k=0
        #print(reglist)
        conf=1
        while k<pos_num or conf>0.4:
            hei=sortlist[k][1][0]
            wid=sortlist[k][1][1]
            cha=sortlist[k][1][2]
            
            ty=reglist[k][0]
            tx=reglist[k][1]
            th=reglist[k][2]
            tw=reglist[k][3]
            
            conf=sortlist[k][0]
            #print(generate_box(hei,wid,cha))
            #print([ty,tx,th,tw])
            #cor_box=correct_box(generate_box(hei,wid,cha),[ty,tx,th,tw])
            #if cor_box[0]>=0 and cor_box[0]<=223 and cor_box[1]>=0 and cor_box[1]<=223 and cor_box[2]>=0 and cor_box[2]<=223 and cor_box[3]>=0 and cor_box[3]<=223:
            outclssample[0][hei][wid][cha][0]=1
            outregsample[0][hei][wid][cha][0]=ty    
            outregsample[0][hei][wid][cha][1]=tx    
            outregsample[0][hei][wid][cha][2]=th    
            outregsample[0][hei][wid][cha][3]=tw    
            k=k+1
    neg_num=0
    gt_boxes=[]
    for i in range(len(loc_data)):
        obj=loc_data[i]
        lefttop_h=obj[1]
        lefttop_w=obj[0]
        rightbottom_h=obj[3]
        rightbottom_w=obj[2]
        gt_boxes.append([lefttop_h,lefttop_w,rightbottom_h,rightbottom_w])
    while neg_num<pos_num*len(gt_boxes):
        
        sjh=np.random.randint(0,14)
        sjw=np.random.randint(0,14)
        sjc=np.random.randint(0,9)
        
        negbox=generate_box(sjh,sjw,sjc)
        if_neg=0
        for i in range(len(gt_boxes)):
            if IoU(gt_boxes[i],negbox)<=0.05:
                if_neg=if_neg+1
                #print(if_neg,'if_neg')
                #print(len(gt_boxes),'gt')
        if if_neg==len(gt_boxes):
            outclssample[0][sjh][sjw][sjc][1]=1
            neg_num=neg_num+1
    for h in range(14):
        for w in range(14):
            for c in range(9):
                if outclssample[0][h][w][c][0]==1 and outclssample[0][h][w][c][1]==1:
                    outclssample[0][h][w][c][0]=0
    for h in range(14):
        for w in range(14):
            for c in range(9):
                if outclssample[0][h][w][c][0]==1 or outclssample[0][h][w][c][1]==1:
                    ocs_no[0][h][w][c]=0
                if outclssample[0][h][w][c][0]==1:
                    reg_no[0][h][w][c]=0
    return outclssample,outregsample,ocs_no,reg_no


def generate_fasthead_train_sample(boxes,loc_data):#boxes[N,4];    loc_data[[[x1,y1,x2,y2,cls][obj2]]]            
    out_clsv=np.zeros((len(boxes),21))
    out_regv=np.zeros((len(boxes),21,4))
    cls_no=np.full((len(boxes)),-1)
    reg_no=np.full((len(boxes),21,4),-1)
    
    all_pos_num=0
    all_neg_num=0
    loc_data=loc_data[0]
    
    poslist=[]
    neglist=[]
    for i in range(len(boxes)):
        boxtmp=boxes[i]
        ifneg=0
        for j in range(len(loc_data)):
            lefttop_h=loc_data[j][1]
            lefttop_w=loc_data[j][0]
            rightbottom_h=loc_data[j][3]
            rightbottom_w=loc_data[j][2]
            gt_cls=loc_data[j][4]
            #print(gt_cls)
            gt_box=[lefttop_h,lefttop_w,rightbottom_h,rightbottom_w]
            #neg_cls=0
            #print(boxtmp,gt_box)
            #################
            #控制正负样本选取#
            #################
            if IoU(boxtmp,gt_box)>=0.65:
                boxcls=int(gt_cls)
                #print(i,boxcls)
                out_clsv[i][boxcls]=1
                
                poslist.append(i)
                
                all_pos_num=all_pos_num+1
                #print(i,boxcls)
                
                anchor_lefttop_h=boxtmp[0]
                anchor_lefttop_w=boxtmp[1]
                anchor_rightbottom_h=boxtmp[2]
                anchor_rightbottom_w=boxtmp[3]
                    
                ya=(anchor_lefttop_h+anchor_rightbottom_h)/2
                xa=(anchor_lefttop_w+anchor_rightbottom_w)/2
                ha=anchor_rightbottom_h-anchor_lefttop_h
                wa=anchor_rightbottom_w-anchor_lefttop_w
                    
                if ha==0:
                    ha=0.01
                if wa==0:
                    wa=0.01
                
                gt_lefttop_h=lefttop_h
                gt_lefttop_w=lefttop_w
                gt_rightbottom_h=rightbottom_h
                gt_rightbottom_w=rightbottom_w
                    
                gt_y=(gt_rightbottom_h+gt_lefttop_h)/2
                gt_x=(gt_rightbottom_w+gt_lefttop_w)/2
                gt_h=gt_rightbottom_h-gt_lefttop_h
                gt_w=gt_rightbottom_w-gt_lefttop_w
                
                ty=(gt_y-ya)/ha
                tx=(gt_x-xa)/wa
                tw=np.log(gt_w/wa)
                th=np.log(gt_h/ha)
                out_regv[i][boxcls][0]=ty
                out_regv[i][boxcls][1]=tx
                out_regv[i][boxcls][2]=th
                out_regv[i][boxcls][3]=tw
            if IoU(boxtmp,gt_box)<=0.3:
                ifneg=ifneg+1
        if ifneg==len(loc_data):
            out_clsv[i][20]=1
            all_neg_num=all_neg_num+1
            neglist.append(i)
    #确保negnum是posnum的三倍
    #print(all_pos_num,all_neg_num)
    #all_neg_num=len(boxes)-all_pos_num
    
    if all_pos_num>=int(round(all_neg_num/3)):
        pos_num=int(round(all_neg_num/3))
        neg_num=int(pos_num*2)
    else:
        pos_num=int(all_pos_num)
        neg_num=int(round(pos_num*0.15))
    ############
    #设定负样本 #
    ############
    if neg_num==0:
        print('no pos')
        if np.random.uniform(0,1)<=0.5:
            neg_num=1
        else:
            neg_num=0
    #print(pos_num,neg_num)
    k=0
    #print(poslist)
    #print(neglist)
    while k<pos_num:
        sjs=np.random.randint(0,len(poslist))
        #print(out_clsv[sjs][20])
        #print(out_clsv[poslist[sjs]])
        if out_clsv[poslist[sjs]][20]!=1:
            cls_no[poslist[sjs]]=0
            objcls=vec2num(out_clsv[poslist[sjs]])
            reg_no[poslist[sjs]][objcls][0]=0
            reg_no[poslist[sjs]][objcls][1]=0
            reg_no[poslist[sjs]][objcls][2]=0
            reg_no[poslist[sjs]][objcls][3]=0
            k=k+1
    l=0
    #print(len(neglist))
    if len(neglist)>0:
        while l<neg_num:
            sjs=np.random.randint(0,len(neglist))
            if out_clsv[neglist[sjs]][20]==1:
                cls_no[neglist[sjs]]=0
                objcls=20
                l=l+1
    '''else:
        while l<1:#neg_num:
            print('fake_neg')
            sjs=np.random.randint(0,len(boxes))
            #if out_clsv[neglist[sjs]][20]==1:
            cls_no[sjs]=0
            objcls=20
            out_clsv[sjs][20]=1
            l=l+1  '''  
        
    out_regv=np.reshape(out_regv,[-1,84])
    reg_no=np.reshape(reg_no,[-1,84])
    if pos_num>0 and neg_num>0:
        flag=1
    else:
        flag=0
    return out_clsv,out_regv,cls_no,reg_no,flag
            #else:
             #   neg_cls=neg_cls+1
           
def vec2num(vec):
    out=20
    for i in range(len(vec)):
         if vec[i]==1:
             out=i
    return out
def all_zero(vector):
    v_co=0
    for i in range(len(vector)):
        if vector[i]==0:
            v_co=v_co+1
    if v_co==len(vector):
        out='all_z'
    else:
        out='some'
    return out                
def correct_box(anchorbox,reg):
    out=[]
    anchor_lefttop_h=anchorbox[0]
    anchor_lefttop_w=anchorbox[1]
    anchor_rightbottom_h=anchorbox[2]
    anchor_rightbottom_w=anchorbox[3]
    ty=reg[0]
    tx=reg[1]
    th=reg[2]
    tw=reg[3]
    ya=(anchor_lefttop_h+anchor_rightbottom_h)/2
    xa=(anchor_lefttop_w+anchor_rightbottom_w)/2
    ha=anchor_rightbottom_h-anchor_lefttop_h
    wa=anchor_rightbottom_w-anchor_lefttop_w
    y=ty*ha+ya
    x=tx*wa+xa
    h=np.power(np.e,th)*ha
    w=np.power(np.e,tw)*wa
    lefttop_h=int(y-h/2)
    lefttop_w=int(x-w/2)
    rightbottom_h=int(y+h/2)
    rightbottom_w=int(x+w/2)
    out=[lefttop_h,lefttop_w,rightbottom_h,rightbottom_w]
    return out
def draw_rpn_boxes(pd,ocs,reg):
    im=pd[0]
    anchor_box=[]
    bbox=[]
    neg_box=[]
    for h in range(14):
        for w in range(14):
            for c in range(9):
                if ocs[0][h][w][c][0]==1:
                    anchor_box.append(generate_box(h,w,c))
                    bbox.append(correct_box(generate_box(h,w,c),reg[0][h][w][c]))
                    #print(generate_box(h,w,c),reg[0][h][w][c],correct_box(generate_box(h,w,c),reg[0][h][w][c]))
                    #print()
                if ocs[0][h][w][c][1]==1:
                    neg_box.append(generate_box(h,w,c))
    pos_box=bbox#anchor_box
    #print(anchor_box)
    #print()
    for i in range(len(anchor_box)):                
        cv2.rectangle(im,(pos_box[i][1],pos_box[i][0]),(pos_box[i][3],pos_box[i][2]),(0,255,0),1)#画出框
        #if pos_box[i][1]<0 or pos_box[i][1]>223:
            #print(pos_box[i][1])
        try:
            cv2.rectangle(im,(neg_box[i][1],neg_box[i][0]),(neg_box[i][3],neg_box[i][2]),(0,0,255),1)#画出框
        except:
            pass
    cv2.imshow("Image", im/255)    
    cv2.waitKey (0)
    cv2.destroyAllWindows()
    im=pd[0]
    return 0

def load_train_data(num):
    pdname='./train_data/pd_'+str(num)+'.npy'
    ocsname='./train_data/rpn_ocs_'+str(num)+'.npy'
    orsname='./train_data/rpn_ors_'+str(num)+'.npy'
    noname='./train_data/rpn_no_'+str(num)+'.npy'
    rnname='./train_data/rpn_rn_'+str(num)+'.npy'
    gldname='./train_data/gld_'+str(num)+'.npy'
    pd=np.load(pdname).astype(np.float32)
    ocs=np.load(ocsname).astype(np.float32)
    ors=np.load(orsname).astype(np.float32)
    no=np.load(noname).astype(np.float32)
    rn=np.load(rnname).astype(np.float32)
    gld=np.load(gldname).astype(np.float32)
    return pd,ocs,ors,no,rn,gld
def split_box(boxes):
    out=[]
    for i in range(len(boxes)):
        lefttop_h=boxes[i][0]
        lefttop_w=boxes[i][1]
        rightbottom_h=boxes[i][2]
        rightbottom_w=boxes[i][3]
        if lefttop_h<0:
            lefttop_h=0
        if lefttop_h>223:
            lefttop_h=223
        if lefttop_w<0:
            lefttop_w=0
        if lefttop_w>223:
            lefttop_w=223
        if rightbottom_h<0:
            rightbottom_h=0
        if rightbottom_h>223:
            rightbottom_h=223
        if rightbottom_w<0:
            rightbottom_w=0
        if rightbottom_w>223:
            rightbottom_w=223
        out.append([lefttop_h,lefttop_w,rightbottom_h,rightbottom_w])
    return out
def generate_fasthead_train_data_v2(loc_data):
    pos_num=0
    loc_data=loc_data[0]
    #out_clsv=np.zeros((samplenum,21))
    #out_regv=np.zeros((samplenum,21,4))
    #cls_no=np.full((samplenum),-1)
    #reg_no=np.full((samplenum,21,4),-1)
    for i in range(len(loc_data)):
        ioulist=[]
        lefttop_h=loc_data[i][1]
        lefttop_w=loc_data[i][0]
        rightbottom_h=loc_data[i][3]
        rightbottom_w=loc_data[i][2]
        gt_cls=loc_data[i][4]
        gt_box=[lefttop_h,lefttop_w,rightbottom_h,rightbottom_w]
        #print(gt_box)
        for h in range(14):
            for w in range(14):
                for c in range(9):
                    boxtmp=generate_box(h,w,c)
                    #ioulist.append([IoU(gt_box,tmpbox),[h,w,c,gt_cls]])
                    iou=IoU(gt_box,boxtmp)
                    if iou>=0.5:
                        pos_num=pos_num+1
    if pos_num>0:
        samplenum=pos_num*2
        
    else:
        print('no pos')
        samplenum=1
    #print(pos_num,samplenum)
    out_clsv=np.zeros((samplenum,21))
    out_regv=np.zeros((samplenum,21,4))
    cls_no=np.full((samplenum),-1)
    reg_no=np.full((samplenum,21,4),-1)
    
    boxlist=[]
    ser=0
    neglist=[]
    for h in range(14):
        for w in range(14):
            for c in range(9):
                boxtmp=generate_box(h,w,c)
                ifneg=0
                #ioulist.append([IoU(gt_box,tmpbox),[h,w,c,gt_cls]])
                for i in range(len(loc_data)):
                    ioulist=[]
                    lefttop_h=loc_data[i][1]
                    lefttop_w=loc_data[i][0]
                    rightbottom_h=loc_data[i][3]
                    rightbottom_w=loc_data[i][2]
                    gt_cls=loc_data[i][4]
                    gt_box=[lefttop_h,lefttop_w,rightbottom_h,rightbottom_w]
                    iou=IoU(gt_box,boxtmp)
                    if iou>=0.5:
                        anchor_lefttop_h=boxtmp[0]
                        anchor_lefttop_w=boxtmp[1]
                        anchor_rightbottom_h=boxtmp[2]
                        anchor_rightbottom_w=boxtmp[3]
                            
                        ya=(anchor_lefttop_h+anchor_rightbottom_h)/2
                        xa=(anchor_lefttop_w+anchor_rightbottom_w)/2
                        ha=anchor_rightbottom_h-anchor_lefttop_h
                        wa=anchor_rightbottom_w-anchor_lefttop_w
                            
                        if ha==0:
                            ha=0.01
                        if wa==0:
                            wa=0.01
                        
                        gt_lefttop_h=lefttop_h
                        gt_lefttop_w=lefttop_w
                        gt_rightbottom_h=rightbottom_h
                        gt_rightbottom_w=rightbottom_w
                            
                        gt_y=(gt_rightbottom_h+gt_lefttop_h)/2
                        gt_x=(gt_rightbottom_w+gt_lefttop_w)/2
                        gt_h=gt_rightbottom_h-gt_lefttop_h
                        gt_w=gt_rightbottom_w-gt_lefttop_w
                        ty=(gt_y-ya)/ha
                        tx=(gt_x-xa)/wa
                        tw=np.log(gt_w/wa)
                        th=np.log(gt_h/ha)
                        #print(ser,gt_cls)
                        gt_cls=int(gt_cls)
                        out_clsv[ser][gt_cls]=1
                        out_regv[ser][gt_cls][0]=ty
                        out_regv[ser][gt_cls][1]=tx
                        out_regv[ser][gt_cls][2]=th
                        out_regv[ser][gt_cls][3]=tw
                        cls_no[ser]=0
                        reg_no[ser][gt_cls][0]=0
                        reg_no[ser][gt_cls][1]=0
                        reg_no[ser][gt_cls][2]=0
                        reg_no[ser][gt_cls][3]=0
                        
                        boxlist.append(boxtmp)
                        ser=ser+1
                    if iou<=0.1:
                        ifneg=ifneg+1
                    if ifneg==len(loc_data):
                        neglist.append(boxtmp)
                        #out_clsv[ser][gt_cls]=1
                        #boxlist.append(boxtmp)
                        #cls_no[ser]=0
                        #reg_no[ser][gt_cls][0]=0
                        #reg_no[ser][gt_cls][1]=0
                        #reg_no[ser][gt_cls][2]=0
                        #reg_no[ser][gt_cls][3]=0
                        #ser=ser+1
    neg_num=0
    while neg_num<samplenum-pos_num:
        #print(ser)
        sjs=np.random.randint(0,len(neglist))
        out_clsv[ser][20]=1
        boxlist.append(neglist[sjs])
        cls_no[ser]=0
        #reg_no[ser][gt_cls][0]=0
        #reg_no[ser][gt_cls][1]=0
        #reg_no[ser][gt_cls][2]=0
        #reg_no[ser][gt_cls][3]=0
        ser=ser+1
        neg_num=neg_num+1
        out_regv=np.reshape(out_regv,[-1,84])
        reg_no=np.reshape(reg_no,[-1,84])
    boxlist=split_box(boxlist)
    return boxlist,out_clsv,out_regv,cls_no,reg_no
                
                    