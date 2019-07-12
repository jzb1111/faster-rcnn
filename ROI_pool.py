# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 16:45:28 2019

@author: ADMIN
"""

import numpy as np
import tensorflow as tf
from runNMS import run_nms
#from runGFV import run_GFV
from GFV_for_train import gfv

class ROIs():
    #我们假设rpn产生的两个特征图直接传给roi
    #输入xs[1,14,14,512]tf
    #    boxes[-1,4]
    def __init__(self,xs,boxes):
        self.xs=xs
        self.boxes=boxes
        
    def _build_model(self):
        #num,hei,wid,cha=self.xs.shape
 
        rois=self.generate_rois(self.boxes)
        
        return rois
    
    def generate_rois(self,boxes):#boxes[N,4]
        #num,le=boxes.shape
        
        lefttop_h=boxes[:,0]#N
        lefttop_w=boxes[:,1]#N
        rightbottom_h=boxes[:,2]#N
        rightbottom_w=boxes[:,3]#N
        
        map_lefttop_h=tf.to_int32(tf.floor(lefttop_h/16))#N
        map_lefttop_w=tf.to_int32(tf.floor(lefttop_w/16))#N
        map_rightbottom_h=tf.to_int32(tf.floor(rightbottom_h/16))#N
        map_rightbottom_w=tf.to_int32(tf.floor(rightbottom_w/16))#N
        
        map_lefttoph_tmp=tf.reshape(map_lefttop_h,[-1,1])
        map_lefttopw_tmp=tf.reshape(map_lefttop_w,[-1,1])
        map_lefttoph_tile=tf.tile(map_lefttoph_tmp,multiples=[1,25])
        map_lefttopw_tile=tf.tile(map_lefttopw_tmp,multiples=[1,25])
        
        map_lefttoph_r=tf.reshape(map_lefttoph_tile,[-1,5,5])
        map_lefttopw_r=tf.reshape(map_lefttopw_tile,[-1,5,5])
        
        hei=map_rightbottom_h-map_lefttop_h#N
        wid=map_rightbottom_w-map_lefttop_w#N
        
        hei_ratio=hei/5#N
        wid_ratio=wid/5#N
        hei_ratio_tmp=hei_ratio
        wid_ratio_tmp=wid_ratio
        
        hei_ratio_tmp=tf.reshape(hei_ratio,[-1,1])
        wid_ratio_tmp=tf.reshape(wid_ratio,[-1,1])
        hei_ratio_tile=tf.tile(hei_ratio_tmp,multiples=[1,25])
        wid_ratio_tile=tf.tile(wid_ratio_tmp,multiples=[1,25])
        
        #hei_ratio
        #print(hei_ratio)
        #print(wid_ratio)
        hei_ratio_r=tf.reshape(hei_ratio_tile,[-1,5,5])
        hei_ratio_r=tf.transpose(hei_ratio_r,[0,2,1])
        wid_ratio_r=tf.reshape(wid_ratio_tile,[-1,5,5])
        hei_wid_ratio=tf.stack([hei_ratio_r,wid_ratio_r],axis=-1)
            
        #1.reshape one rois on map2.where equal 1 to generate
        rois_tabel=self.generate_tabel(boxes)
        rois_tabel_h=rois_tabel[:,:,:,0]+map_lefttoph_r
        rois_tabel_w=rois_tabel[:,:,:,1]+map_lefttopw_r
        rois_tabel=tf.stack([rois_tabel_h,rois_tabel_w],axis=-1)
        #rois_on_map_h=tf.to_int32(tf.round(tf.to_float(rois_tabel[:,:,:,0])*tf.to_float(hei_ratio)))#int(round(h*hei_ratio))
        #rois_on_map_w=tf.to_int32(tf.round(tf.to_float(rois_tabel[:,:,:,1])*tf.to_float(wid_ratio)))
        #rois_on_map=tf.stack([rois_on_map_h,rois_on_map_w],axis=-1)
        rois_on_map=tf.to_int32(tf.round(tf.to_float(rois_tabel)*tf.to_float(hei_wid_ratio)))
        
        fvs=gfv(self.xs,rois_on_map)
        fv_res=tf.reshape(fvs,[-1,5*5*512])
        #roilist.append(fv_res)
        return fv_res
        
    def generate_tabel(self,a):
        a_h=a[:,0]
        onelike=tf.ones_like(a_h)
        tmp=onelike
        for h in range(5*5-1):
            tmp=tf.concat([tmp,onelike],axis=0)
        num=tmp
        out=tf.where(tf.equal(num,1))
        out_h=tf.to_int32(out/5-tf.floor(out/25)*5)    
        out_w=tf.to_int32(out%5)
        out=tf.stack([out_h,out_w],axis=1)
        out=tf.reshape(out,[-1,5,5,2])
        #out=tf.zeros((num,2))
        
        return out        
    
class ROIs_v2():
    #我们假设rpn产生的两个特征图直接传给roi
    #输入xs[1,14,14,512]tf
    #    boxes[-1,4]
    def __init__(self,xs,boxes):
        self.xs=xs
        self.boxes=boxes
        
    def _build_model(self):
        #num,hei,wid,cha=self.xs.shape
 
        rois=self.generate_rois(self.boxes)
        
        return rois
    
    def generate_rois(self,boxes):#boxes[N,4]
        #num,le=boxes.shape
        boxes=tf.to_float(boxes)
        
        lefttop_h=boxes[:,0]/224#N
        lefttop_w=boxes[:,1]/224#N
        rightbottom_h=boxes[:,2]/224#N
        rightbottom_w=boxes[:,3]/224#N
        
        t_box=tf.stack([lefttop_h,lefttop_w,rightbottom_h,rightbottom_w],axis=1)
        ind=tf.to_int32(tf.zeros_like(lefttop_h))
        rois=tf.image.crop_and_resize(self.xs,t_box,box_ind=ind,crop_size=[5,5],method='bilinear')
        rois=tf.reshape(rois,[-1,25*512])
        return rois
        
        

    
class roi_box():#np,np
    def __init__(self,rpn_cls,rpn_reg):
        self.rpn_cls=rpn_cls
        self.rpn_reg=rpn_reg
        
    def _build_model(self):
        hei=14
        wid=14
        cha=9
        clsconf=self.calc_conf()#[14,14,9]
        boxlist,conflist=self.calc_anchorbox(clsconf,hei,wid,cha)
        boxlist=np.array(boxlist).astype(np.float32)
        conflist=np.array(conflist).astype(np.float32)
        selected_boxes=run_nms(boxlist,conflist,200,0.01)#改变iou不知道会有什么后果
        selected_boxes=self.split_box(selected_boxes)
        return selected_boxes
    
    def calc_conf(self):
        out=self.rpn_cls[0,:,:,:,0]-self.rpn_cls[0,:,:,:,1]#out[14,14,9]
        return out
    def split_box(self,boxes):
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
    def generate_anchorbox(self,hei,wid,cha):
        out=[]
        scalelist=[40,79,158]
        scale=scalelist[int(cha/3)]
        ratiolist=[1,1/1.4,1.4]
        ratio=ratiolist[cha%3]
        lefttop_h=(hei+0.5)*16-(scale*ratio)/2
        lefttop_w=(wid+0.5)*16-(scale/ratio)/2
        
        rightbottom_h=(hei+0.5)*16+(scale*ratio)/2
        rightbottom_w=(wid+0.5)*16+(scale/ratio)/2
        #lefttop=[lefttop_h,lefttop_w]
        #rightbottom=[rightbottom_h,rightbottom_w]
        out=[int(lefttop_h),int(lefttop_w),int(rightbottom_h),int(rightbottom_w)]
        return out
    
    def correct_box(self,anchorbox,reg):
        out=[]
        anchor_lefttop_h=anchorbox[0]
        anchor_lefttop_w=anchorbox[1]
        anchor_rightbottom_h=anchorbox[2]
        anchor_rightbottom_w=anchorbox[3]
        ty=reg[0]
        tx=reg[1]
        th=reg[2]
        tw=reg[3]
        ya=(anchor_lefttop_h+anchor_rightbottom_h)/2#>0
        xa=(anchor_lefttop_w+anchor_rightbottom_w)/2#>0
        ha=anchor_rightbottom_h-anchor_lefttop_h#>0
        wa=anchor_rightbottom_w-anchor_lefttop_w#>0
        y=ty*ha+ya
        x=tx*wa+xa
        h=np.power(np.e,th)*ha
        #if h<=0:
         #   print(h,'h')
        w=np.power(np.e,tw)*wa
        #if w<=0:
         #   print(w,'w')
        lefttop_h=int(y-h/2)
        lefttop_w=int(x-w/2)
        rightbottom_h=int(y+h/2)
        rightbottom_w=int(x+w/2)
        out=[lefttop_h,lefttop_w,rightbottom_h,rightbottom_w]
        return out
    def calc_anchorbox(self,clsconf,hei,wid,cha):
        clslist=[]
        #reglist=[]
        for h in range(hei):
            for w in range(wid):
                for c in range(cha):
                    clslist.append([clsconf[h][w][c],[h,w,c]])
                    #reglist.append(self.rpn_reg[h][w][c],[h,w,c])
        sortcls=sorted(clslist,reverse=True)
        
        boxlist=[]
        conflist=[]
        for i in range(1000):
            hei=sortcls[i][1][0]
            wid=sortcls[i][1][1]
            cha=sortcls[i][1][2]
            conf=sortcls[i][0]
            anchorboxestmp=self.generate_anchorbox(hei,wid,cha)
            reginf=self.rpn_reg[0][hei][wid][cha]
            #print(anchorboxestmp,'anchor')
            #print(reginf,'reginf')
            box=self.correct_box(anchorboxestmp,reginf)
            if box[0]>=0 and box[0]<=223 and box[1]>=0 and box[1]<=223 and box[2]>=0 and box[2]<=223 and box[3]>=0 and box[3]<=223:
                boxlist.append(box)
                conflist.append(conf)
        return boxlist,conflist