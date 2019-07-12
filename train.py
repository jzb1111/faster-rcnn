# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:06:58 2019

@author: ADMIN
"""

import tensorflow as tf
from VGG import vgg16
from resnet import ResNet
from RPN import R_P_N
from ROI_pool import ROIs,ROIs_v2,roi_box
from fast_head import fasthead
from nms_for_train import nms
from support.get_loss import rpn_cls_loss,rpn_reg_loss,fasthead_cls_loss,fasthead_reg_loss
from support.read_data import generate_fasthead_train_sample,file_name,load_train_data,generate_fasthead_train_data_v2
import numpy as np

lr=0.001

xs=tf.placeholder(tf.float32,[1,224,224,3],name='input_xs')

eval_boxes=tf.placeholder(tf.int32,[None,4],name='boxes')
eval_boxes=tf.reshape(eval_boxes,[-1,4])

gt_clsv=tf.placeholder(tf.float32,[None,21])
gt_regv=tf.placeholder(tf.float32,[None,84])
gt_clsvn=tf.placeholder(tf.float32,[None])
gt_regvn=tf.placeholder(tf.float32,[None,84])

gt_rpncls=tf.placeholder(tf.float32,[1,14,14,9,2])
gt_rpnreg=tf.placeholder(tf.float32,[1,14,14,9,4])
gt_clsno=tf.placeholder(tf.float32,[1,14,14,9])
gt_regno=tf.placeholder(tf.float32,[1,14,14,9])

#vgg_net=vgg16(xs)
#vggout=vgg_net._build_model_()

vgg_net=ResNet(xs)
vggout=vgg_net._build_model()

rpn_net=R_P_N(vggout)
clsmap,regmap=rpn_net._build_model_()

clsmap=tf.reshape(clsmap,[1,14,14,9,2],name='clsmap')
regmap=tf.reshape(regmap,[1,14,14,9,4],name='regmap')

#for train:
#eval_boxes:the boxes which closest with gt_box

#roipool=ROIs(vggout,eval_boxes)#tf,tf#for train,we input some gt_box,and for use we input the real box
#rois=roipool._build_model()

roipool=ROIs_v2(vggout,eval_boxes)#tf,tf#for train,we input some gt_box,and for use we input the real box
rois=roipool._build_model()

fast_h=fasthead(rois)
clsv,regv=fast_h._build_model_()

clsv_=tf.reshape(clsv,[-1,21],name='clsv')
regv_=tf.reshape(regv,[-1,84],name='regv')

nms_f_t=nms(clsv,regv,eval_boxes)
clsout,boxout=nms_f_t._build_model_()

clsout=tf.reshape(clsout,[-1],name='clsout')
boxout=tf.reshape(boxout,[-1,4],name='boxout')

rpnclsloss=rpn_cls_loss(clsmap,gt_rpncls,gt_clsno)
rpnregloss=rpn_reg_loss(regmap,gt_rpnreg,gt_regno)

rpnloss=rpnclsloss+rpnregloss

fastheadclsloss=fasthead_cls_loss(clsv,gt_clsv,gt_clsvn)
fastheadregloss=fasthead_reg_loss(regv,gt_regv,gt_regvn)

fastheadloss=fastheadclsloss+fastheadregloss

totalloss=rpnloss+fastheadloss
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(totalloss)

r,d,files=file_name('./VOC2007/Annotations/')

#saver=tf.train.Saver()

init=tf.initialize_all_variables()

clsloss=[]
regloss=[]

with tf.Session() as sess:
    sess.run(init)
    for i in range(500001):
        sjs=np.random.randint(0,len(files))
        pd,ocs,ors,no,rn,gld=load_train_data(sjs)
        eva_clsmap=sess.run(clsmap,feed_dict={xs:pd})
        eva_regmap=sess.run(regmap,feed_dict={xs:pd})
        
        ###########################################
        #these four are for test
        roibox=roi_box(eva_clsmap,eva_regmap)
        s_boxes=roibox._build_model()
        s_boxes=np.array(s_boxes)
        s_boxes=s_boxes.astype(np.int32)
        
        gtclsv,gtregv,clsvno,regvno,flag=generate_fasthead_train_sample(s_boxes,gld)
        
        if flag==1:
            gtclsv=gtclsv.astype(np.float32)
            gtregv=gtregv.astype(np.float32)
            clsvno=clsvno.astype(np.float32)
            regvno=regvno.astype(np.float32)
            ###########################################
            
            ###########################################
            #these four are for train
            #s_boxes,gtclsv,gtregv,clsvno,regvno=generate_fasthead_train_data_v2(gld)
            #s_boxes=np.array(s_boxes).astype(np.int32)
            #gtclsv=gtclsv.astype(np.float32)
            #gtregv=gtregv.astype(np.float32)
            #clsvno=clsvno.astype(np.float32)
            #regvno=regvno.astype(np.float32)
            
            sess.run(train_step,feed_dict={xs:pd,gt_rpncls:ocs,gt_rpnreg:ors,gt_clsno:no,gt_regno:rn,eval_boxes:s_boxes,gt_clsv:gtclsv,gt_regv:gtregv,gt_clsvn:clsvno,gt_regvn:regvno})
            onerpnclsloss=sess.run(rpnclsloss,feed_dict={xs:pd,gt_rpncls:ocs,gt_rpnreg:ors,gt_clsno:no,gt_regno:rn})
            onerpnregloss=sess.run(rpnregloss,feed_dict={xs:pd,gt_rpncls:ocs,gt_rpnreg:ors,gt_clsno:no,gt_regno:rn})
            twofhclsloss=sess.run(fastheadclsloss,feed_dict={xs:pd,eval_boxes:s_boxes,gt_clsv:gtclsv,gt_regv:gtregv,gt_clsvn:clsvno,gt_regvn:regvno})
            twofhregloss=sess.run(fastheadregloss,feed_dict={xs:pd,eval_boxes:s_boxes,gt_clsv:gtclsv,gt_regv:gtregv,gt_clsvn:clsvno,gt_regvn:regvno})
            threetotalloss=sess.run(totalloss,feed_dict={xs:pd,gt_rpncls:ocs,gt_rpnreg:ors,gt_clsno:no,gt_regno:rn,eval_boxes:s_boxes,gt_clsv:gtclsv,gt_regv:gtregv,gt_clsvn:clsvno,gt_regvn:regvno})
            print(i)
            print(threetotalloss,onerpnclsloss,onerpnregloss,twofhclsloss,twofhregloss)
        else:
            print('no pos')
        if i%10000==0:
            output_graph_def1=tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,output_node_names=['clsmap','regmap'])#sess.graph_def
            output_graph_def2=tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,output_node_names=['clsv','regv'])#sess.graph_def
            
            #tflite_model = tf.lite.toco_convert(output_graph_def, [xs], [gpoutput])   #这里[input], [out]这里分别是输入tensor或者输出tensor的集合,是变量实体不是名字
            #open("./model/for_lite/model_mobile"+str(i)+".pb", "wb").write(output_graph_def)
            with tf.gfile.FastGFile('./model/rpn'+str(i)+'.pb', mode = 'wb') as f:
                f.write(output_graph_def1.SerializeToString())
            with tf.gfile.FastGFile('./model/fasthead'+str(i)+'.pb', mode = 'wb') as f:
                f.write(output_graph_def2.SerializeToString())
            
            
    