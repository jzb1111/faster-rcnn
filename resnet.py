import tensorflow as tf
import tensorflow.contrib as tc

class ResNet():
	def __init__(self,xdata,is_training=True,input_size=224):
		self.input_size=input_size
		self.is_training=is_training
		self.normalizer=tc.layers.batch_norm
		self.bn_params={'is_training':self.is_training}
		self.xs=xdata
		#with tf.variable_scope('ResNet'):
        #self._build_model()
	
	def _build_model(self):#resnet101
		with tf.variable_scope('init_conv'):
			paddings=tf.constant([[0,0],[3,3],[3,3],[0,0]])
			self.xs=tf.pad(self.xs,paddings,"CONSTANT")
			output=tc.layers.conv2d(self.xs,64,7,
									stride=2,padding='VALID',normalizer_fn=self.normalizer,
									normalizer_params=self.bn_params)#112
			#output=tc.layers.max_pool2d(output,[3,3],2,padding='SAME')#56
		output=self.res_block(output,[64,64],'conv2_x',3,first_block=True)#56
		print(output)
		output=self.res_block(output,[128,128],'conv3_x',4)#28
		print(output)
		output=self.res_block(output,[256,256],'conv4_x',6)#14
		print(output)
		output=self.res_block(output,[512,512],'conv5_x',3)#7
		
		return output
		
		
		
		
	def identity_block(self,inputs,conv_num,name):
		#inputs:输入
		#conv_num:每个卷积层的num
		#name:name		
		with tf.variable_scope('id_block{}'.format(name)):
					
			output=tc.layers.conv2d(inputs,conv_num[0],1,
									stride=1,padding='VALID',normalizer_fn=self.normalizer,
									normalizer_params=self.bn_params)
			output=tc.layers.conv2d(output,conv_num[0],3,
									stride=1,padding='SAME',normalizer_fn=self.normalizer,
									normalizer_params=self.bn_params)
			output=tc.layers.conv2d(output,conv_num[1],1,
									stride=1,padding='VALID',activation_fn=None,
									normalizer_fn=self.normalizer,
									normalizer_params=self.bn_params)
			
			output=tf.nn.relu(inputs+output)
			
			return output
	
			
	def conv_block(self,inputs,conv_num,name,first_block):
		#inputs:输入
		#conv_num:每个卷积层的num
		#name:name
		#conv_block的hwc均发生变化
		input_channel=inputs.get_shape().as_list()[-1]
		
		if first_block is True:
			stride_f=1
		else:
			stride_f=2
		with tf.variable_scope('conv_block{}'.format(name)):
					
			output=tc.layers.conv2d(inputs,conv_num[0],1,
									stride=stride_f,padding='VALID',normalizer_fn=self.normalizer,
									normalizer_params=self.bn_params)
			output=tc.layers.conv2d(output,conv_num[0],3,
									stride=1,padding='SAME',normalizer_fn=self.normalizer,
									normalizer_params=self.bn_params)
			output=tc.layers.conv2d(output,conv_num[1],1,
									stride=1,padding='VALID',activation_fn=None,
									normalizer_fn=self.normalizer,
									normalizer_params=self.bn_params)
			
			inputs=tc.layers.conv2d(inputs,conv_num[1],1,
								   stride=stride_f,padding='VALID',activation_fn=None,
								   normalizer_fn=self.normalizer,
							       normalizer_params=self.bn_params)
			output=tf.nn.relu(inputs+output)
			return output
			
	def res_block(self,inputs,conv_num,name,repeat_num,first_block=False):
		with tf.variable_scope(name):
			output=self.conv_block(inputs,conv_num,'0',first_block)
			for i in range(repeat_num-1):
				output=self.identity_block(output,conv_num,str(i+1))
		return output
		