import numpy as np
import tensorflow as tf
import tflearn
from tensorflow.contrib.layers.python.layers import batch_norm
import random
import pickle 
import scipy.ndimage as nd 
import scipy 
import math
import svgwrite
from svgwrite.image import Image as svgimage
from PIL import Image
import sys 
import os 
from resnet import resblock as residual_block
from resnet import relu
from resnet import batch_norm as batch_norm_resnet  

import tf_common_layer as common

MAX_DEGREE=6


class Sat2GraphModel():
	def __init__(self, sess, image_size=352, image_ch = 3, downsample_level = 1, batchsize = 8, resnet_step=8, channel=12, mode = "train", joint_with_seg=True):
		self.sess = sess 
		self.train_seg = False
		self.image_size = image_size
		self.image_ch = image_ch 
		self.channel = channel
		self.joint_with_seg = joint_with_seg
		self.mode = mode 
		#self.model_name = model_name
		self.batchsize = batchsize
		self.resnet_step = resnet_step

		self.input_sat = tf.placeholder(tf.float32, shape = [self.batchsize, self.image_size, self.image_size, self.image_ch], name="input")

		self.input_seg_gt = tf.placeholder(tf.float32, shape = [self.batchsize, self.image_size, self.image_size, 1])
		self.input_seg_gt_target = tf.concat([self.input_seg_gt+0.5, 0.5 - self.input_seg_gt], axis=3)

		self.target_prob = tf.placeholder(tf.float32, shape = [self.batchsize, self.image_size, self.image_size, 2 * (MAX_DEGREE + 1)])
		self.target_vector = tf.placeholder(tf.float32, shape = [self.batchsize, self.image_size, self.image_size, 2 * (MAX_DEGREE)])

		self.np_mask = np.ones((self.batchsize,self.image_size, self.image_size,1))
		self.np_mask[:,32:self.image_size-32,32:self.image_size-32,:] =0.0 

		self.lr = tf.placeholder(tf.float32, shape=[])
		
		self.is_training = tf.placeholder(tf.bool, name="istraining")


		if self.train_seg:
			self.linear_output = self.BuildDeepLayerAggregationNetWithResnet(self.input_sat, input_ch = image_ch, output_ch = 2, ch = channel)

			num_unet = len(tf.trainable_variables())
			print("Weights", num_unet)

			self.output = tf.nn.softmax(self.linear_output)
			self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(self.input_seg_gt_target, self.linear_output))

			self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

		else:
			
			self.imagegraph_output = self.BuildDeepLayerAggregationNetWithResnet(self.input_sat, input_ch = image_ch, output_ch =2 + MAX_DEGREE * 4 + (2 if self.joint_with_seg==True else 0), ch=channel)

			x = self.imagegraph_output

			num_unet = len(tf.trainable_variables())
			print("Number of Weights", num_unet)
			
			self.output = self.SoftmaxOutput(self.imagegraph_output)

			target = self.Merge(self.target_prob, self.target_vector)
			
			self.keypoint_prob_loss, self.direction_prob_loss, self.direction_vector_loss, self.seg_loss = self.SupervisedLoss(self.imagegraph_output, self.target_prob, self.target_vector)
			
			self.prob_loss = (self.keypoint_prob_loss + self.direction_prob_loss)
			if self.joint_with_seg:
				self.loss = self.prob_loss + self.direction_vector_loss + self.seg_loss
			else:
				self.loss = self.prob_loss + self.direction_vector_loss

			self.l2loss_grad = tf.gradients(self.loss, tf.trainable_variables())
			self.l2loss_grad_max = tf.reduce_max(self.l2loss_grad[0])
			#self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).apply_gradients(zip(self.l2loss_grad, tf.trainable_variables()))
			self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)


		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver(max_to_keep=10)

		self.summary_loss = []
		
		self.test_loss =  tf.placeholder(tf.float32)
		self.train_loss =  tf.placeholder(tf.float32)
		self.l2_grad = tf.placeholder(tf.float32)

		self.summary_loss.append(tf.summary.scalar('loss/test', self.test_loss))
		self.summary_loss.append(tf.summary.scalar('loss/train', self.train_loss))
		self.summary_loss.append(tf.summary.scalar('grad/l2', self.l2_grad))

		self.merged_summary = tf.summary.merge_all()


	def class_reduce_block(self, x, in_ch, out_ch, name, resnet_step = 0, k = 3):
		x, _, _ = common.create_conv_layer(name+"_1", x, in_ch, out_ch, kx = k, ky = k, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = True)
		x, _, _ = common.create_conv_layer(name+"_2", x, out_ch, out_ch, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = self.is_training, batchnorm = True)

		return x 

	def class_resnet_blocks(self, x, ch, name, resnet_step=0):
		if resnet_step > 0:
			for i in range(resnet_step):
				x = residual_block(x, channels=ch, is_training=self.is_training, downsample=False, scope=name+"_residual_block_decode_%d" % i)

			#x = batch_norm_resnet(x, is_training = self.is_training, scope = name+"_decode_0_batch_norm")
			x = batch_norm_resnet(x, scope = name+"_decode_0_batch_norm") # roll back

			x = tf.nn.relu(x)
		return x 


	def class_aggregate_block(self, x1, x2, in_ch1, in_ch2, out_ch, name, batchnorm=True, k = 3):
		x2, _, _ = common.create_conv_layer(name+"_1", x2, in_ch2, in_ch2, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = self.is_training, batchnorm = batchnorm, deconv=True)
		
		x = tf.concat([x1,x2], axis=3) # in_ch1 + in_ch2

		x, _, _ = common.create_conv_layer(name+"_2", x, in_ch1 + in_ch2, out_ch, kx = k, ky = k, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = batchnorm)
		x, _, _ = common.create_conv_layer(name+"_3", x, out_ch, out_ch, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = batchnorm)

		return x 



	def unstack(self, tensor, axis = 3, size = None):
		ts = tf.unstack(tensor, axis = 3)
		new_ts = []

		for t in ts:
			if size is None:
				new_ts.append(tf.reshape(t,shape=[-1,self.image_size,self.image_size,1]))
			else:
				new_ts.append(tf.reshape(t,shape=[-1,size,size,1]))

		return new_ts 


	def DownSampleSimilarityLoss(self, output, target):
		
		y = tf.nn.avg_pool(output, [1,3,3,1], strides=[1,2,2,1],padding='SAME')
		y = tf.nn.avg_pool(y, [1,3,3,1], strides=[1,2,2,1],padding='SAME')
		y = tf.nn.avg_pool(y, [1,3,3,1], strides=[1,2,2,1],padding='SAME')
		y = tf.nn.avg_pool(y, [1,3,3,1], strides=[1,2,2,1],padding='SAME')
		
		x = tf.nn.avg_pool(target, [1,3,3,1], strides=[1,2,2,1],padding='SAME')
		x = tf.nn.avg_pool(x, [1,3,3,1], strides=[1,2,2,1],padding='SAME')
		x = tf.nn.avg_pool(x, [1,3,3,1], strides=[1,2,2,1],padding='SAME')
		x = tf.nn.avg_pool(x, [1,3,3,1], strides=[1,2,2,1],padding='SAME')


		return tf.reduce_mean(tf.nn.l2_loss(x-y))


	def DownSampleSimilarityLossOnProbs(self, softmax_outputs, target_prob):

		channels = self.unstack(softmax_outputs, axis = 3)

		new_list = []
		new_list += channels[0:2]

		for i in range(MAX_DEGREE):
			new_list += channels[2+4*i:4+4*i]

		return self.DownSampleSimilarityLoss(tf.concat(new_list, axis=3), target_prob)


	def SupervisedLoss(self, imagegraph_output, imagegraph_target_prob, imagegraph_target_vector):

		imagegraph_outputs = self.unstack(imagegraph_output, axis = 3)
		imagegraph_target_probs = self.unstack(imagegraph_target_prob, axis = 3)
		imagegraph_target_vectors = self.unstack(imagegraph_target_vector, axis = 3)

		soft_mask = tf.clip_by_value(imagegraph_target_probs[0]-0.01, 0.0, 0.99)
		soft_mask = soft_mask + 0.01 

		soft_mask2 = tf.reshape(soft_mask, [self.batchsize, self.image_size, self.image_size])


		#seg_mask = tf.clip_by_value(self.input_seg_gt+0.5, 0.2, 0.8) * 5.0 


		keypoint_prob_loss = 0

		keypoint_prob_output = tf.concat(imagegraph_outputs[0:2], axis=3) 
		keypoint_prob_target = tf.concat(imagegraph_target_probs[0:2], axis=3) 

		#keypoint_prob_loss = tf.reduce_sum(tf.losses.softmax_cross_entropy(keypoint_prob_target, keypoint_prob_output, reduction=tf.losses.Reduction.NONE))


		keypoint_prob_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(keypoint_prob_target, keypoint_prob_output))

		# direction prob loss
		direction_prob_loss = 0

		for i in range(MAX_DEGREE):
			prob_output = tf.concat(imagegraph_outputs[2 + i*4 : 2 + i*4 + 2], axis=3)
			prob_target = tf.concat(imagegraph_target_probs[2 + i*2 : 2 + i*2 + 2], axis=3)

			#direction_prob_loss += tf.reduce_mean(tf.multiply((self.input_seg_gt+0.5), tf.losses.softmax_cross_entropy(prob_target, prob_output)))

			# only at key points! 
			direction_prob_loss += tf.reduce_mean(tf.multiply((soft_mask2), tf.losses.softmax_cross_entropy(prob_target, prob_output, reduction=tf.losses.Reduction.NONE)))

		direction_prob_loss /= MAX_DEGREE

		# direction vector loss 
		direction_vector_loss = 0

		for i in range(MAX_DEGREE):
			vector_output = tf.concat(imagegraph_outputs[2 + i*4 + 2 : 2 + i*4 + 4], axis=3)
			vector_target = tf.concat(imagegraph_target_vectors[i*2:i*2+2], axis=3)

			#direction_vector_loss += tf.reduce_mean(tf.square(vector_output - vector_target))
			
			# only at key points! 
			direction_vector_loss += tf.reduce_mean(tf.multiply((soft_mask), tf.square(vector_output - vector_target)))

		direction_vector_loss /= MAX_DEGREE 


		if self.joint_with_seg:

			seg_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(self.input_seg_gt_target, tf.concat([imagegraph_outputs[2+MAX_DEGREE*4], imagegraph_outputs[2+MAX_DEGREE*4+1]], axis=3)))

			return keypoint_prob_loss, direction_prob_loss*10.0, direction_vector_loss * 1000.0 , seg_loss * 0.1
		
		else:

			return keypoint_prob_loss, direction_prob_loss* 10.0, direction_vector_loss * 1000.0, keypoint_prob_loss-keypoint_prob_loss


	def Merge(self, imagegraph_target_prob, imagegraph_target_vector):
		imagegraph_target_probs = self.unstack(imagegraph_target_prob, axis = 3)
		imagegraph_target_vectors = self.unstack(imagegraph_target_vector, axis = 3)

		new_list = []

		new_list += imagegraph_target_probs[0:2]

		for i in range(MAX_DEGREE):
			new_list += imagegraph_target_probs[2+i*2:2+i*2+2]
			new_list += imagegraph_target_vectors[i*2:i*2+2]
		
		return tf.concat(new_list, axis=3)

	
	def SoftmaxOutput(self, imagegraph_output):
		imagegraph_outputs = self.unstack(imagegraph_output, axis = 3)

		new_outputs = []

		new_outputs.append(tf.nn.sigmoid(imagegraph_outputs[0]-imagegraph_outputs[1]))
		new_outputs.append(1.0 - new_outputs[-1])
			

		#new_outputs.append(tf.nn.softmax(tf.concat(imagegraph_outputs[0:2], axis=3)))

		for i in range(MAX_DEGREE):
			#new_outputs.append(tf.nn.softmax(tf.concat(imagegraph_outputs[2+i*4:2+i*4+2], axis=3)))

			new_outputs.append(tf.nn.sigmoid(imagegraph_outputs[2+i*4]-imagegraph_outputs[2+i*4+1]))
			new_outputs.append(1.0 - new_outputs[-1])
				
			new_outputs.append(tf.concat(imagegraph_outputs[2+i*4+2:2+i*4+4], axis=3))


		if self.joint_with_seg:
			new_outputs.append(tf.nn.sigmoid(imagegraph_outputs[2+4*MAX_DEGREE]-imagegraph_outputs[2+4*MAX_DEGREE+1]))
			new_outputs.append(1.0 - new_outputs[-1])
			


		return tf.concat(new_outputs, axis=3, name="output")



	def BuildDeepLayerAggregationNetUNET(self, net_input, input_ch = 3, output_ch = 26, ch = 32):
		## 
		conv1, _, _ = common.create_conv_layer('cnn_l1', net_input, input_ch, ch, kx = 5, ky = 5, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = False)
		conv2, _, _ = common.create_conv_layer('cnn_l2', conv1, ch, ch*2, kx = 5, ky = 5, stride_x = 2, stride_y = 2, is_training = self.is_training, batchnorm = True)
		# 2s * 2ch

		def reduce_block(x, in_ch, out_ch, name):
			x, _, _ = common.create_conv_layer(name+"_1", x, in_ch, in_ch, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = True)
			x, _, _ = common.create_conv_layer(name+"_2", x, in_ch, out_ch, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = self.is_training, batchnorm = True)


			return x 

		def aggregate_block(x1, x2, in_ch1, in_ch2, out_ch, name, batchnorm=True):
			x2, _, _ = common.create_conv_layer(name+"_1", x2, in_ch2, in_ch2, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = self.is_training, batchnorm = batchnorm, deconv=True)
			
			x = tf.concat([x1,x2], axis=3) # in_ch1 + in_ch2

			x, _, _ = common.create_conv_layer(name+"_2", x, in_ch1 + in_ch2, in_ch1 + in_ch2, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = batchnorm)
			x, _, _ = common.create_conv_layer(name+"_3", x, in_ch1 + in_ch2, out_ch, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = batchnorm)

			return x 


		x_4s = reduce_block(conv2, ch*2, ch*4, "x_4s")
		x_8s = reduce_block(x_4s, ch*4, ch*8, "x_8s")
		x_16s = reduce_block(x_8s, ch*8, ch*16, "x_16s")
		x_32s = reduce_block(x_16s, ch*16, ch*32, "x_32s")

		
		a1_16s = aggregate_block(x_16s, x_32s, ch*16, ch*32, ch*32, "a1_16s")		
		a2_8s = aggregate_block(x_8s, a1_16s, ch*8, ch*32, ch*16, "a2_8s")
		a3_4s = aggregate_block(x_4s, a2_8s, ch*4, ch*16, ch*8, "a3_4s")
		a4_2s = aggregate_block(conv2, a3_4s, ch*2, ch*8, ch*8, "a4_2s") # 2s 8ch 


		a5_2s, _, _ = common.create_conv_layer('a5_2s', a4_2s, ch*8, ch*4, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = True)
		
		a_out = aggregate_block(conv1, a5_2s, ch, ch*4, ch*4, "a_out", batchnorm=False)
		
		a_out, _, _ = common.create_conv_layer('out', a_out, ch*4, output_ch, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = False, activation = "linear")
		

		return a_out 


	def BuildDeepLayerAggregationNetWithResnet(self, net_input, input_ch = 3, output_ch = 26, ch = 24):
		
		print("channel: ", ch)

		resnet_step = self.resnet_step

		## 
		conv1, _, _ = common.create_conv_layer('cnn_l1', net_input, input_ch, ch, kx = 5, ky = 5, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = False)
		conv2, _, _ = common.create_conv_layer('cnn_l2', conv1, ch, ch*2, kx = 5, ky = 5, stride_x = 2, stride_y = 2, is_training = self.is_training, batchnorm = True)
		# 2s * 2ch

		def reduce_block(x, in_ch, out_ch, name, resnet_step = 0):
			x, _, _ = common.create_conv_layer(name+"_1", x, in_ch, in_ch, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = True)
			x, _, _ = common.create_conv_layer(name+"_2", x, in_ch, out_ch, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = self.is_training, batchnorm = True)

			return x 

		def resnet_blocks(x, ch, name, resnet_step=0):
			if resnet_step > 0:
				for i in range(resnet_step):
					x = residual_block(x, channels=ch, is_training=self.is_training, downsample=False, scope=name+"_residual_block_decode_%d" % i)

				x = batch_norm_resnet(x, is_training=self.is_training, scope = name+"_decode_0_batch_norm") # mark
				
				x = tf.nn.relu(x)
			return x 


		def aggregate_block(x1, x2, in_ch1, in_ch2, out_ch, name, batchnorm=True):
			x2, _, _ = common.create_conv_layer(name+"_1", x2, in_ch2, in_ch2, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = self.is_training, batchnorm = batchnorm, deconv=True)
			
			x = tf.concat([x1,x2], axis=3) # in_ch1 + in_ch2

			x, _, _ = common.create_conv_layer(name+"_2", x, in_ch1 + in_ch2, in_ch1 + in_ch2, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = batchnorm)
			x, _, _ = common.create_conv_layer(name+"_3", x, in_ch1 + in_ch2, out_ch, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = batchnorm)

			return x 


		x_4s = reduce_block(conv2, ch*2, ch*4, "x_4s")
		x_4s = resnet_blocks(x_4s, ch*4, "x_4s", int(resnet_step/8))
		
		x_8s = reduce_block(x_4s, ch*4, ch*8, "x_8s")
		x_8s = resnet_blocks(x_8s, ch*8, "x_8s", int(resnet_step/4))
		
		x_16s = reduce_block(x_8s, ch*8, ch*16, "x_16s")
		x_16s = resnet_blocks(x_16s, ch*16, "x_16s", int(resnet_step/2))
		
		x_32s = reduce_block(x_16s, ch*16, ch*32, "x_32s")
		x_32s = resnet_blocks(x_32s, ch*32, "x_32s",resnet_step = resnet_step)  # 8


		a1_2s = aggregate_block(conv2, x_4s, ch*2, ch*4, ch*4, "a1_2s")
		a1_4s = aggregate_block(x_4s, x_8s, ch*4, ch*8, ch*8, "a1_4s")
		a1_8s = aggregate_block(x_8s, x_16s, ch*8, ch*16, ch*16, "a1_8s")
		a1_16s = aggregate_block(x_16s, x_32s, ch*16, ch*32, ch*32, "a1_16s")
		a1_16s = resnet_blocks(a1_16s, ch*32, "a1_16s",resnet_step = int(resnet_step/2)) # 4 

		a2_2s = aggregate_block(a1_2s, a1_4s, ch*4, ch*8, ch*4, "a2_2s")
		a2_4s = aggregate_block(a1_4s, a1_8s, ch*8, ch*16, ch*8, "a2_4s")
		a2_8s = aggregate_block(a1_8s, a1_16s, ch*16, ch*32, ch*16, "a2_8s")
		a2_8s = resnet_blocks(a2_8s, ch*16, "a2_8s",resnet_step = int(resnet_step/4)) # 2

		a3_2s = aggregate_block(a2_2s, a2_4s, ch*4, ch*8, ch*4, "a3_2s")
		a3_4s = aggregate_block(a2_4s, a2_8s, ch*8, ch*16, ch*8, "a3_4s")
		a3_4s = resnet_blocks(a3_4s, ch*8, "a3_4s",resnet_step = int(resnet_step/8)) # 1


		a4_2s = aggregate_block(a3_2s, a3_4s, ch*4, ch*8, ch*8, "a4_2s") # 2s 8ch 

		a5_2s, _, _ = common.create_conv_layer('a5_2s', a4_2s, ch*8, ch*4, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = True)
		
		a_out = aggregate_block(conv1, a5_2s, ch, ch*4, ch*4, "a_out", batchnorm=False)
		
		a_out, _, _ = common.create_conv_layer('out', a_out, ch*4, output_ch, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = False, activation = "linear")
		

		return a_out 

	
	def Train(self, inputdata, target_prob, target_vector, input_seg_gt, lr):
		feed_dict = {
			self.input_sat : inputdata,
			self.target_prob : target_prob,
			self.target_vector : target_vector,
			self.input_seg_gt : input_seg_gt, 
			self.lr : lr,
			self.is_training : True
		}

		ops = [self.loss, self.l2loss_grad_max, self.prob_loss, self.direction_vector_loss, self.seg_loss, self.train_op]
		
		return self.sess.run(ops, feed_dict=feed_dict)


	def TrainSegmentation(self, inputdata, target_prob, target_vector, input_seg_gt, lr):
		feed_dict = {
			self.input_sat : inputdata,
			self.target_prob : target_prob,
			self.target_vector : target_vector,
			self.input_seg_gt : input_seg_gt, 
			self.lr : lr,
			self.is_training : True
		}

		return self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
	


	def Evaluate(self, inputdata, target_prob, target_vector, input_seg_gt):
		feed_dict = {
			self.input_sat : inputdata,
			self.target_prob : target_prob,
			self.target_vector : target_vector,
			self.input_seg_gt : input_seg_gt, 
			self.is_training : False
		}

		ops = [self.loss, self.output]
		
		return self.sess.run(ops, feed_dict=feed_dict)

	def EvaluateSegmentation(self, inputdata, target_prob, target_vector, input_seg_gt):
		feed_dict = {
			self.input_sat : inputdata,
			self.target_prob : target_prob,
			self.target_vector : target_vector,
			self.input_seg_gt : input_seg_gt, 
			self.is_training : False
		}
		
		return self.sess.run([self.loss, self.output], feed_dict=feed_dict)


	def saveModel(self, path):
		self.saver.save(self.sess, path)

	def restoreModel(self, path):
		self.saver.restore(self.sess, path)

	def addLog(self, test_loss, train_loss, l2_grad):
		feed_dict = {
			self.test_loss : test_loss,
			self.train_loss : train_loss,
			self.l2_grad : l2_grad,
		}
		return self.sess.run(self.merged_summary, feed_dict=feed_dict)

