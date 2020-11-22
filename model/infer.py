# usage
# python infer.py input_image.png output_prefix
#
#

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json 
import os 
import os.path  
import scipy.ndimage 
import scipy.misc 
import math 
import cv2
import numpy as np 
import tensorflow as tf 
from time import time 
import sys 
from PIL import Image 
 
from model import Sat2GraphModel
from decoder import DecodeAndVis 
from douglasPeucker import simpilfyGraph 
import json
import pickle 


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

model = Sat2GraphModel(sess, image_size=352, resnet_step = 8, batchsize = 1, channel = 12, mode = "test")
model.restoreModel("/data/songtao/qcriStartup/Sat2Graph/model/modelv1run2_352_8__channel12/model110000")

params = model.get_params()

weights = {}

for i in range(len(model.variables_names)):
	weights[model.variables_names[i]] = params[i]
	print(i, model.variables_names[i], np.shape(params[i]), np.amax(params[i]), np.amin(params[i]))
	if np.isnan(params[i]).any():
		print("NAN Detected!!!!!!!!!!!!")
		print("")
	if np.isinf(params[i]).any():
		print("INF Detected!!!!!!!!!!!!")
		print("")
		
pickle.dump(weights, open("weights.p","w"))





# L18-13904E-7800N_1m.png
# L18-13906E-7800N_1m.png
# L18-13907E-7800N_1m.png
# L18-13905E-7790N_1m.png
# L18-13905E-7791N_1m.png

gt_prob_placeholder = np.zeros((1,352,352,14))
gt_vector_placeholder = np.zeros((1,352,352,12))
gt_seg_placeholder = np.zeros((1,352,352,1))



input_file = sys.argv[1]
if len(sys.argv) <= 2:
	output_file = sys.argv[1].replace(".png", "")
else:
	output_file = sys.argv[2]

v_thr = 0.01
e_thr = 0.01
snap_dist = 15
snap_w = 50


# run the model 

sat_img = scipy.ndimage.imread(input_file)
dim = np.shape(sat_img)
s = (dim[0]-2048) // 2
sat_img = sat_img[s:s+2048, s:s+2048,:]

#sat_img = scipy.misc.imresize(sat_img, (2048,2048)).astype(np.float)

max_v = 255
sat_img = (sat_img.astype(np.float)/ max_v - 0.5) * 0.9 
sat_img = sat_img.reshape((1,2048,2048,3))

image_size = 352 

weights = np.ones((image_size,image_size, 2+4*6 + 2)) * 0.001 
weights[32:image_size-32,32:image_size-32, :] = 0.5 
weights[56:image_size-56,56:image_size-56, :] = 1.0 
weights[88:image_size-88,88:image_size-88, :] = 1.5 

mask = np.zeros((2048+64, 2048+64, 2+4*6 + 2))
output = np.zeros((2048+64, 2048+64, 2+4*6 + 2))
sat_img = np.pad(sat_img, ((0,0),(32,32),(32,32),(0,0)), 'constant')
				

t0 = time()
for x in range(0,352*6-176-88,176/2):	
	print(x)
	for y in range(0,352*6-176-88,176/2):

		alloutputs  = model.Evaluate(sat_img[:,x:x+image_size, y:y+image_size,:], gt_prob_placeholder, gt_vector_placeholder, gt_seg_placeholder)
		_output = alloutputs[1]

		mask[x:x+image_size, y:y+image_size, :] += weights
		output[x:x+image_size, y:y+image_size,:] += np.multiply(_output[0,:,:,:], weights)


print("GPU time:", time() - t0)
t0 = time()

output = np.divide(output, mask)
output = output[32:2048+32,32:2048+32,:]
# alloutputs  = model.Evaluate(sat_img, gt_prob_placeholder, gt_vector_placeholder, gt_seg_placeholder)
# output = alloutputs[1][0,:,:,:]

#graph = DecodeAndVis(output, output_file, thr=0.01, edge_thr = 0.1, angledistance_weight=50, snap=True, imagesize = 704)
graph = DecodeAndVis(output, output_file, thr=v_thr, edge_thr = e_thr, angledistance_weight=snap_w, snap_dist = snap_dist, snap=True, imagesize = 2048)

print("Decode time:", time() - t0)
t0 = time()

graph = simpilfyGraph(graph)

print("Graph simpilfy time:", time() - t0)
t0 = time()


# vis 
sat_img = scipy.ndimage.imread(input_file)
dim = np.shape(sat_img)
s = (dim[0]-2048) // 2
sat_img = sat_img[s:s+2048, s:s+2048,:]

sat_img = scipy.misc.imresize(sat_img, (2048,2048))

for k,v in graph.iteritems():
	n1 = k 
	for n2 in v:
		cv2.line(sat_img, (n1[1], n1[0]), (n2[1], n2[0]), (255,255,0),2)

Image.fromarray(sat_img).save(output_file+"_vis.png")














