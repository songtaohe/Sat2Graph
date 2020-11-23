# usage
# python infer.py input_image.png output_prefix
#
#
import sys 
USE_CPU = False

import os

model_name = "/data/songtao/Sat2GraphLib/globalmodel20200810_dla_mapbox_highway_new_352_8__channel24/model1000000"
model_fp_name = model_name.replace("/", "_")
if not os.path.isfile(model_fp_name):
	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
	USE_CPU = True


import json 
import os 
import os.path  
import scipy.ndimage 
import scipy.misc 
import math 
import cv2
import numpy as np 
import tensorflow as tf 
from time import time, sleep 
import sys 
from PIL import Image 
from subprocess import Popen 
from model import Sat2GraphModel
from decoder import DecodeAndVis 
from douglasPeucker import simpilfyGraph 
import json
import pickle 
import tifffile


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
#gpu_options = tf.GPUOptions()
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

model = Sat2GraphModel(sess, image_size=352, resnet_step = 8, batchsize = 1, channel = 24, mode = "test")

retry_counter = 0
model.restoreModel(model_name)


while True:
	print("loading counter", retry_counter)
	retry_counter += 1

	
	params = model.get_params()

	weights = {}
	fingerprint = ""

	hasBadWeights = False

	for i in range(len(model.variables_names)):
		weights[model.variables_names[i]] = params[i]
		fingerprint += str(i) + " "+ str(model.variables_names[i]) + " " + str(np.amax(params[i]))+" "+ str(np.amin(params[i])) + "\n"
		if np.isnan(params[i]).any():
			print(i, model.variables_names[i], np.shape(params[i]), np.amax(params[i]), np.amin(params[i]))
			print("NAN Detected!!!!!!!!!!!!")
			print("")
			hasBadWeights = True
			

		elif np.isinf(params[i]).any():
			print(i, model.variables_names[i], np.shape(params[i]), np.amax(params[i]), np.amin(params[i]))
			print("INF Detected!!!!!!!!!!!!")
			print("")
			hasBadWeights = True

		if np.amax(params[i]) > 10**5 or np.amin(params[i]) < -10**5:
			print(i, model.variables_names[i], np.shape(params[i]), np.amax(params[i]), np.amin(params[i]))
			print("Very Large/Small Numbers Detected!!!!!!!!!!!!")
			print("")
			hasBadWeights = True
			
	model_fp_name = model_name.replace("/", "_")

	if os.path.isfile(model_fp_name) and USE_CPU == False:
		with open("weightsfp.txt","w") as fout:
			fout.write(fingerprint)
		print("model fingerprint diff ")
		cpu_weights = pickle.load(open("weights_"+model_fp_name+".p"))

		diff = 0
		for k, v1 in cpu_weights.iteritems():
			v2 = weights[k]

			
			if abs(np.mean((v1-v2)))> 0.0001:
				diff += 1
				print("diff", k, abs(np.mean((v1-v2))))
	
		pickle.dump(weights, open("weights.p","w"))
		
		Popen("diff "+model_fp_name + " "+ "weightsfp.txt", shell=True).wait()


		retry = 0
		while True:
			print("retry", retry)

			# do this params by params
			failed_cc = 0 
			batch_size = 4
			for i in range(0, len(model.variables_names), batch_size):
				print("configuring parameters", i)

				idxs = [x for x in range(i, min(i+batch_size, len(model.variables_names)))]

				succ = False 

				while not succ:
					model.set_batch_param(cpu_weights, idxs)
					check_param = model.get_batch_param(idxs)
					
					diff_cc = 0
					for j in idxs: 
						diff_cc +=  len(np.where((check_param[j-i] - cpu_weights[model.variables_names[j]])!=0)[0])
					
					if diff_cc > 0:
						succ = False
						failed_cc += 1
						sleep(0.1)
						#if failed_cc % 100 == 0:
						print(i, diff_cc, failed_cc)
					else:
						succ = True 

			print("failed_cc", failed_cc)

			# don't check, just break!
			break
			# model.set_params(cpu_weights)
			# check_params = model.get_params()

			# wrong = False
			# for i in range(len(check_params)):
			# 	if np.mean(check_params[i] - cpu_weights[model.variables_names[i]]) != 0:
			# 		bugs = np.where((check_params[i] - cpu_weights[model.variables_names[i]]) !=0 )
			# 		print(i, model.variables_names[i], np.shape(cpu_weights[model.variables_names[i]]), np.shape(check_params[i]), np.mean(check_params[i] - cpu_weights[model.variables_names[i]]), len(bugs[0]) )
			# 		wrong = True 
			# if wrong:
			# 	retry += 1
			# 	if retry < 10:
			# 		continue 

			# 	print("something wrong...")
			# 	sess.close()
			# 	exit()
			# else:
			# 	print("Passed the weights test!")
			# 	break

		 
	
		#hasBadWeights = False

		print("Use weights from CPU loader")
		print("Restoring models using GPU has some wired bugs, so we should always load weights on CPU first. [I guess this is a bug in tensorflow 1.13.1]")
		
		break

		if diff > 0:
			continue

	elif hasBadWeights == False:
		with open(model_fp_name,"w") as fout:
			fout.write(fingerprint)

		pickle.dump(weights, open("weights_"+model_fp_name+".p","w"))

	if hasBadWeights:
		continue 
		sess.close()
		exit()

	if USE_CPU :
		sess.close()
		exit()

	break 




# L18-13904E-7800N_1m.png
# L18-13906E-7800N_1m.png
# L18-13907E-7800N_1m.png
# L18-13905E-7790N_1m.png
# L18-13905E-7791N_1m.png

gt_prob_placeholder = np.zeros((1,352,352,14))
gt_vector_placeholder = np.zeros((1,352,352,12))
gt_seg_placeholder = np.zeros((1,352,352,1))



input_file = sys.argv[1]
#if len(sys.argv) <= 2:
output_file = sys.argv[1].replace(".png", "")
#else:
#	output_file = sys.argv[2]

v_thr = 0.05
e_thr = 0.05
snap_dist = 15
snap_w = 50



for input_file in sys.argv[1:]:
	print(input_file)
	if "png" in input_file:
		output_file = input_file.replace(".png", "")
	else:
		output_file = input_file.replace(".tif", "")

	# run the model 
	if ".tif" in input_file:
		sat_img = tifffile.imread(input_file)[:,:,0:3]
	else:
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
	graph = DecodeAndVis(output, output_file, thr=v_thr, edge_thr = e_thr, angledistance_weight=snap_w, snap_dist = snap_dist, snap=True, imagesize = 2048, spurs_thr = 200, isolated_thr= 500)

	print("Decode time:", time() - t0)
	t0 = time()

	#graph = simpilfyGraph(graph)

	print("Graph simpilfy time:", time() - t0)
	t0 = time()


	# vis 
	if ".tif" in input_file:
		sat_img = tifffile.imread(input_file)[:,:,0:3]
	else:
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

sess.close()












