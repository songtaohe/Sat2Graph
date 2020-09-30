from model import Sat2GraphModel
#from dataloader import Sat2GraphDataLoader as Sat2GraphDataLoaderOSM
from dataloader_lowres import Sat2GraphDataLoader
#from dataloader_spacenet import Sat2GraphDataLoader as Sat2GraphDataLoaderSpacenet
from subprocess import Popen 
import numpy as np 
from time import time 
import tensorflow as tf 
from decoder import DecodeAndVis 
from PIL import Image 
import sys   
import random 
import json 
import argparse
import tifffile
import cv2 
import pickle 
import os 


datafiles = json.load(open("train_prep_RE_18_20_CHN_KZN_250.json"))['data']
basefolder = "/data/songtao/harvardDataset5m/"
basefolderTesting = "/data/songtao/harvardDataset5mTesting/"
testingfiles = os.listdir(basefolderTesting)
outputFolder = "/data/songtao/harvardDataset5mTestingResults/"


testfiles = []

for item in datafiles:
	if item[-1] == 'test':
		filepath = basefolder+item[1]
		testfiles.append(item[1].replace(".tif",""))
prefixs = []
for name in testfiles:
	p = name.split("Analytic_")[0]
	if p not in prefixs:
		prefixs.append(p)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
	model = Sat2GraphModel(sess, image_size=256, image_ch=5, resnet_step = 8, batchsize = 1, channel = 12, mode = "test")
	model.restoreModel(sys.argv[1])

	#prefixs = ["4459815_2019-10-13_RE1_3A_", "4659204_2019-07-03_RE4_3A_", "4459815_2018-07-07_RE2_3A_", "4559216_2018-07-20_RE2_3A_", "4559325_2019-08-24_RE3_3A_", "4559325_2019-09-28_RE1_3A_", "4559325_2019-07-03_RE4_3A_"]
	#prefixs = ["4459815_2019-10-13_RE1_3A_"]
	cc = 0
	t0 = time() 
	for prefix in prefixs:
		print(cc, len(prefixs), time() - t0)
		cc += 1
		t0 = time() 

		input_img = np.zeros((5120,5120,5))
		input_mask = np.zeros((5120, 5120))

		for item in testingfiles:
			if prefix in item:
				#print(item)

				items = item.split("Analytic_")
				xy = items[-1].split("_")[0].split("-")
				x = int(xy[1]) + 60
				y = int(xy[0]) + 60 

				file = basefolderTesting + "/" + item

				img = tifffile.imread(file)
				sat_img = img[:,:,0:5].astype(np.float)/(16384) - 0.5 

				input_img[x:x+250, y:y+250, :] = sat_img
				input_mask[x:x+250, y:y+250] = 1.0 

		input_img = np.clip(input_img, -0.5, 0.5)
		Image.fromarray(((input_img[:,:,0:3]+0.5) * 255.0).astype(np.uint8)).save("output/"+prefix+"rgb.png")

		#continue
	
		output = np.zeros((5120,5120,26))
		weights = np.zeros((5120,5120,26))+0.00001
		localweights = np.zeros((256,256,26)) + 0.00001 
		localweights[32:224,32:224,:] = 0.5
		localweights[64:192,64:192,:] = 1.0 


		gt_prob = np.zeros((1,256,256,14))
		gt_vector = np.zeros((1,256,256,12))
		gt_seg = np.zeros((1,256,256,1))

		for x in range(0, 5120-192, 64):
			if x % 512 == 0:
				print(x)
			for y in range(0,5120-192,64):
				if np.sum(input_mask[x:x+256,y:y+256]) > 10.0:
					_, output_ = model.Evaluate(input_img[x:x+256,y:y+256,:].reshape((1,256,256,5)), gt_prob, gt_vector, gt_seg)

					output[x:x+256,y:y+256,:] += output_[0,:,:,0:26] * localweights
					weights[x:x+256,y:y+256,:] += localweights

		output = np.divide(output, weights)

		DecodeAndVis(output, outputFolder+prefix+"output", thr=0.01, edge_thr=0.03, snap=True, imagesize = 5120)

		img = cv2.imread(outputFolder+prefix+"rgb.png")
		graph = pickle.load(open(outputFolder+prefix+"output_graph.p"))

		for node, nei in graph.iteritems():
			y1,x1 = int(node[0]),int(node[1])

			for nn in nei:
				y2,x2 = int(nn[0]),int(nn[1])

				cv2.line(img, (x1,y1),(x2,y2),(0,255,255),2)

		cv2.imwrite(outputFolder+prefix+"graph_vis.png", img)









