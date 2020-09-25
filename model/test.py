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


datafiles = json.load(open("train_prep_RE_18_20_CHN_KZN_250.json"))['data']
basefolder = "/data/songtao/harvardDataset5m/"

prefix = "4559325_2019-07-03_RE4_3A_"

input_img = np.zeros((5120,5120,5))
input_mask = np.zeros((5120, 5120))

for item in datafiles:
	if prefix in item[1]:
		#print(item)

		items = item[1].split("Analytic_")
		xy = items[-1].split("_")[0].split("-")
		x = int(xy[1]) + 60
		y = int(xy[0]) + 60 

		file = basefolder + "/" + item[1]

		img = tifffile.imread(file)
		sat_img = img[:,:,0:5].astype(np.float)/(16384) - 0.5 

		input_img[x:x+250, y:y+250, :] = sat_img
		input_mask[x:x+250, y:y+250] = 1.0 


Image.fromarray(((input_img[:,:,0:3]+0.5) * 255.0).astype(np.uint8)).save("output/"+prefix+"rgb.png")


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
	model = Sat2GraphModel(sess, image_size=256, image_ch=5, resnet_step = 8, batchsize = 1, channel = 12, mode = "test")
	model.restoreModel(sys.argv[1])

	output = np.zeros((5120,5120,26))
	weights = np.zeros((5120,5120))+0.00001

	gt_prob = np.zeros_like((1,256,256,14))
	gt_vector = np.zeros_like((1,256,256,12))
	gt_seg = np.zeros_like((1,256,256,1))

	for x in range(0, 5120-256, 128):
		print(x)
		for y in range(0,5120-256,128):
			if np.sum(input_mask[x:x+256,y:y+256]) > 10.0:
				_, output_ = model.Evaluate(input_img[x:x+256,y:y+256,:].reshape((1,256,256,5)), gt_prob, gt_vector, gt_seg)

				output[x:x+256,y:y+256,:] += output_[0,:,:,:]
				weights[x:x+256,y:y+256,:] += 1.0 

	output = np.divide(output, weights)

	DecodeAndVis(output, "output/"+prefix+"output", thr=0.05, snap=True, imagesize = 5120)

							







