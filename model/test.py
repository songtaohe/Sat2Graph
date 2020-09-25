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

input_img = np.zeros((5000,5000,5))


for item in datafiles:
	if prefix in item[1]:
		print(item)

		items = item[1].split("Analytic_")
		xy = items[-1].split("_")[0].split("-")
		x = int(xy[0])
		y = int(xy[1])

		file = basefolder + "/" + item[1]

		img = tifffile.imread(filename+".tif")
		sat_img = img[:,:,0:5].astype(np.float)/(16384) - 0.5 

		input_img[x:x+250, y:y+250, :] = sat_img


Image.fromarray(((input_img[:,:,0:3]+0.5) * 255.0).astype(np.uint8)).save(prefix+"rgb.png")





# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
# with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
# 	model = Sat2GraphModel(sess, image_size=image_size, image_ch=5, resnet_step = args.resnet_step, batchsize = batch_size, channel = args.channel, mode = args.mode)
# 	model.restoreModel(sys.argv[1])

