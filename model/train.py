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


# This file supports training and testing Sat2Graph models on both the 20cities dataset and Spacenet dataset.  
# 
# -> Train Sat2Graph model on the 20cities dataset
# time python train.py -model_save tmp -instance_id test -image_size 352
# 
# -> Train Sat2Graph model on the 20cities dataset from the pre-trained model
# time python train.py -model_save tmp -instance_id test -image_size 352 -model_recover ../data/20citiesModel/model
#
# -> Test Sat2Graph model on the 20cities dataset
# time python train.py -model_save tmp -instance_id test -image_size 352 -model_recover ../data/20citiesModel/model -mode test
# 

parser = argparse.ArgumentParser()

parser.add_argument('-model_save', action='store', dest='model_save', type=str,
                    help='model save folder ', required =True)

parser.add_argument('-instance_id', action='store', dest='instance_id', type=str,
                    help='instance_id ', required =True)

parser.add_argument('-model_recover', action='store', dest='model_recover', type=str,
                    help='model recover ', required =False, default=None)


parser.add_argument('-image_size', action='store', dest='image_size', type=int,
                    help='instance_id ', required =False, default=256)

parser.add_argument('-lr', action='store', dest='lr', type=float,
                    help='learning rate', required =False, default=0.001)

parser.add_argument('-lr_decay', action='store', dest='lr_decay', type=float,
                    help='learning rate decay', required =False, default=0.5)

parser.add_argument('-lr_decay_step', action='store', dest='lr_decay_step', type=int,
                    help='learning rate decay step', required =False, default=50000)

parser.add_argument('-init_step', action='store', dest='init_step', type=int,
                    help='initial step size ', required =False, default=0)


# parser.add_argument('-model_name', action='store', dest='model_name', type=str,
#                     help='instance_id ', required =False, default="UNET_resnet")

parser.add_argument('-resnet_step', action='store', dest='resnet_step', type=int,
                    help='instance_id ', required =False, default=8)

# parser.add_argument('-train_segmentation', action='store', dest='train_segmentation', type=bool,
#                     help='train_segmentation', required =False, default=False)

parser.add_argument('-spacenet', action='store', dest='spacenet', type=str,
                    help='spacenet folder', required =False, default="")

parser.add_argument('-channel', action='store', dest='channel', type=int,
                    help='channel', required =False, default=12)

parser.add_argument('-mode', action='store', dest='mode', type=str,
                    help='mode [train][test][validate]', required =False, default="train")

args = parser.parse_args()

print(args)

log_folder = "alllogs"

from datetime import datetime
instance_id = args.instance_id + "_" + str(args.image_size) + "_" + str(args.resnet_step) + "_" + "_channel%d" % args.channel
run = "run-"+datetime.today().strftime('%Y-%m-%d-%H-%M-%S')+"-"+instance_id


osmdataset = "../data/20cities/"
spacenetdataset = "../data/spacenet/"

image_size = args.image_size

batch_size = 4 # 256 256  

# if args.image_size == 384:
# 	batch_size = 4

if args.mode != "train":
	batch_size = 1

validation_folder = "validation_" + instance_id 
Popen("mkdir -p "+validation_folder, shell=True).wait()

model_save_folder = args.model_save + instance_id + "/"

max_degree = 6

Popen("mkdir -p %s" % model_save_folder, shell=True).wait()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
	model = Sat2GraphModel(sess, image_size=image_size, image_ch=5, resnet_step = args.resnet_step, batchsize = batch_size, channel = args.channel, mode = args.mode)
	
	if args.model_recover is not None:
		print("model recover", args.model_recover)
		try:
			model.restoreModel(args.model_recover)
		except:
			from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
			print_tensors_in_checkpoint_file(file_name=args.model_recover, tensor_name='', all_tensors=True)
			exit()
	writer = tf.summary.FileWriter(log_folder+"/"+run, sess.graph)

	if args.spacenet == "":
		if args.mode == "train":

			datafiles = json.load(open("train_prep_RE_18_20_CHN_KZN_250.json"))['data']
			basefolder = "/data/songtao/harvardDataset5m/"

			trainfiles = []
			validfiles = []
			for item in datafiles:
				if item[1].endswith("4.tif") or item[1].endswith("3.tif") or item[1].endswith("2.tif") or item[1].endswith("1.tif"):
					if item[-1] == 'train':
						trainfiles.append(basefolder+item[1].replace(".tif",""))
					elif item[-1] == 'valid':
						validfiles.append(basefolder+item[1].replace(".tif",""))

			print("train size", len(trainfiles))
			print("valid size", len(validfiles))

			dataloader_train = Sat2GraphDataLoader(osmdataset, trainfiles, imgsize = image_size, preload_tiles = 400, testing = False, random_mask=True)
			dataloader_train.preload()

			dataloader_test = Sat2GraphDataLoader(osmdataset, validfiles, imgsize = image_size, preload_tiles = 200, random_mask=False, testing=True)
			dataloader_test.preload()

	else:
		# CLEANING UP ... 
		pass

		# print("train with spacenet", args.spacenet)

		# datasplit = json.load(open(args.spacenet + "/dataset.json", "r"))
		# indrange_train = datasplit["train"]
		# indrange_test = datasplit["test"]
		# indrange_validation = datasplit["validation"]

		# if args.mode == "train":

		# 	dataloader_train = HGGDataLoaderSpacenet(args.spacenet, indrange_train, imgsize = image_size, preload_tiles = 100, loadseg = train_seg, load_rgb_image = use_SAT,random_mask=False if args.noda else True, dataset_image_size=400)
		# 	dataloader_train.preload(num=1024)

		# 	dataloader_test = HGGDataLoaderSpacenet(args.spacenet, indrange_validation, imgsize = image_size, preload_tiles = 100, loadseg = train_seg, load_rgb_image = use_SAT, random_mask=False,dataset_image_size=400)
		# 	dataloader_test.preload(num=128)

		# else:
		# 	dataloader = HGGDataLoaderSpacenet(args.spacenet, [], imgsize = image_size, preload_tiles = 1, loadseg = train_seg, load_rgb_image = use_SAT,random_mask=False, testing=True, dataset_image_size=400)

		# 	t0 = time()
		# 	cc = 0

		# 	Popen("mkdir -p /data/songtao/RoadMaster/rawoutputs_%s" % (args.instance_id), shell=True).wait()


		# 	for prefix in indrange_test:
				
		# 		dataloader.preload(num=1, FixIndrange=[prefix])

		# 		input_sat, gt_prob, gt_vector, gt_seg, gan_noisy, gt_angle = dataloader.getBatch(1, get_angle = True)

		# 		if args.train_segmentation:
		# 			alloutputs = model.EvaluateSegmentation(input_sat, gt_prob, gt_vector, gt_seg, input_angle_gt = gt_angle)
		# 			output = alloutputs[1][0,:,:,:]
								
		# 			output_img = (output[:,:,0] * 255.0).reshape((352, 352)).astype(np.uint8)
		# 			Image.fromarray(output_img).save("/data/songtao/RoadMaster/rawoutputs_%s/%s_seg.png" % (args.instance_id, prefix))

		# 		else:
		# 			alloutputs  = model.Evaluate(input_sat, gt_prob, gt_vector, gt_seg)

		# 			_output = alloutputs[1]
		# 			output = _output[0,:,:,:]

		# 			ethr = thr = "0.01"

		# 			np.save("/data/songtao/RoadMaster/rawoutputs_%s/%s_output_raw" % (args.instance_id, prefix), output)


		# 		print(cc, len(indrange_test),prefix, time() - t0)
		# 		t0 = time()
		# 		cc += 1	


		# 	exit()


	validation_data = []

	test_size = 64

	for j in range(test_size/batch_size):
		input_sat, gt_prob, gt_vector, gt_seg, gt_class= dataloader_test.getBatch(batch_size)
		validation_data.append([np.copy(input_sat), np.copy(gt_prob), np.copy(gt_vector), np.copy(gt_seg), np.copy(gt_class)])


	step = args.init_step
	lr = args.lr
	sum_loss = 0 

	gt_imagegraph = np.zeros((batch_size, image_size, image_size, 2 + 4*max_degree))


	t_load = 0 
	t_last = time() 
	t_train = 0 

	test_loss = 0
	
	sum_prob_loss = 0
	sum_vector_loss = 0
	sum_seg_loss = 0 

	while True:
		t0 = time()
		input_sat, gt_prob, gt_vector, gt_seg, gt_class = dataloader_train.getBatch(batch_size)
		t_load += time() - t0 
		
		t0 = time()

		loss, grad_max, prob_loss, vector_loss,seg_loss, _ = model.Train(input_sat, gt_prob, gt_vector, gt_seg, gt_class, lr)

		sum_loss += loss

		sum_prob_loss += prob_loss
		sum_vector_loss += vector_loss
		sum_seg_loss += seg_loss

		if step < 1:
			if args.model_recover is not None:
				print("load model ", args.model_recover)
				model.restoreModel(args.model_recover)

		t_train += time() - t0 			

		if step % 10 == 0:
			sys.stdout.write("\rbatch:%d "%step + ">>" * ((step - (step/200)*200)/10) + "--" * (((step/200+1)*200-step)/10))
			sys.stdout.flush()

		if step > -1 and step % 200 == 0:
			sum_loss /= 200
			
			if step % 1000 == 0 or (step < 1000 and step % 200 == 0):
				test_loss = 0

				for j in range(-1,test_size/batch_size):
					if j >= 0:
						input_sat, gt_prob, gt_vector, gt_seg, gt_class = validation_data[j][0], validation_data[j][1], validation_data[j][2], validation_data[j][3], validation_data[j][4]
					if j == 0:
						test_loss = 0
						test_gan_loss = 0

					gt_imagegraph[:,:,:,0:2] = gt_prob[:,:,:,0:2]
					for k in range(max_degree):
						gt_imagegraph[:,:,:,2+k*4:2+k*4+2] = gt_prob[:,:,:,2+k*2:2+k*2+2]
						gt_imagegraph[:,:,:,2+k*4+2:2+k*4+4] = gt_vector[:,:,:,k*2:k*2+2]

					_test_loss, output = model.Evaluate(input_sat, gt_prob, gt_vector, gt_seg, gt_class)

					test_loss += _test_loss

					if step == 1000 or step % 2000 == 0 or (step < 1000 and step % 200 == 0):
						for k in range(batch_size):
							
							input_sat_img = ((input_sat[k,:,:,0:3] + 0.5) * 255.0).reshape((image_size,image_size,3)).astype(np.uint8)
							input_gt_seg = ((gt_seg[k,:,:,0]+0.5)*255.0).astype(np.uint8)

							# segmentation output (joint training)
							output_img = (output[k,:,:,26] * 255.0).reshape((image_size,image_size)).astype(np.uint8)
							Image.fromarray(output_img).save(validation_folder+"/tile%d_output_seg.png" % (j*batch_size+k))
							Image.fromarray(((gt_seg[k,:,:,0] + 0.5) * 255.0).reshape((image_size, image_size)).astype(np.uint8)).save(validation_folder+"/tile%d_gt_seg.png" % (j*batch_size+k))

							# road class output (joint training)

							output_img = np.zeros((image_size,image_size,3), dtype=np.uint8)

							output_img[:,:,2:3] = output[k,:,:,28:29] * 255 
							output_img[:,:,1:2] = output[k,:,:,29:30] * 255 + output[k,:,:,30:31] * 255 
							output_img[:,:,0:1] = output[k,:,:,31:32] * 255 + output[k,:,:,32:33] * 255 

							output_img = output_img.astype(np.uint8)

							Image.fromarray(output_img).save(validation_folder+"/tile%d_output_road_class.png" % (j*batch_size+k))
							Image.fromarray((gt_class * 60).astype(np.uint8).reshape((image_size,image_size))).save(validation_folder+"/tile%d_gt_road_class.png" % (j*batch_size+k))
							

							# keypoints 
							output_keypoints_img = (output[k,:,:,0] * 255.0).reshape((image_size,image_size)).astype(np.uint8)
							Image.fromarray(output_keypoints_img).save(validation_folder+"/tile%d_output_keypoints.png" % (j*batch_size+k))

							# input satellite
							Image.fromarray(input_sat_img).save(validation_folder+"/tile%d_input_sat.png" % (j*batch_size+k))
								
							# input gt seg 
							Image.fromarray(input_gt_seg).save(validation_folder+"/tile%d_input_gt_seg.png" % (j*batch_size+k))
							

							# todo 			
							#ImageGraphVis(output[k,:,:,0:2 + 4*max_degree].reshape((image_size, image_size, 2 + 4*max_degree )), validation_folder+"/tile%d_output_graph_0.01.png" % (j*batch_size+k), thr=0.01, imagesize = image_size)
							DecodeAndVis(output[k,:,:,0:2 + 4*max_degree].reshape((image_size, image_size, 2 + 4*max_degree )), validation_folder+"/tile%d_output_graph_0.01_snap.png" % (j*batch_size+k), thr=0.01, snap=True, imagesize = image_size)
							DecodeAndVis(gt_imagegraph[k,:,:,:].reshape((image_size, image_size, 2 + 4*max_degree )), validation_folder+"/tile%d_output_graph_gt.png" % (j*batch_size+k), thr=0.5, imagesize = image_size, use_graph_refine = False)


				test_loss /= test_size/batch_size
				
			print("")
			print("step", step, "loss", sum_loss, "test_loss", test_loss, "prob_loss", sum_prob_loss/200.0, "vector_loss", sum_vector_loss/200.0, "seg_loss", sum_seg_loss/200.0)
			
			summary = model.addLog(test_loss, sum_loss, lr)
			writer.add_summary(summary, step)

			sum_prob_loss = 0
			sum_vector_loss = 0
			sum_seg_loss = 0
			sum_loss = 0 


		if step > 0 and step % 400 == 0:
			dataloader_train.preload(num=1024)



		if step > 0 and step %2000 == 0:
			
			print(time() - t_last, t_load, t_train)
			t_last = time() 
			t_load = 0 
			t_train = 0 

		if step > 0 and (step % 10000 == 0):
			model.saveModel(model_save_folder + "model%d" % step)

		if step > 0 and step % args.lr_decay_step == 0:
			lr = lr * args.lr_decay

		step += 1
		if step == 500000+2:
			break 
