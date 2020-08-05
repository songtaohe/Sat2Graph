from model import Sat2GraphModel
from dataloader import Sat2GraphDataLoader as Sat2GraphDataLoaderOSM
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



# -> Train Sat2Graph model on the global dataset (938 2k by 2k tiles)
# time python train.py -model_save tmp -instance_id test -image_size 352

# python train.py -model_save /data/songtao/Sat2GraphLib/globalmodel -instance_id 20200804v1 -image_size 352 
# python train.py -model_save /data/songtao/Sat2GraphLib/globalmodel -instance_id 20200804v2UNET -image_size 352 -channel 32 -model_name unet

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
                    help='learning rate decay step', required =False, default=100000)

parser.add_argument('-init_step', action='store', dest='init_step', type=int,
                    help='initial step size ', required =False, default=0)


parser.add_argument('-model_name', action='store', dest='model_name', type=str,
                    help='instance_id ', required =False, default="DLA")

parser.add_argument('-resnet_step', action='store', dest='resnet_step', type=int,
                    help='instance_id ', required =False, default=8)

# parser.add_argument('-train_segmentation', action='store', dest='train_segmentation', type=bool,
#                     help='train_segmentation', required =False, default=False)

parser.add_argument('-spacenet', action='store', dest='spacenet', type=str,
                    help='spacenet folder', required =False, default="")

parser.add_argument('-channel', action='store', dest='channel', type=int,
                    help='channel', required =False, default=24)

parser.add_argument('-mode', action='store', dest='mode', type=str,
                    help='mode [train][test][validate]', required =False, default="train")

args = parser.parse_args()

print(args)

log_folder = "alllogs"

from datetime import datetime
instance_id = args.instance_id + "_" + str(args.image_size) + "_" + str(args.resnet_step) + "_" + "_channel%d" % args.channel
run = "run-"+datetime.today().strftime('%Y-%m-%d-%H-%M-%S')+"-"+instance_id


osmdataset = "../data/20cities/"
osmdataset = "/data/songtao/Sat2GraphLib/download/global_dataset/"
spacenetdataset = "../data/spacenet/"

image_size = args.image_size

batch_size = 2 # 352 * 352 
batch_divide = 32 # 32*2  

# if args.image_size == 384:
# 	batch_size = 4

if args.mode != "train":
	batch_size = 1

validation_folder = "validation_" + instance_id 
Popen("mkdir -p "+validation_folder, shell=True).wait()

model_save_folder = args.model_save + instance_id + "/"

max_degree = 6

Popen("mkdir -p %s" % model_save_folder, shell=True).wait()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.65)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
	model = Sat2GraphModel(sess, image_size=image_size, resnet_step = args.resnet_step, batchsize = batch_size, batchdivide=batch_divide, channel = args.channel, mode = args.mode, model_name = args.model_name)
	
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
		print("Use the 20-city datasets")
		# dataset partition
		indrange_train = []
		indrange_test = []
		indrange_validation = []

		for x in range(180):
			if x % 10 < 8 :
				indrange_train.append(x)

			if x % 10 == 9:
				indrange_test.append(x)

			if x % 20 == 18:
				indrange_validation.append(x)

			if x % 20 == 8:
				indrange_test.append(x)

		for x in range(180, 938):
			if x % 50 < 48 :
				indrange_train.append(x)

			if x % 50 == 48:
				indrange_test.append(x)

			if x % 50 == 49:
				indrange_validation.append(x)


		print("training set", len(indrange_train))
		print("testing set", len(indrange_test))
		print("validation set", len(indrange_validation))
		

		if args.mode == "train":
			dataloader_train = Sat2GraphDataLoaderOSM(osmdataset, indrange_train, imgsize = image_size, preload_tiles = 4, testing = False, random_mask=True)
			dataloader_train.preload(num=1024)

			dataloader_test = Sat2GraphDataLoaderOSM(osmdataset, indrange_validation, imgsize = image_size, preload_tiles = len(indrange_validation), random_mask=False, testing=True)
			dataloader_test.preload(num=128)


		else:
			dataloader = Sat2GraphDataLoaderOSM(osmdataset, [], imgsize = image_size, preload_tiles = 1, random_mask=False)

			tiles = indrange_test
			if args.mode == "validate":
				tiles = indrange_validation
			
			Popen("mkdir -p rawoutputs_%s" % (args.instance_id), shell=True).wait()
			Popen("mkdir -p outputs", shell=True).wait() 

			for tile_id in tiles:
				t0 = time()

				input_sat, gt_prob, gt_vector = dataloader.loadtile(tile_id)
				#print(np.shape(input_sat))
				gt_seg = np.zeros((1,image_size,image_size,1))
			

				gt_imagegraph = np.zeros((2048,2048,26))

				gt_imagegraph[:,:,0:2] = gt_prob[0,:,:,0:2]
				for k in range(max_degree):
					gt_imagegraph[:,:,2+k*4:2+k*4+2] = gt_prob[0,:,:,2+k*2:2+k*2+2]
					gt_imagegraph[:,:,2+k*4+2:2+k*4+4] = gt_vector[0,:,:,k*2:k*2+2]

				x, y = 0, 0 

				output = np.zeros((2048+64, 2048+64, 2+4*6 + 2))

				mask = np.ones((2048+64,2048+64, 2+4*6 + 2)) * 0.001
				weights = np.ones((image_size,image_size, 2+4*6 + 2)) * 0.001 
				weights[32:image_size-32,32:image_size-32, :] = 0.5 
				weights[56:image_size-56,56:image_size-56, :] = 1.0 
				weights[88:image_size-88,88:image_size-88, :] = 1.5 
				
				input_sat = np.pad(input_sat, ((0,0),(32,32),(32,32),(0,0)), 'constant')
				gt_vector = np.pad(gt_vector, ((0,0),(32,32),(32,32),(0,0)), 'constant')
				gt_prob = np.pad(gt_prob,((0,0),(32,32),(32,32),(0,0)), 'constant')
				
				for x in range(0,352*6-176-88,176/2):
					
					progress = x/88

					sys.stdout.write("\rProcessing Tile %d ...  "%tile_id + ">>" * progress + "--" * (20-progress))
					sys.stdout.flush()

					for y in range(0,352*6-176-88,176/2):

						alloutputs  = model.Evaluate(input_sat[:,x:x+image_size, y:y+image_size,:], gt_prob[:,x:x+image_size, y:y+image_size,:], gt_vector[:,x:x+image_size, y:y+image_size,:], gt_seg)
						_output = alloutputs[1]
					
						mask[x:x+image_size, y:y+image_size, :] += weights
						output[x:x+image_size, y:y+image_size,:] += np.multiply(_output[0,:,:,:], weights)


				output = np.divide(output, mask)

				output = output[32:2048+32,32:2048+32,:]
				input_sat = input_sat[:,32:2048+32,32:2048+32,:]

				output_keypoints_img = (output[:,:,0] * 255.0).reshape((2048,2048)).astype(np.uint8)
				Image.fromarray(output_keypoints_img).save("outputs/region_%d_output_keypoints.png" % tile_id)

				input_sat_img = ((input_sat[0,:,:,:]+0.5) * 255.0).reshape((2048,2048,3)).astype(np.uint8)
				Image.fromarray(input_sat_img).save("outputs/region_%d_input.png" % tile_id)

				DecodeAndVis(output, "outputs/region_%d_output" % (tile_id), thr=0.05, snap=True, imagesize = 2048)
							
				np.save("rawoutputs_%s/region_%d_output_raw" % (args.instance_id, tile_id), output)
									
				print(" done!  time: %.2f seconds"%(time() - t0))

			exit()

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

	test_size = 96

	for j in range(test_size/batch_size):
		input_sat, gt_prob, gt_vector, gt_seg= dataloader_test.getBatch(batch_size)
		validation_data.append([np.copy(input_sat), np.copy(gt_prob), np.copy(gt_vector), np.copy(gt_seg)])


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
		input_sat, gt_prob, gt_vector, gt_seg = dataloader_train.getBatch(batch_size)
		t_load += time() - t0 
		
		t0 = time()

		loss, grad_max, prob_loss, vector_loss,seg_loss, _ = model.Train(input_sat, gt_prob, gt_vector, gt_seg, lr)

		if step > 0 and step % batch_divide == 0 :
			model.ApplyGradient(lr)


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
						input_sat, gt_prob, gt_vector, gt_seg = validation_data[j][0], validation_data[j][1], validation_data[j][2], validation_data[j][3]
					if j == 0:
						test_loss = 0
						test_gan_loss = 0

					gt_imagegraph[:,:,:,0:2] = gt_prob[:,:,:,0:2]
					for k in range(max_degree):
						gt_imagegraph[:,:,:,2+k*4:2+k*4+2] = gt_prob[:,:,:,2+k*2:2+k*2+2]
						gt_imagegraph[:,:,:,2+k*4+2:2+k*4+4] = gt_vector[:,:,:,k*2:k*2+2]

					_test_loss, output = model.Evaluate(input_sat, gt_prob, gt_vector, gt_seg)

					test_loss += _test_loss

					if step == 1000 or step % 2000 == 0 or (step < 1000 and step % 200 == 0):
						for k in range(batch_size):
							
							input_sat_img = ((input_sat[k,:,:,:] + 0.5) * 255.0).reshape((image_size,image_size,3)).astype(np.uint8)

							# segmentation output (joint training)
							output_img = (output[k,:,:,-2] * 255.0).reshape((image_size,image_size)).astype(np.uint8)
							Image.fromarray(output_img).save(validation_folder+"/tile%d_output_seg.png" % (j*batch_size+k))
							Image.fromarray(((gt_seg[k,:,:,0] + 0.5) * 255.0).reshape((image_size, image_size)).astype(np.uint8)).save(validation_folder+"/tile%d_gt_seg.png" % (j*batch_size+k))

							# keypoints 
							output_keypoints_img = (output[k,:,:,0] * 255.0).reshape((image_size,image_size)).astype(np.uint8)
							Image.fromarray(output_keypoints_img).save(validation_folder+"/tile%d_output_keypoints.png" % (j*batch_size+k))

							# input satellite
							Image.fromarray(input_sat_img).save(validation_folder+"/tile%d_input_sat.png" % (j*batch_size+k))
								
							# todo 			
							#ImageGraphVis(output[k,:,:,0:2 + 4*max_degree].reshape((image_size, image_size, 2 + 4*max_degree )), validation_folder+"/tile%d_output_graph_0.01.png" % (j*batch_size+k), thr=0.01, imagesize = image_size)
							DecodeAndVis(output[k,:,:,0:2 + 4*max_degree].reshape((image_size, image_size, 2 + 4*max_degree )), validation_folder+"/tile%d_output_graph_0.01_snap.png" % (j*batch_size+k), thr=0.01, snap=True, imagesize = image_size, use_graph_refine=False)
							#ImageGraphVis(gt_imagegraph[k,:,:,:].reshape((image_size, image_size, 2 + 4*max_degree )), validation_folder+"/tile%d_output_graph_gt.png" % (j*batch_size+k), thr=0.5, imagesize = image_size)
							DecodeAndVis(gt_imagegraph[k,:,:,:].reshape((image_size, image_size, 2 + 4*max_degree )), validation_folder+"/tile%d_output_graph_gt.png" % (j*batch_size+k), thr=0.5, snap=True, imagesize = image_size, use_graph_refine=False)
							

				test_loss /= test_size/batch_size
				
			print("")
			print("step", step, "loss", sum_loss, "test_loss", test_loss, "prob_loss", sum_prob_loss/200.0, "vector_loss", sum_vector_loss/200.0, "seg_loss", sum_seg_loss/200.0)
			
			summary = model.addLog(test_loss, sum_loss, 0)
			writer.add_summary(summary, step)


			sum_prob_loss = 0
			sum_vector_loss = 0
			sum_seg_loss = 0
			sum_loss = 0 


		if step > 0 and step % 200 == 0:
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
		if step == 600000+2:
			break 
