# from http.server import BaseHTTPRequestHandler
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import logging
import json 
import os 
import os.path  
import scipy.ndimage 
import math 
import cv2
import numpy as np 
import tensorflow as tf 
from time import time 
 
from model import Sat2GraphModel
from decoder import DecodeAndVis 
from douglasPeucker import simpilfyGraph 

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.34)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

model = Sat2GraphModel(sess, image_size=352, resnet_step = 8, batchsize = 1, channel = 24, mode = "test", model_name="unet")
model.restoreModel("/data/songtao/Sat2GraphLib/globalmodel20200804v3UNET_352_8__channel24/model560000")

gt_prob_placeholder = np.zeros((1,352,352,14))
gt_vector_placeholder = np.zeros((1,352,352,12))
gt_seg_placeholder = np.zeros((1,352,352,1))



class S(BaseHTTPRequestHandler):
	def _set_response(self):
		self.send_response(200)
		self.send_header('Content-type', 'text/html')
		self.end_headers()

	def do_GET(self):
		logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
		self._set_response()
		self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))

	def do_POST(self):
		global model 
		global gt_prob_placeholder
		global gt_vector_placeholder
		global gt_seg_placeholder

		content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
		post_data = self.rfile.read(content_length) # <--- Gets the data itself
		logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\n\n",
				str(self.path), str(self.headers))

		return_str = ""

		#try:
		data = json.loads(post_data.decode('utf-8'))

		input_file = data["img_in"]
		output_file = data["output_json"]

		# run the model 

		sat_img = scipy.ndimage.imread(input_file).astype(np.float)
		max_v = 255
		sat_img = (sat_img.astype(np.float)/ max_v - 0.5) * 0.9 
		sat_img = sat_img.reshape((1,704,704,3))

		image_size = 352 

		weights = np.ones((image_size,image_size, 2+4*6 + 2)) * 0.001 
		weights[32:image_size-32,32:image_size-32, :] = 0.5 
		weights[56:image_size-56,56:image_size-56, :] = 1.0 
		weights[88:image_size-88,88:image_size-88, :] = 1.5 

		mask = np.zeros((704, 704, 2+4*6 + 2))
		output = np.zeros((704, 704, 2+4*6 + 2))

		t0 = time()
		for x in range(0,704-176-88,176/2):		
			for y in range(0,704-176-88,176/2):

				alloutputs  = model.Evaluate(sat_img[:,x:x+image_size, y:y+image_size,:], gt_prob_placeholder, gt_vector_placeholder, gt_seg_placeholder)
				_output = alloutputs[1]

				mask[x:x+image_size, y:y+image_size, :] += weights
				output[x:x+image_size, y:y+image_size,:] += np.multiply(_output[0,:,:,:], weights)

		print("GPU time:", time() - t0)
		t0 = time()

		output = np.divide(output, mask)

		# alloutputs  = model.Evaluate(sat_img, gt_prob_placeholder, gt_vector_placeholder, gt_seg_placeholder)
		# output = alloutputs[1][0,:,:,:]

		# for 20 cities model
		#graph = DecodeAndVis(output, output_file, thr=0.01, edge_thr = 0.1, angledistance_weight=50, snap=True, imagesize = 704)

		# for global model
		graph = DecodeAndVis(output, output_file, thr=0.05, edge_thr = 0.15, angledistance_weight=50, snap=True, imagesize = 704)

		print("Decode time:", time() - t0)
		t0 = time()

		graph = simpilfyGraph(graph)

		print("Graph simpilfy time:", time() - t0)
		t0 = time()

		lines = []
		points = []

		biasx = -102
		biasy = -102

		def addbias(loc):
			return (loc[0]+biasx, loc[1]+biasy)

		def inrange(loc):
			if loc[0] > 102 and loc[0] < 602 and loc[1] > 102 and loc[1] < 602:
				return True 
			else:
				return False 

		for nid, nei in graph.iteritems():
			for nn in nei:
				if inrange(nn) or inrange(nid):
					edge = (addbias(nid), addbias(nn))
					edge_ = (addbias(nn), addbias(nid))
					if edge not in lines and edge_ not in lines:
						lines.append(edge)  

			if inrange(nid) and len(nei)!=2:
				points.append(addbias(nid))


		# graph to json 

		
		return_str = json.dumps({"graph":[lines, points], "success":"true"})


		# except:
		# 	return_str = json.dumps({"success":"false"})
		# 	print("parse json data failed")


		self._set_response()
		self.wfile.write(return_str.encode('utf-8'))

def run(server_class=HTTPServer, handler_class=S, port=8080):
	logging.basicConfig(level=logging.INFO)
	server_address = ('', port)
	httpd = server_class(server_address, handler_class)
	logging.info('Starting httpd...\n')
	try:
		httpd.serve_forever()
	except KeyboardInterrupt:
		pass
	httpd.server_close()
	logging.info('Stopping httpd...\n')

if __name__ == '__main__':
	from sys import argv

	if len(argv) == 2:
		run(port=int(argv[1]))
	else:
		run()