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
 
from model import Sat2GraphModel
from decoder import DecodeAndVis 


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

model = Sat2GraphModel(sess, image_size=352, resnet_step = 12, batchsize = 1, channel = 18, mode = "test")
model.restoreModel("../data/20citiesModel/model")

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
		sat_img = sat_img.reshape((1,352,352,3))


		alloutputs  = model.Evaluate(sat_img, gt_prob_placeholder, gt_vector_placeholder, gt_seg_placeholder)
		output = alloutputs[1][0,:,:,:]

		graph = DecodeAndVis(output, output_file, thr=0.05, snap=True, imagesize = 352)

		lines = []
		points = []

		

		for nid, nei in graph.iteritems():
			for nn in nei:
				edge = (nid, nn)
				edge_ = (nn, nid)
				if edge not in lines and edge_ not in lines:
					lines.append(edge)  

			points.append(nid)


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