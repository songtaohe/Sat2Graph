import tensorflow as tf 
import sys  
import scipy.ndimage
from time import time 
import numpy as np 
import json 
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


pbfile = '/data/songtao/spatialCNN/models/roaddla/graph_dyn_frozen.pb'

with tf.gfile.GFile(pbfile, 'rb') as f:
    graph_def_optimized = tf.GraphDef()
    graph_def_optimized.ParseFromString(f.read())

G = tf.Graph()

# fix batch norm 
for node in graph_def_optimized.node:            
    if node.op == 'RefSwitch':
        node.op = 'Switch'
        for index in xrange(len(node.input)):
            if 'moving_' in node.input[index]:
                node.input[index] = node.input[index] + '/read'
    elif node.op == 'AssignSub':
        node.op = 'Sub'
        if 'use_locking' in node.attr: del node.attr['use_locking']
    elif node.op == 'AssignAdd':
        node.op = 'Add'
        if 'use_locking' in node.attr: del node.attr['use_locking']

gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(graph=G, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    y, = tf.import_graph_def(graph_def_optimized, return_elements=['output:0'])
    #print('Operations in Optimized Graph:')
    #print([op.name for op in G.get_operations()])

    x = G.get_tensor_by_name('import/input:0')
    istraining = G.get_tensor_by_name('import/istraining:0')


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

        crop_dim = [dim[0] / 32 * 32, dim[1] / 32 * 32]
        sat_img_vis = np.copy(sat_img)
        sat_img = sat_img[0:0+crop_dim[0], 0:0+crop_dim[1],:]
        

        max_v = 255
        sat_img = (sat_img.astype(np.float)/ max_v - 0.5) * 0.9 
        sat_img_ = sat_img.reshape((1,crop_dim[0],crop_dim[1],3))

        t0 = time()
        output = sess.run(y, feed_dict={x: sat_img_, istraining: False})
        print(np.shape(output))
        print("gpu done", time()-t0)
        graph = DecodeAndVis(output[0,:,:,:28], output_file, thr=v_thr, edge_thr = e_thr, angledistance_weight=snap_w, snap_dist = snap_dist, snap=True, imagesize = crop_dim[0], spurs_thr = 100, isolated_thr= 500, connect_deadend_thr=0)
        graph = simpilfyGraph(graph)
        print("all done", time()-t0)


        sat_img = sat_img_vis
        for k,v in graph.iteritems():
            n1 = k 
            for n2 in v:
                cv2.line(sat_img, (n1[1], n1[0]), (n2[1], n2[0]), (255,255,0),2)
        
        Image.fromarray(sat_img).save(output_file+"_vis.png")

        jsongraph = {}
        for k,v in graph.iteritems():
            sk = "%d_%d" % (k[0], k[1])
            jsongraph[sk] = []
            for n2 in v:
                jsongraph[sk].append("%d_%d" % (n2[0], n2[1]))

        json.dump(jsongraph, open(output_file+"_graph.json","w"), indent=2)

        


    
