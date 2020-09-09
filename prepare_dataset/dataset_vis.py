import pickle 
import cv2 
import numpy as np 
import sys  

def drawgraph(graph, filename):
	img = np.ones((2048,2048,3), dtype=np.uint8)*255

	for n, v in graph.iteritems():
		for nei in v:
			p1 = (int(n[1]), int(n[0]))
			p2 = (int(nei[1]), int(nei[0]))

			img = cv2.line(img, p1, p2, (0,0,0),2)

	for n, v in graph.iteritems():
		p1 = (int(n[1]), int(n[0]))
		img = cv2.circle(img, p1, 2, (0,0,255),-1)

	cv2.imwrite(filename, img)


graph1 = pickle.load(open("global_dataset_mapbox_no_service_road/region_%s_graph_gt.pickle" % sys.argv[1]))
graph2 = pickle.load(open("global_dataset_mapbox_no_service_road/region_%s_refine_gt_graph.p" % sys.argv[1]))


drawgraph(graph1, "org.png")
drawgraph(graph2, "refine.png")

