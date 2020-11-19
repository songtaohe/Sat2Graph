import pickle
import numpy as np 
import cv2 
import scipy.ndimage 
import sys 
import graph_ops as graphlib 

nodeneighbor = pickle.load(open(sys.argv[1]))
classmap = scipy.ndimage.imread(sys.argv[2])
output_fn = sys.argv[3]


edgeClass = {}

for nloc, neis in nodeneighbor.iteritems():
	for nei in neis:
		edgeK = (nloc, nei)

		if (nloc, nei) in edgeClass or (nei, nloc) in edgeClass:
			continue 

		l = graphlib.PixelDistance(nloc, nei)

		class1count = 1.0
		class2count = 1.0 

		for i in range(int(l)+1):
			alpha = float(i) / int(l)
			x = nloc[0] * alpha + nei[0] * (1-alpha)
			y = nloc[1] * alpha + nei[1] * (1-alpha)

			x = int(x)
			y = int(y)

			class1count += classmap[x][y][0]
			class2count += classmap[x][y][1]

		p1 = class1count / (class1count+class2count)
		p2 = class2count / (class1count+class2count)

		edgeClass[edgeK] = (p1,p2)

pickle.dump(edgeClass, open(output_fn, "w"))




