import pickle
import numpy as np 
import cv2 
import sys 

nodeneighbor = pickle.load(open(sys.argv[1]))
edgeClass = pickle.load(open(sys.argv[2]))
output_fn = sys.argv[3]

img = np.zeros((2048, 2048, 3), dtype=np.uint8)

for nloc, neis in nodeneighbor.iteritems():
	for nei in neis:
		if (nloc, nei) in edgeClass:
			probs = edgeClass[(nloc, nei)]

			if probs[0] >= 0.5:
				color = (255,0,0)
			else:
				color = (0,255,0)


			cv2.line(img, (int(nloc[0]),int(nloc[1])) , (int(nei[0]), int(nei[1])), color, 3)

for nloc, neis in nodeneighbor.iteritems():
	cv2.circle(img, (int(nloc[0]),int(nloc[1])), 3, (255,255,0), -1)

cv2.imwrite(output_fn, img)

		
