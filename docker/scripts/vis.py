import cv2 
import graph
import json 
import sys 
import numpy as np  

tilesize = int(sys.argv[1])
infile = sys.argv[2]
outfile = sys.argv[3]

edges = json.load(open(infile,"r"))

img = np.ones((tilesize, tilesize, 3), dtype=np.uint8) * 255
for edge in edges:
    n1 = (int(edge[0][1]), int(edge[0][0]))
    n2 = (int(edge[1][1]), int(edge[1][0]))
    cv2.line(img, n1, n2, (0,0,0),3)

cv2.imwrite(outfile, img)
