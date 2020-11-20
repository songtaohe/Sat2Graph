#import rdp
# Code Copied From Favyen

import scipy.ndimage
import skimage.morphology
import os
from PIL import Image
import math
import numpy
import numpy as np 
from multiprocessing import Pool
import subprocess
import sys
from math import sqrt
import pickle
import json
import graph_ops as graphlib 
import cv2 
from decoder import graph_refine

def distance(a, b):
    return  sqrt(float((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))

def point_line_distance(point, start, end):
    if (start == end):
        return distance(point, start)
    else:
        n = abs(
            (end[0] - start[0]) * (start[1] - point[1]) - (start[0] - point[0]) * (end[1] - start[1])
        )
        d = sqrt(
            float((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        )
        return n / d

def rdp(points, epsilon):
    """
    Reduces a series of points to a simplified version that loses detail, but
    maintains the general shape of the series.
    """
    dmax = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = point_line_distance(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d
    if dmax >= epsilon:
        results = rdp(points[:index+1], epsilon)[:-1] + rdp(points[index:], epsilon)
    else:
        results = [points[0], points[-1]]
    return results

PADDING = 30

in_fname = sys.argv[1]
#threshold = int(sys.argv[2])
threshold = int(256*0.15)  # 0.15 
out_fname = sys.argv[2]

im = scipy.ndimage.imread(in_fname)

if len(im.shape) == 3:
	print 'warning: bad shape {}, using first channel only'.format(im.shape)
	im = im[:, :, 0]

im = numpy.swapaxes(im, 0, 1)

im = (im >= threshold).astype(np.uint8)
im = scipy.ndimage.binary_closing(im)

cv2.imwrite("debugvis0.png", im.astype(np.uint8)*255)

im = skimage.morphology.thin(im)
im = im.astype('uint8')

cv2.imwrite("debugvis1.png", im.astype(np.uint8)*255)

#print("thinning done")

vertices = []
edges = set()
def add_edge(src, dst):
	if (src, dst) in edges or (dst, src) in edges:
		return
	elif src == dst:
		return
	edges.add((src, dst))


point_to_neighbors = {}
q = []
while True:
	if len(q) > 0:
		lastid, i, j = q.pop()
		path = [vertices[lastid], (i, j)]
		if im[i, j] == 0:
			continue
		point_to_neighbors[(i, j)].remove(lastid)
		if len(point_to_neighbors[(i, j)]) == 0:
			del point_to_neighbors[(i, j)]
	else:
		w = numpy.where(im > 0)
		#print(len(w[0]))
		if len(w[0]) == 0:
			break
		i, j = w[0][0], w[1][0]
		lastid = len(vertices)
		vertices.append((i, j))
		path = [(i, j)]

	while True:
		im[i, j] = 0
		neighbors = []
		for oi in [-1, 0, 1]:
			for oj in [-1, 0, 1]:
				ni = i + oi
				nj = j + oj
				if ni >= 0 and ni < im.shape[0] and nj >= 0 and nj < im.shape[1] and im[ni, nj] > 0:
					neighbors.append((ni, nj))
		if len(neighbors) == 1 and (i, j) not in point_to_neighbors:
			ni, nj = neighbors[0]
			path.append((ni, nj))
			i, j = ni, nj
		else:
			if len(path) > 1:
				path = rdp(path, 2)
				if len(path) > 2:
					for point in path[1:-1]:
						curid = len(vertices)
						vertices.append(point)
						add_edge(lastid, curid)
						lastid = curid
				neighbor_count = len(neighbors) + len(point_to_neighbors.get((i, j), []))
				if neighbor_count == 0 or neighbor_count >= 2:
					curid = len(vertices)
					vertices.append(path[-1])
					add_edge(lastid, curid)
					lastid = curid
			for ni, nj in neighbors:
				if (ni, nj) not in point_to_neighbors:
					point_to_neighbors[(ni, nj)] = set()
				point_to_neighbors[(ni, nj)].add(lastid)
				q.append((lastid, ni, nj))
			for neighborid in point_to_neighbors.get((i, j), []):
				add_edge(neighborid, lastid)
			break
neighbors = {}

vertex = vertices

for edge in edges:

	nk1 = (vertex[edge[0]][1],vertex[edge[0]][0])
	nk2 = (vertex[edge[1]][1],vertex[edge[1]][0])
	
	if nk1 != nk2:
		if nk1 in neighbors:
			if nk2 in neighbors[nk1]:
				pass
			else:
				neighbors[nk1].append(nk2)
		else:
			neighbors[nk1] = [nk2]

		if  nk2 in neighbors:
			if nk1 in neighbors[nk2]:
				pass 
			else:
				neighbors[nk2].append(nk1)
		else:
			neighbors[nk2] = [nk1]


# img = np.zeros((5120, 5120, 3), dtype=np.uint8)
# for nloc, neis in neighbors.iteritems():
# 	for nei in neis:
# 		color = (255,255,255)
# 		cv2.line(img, (int(nloc[0]),int(nloc[1])) , (int(nei[0]), int(nei[1])), color, 3)

# cv2.imwrite(out_fname.replace(".p", "_graphvis0.png"), img)

	

node_neighbor = graphlib.graphDensify(neighbors, density = 20, distFunc = graphlib.PixelDistance)
node_neighbor = graph_refine(node_neighbor, isolated_thr = 250, spurs_thr = 50, three_edge_loop_thr = 70)
node_neighbor = graph_refine(node_neighbor, isolated_thr = 250, spurs_thr = 50, three_edge_loop_thr = 70)
node_neighbor = graph_refine(node_neighbor, isolated_thr = 250, spurs_thr = 50, three_edge_loop_thr = 70)

#node_neighbor = neighbors
pickle.dump(node_neighbor, open(out_fname, "w"))


img = np.zeros((5120, 5120, 3), dtype=np.uint8)

for nloc, neis in node_neighbor.iteritems():
	for nei in neis:
		color = (255,255,255)
		cv2.line(img, (int(nloc[1]),int(nloc[0])) , (int(nei[1]), int(nei[0])), color, 3)

cv2.imwrite(out_fname.replace(".p", "_graphvis.png"), img)














