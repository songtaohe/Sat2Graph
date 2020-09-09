import math 
import numpy as np 
import scipy.misc 
from PIL import Image 
import cv2 
import sys 
import pickle 
from rtree import index
from common import * 

def GPSDistance(p1, p2):
	a = p1[0] - p2[0]
	b = (p1[1] - p2[1]) * math.cos(math.radians(p1[0]))

	return math.sqrt(a*a + b*b)


def graphInsert(node_neighbor, n1key, n2key):
	if n1key != n2key:
		if n1key in node_neighbor:
			if n2key in node_neighbor[n1key]:
				pass 
			else:
				node_neighbor[n1key].append(n2key)
		else:
			node_neighbor[n1key] = [n2key]


		if n2key in node_neighbor:
			if n1key in node_neighbor[n2key]:
				pass 
			else:
				node_neighbor[n2key].append(n1key)
		else:
			node_neighbor[n2key] = [n1key]

	return node_neighbor


def graphDensify(node_neighbor, density = 0.00020):
	visited = []

	new_node_neighbor = {}

	for node, node_nei in node_neighbor.iteritems():
		if len(node_nei) == 1 or len(node_nei) > 2:
			if node in visited:
				continue

			# search node_nei 

			for next_node in node_nei:
				if next_node in visited:
					continue

				node_list = [node, next_node]

				current_node = next_node 

				while True:
					if len(node_neighbor[node_list[-1]]) == 2:
						if node_neighbor[node_list[-1]][0] == node_list[-2]:
							node_list.append(node_neighbor[node_list[-1]][1])
						else:
							node_list.append(node_neighbor[node_list[-1]][0])
					else:
						break

				for i in range(len(node_list)-1):
					if node_list[i] not in visited:
						visited.append(node_list[i])

				# interpolate
				# partial distance 
				pd = [0]

				for i in range(len(node_list)-1):
					pd.append(pd[-1]+GPSDistance(node_list[i], node_list[i+1]))

				interpolate_N = int(pd[-1]/density)

				last_loc = node_list[0]

				for i in range(interpolate_N):
					int_d = pd[-1]/(interpolate_N+1)*(i+1)
					for j in range(len(node_list)-1):
						if pd[j] <= int_d and pd[j+1] > int_d:
							a = (int_d-pd[j]) / (pd[j+1] - pd[j])

							loc = ((1-a) * node_list[j][0] + a * node_list[j+1][0], (1-a) * node_list[j][1] + a * node_list[j+1][1])

							new_node_neighbor = graphInsert(new_node_neighbor, last_loc, loc)
							last_loc = loc 

				new_node_neighbor = graphInsert(new_node_neighbor, last_loc, node_list[-1])

	return new_node_neighbor

def graph2RegionCoordinate(node_neighbor, region):
	new_node_neighbor = {}

	for node, nei in node_neighbor.iteritems():
		loc0 = node 
		for loc1 in nei:
			x0 = (loc0[1] - region[1])/(region[3]-region[1])*2048
			y0 = (region[2]-loc0[0])/(region[2]-region[0])*2048
			x1 = (loc1[1] - region[1])/(region[3]-region[1])*2048
			y1 = (region[2]-loc1[0])/(region[2]-region[0])*2048

			n1key = (y0,x0)
			n2key = (y1,x1)

			new_node_neighbor = graphInsert(new_node_neighbor, n1key, n2key)

	return new_node_neighbor 


def graphVis2048(node_neighbor, region, filename):
	img = np.zeros((2048,2048,3),dtype=np.uint8)
	img = img + 255 

	for node, nei in node_neighbor.iteritems():
		loc0 = node 
		for loc1 in nei:
			x0 = int((loc0[1] - region[1])/(region[3]-region[1])*2048)
			y0 = int((region[2]-loc0[0])/(region[2]-region[0])*2048)
			x1 = int((loc1[1] - region[1])/(region[3]-region[1])*2048)
			y1 = int((region[2]-loc1[0])/(region[2]-region[0])*2048)

			cv2.line(img, (x0,y0), (x1,y1), (0,0,0),2)


	for node, nei in node_neighbor.iteritems():
		loc0 = node 
		x0 = int((loc0[1] - region[1])/(region[3]-region[1])*2048)
		y0 = int((region[2]-loc0[0])/(region[2]-region[0])*2048)
		
		cv2.circle(img, (x0,y0),3, (0,0,255),-1)

	cv2.imwrite(filename, img)

def graphVis2048Segmentation(node_neighbor, region, filename, size=2048):
	img = np.zeros((size,size),dtype=np.uint8)
	

	for node, nei in node_neighbor.iteritems():
		loc0 = node 
		for loc1 in nei:
			x0 = int((loc0[1] - region[1])/(region[3]-region[1])*size)
			y0 = int((region[2]-loc0[0])/(region[2]-region[0])*size)
			x1 = int((loc1[1] - region[1])/(region[3]-region[1])*size)
			y1 = int((region[2]-loc1[0])/(region[2]-region[0])*size)

			cv2.line(img, (x0,y0), (x1,y1), (255),2)


	cv2.imwrite(filename, img)

def graphVisStackingRoad(node_neighbor, region, filename, size=2048):
	img = np.zeros((size,size),dtype=np.uint8)
	
	crossing_point, adjustment = locate_stacking_road(node_neighbor)

	for ip in crossing_point.values():
		loc0 = ip 
		x0 = int((loc0[1] - region[1])/(region[3]-region[1])*size)
		y0 = int((region[2]-loc0[0])/(region[2]-region[0])*size)
		
		cv2.circle(img, (x0,y0), 5, (255),-1)

	cv2.imwrite(filename, img)

def graphVisIntersection(node_neighbor, region, filename, size=2048):
	img = np.zeros((size,size),dtype=np.uint8)
	
	for node, nei in node_neighbor.iteritems():
		loc0 = node 

		if len(nei) != 2:
			x0 = int((loc0[1] - region[1])/(region[3]-region[1])*size)
			y0 = int((region[2]-loc0[0])/(region[2]-region[0])*size)
			
			cv2.circle(img, (x0,y0), 5, (255),-1)

	cv2.imwrite(filename, img)


def locate_stacking_road(graph):

	idx = index.Index()

	edges = []
	
	for n1, v in graph.items():
		for n2 in v:
			if (n1,n2) in edges or (n2,n1) in edges:
				continue

			x1 = min(n1[0], n2[0])
			x2 = max(n1[0], n2[0])

			y1 = min(n1[1], n2[1])
			y2 = max(n1[1], n2[1])

			idx.insert(len(edges), (x1,y1,x2,y2))

			edges.append((n1,n2))

	adjustment = {}

	crossing_point = {}


	for edge in edges:
		n1 = edge[0]
		n2 = edge[1]

		x1 = min(n1[0], n2[0])
		x2 = max(n1[0], n2[0])

		y1 = min(n1[1], n2[1])
		y2 = max(n1[1], n2[1])

		candidates = list(idx.intersection((x1,y1,x2,y2)))

		for _candidate in candidates:
			# todo mark the overlap point 
			candidate = edges[_candidate]


			if n1 == candidate[0] or n1 == candidate[1] or n2 == candidate[0] or n2 == candidate[1]:
				continue

			if intersect(n1,n2,candidate[0], candidate[1]):

				ip = intersectPoint(n1,n2,candidate[0], candidate[1])

				if (candidate, edge) not in crossing_point:
					crossing_point[(edge, candidate)] = ip

				#release points 
				d = distance(ip, n1)
				thr = 9.5 # was 5.0
				if d < thr:
					vec = neighbors_norm(graph, n1, n2)
					weight = (thr-d)/thr
					vec = (vec[0] * weight, vec[1] * weight)
					
					
					if n1 not in adjustment:
						adjustment[n1] = [vec] 
					else:
						adjustment[n1].append(vec)

				d = distance(ip, n2)
				if d < thr:
					vec = neighbors_norm(graph, n2, n1)
					weight = (thr-d)/thr
					vec = (vec[0] * weight, vec[1] * weight)
					
					
					if n2 not in adjustment:
						adjustment[n2] = [vec] 
					else:
						adjustment[n2].append(vec)


				c1 = candidate[0]
				c2 = candidate[1]


				d = distance(ip, c1)
				if d < thr:
					vec = neighbors_norm(graph, c1, c2)
					weight = (thr-d)/thr
					vec = (vec[0] * weight, vec[1] * weight)
					
					
					if c1 not in adjustment:
						adjustment[c1] = [vec] 
					else:
						adjustment[c1].append(vec)

				d = distance(ip, c2)
				if d < thr:
					vec = neighbors_norm(graph, c2, c1)
					weight = (thr-d)/thr
					vec = (vec[0] * weight, vec[1] * weight)
					
					
					if c2 not in adjustment:
						adjustment[c2] = [vec] 
					else:
						adjustment[c2].append(vec)

	# apply adjustment 
	# move 2 pixels each time 


	return crossing_point, adjustment

def locate_parallel_road(graph):

	idx = index.Index()

	edges = []
	
	for n1, v in graph.items():
		for n2 in v:
			if (n1,n2) in edges or (n2,n1) in edges:
				continue

			x1 = min(n1[0], n2[0])
			x2 = max(n1[0], n2[0])

			y1 = min(n1[1], n2[1])
			y2 = max(n1[1], n2[1])

			idx.insert(len(edges), (x1,y1,x2,y2))

			edges.append((n1,n2))

	adjustment = {}

	crossing_point = {}

	parallel_road = []

	for edge in edges:
		n1 = edge[0]
		n2 = edge[1]

		if distance(n1,n2) < 10:
			continue

		x1 = min(n1[0], n2[0]) - 20
		x2 = max(n1[0], n2[0]) + 20

		y1 = min(n1[1], n2[1]) - 20
		y2 = max(n1[1], n2[1]) + 20

		candidates = list(idx.intersection((x1,y1,x2,y2)))

		for _candidate in candidates:
			# todo mark the overlap point 
			candidate = edges[_candidate]

			if n1 == candidate[0] or n1 == candidate[1] or n2 == candidate[0] or n2 == candidate[1]:
				continue

			flag = False

			for nei in graph[n1]:
				if candidate[0] in graph[nei]:
					flag = True 
					continue

				if candidate[1] in graph[nei]:
					flag = True 
					continue

			for nei in graph[n2]:
				if candidate[0] in graph[nei]:
					flag = True 
					continue

				if candidate[1] in graph[nei]:
					flag = True 
					continue

			if flag :
				continue

			p = abs(neighbors_cos(graph, (0,0), (n2[0] - n1[0], n2[1] - n1[1]), (candidate[1][0] - candidate[0][0], candidate[1][1] - candidate[0][1])))

			if p > 0.985:
				if n1 not in parallel_road:
					parallel_road.append(n1)





	return parallel_road 



def apply_adjustment(graph, adjustment):
	current_graph = graph
	counter = 0 

	for k,v in adjustment.items():
		#print(k)
		new_graph = {}

		vec = [0,0]

		for vv in v:
			vec[0] += vv[0]
			vec[1] += vv[1]

		vl = vec[0]*vec[0] + vec[1]*vec[1]

		vl = np.sqrt(vl)

		if vl == 0:
			continue

		if vl > 1.0:
			vec[0]/=vl
			vec[1]/=vl

		for l in [1.5,1.0]:

			#new_k = (k[0] + int(vec[0]*l), k[1] + int(vec[1]*l))
			new_k = (k[0] + (vec[0]*l), k[1] + (vec[1]*l))

			if new_k == k:
				continue

			if new_k not in current_graph:

				neighbors = list(current_graph[k])

				del current_graph[k]

				current_graph[new_k] = neighbors

				for nei in neighbors:
					new_nei = []

					for n in current_graph[nei]:
						if n == k:
							new_nei.append(new_k)
						else:
							new_nei.append(n)

					current_graph[nei] = new_nei

				#print(k, "-->", new_k)

				counter += 1

				break
			else:
				continue


	print("adjusted ", counter, " nodes")

	return current_graph, counter 


	pass

def graph_move_node(graph, old_n, new_n):
	nei = list(graph[old_n])
	del graph[old_n]

	graph[new_n] = nei 

	for nn in nei:
		for i in range(len(graph[nn])):
			if graph[nn][i] == old_n:
				graph[nn][i] = new_n

	return graph 


def apply_adjustment_delete_closeby_nodes(graph, adjustment):
	# delete the node and push its two neighbors closer ...
	
	# thr = 9.5 
	thr = 9.5 
	for k,v in adjustment.items(): 
		if len(v) >= 4: # duplicated ... 
			ds = []
			for vv in v:
				ds.append((1.0-distance(vv, (0,0))) * thr)
			sorted(ds)
			gap = sum(ds[0:4])/2.0

			# delete the node and push its two neighbors closer ...
			if gap < 12 and len(graph[k]) == 2:
				nei1 = graph[k][0]
				nei2 = graph[k][1]

				del graph[k]
				print("delete a node", k)

				for i in range(len(graph[nei1])):
					if graph[nei1][i] == k:
						graph[nei1][i] = nei2

				for i in range(len(graph[nei2])):
					if graph[nei2][i] == k:
						graph[nei2][i] = nei1

				# move nei1/nei2 closer?
				if nei1 not in adjustment:
					vec = neighbors_norm(graph, k, nei1)
					new_nei1 = (nei1[0] + vec[0] * 5.0, nei1[1] + vec[1] * 5.0)
					graph = graph_move_node(graph, nei1, new_nei1)

				if nei2 not in adjustment:
					vec = neighbors_norm(graph, k, nei2)
					new_nei2 = (nei2[0] + vec[0] * 5.0, nei2[1] + vec[1] * 5.0)
					graph = graph_move_node(graph, nei2, new_nei2)

	return graph 



def graphGroundTruthPreProcess(graph):
	for it in range(40): # was 8 
		cp, adj = locate_stacking_road(graph)
		if it % 5 == 0 and it != 0 :
			graph = apply_adjustment_delete_closeby_nodes(graph, adj)
		else:
			graph, c = apply_adjustment(graph, adj)
			if c == 0:
				break

	sample_points = {}

	sample_points['parallel_road'] = locate_parallel_road(graph)
	sample_points['complicated_intersections'] = []

	for k,v in graph.items():
		degree = len(v)
		if degree > 4:
			sample_points['complicated_intersections'].append(k)

	sample_points['overpass'] = []

	for k, v in cp.items():
		sample_points['overpass'].append((int(v[0]), int(v[1])))


	return graph, sample_points 




