import cv2 
import numpy as np 
from PIL import Image 
import scipy 
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
import numpy as np 
from rtree import index 
import sys 
import pickle 
from common import * 


vector_norm = 25.0 

def vNorm(v1):
	l = distance(v1,(0,0))+0.0000001
	return (v1[0]/l, v1[1]/l)

def anglediff(v1, v2):
	v1 = vNorm(v1)
	v2 = vNorm(v2)

	return v1[0]*v2[0] + v1[1] * v2[1]



def graph_refine(graph, isolated_thr = 150, spurs_thr = 30, three_edge_loop_thr = 70):
	neighbors = graph

	gid = 0 
	grouping = {}

	for k, v in neighbors.iteritems():
		if k not in grouping:
			# start a search 

			queue = [k]

			while len(queue) > 0:
				n = queue.pop(0)

				if n not in grouping:
					grouping[n] = gid 
					for nei in neighbors[n]:
						queue.append(nei)

			gid += 1 

	group_count = {}

	for k, v in grouping.items():
		if v not in group_count:
			group_count[v] = (1,0)
		else:
			group_count[v] = (group_count[v][0] + 1, group_count[v][1])


		for nei in neighbors[k]:
			a = k[0] - nei[0]
			b = k[1] - nei[1]

			d = np.sqrt(a*a + b*b)

			group_count[v] = (group_count[v][0], group_count[v][1] + d/2)

	# short spurs
	remove_list = []
	for k, v in neighbors.items():
		if len(v) == 1:
			if len(neighbors[v[0]]) >= 3:
				a = k[0] - v[0][0]
				b = k[1] - v[0][1]

				d = np.sqrt(a*a + b*b)	

				if d < spurs_thr:
					remove_list.append(k)


	remove_list2 = []
	remove_counter = 0
	new_neighbors = {}

	def isRemoved(k):
		gid = grouping[k]
		if group_count[gid][0] <= 1:
			return True 
		elif group_count[gid][1] <= isolated_thr:
			return True 
		elif k in remove_list:
			return True 
		elif k in remove_list2:
			return True
		else:
			return False

	for k, v in neighbors.items():
		if isRemoved(k): 
			remove_counter += 1
			pass
		else:
			new_nei = []
			for nei in v:
				if isRemoved(nei):
					pass 
				else:
					new_nei.append(nei)

			new_neighbors[k] = list(new_nei)

	#print(len(new_neighbors), "remove", remove_counter, "nodes")

	return new_neighbors


def graph_shave(graph, spurs_thr = 50):
	neighbors = graph

	# short spurs
	remove_list = []
	for k, v in neighbors.items():
		if len(v) == 1:
			d = distance(k,v[0])
			cur = v[0]
			l = [k]
			while True:
				if len(neighbors[cur]) >= 3:
					break
				elif len(neighbors[cur]) == 1:
					l.append(cur)
					break

				else:

					if neighbors[cur][0] == l[-1]:
						next_node = neighbors[cur][1]
					else:
						next_node = neighbors[cur][0]

					d += distance(cur, next_node)
					l.append(cur)

					cur = next_node 

			if d < spurs_thr:
				for n in l:
					if n not in remove_list:
						remove_list.append(n)

	
	def isRemoved(k):
		if k in remove_list:
			return True 
		else:
			return False

	new_neighbors = {}
	remove_counter = 0

	for k, v in neighbors.items():
		if isRemoved(k): 
			remove_counter += 1
			pass
		else:
			new_nei = []
			for nei in v:
				if isRemoved(nei):
					pass 
				else:
					new_nei.append(nei)

			new_neighbors[k] = list(new_nei)

	#print("shave", len(new_neighbors), "remove", remove_counter, "nodes")

	return new_neighbors


def graph_refine_deloop(neighbors, max_step = 10, max_length = 200, max_diff = 5):

	removed = []
	impact = []

	remove_edge = []
	new_edge = []

	for k, v in neighbors.items():
		if k in removed:
			continue

		if k in impact:
			continue


		if len(v) < 2:
			continue


		for nei1 in v:
			if nei1 in impact:
				continue

			if k in impact:
				continue
			
			for nei2 in v:
				if nei2 in impact:
					continue
				if nei1 == nei2 :
					continue



				if neighbors_cos(neighbors, k, nei1, nei2) > 0.984:
					l1 = neighbors_dist(neighbors, k, nei1)
					l2 = neighbors_dist(neighbors, k, nei2)

					#print("candidate!", l1,l2,neighbors_cos(neighbors, k, nei1, nei2))

					if l2 < l1:
						nei1, nei2 = nei2, nei1 

					remove_edge.append((k,nei2))
					remove_edge.append((nei2,k))

					new_edge.append((nei1, nei2))

					impact.append(k)
					impact.append(nei1)
					impact.append(nei2)

					break

	new_neighbors = {}

	def isRemoved(k):
		if k in removed:
			return True 
		else:
			return False

	for k, v in neighbors.items():
		if isRemoved(k): 
			pass
		else:
			new_nei = []
			for nei in v:
				if isRemoved(nei):
					pass 
				elif (nei, k) in remove_edge:
					pass
				else:
					new_nei.append(nei)

			new_neighbors[k] = list(new_nei)

	for new_e in new_edge:
		nk1 = new_e[0]
		nk2 = new_e[1]

		if nk2 not in new_neighbors[nk1]:
			new_neighbors[nk1].append(nk2)
		if nk1 not in new_neighbors[nk2]:
			new_neighbors[nk2].append(nk1)



	#print("remove %d edges" % len(remove_edge))

	return new_neighbors, len(remove_edge)

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
				thr = 5.0
				if d < thr:
					vec = neighbors_norm(graph, n1, n2)
					#vec = (vec[0] * (thr-d), vec[1] * (thr-d))
					
					
					if n1 not in adjustment:
						adjustment[n1] = [vec] 
					else:
						adjustment[n1].append(vec)

				d = distance(ip, n2)
				if d < thr:
					vec = neighbors_norm(graph, n2, n1)
					#vec = (vec[0] * (thr-d), vec[1] * (thr-d))
					
					
					if n2 not in adjustment:
						adjustment[n2] = [vec] 
					else:
						adjustment[n2].append(vec)


				c1 = candidate[0]
				c2 = candidate[1]


				d = distance(ip, c1)
				if d < thr:
					vec = neighbors_norm(graph, c1, c2)
					#vec = (vec[0] * (thr-d), vec[1] * (thr-d))
					
					
					if c1 not in adjustment:
						adjustment[c1] = [vec] 
					else:
						adjustment[c1].append(vec)

				d = distance(ip, c2)
				if d < thr:
					vec = neighbors_norm(graph, c2, c1)
					#vec = (vec[0] * (thr-d), vec[1] * (thr-d))
					
					
					if c2 not in adjustment:
						adjustment[c2] = [vec] 
					else:
						adjustment[c2].append(vec)


	return crossing_point, adjustment


def _vis(_node_neighbors, save_file, size=2048, bk=None, draw_intersection = False):
	node_neighbors = _node_neighbors

	img = np.ones((size, size, 3), dtype=np.uint8) * 255

	color_node = (255,0,0)


	if bk is not None:
		img = scipy.ndimage.imread(bk)

		img = img.astype(np.float)
		img = (img - 127)*0.75 + 127 
		img = img.astype(np.uint8)

		color_edge = (0,255,255) # yellow
	else:
		color_edge = (0,0,0) # black
	
	edge_width = 2
	
	
	for k,v in node_neighbors.iteritems():
		n1 = k 
		for n2 in v:
			cv2.line(img, (n1[1], n1[0]), (n2[1], n2[0]), color_edge,edge_width)

	scale = 1
	
	for k,v in node_neighbors.iteritems():
		n1 = k 
		cv2.circle(img, (int(n1[1]) * scale,int(n1[0]) * scale), 2, (255,0,0),-1)
		

	cp, _ = locate_stacking_road(node_neighbors)


	for k, v in cp.iteritems():
		e1 = k[0]
		e2 = k[1]

		if draw_intersection == True:
			cv2.line(img, (int(e1[0][1]),int(e1[0][0])), (int(e1[1][1]),int(e1[1][0])), (0,255,0),edge_width)
			cv2.line(img, (int(e2[0][1]),int(e2[0][0])), (int(e2[1][1]),int(e2[1][0])), (0,0,255),edge_width)

	
	Image.fromarray(img).save(save_file)


def detect_local_minima(arr, mask, threshold = 0.5):
	# https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
	"""
	Takes an array and detects the troughs using the local maximum filter.
	Returns a boolean mask of the troughs (i.e. 1 when
	the pixel's value is the neighborhood maximum, 0 otherwise)
	"""
	# define an connected neighborhood
	# http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
	neighborhood = morphology.generate_binary_structure(len(arr.shape),2)
	# apply the local minimum filter; all locations of minimum value 
	# in their neighborhood are set to 1
	# http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
	local_min = (filters.minimum_filter(arr, footprint=neighborhood)==arr)
	# local_min is a mask that contains the peaks we are 
	# looking for, but also the background.
	# In order to isolate the peaks we must remove the background from the mask.
	# 
	# we create the mask of the background
	background = (arr==0)
	# 
	# a little technicality: we must erode the background in order to 
	# successfully subtract it from local_min, otherwise a line will 
	# appear along the background border (artifact of the local minimum filter)
	# http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
	eroded_background = morphology.binary_erosion(
		background, structure=neighborhood, border_value=1)
	# 
	# we obtain the final mask, containing only peaks, 
	# by removing the background from the local_min mask
	detected_minima = local_min ^ eroded_background
	return np.where((detected_minima & (mask > threshold)))  
	#return np.where(detected_minima)


def DrawKP(imagegraph, filename, imagesize=256, max_degree=6):
	vertexness = imagegraph[:,:,0].reshape((imagesize, imagesize))

	for i in range(max_degree):
		vertexness = np.maximum(vertexness, imagegraph[:,:,2+4*i].reshape((imagesize, imagesize)))


	kp = np.copy(vertexness)
	smooth_kp = scipy.ndimage.filters.gaussian_filter(np.copy(kp), 1)

	smooth_kp = smooth_kp / max(np.amax(smooth_kp),0.001)

	Image.fromarray((smooth_kp*255.0).astype(np.uint8)).save(filename)



# Main function 
def DecodeAndVis(imagegraph, filename, imagesize=256, max_degree=6, thr=0.5, edge_thr = 0.5, snap=False, kp_limit = 500, drop=True, use_graph_refine=True, testing=False, spacenet = False, angledistance_weight = 100, snap_dist = 15):
	kp_limit = 10000000

	# for training 
	if imagesize < 600:
		kp_limit = 500

	if testing :
		kp_limit = 10000000

	# for visualization 
	if snap :
		rgb = np.zeros((imagesize*4, imagesize*4, 3), dtype=np.uint8)
		rgb2 = np.zeros((imagesize*4, imagesize*4, 3), dtype=np.uint8)

	else:
		rgb = 255 * np.ones((imagesize*4, imagesize*4, 3), dtype=np.uint8)
		rgb2 = 255 * np.ones((imagesize*4, imagesize*4, 3), dtype=np.uint8)


	# Step-1: Find vertices 
	vertexness = imagegraph[:,:,0].reshape((imagesize, imagesize))

	kp = np.copy(vertexness)
	smooth_kp = scipy.ndimage.filters.gaussian_filter(np.copy(kp), 1)
	smooth_kp = smooth_kp / max(np.amax(smooth_kp),0.001)

	keypoints = detect_local_minima(-smooth_kp, smooth_kp, thr)

	cc = 0 


	# locate edge endpoints
	# we do this becasue the local maima may not represent all the vertices
	edgeEndpointMap = np.zeros((imagesize, imagesize))

	for i in range(len(keypoints[0])):
		if cc > kp_limit:
			break 
		cc += 1

		x,y = keypoints[0][i], keypoints[1][i]

		for j in range(max_degree):

			if imagegraph[x,y,2+4*j] * imagegraph[x,y,0] > thr * thr: # or thr < 0.2:
				
				x1 = int(x + vector_norm * imagegraph[x,y,2+4*j+2])
				y1 = int(y + vector_norm * imagegraph[x,y,2+4*j+3])

				if x1 >= 0 and x1 < imagesize and y1 >= 0 and y1 < imagesize:
					edgeEndpointMap[x1,y1] = imagegraph[x,y,2+4*j] * imagegraph[x,y,0]

	edgeEndpointMap = scipy.ndimage.filters.gaussian_filter(edgeEndpointMap, 3)
	edgeEndpoints = detect_local_minima(-edgeEndpointMap, edgeEndpointMap, thr*thr*thr)

	# create rtree index to speed up the queries. 
	idx = index.Index()

	if snap == True:
		cc = 0

		# insert keypoints to the rtree
		for i in range(len(keypoints[0])):
			if cc > kp_limit:
				break 

			x,y = keypoints[0][i], keypoints[1][i]

			idx.insert(i,(x-1,y-1,x+1,y+1))

			cc += 1

		# insert edge endpoints (the other vertex of the edge) to the rtree
		for i in range(len(edgeEndpoints[0])):
			if cc > kp_limit*2:
				break 

			x,y = edgeEndpoints[0][i], edgeEndpoints[1][i]

			candidates = list(idx.intersection((x-5,y-5,x+5,y+5)))

			if len(candidates) == 0:
				idx.insert(i + len(keypoints[0]),(x-1,y-1,x+1,y+1))
			cc += 1


	# endpoint lookup 
	neighbors = {}

	cc = 0
	for i in range(len(keypoints[0])):

		if cc > kp_limit:
			break 

		x,y = keypoints[0][i], keypoints[1][i]

		for j in range(max_degree):
			# imagegraph[x,y,2+4*j] --> edgeness
			# imagegraph[x,y,0] --> vertexness
			if imagegraph[x,y,2+4*j] * imagegraph[x,y,0] > thr*edge_thr and imagegraph[x,y,2+4*j] > edge_thr: 
				
				x1 = int(x + vector_norm * imagegraph[x,y,2+4*j+2])
				y1 = int(y + vector_norm * imagegraph[x,y,2+4*j+3])

				skip = False

				l = vector_norm * np.sqrt(imagegraph[x,y,2+4*j+2]*imagegraph[x,y,2+4*j+2] + imagegraph[x,y,2+4*j+3]*imagegraph[x,y,2+4*j+3])

				if snap==True:
					# Pass-One (restrict distance metric)

					best_candidate = -1 
					min_distance = snap_dist #15.0 

					candidates = list(idx.intersection((x1-20,y1-20,x1+20,y1+20)))

					for candidate in candidates:
						# only snap to keypoints 
						if candidate >= len(keypoints[0]):
							continue

						if candidate < len(keypoints[0]):
							x_c = keypoints[0][candidate]
							y_c = keypoints[1][candidate]
						else:
							x_c = edgeEndpoints[0][candidate-len(keypoints[0])]
							y_c = edgeEndpoints[1][candidate-len(keypoints[0])]

						d = distance((x_c,y_c), (x1,y1))
						if d > l :
							continue 

						# vector from the edge endpoint (the other side of the edge) to the current vertex. 
						v0 = (x - x_c, y - y_c)

						min_sd = angledistance_weight

						for jj in range(max_degree):
							if imagegraph[x_c,y_c,2+4*jj] * imagegraph[x_c,y_c,0] > thr*edge_thr and imagegraph[x,y,2+4*jj] > edge_thr:
								vc = (vector_norm * imagegraph[x_c,y_c,2+4*jj+2], vector_norm * imagegraph[x_c,y_c,2+4*jj+3])

								# cosine distance 
								ad = 1.0 - anglediff(v0,vc)
								ad = ad * angledistance_weight 

								if ad < min_sd:
									min_sd = ad 

						d = d + min_sd


						# cosine distance between the original output edge direction and the edge direction after snapping.  
						v1 = (x_c - x, y_c - y)
						v2 = (x1 - x, y1 - y)
						# cosine distance 
						ad = 1.0 - anglediff(v1,v2) # -1 to 1

						d = d + ad * angledistance_weight # 0.15 --> 15

						if d < min_distance:
							min_distance = d 
							best_candidate = candidate

					# Pass-Two (release the distance metric)
					min_distance = snap_dist #15.0 
					# only need the second pass when there is no good candidate found in the first pass. 
					if best_candidate == -1:
						for candidate in candidates:
							# only snap to keypoints 
							if candidate >= len(keypoints[0]):
								continue

							if candidate < len(keypoints[0]):
								x_c = keypoints[0][candidate]
								y_c = keypoints[1][candidate]
							else:
								x_c = edgeEndpoints[0][candidate-len(keypoints[0])]
								y_c = edgeEndpoints[1][candidate-len(keypoints[0])]

							d = distance((x_c,y_c), (x1,y1))
							if d > l*0.5 :
								continue 

							# cosine distance between the original output edge direction and the edge direction after snapping.  
							v1 = (x_c - x, y_c - y)
							v2 = (x1 - x, y1 - y)

							ad = 1.0 - anglediff(v1,v2) # -1 to 1
							d = d + ad * angledistance_weight * 2 # 0.15 --> 30

							if d < min_distance:
								min_distance = d 
								best_candidate = candidate

					# Pass-Three (release the distance metric even more)
					if best_candidate == -1:
						for candidate in candidates:
							# only snap to edge endpoints 
							if candidate < len(keypoints[0]):
								continue

							if candidate < len(keypoints[0]):
								x_c = keypoints[0][candidate]
								y_c = keypoints[1][candidate]
							else:
								x_c = edgeEndpoints[0][candidate-len(keypoints[0])]
								y_c = edgeEndpoints[1][candidate-len(keypoints[0])]

							d = distance((x_c,y_c), (x1,y1))
							if d > l :
								continue 


							v1 = (x_c - x, y_c - y)
							v2 = (x1 - x, y1 - y)

							ad = 1.0 - anglediff(v1,v2) # -1 to 1
							d = d + ad * angledistance_weight # 0.15 --> 15

							if d < min_distance:
								min_distance = d 
								best_candidate = candidate

					if best_candidate != -1 :
						if best_candidate < len(keypoints[0]):
							x1 = keypoints[0][best_candidate]
							y1 = keypoints[1][best_candidate]
						else:
							x1 = edgeEndpoints[0][best_candidate-len(keypoints[0])]
							y1 = edgeEndpoints[1][best_candidate-len(keypoints[0])]
					else:
						skip = True
							

				# visualization 
				c = int(imagegraph[x,y,2+4*j] * 200.0) + 55
				color = (c,c,c)

				if snap == False: 
					color = (255-c, 255-c, 255-c)

				w = 2

				if skip == False or drop==False:

					nk1 = (x1,y1)
					nk2 = (x,y)

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

					cv2.line(rgb,(y1*4,x1*4), (y*4,x*4), color,w)
		cc += 1


	
	# refine the graph

	spurs_thr = 50 
	isolated_thr = 200

	# spacenet's tiles are small
	if spacenet :
		spurs_thr = 25
		isolated_thr = 100

	if imagesize < 400:
		spurs_thr = 25
		isolated_thr = 100

	if use_graph_refine:
		graph = graph_refine(neighbors, isolated_thr=isolated_thr, spurs_thr=spurs_thr)
		
		_vis(neighbors , filename+"_norefine_bk.png", size=imagesize)

		rc = 100
		while rc > 0:
			if spacenet :
				isolated_thr = 0
				spurs_thr = 0
			
			if imagesize < 400:
				spurs_thr = 25
				isolated_thr = 100

			graph, rc = graph_refine_deloop(graph_refine(graph, isolated_thr=isolated_thr, spurs_thr=spurs_thr))

		if spacenet :
			spurs_thr = 25
			isolated_thr = 100
		
		if imagesize < 400:
			spurs_thr = 25
			isolated_thr = 100

		graph = graph_shave(graph, spurs_thr = spurs_thr)
	else:
		graph = neighbors 

	_vis(graph, filename+"_refine_bk.png", size=imagesize, draw_intersection=True)


	cc = 0

	if snap == False:

		for i in range(len(keypoints[0])):
			if cc > kp_limit:
				break 

			x,y = keypoints[0][i], keypoints[1][i]

			cv2.circle(rgb, (y*4,x*4), 5, (255,0,0), -1)
			cc += 1

			d = 0

			for j in range(max_degree):

				#if imagegraph[x,y,2+4*j] * imagegraph[x,y,0] > thr * thr: # or thr < 0.2:
				if imagegraph[x,y,2+4*j] * imagegraph[x,y,0] > thr*0.5: # or thr < 0.2:
					d += 1

			color = (255,0,0)
			if d == 2:
				color = (0,255,0)
			if d == 3:
				color = (0,128,128)

			if d >= 4:
				color = (0,0,255)

			cv2.circle(rgb2, (y*4,x*4), 8, color, -1)
		


	else:
		for i in range(len(keypoints[0])):
			if cc > kp_limit:
				break 

			x,y = keypoints[0][i], keypoints[1][i]

			cv2.circle(rgb, (y*4,x*4), 5, (255,0,0), -1)
			cc += 1

			d = 0

			for j in range(max_degree):

				#if imagegraph[x,y,2+4*j] * imagegraph[x,y,0] > thr * thr: # or thr < 0.2:
				if imagegraph[x,y,2+4*j] * imagegraph[x,y,0] > thr*0.5: # or thr < 0.2:
					d += 1

			color = (255,0,0)
			if d == 2:
				color = (0,255,0)
			if d == 3:
				color = (0,128,128)

			if d >= 4:
				color = (0,0,255)

			cv2.circle(rgb2, (y*4,x*4), 8, color, -1)
		
		for i in range(len(edgeEndpoints[0])):
			x,y = edgeEndpoints[0][i], edgeEndpoints[1][i]
			cv2.circle(rgb, (y*4,x*4), 3, (0,255,0), -1)



	#ImageGraphVisIntersection(imagegraph, filename, imagesize=imagesize)
	Image.fromarray(rgb).save(filename+"_imagegraph.png")
	Image.fromarray(rgb2).save(filename+"_intersection_node.png")

	pickle.dump(graph, open(filename+"_graph.p","w"))

	return graph

