import scipy 
import numpy as np 
import random 
import pickle 
import math 
from PIL import Image 
import json 
import scipy.ndimage 
import math 
import cv2


image_size = 256 
vector_norm = 25.0 

# rotate 
# (1) sat_img
# (2) gt_seg 
# (3) neighbors
# (4) sample point
# (5) sample mask?

# angle is in degrees 
def rotate(sat_img, gt_seg, neighbors, samplepoints, angle=0, size=2048):

	mask = np.zeros(np.shape(gt_seg))

	mask[256:size-256,256:size-256] = 1


	sat_img = scipy.ndimage.rotate(sat_img, angle, reshape=False)
	gt_seg = scipy.ndimage.rotate(gt_seg, angle, reshape=False)

	mask = scipy.ndimage.rotate(mask, angle, reshape=False)


	new_neighbors = {}
	new_samplepoints = {}

	def transfer(pos, angle):
		x = pos[0] - int(size/2)
		y = pos[1] - int(size/2)

		new_x = x * math.cos(math.radians(angle)) - y * math.sin(math.radians(angle))
		new_y = x * math.sin(math.radians(angle)) + y * math.cos(math.radians(angle))

		return (int(new_x + int(size/2)), int(new_y + int(size/2)))

	def inrange(pos, m):
		if pos[0] > m and pos[0] < size-1-m and pos[1]>m and pos[1]<size-1-m:
			return True
		else:
			return False

	for k,n in neighbors.items():
		nk = transfer(k, angle)

		if inrange(nk,0) == False:
			continue

		new_neighbors[nk] = []

		for nei in n:
			nn = transfer(nei, angle) 
			if inrange(nn,0):
				new_neighbors[nk].append(nn)


	for k,vs in samplepoints.items():

		new_samplepoints[k] = []

		for v in vs:
			nv = transfer(v, angle)

			if inrange(nv, 256):
				new_samplepoints[k].append(nv)

	return sat_img, gt_seg, new_neighbors, new_samplepoints, mask



def neighbor_transpos(n_in):
	n_out = {}

	for k, v in n_in.items():
		nk = (k[1], k[0])
		nv = []

		for _v in v :
			nv.append((_v[1],_v[0]))

		n_out[nk] = nv 

	return n_out 

def neighbor_to_integer(n_in):
	n_out = {}

	for k, v in n_in.items():
		nk = (int(k[0]), int(k[1]))
		
		if nk in n_out:
			nv = n_out[nk]
		else:
			nv = []

		for _v in v :
			new_n_k = (int(_v[0]),int(_v[1]))

			if new_n_k in nv:
				pass
			else:
				nv.append(new_n_k)

		n_out[nk] = nv 

	return n_out



class Sat2GraphDataLoader():
	def __init__(self, folder, indrange = [0,10],imgsize = 256, preload_tiles = 4, max_degree = 6, loadseg = False, random_mask=True, testing=False, dataset_image_size = 2048, transpose=False):
		self.folder = folder
		self.indrange = indrange
		self.random_mask = random_mask 
		self.dataset_image_size = dataset_image_size
		self.transpose = transpose 

		self.preload_tiles = preload_tiles
		self.max_degree = max_degree
		self.num = 0
		self.loadseg = loadseg
	
		self.image_size = imgsize 
		self.testing = testing
		global image_size 
		image_size = imgsize 

		self.input_sat = np.zeros((8,image_size,image_size,3))
		
		self.gt_seg = np.zeros((8,image_size,image_size,1))
		self.target_prob = np.zeros((8,image_size,image_size,2*(max_degree + 1)))
		self.target_vector = np.zeros((8,image_size,image_size,2*(max_degree)))

		self.noise_mask = (np.random.rand(64,64,3) - 0.5) * 0.8


		random.seed(1)


	def loadtile(self, ind):
		
		try:
			sat_img = scipy.ndimage.imread(self.folder + "/region_%d_sat.png" % ind).astype(np.float)
		except:
			sat_img = scipy.ndimage.imread(self.folder + "/region_%d_sat.jpg" % ind).astype(np.float)
					
		max_v = np.amax(sat_img) + 0.0001 

		sat_img = (sat_img.astype(np.float)/ max_v - 0.5) * 0.9 

		sat_img = sat_img.reshape((1,self.dataset_image_size,self.dataset_image_size,3))

		#Image.fromarray(((sat_img[0,:,:,:] + 0.5) * 255.0).astype(np.uint8)).save("outputs/test.png")

		#print(np.shape(sat_img))
		

		tiles_prob = np.zeros((1,self.dataset_image_size, self.dataset_image_size, 2 * (self.max_degree + 1)))
		tiles_vector = np.zeros((1, self.dataset_image_size, self.dataset_image_size, 2 * (self.max_degree)))

		tiles_prob[:,:,:,0::2] = 0
		tiles_vector[:,:,:,1::2] = 1
	
		try:
			neighbors = pickle.load(open(self.folder + "/region_%d_refine_gt_graph.p" % ind))
			neighbors = neighbor_to_integer(neighbors)

			if self.transpose:
				neighbors = neighbor_transpos(neighbors)

			r = 1
			i = 0

			#tiles_angle = np.zeros((self.dataset_image_size, self.dataset_image_size, 1), dtype=np.uint8)
				
			for loc, n_locs in neighbors.items():
				if loc[0] < 16 or loc[1] < 16 or loc[0] > self.dataset_image_size - 16 or loc[1] > self.dataset_image_size - 16 :
					continue

				
				tiles_prob[i,loc[0],loc[1],0] = 1
				tiles_prob[i,loc[0],loc[1],1] = 0

				for x in range(loc[0]-r, loc[0]+r+1):
					for y in range(loc[1]-r, loc[1]+r+1):
						tiles_prob[i,x,y,0] = 1
						tiles_prob[i,x,y,1] = 0


				for n_loc in n_locs:
					if n_loc[0] < 16 or n_loc[1] < 16 or n_loc[0] > self.dataset_image_size - 16 or n_loc[1] > self.dataset_image_size - 16 :
						continue
					d = math.atan2(n_loc[1] - loc[1], n_loc[0] - loc[0]) + math.pi 

					j = int(d/(math.pi/3.0)) % self.max_degree

					for x in range(loc[0]-r, loc[0]+r+1):
						for y in range(loc[1]-r, loc[1]+r+1):
							tiles_prob[i,x,y,2+2*j] = 1
							tiles_prob[i,x,y,2+2*j+1] = 0

							tiles_vector[i,x,y,2*j] = (n_loc[0] - loc[0])/vector_norm
							tiles_vector[i,x,y,2*j+1] = (n_loc[1] - loc[1])/vector_norm
		except:
			pass

		return sat_img, tiles_prob, tiles_vector


	def preload(self, num = 1024, seg_only = False):
		self.noise_mask = (np.random.rand(64,64,3)) * 1.0 + 0.5

		image_size = self.image_size

		tiles = []

		
		self.tiles_input = np.zeros((self.preload_tiles, self.dataset_image_size, self.dataset_image_size,3))
		
		self.tiles_gt_seg = np.zeros((self.preload_tiles, self.dataset_image_size, self.dataset_image_size,1))
		
		self.tiles_prob = np.zeros((self.preload_tiles, self.dataset_image_size, self.dataset_image_size, 2 * (self.max_degree + 1)))
		self.tiles_vector = np.zeros((self.preload_tiles, self.dataset_image_size, self.dataset_image_size, 2 * (self.max_degree)))

		
		self.tiles_prob[:,:,:,0::2] = 0
		self.tiles_prob[:,:,:,1::2] = 1

		self.rotmask = np.ones((self.preload_tiles, self.dataset_image_size, self.dataset_image_size))

		self.samplepoints = []

		for i in range(self.preload_tiles):
			ind = random.choice(self.indrange)

			# load sample points 
			samplepoints = json.load(open(self.folder + "/region_%d_refine_gt_graph_samplepoints.json" % ind,"r"))
			self.samplepoints.append(samplepoints)

			
			
			try:
				sat_img = scipy.ndimage.imread(self.folder + "/region_%d_sat.png" % ind)
			except:
				sat_img = scipy.ndimage.imread(self.folder + "/region_%d_sat.jpg" % ind)

			max_v = np.amax(sat_img) + 0.0001 

			# rotate 
			# (1) sat_img
			# (2) gt_seg 
			# (3) neighbors
			# (4) sample point
			# (5) sample mask?


			neighbors = pickle.load(open(self.folder + "/region_%d_refine_gt_graph.p" % ind))
			neighbors = neighbor_to_integer(neighbors)

			if self.transpose:
				neighbors = neighbor_transpos(neighbors)

			gt_seg = scipy.ndimage.imread(self.folder + "/region_%d_gt.png" % ind)

			self.rotmask[i,:,:] = np.ones((self.dataset_image_size, self.dataset_image_size))

			# random rotation augmentation 
			if self.testing == False and random.randint(0,5) < 4:
				angle = random.randint(0,3) * 90 + random.randint(-30,30)
				sat_img, gt_seg, neighbors, samplepoints, rotmask = rotate(sat_img, gt_seg, neighbors, samplepoints, angle= angle, size = self.dataset_image_size)
				self.rotmask[i,:,:] = rotmask


			self.tiles_input[i,:,:,:] = sat_img.astype(np.float)/ max_v - 0.5
			self.tiles_gt_seg[i,:,:,0] = gt_seg.astype(np.float)/255.0 - 0.5
			
			
			r = 1

			for loc, n_locs in neighbors.items():

				if loc[0] < 16 or loc[1] < 16 or loc[0] > self.dataset_image_size - 16 or loc[1] > self.dataset_image_size - 16 :
					continue

				
				self.tiles_prob[i,loc[0],loc[1],0] = 1
				self.tiles_prob[i,loc[0],loc[1],1] = 0

				
				for x in range(loc[0]-r, loc[0]+r+1):
					for y in range(loc[1]-r, loc[1]+r+1):

						self.tiles_prob[i,x,y,0] = 1
						self.tiles_prob[i,x,y,1] = 0


				for n_loc in n_locs:

					if n_loc[0] < 16 or n_loc[1] < 16 or n_loc[0] > self.dataset_image_size - 16 or n_loc[1] > self.dataset_image_size - 16 :
						continue

					d = math.atan2(n_loc[1] - loc[1], n_loc[0] - loc[0]) + math.pi 
					j = int(d/(math.pi/3.0)) % self.max_degree

					for x in range(loc[0]-r, loc[0]+r+1):
						for y in range(loc[1]-r, loc[1]+r+1):
							self.tiles_prob[i,x,y,2+2*j] = 1
							self.tiles_prob[i,x,y,2+2*j+1] = 0

							self.tiles_vector[i,x,y,2*j] = (n_loc[0] - loc[0]) / vector_norm
							self.tiles_vector[i,x,y,2*j+1] = (n_loc[1] - loc[1]) / vector_norm


			# random rotation augmentation 
			if self.testing == False:
				self.tiles_input[i,:,:,:] = self.tiles_input[i,:,:,:] * (0.8 + 0.2 * random.random()) - (random.random() * 0.4 - 0.2)
				self.tiles_input[i,:,:,:] = np.clip(self.tiles_input[i,:,:,:], -0.5, 0.5)

				self.tiles_input[i,:,:,0] = self.tiles_input[i,:,:,0] * (0.8 + 0.2 * random.random())
				self.tiles_input[i,:,:,1] = self.tiles_input[i,:,:,1] * (0.8 + 0.2 * random.random())
				self.tiles_input[i,:,:,2] = self.tiles_input[i,:,:,2] * (0.8 + 0.2 * random.random())


	def getBatch(self, batchsize = 64, st = None):
		
		image_size = self.image_size

		for i in range(batchsize):
			c = 0
			while True:
				tile_id = random.randint(0,self.preload_tiles-1)

				coin = random.randint(0,99)

				if coin < 20: # 20%
					while True:
						
						x = random.randint(256, self.dataset_image_size-256-image_size)
						y = random.randint(256, self.dataset_image_size-256-image_size)


						if self.rotmask[tile_id,x,y] > 0.5:
							break

				elif coin < 40: # complicated intersections 
					sps =  self.samplepoints[tile_id]['complicated_intersections']

					if len(sps) == 0:
						c += 1 
						continue

					ind = random.randint(0, len(sps)-1)

					x = sps[ind][0] - image_size/2
					y = sps[ind][1] - image_size/2

					x = np.clip(x, 256, self.dataset_image_size-256-image_size)
					y = np.clip(y, 256, self.dataset_image_size-256-image_size)
					
				elif coin < 60: # parallel roads 
					sps =  self.samplepoints[tile_id]['parallel_road']

					if len(sps) == 0:
						c += 1 
						continue

					ind = random.randint(0, len(sps)-1)

					x = sps[ind][0] - image_size/2
					y = sps[ind][1] - image_size/2

					x = np.clip(x, 256, self.dataset_image_size-256-image_size)
					y = np.clip(y, 256, self.dataset_image_size-256-image_size)

				else: # overpass
					sps =  self.samplepoints[tile_id]['overpass']

					if len(sps) == 0:
						c += 1 
						continue

					ind = random.randint(0, len(sps)-1)

					x = sps[ind][0] - image_size/2
					y = sps[ind][1] - image_size/2

					x = np.clip(x, 256, self.dataset_image_size-256-image_size)
					y = np.clip(y, 256, self.dataset_image_size-256-image_size)

				x = int(x)
				y = int(y)


				c += 1
				if np.sum(self.tiles_gt_seg[tile_id,x:x+image_size, y:y+image_size,:]+0.5) < 20*20 and c < 10:
					continue

				self.input_sat[i,:,:,:] = self.tiles_input[tile_id, x:x+image_size, y:y+image_size,:]
				
				if random.randint(0,100) < 50 and self.random_mask==True:

					# add noise
					for it in range(random.randint(1,5)):
						xx = random.randint(0,image_size-64-1)
						yy = random.randint(0,image_size-64-1)

						self.input_sat[i,xx:xx+64,yy:yy+64,:] =  np.multiply(self.input_sat[i,xx:xx+64,yy:yy+64,:] + 0.5, self.noise_mask) - 0.5
					
					# add more noise 
					for it in range(random.randint(1,3)):
						xx = random.randint(0,image_size-64-1)
						yy = random.randint(0,image_size-64-1)

						self.input_sat[i,xx:xx+64,yy:yy+64,:] =  (self.noise_mask - 1.0) 
					

				self.target_prob[i,:,:,:] = self.tiles_prob[tile_id, x:x+image_size, y:y+image_size,:]
				self.target_vector[i,:,:,:] = self.tiles_vector[tile_id, x:x+image_size, y:y+image_size,:]
				self.gt_seg[i,:,:,:] = self.tiles_gt_seg[tile_id,x:x+image_size, y:y+image_size,:]

				break

		st = 0

		return self.input_sat[st:st+batchsize,:,:,:], self.target_prob[st:st+batchsize,:,:,:], self.target_vector[st:st+batchsize,:,:,:], self.gt_seg[st:st+batchsize,:,:,:]
		