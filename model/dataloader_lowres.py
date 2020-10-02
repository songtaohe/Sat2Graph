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
import tifffile 


image_size = 256 # original 250 
vector_norm = 25.0 


def neighbor_transpos(n_in):
	n_out = {}

	for k, v in n_in.items():
		nk = (k[1], k[0])
		nv = []

		for _v in v :
			nv.append((_v[1],_v[0]))

		n_out[nk] = nv 

	return n_out 


def rotate(sat_img, gt_seg, neighbors, road_class, angle=0, size=256):
	sat_img = scipy.ndimage.rotate(sat_img, angle, reshape=False)
	gt_seg = scipy.ndimage.rotate(gt_seg, angle, reshape=False)
	road_class = scipy.ndimage.rotate(road_class, angle, reshape=False)

	new_neighbors = {}

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

	return sat_img, gt_seg, new_neighbors, road_class



class Sat2GraphDataLoader():
	def __init__(self, folder, filelist, imgsize = 256, preload_tiles = 100, max_degree = 6, loadseg = False, random_mask=True, testing=False, dataset_image_size = 256, transpose=False):
		self.folder = folder
		self.filelist = filelist
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

		self.input_sat = np.zeros((8,image_size,image_size,5))
		
		self.gt_seg = np.zeros((8,image_size,image_size,1))
		self.gt_class = np.zeros((8,image_size,image_size,1))
		self.target_prob = np.zeros((8,image_size,image_size,2*(max_degree + 1)))
		self.target_vector = np.zeros((8,image_size,image_size,2*(max_degree)))

		self.noise_mask = (np.random.rand(16,16,5) - 0.5) * 0.8


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
		self.noise_mask = (np.random.rand(16,16,5)) * 1.0 + 0.5

		image_size = self.image_size

		tiles = []

		self.tiles_input = np.zeros((self.preload_tiles, self.dataset_image_size, self.dataset_image_size,5))
		self.tiles_gt_seg = np.zeros((self.preload_tiles, self.dataset_image_size, self.dataset_image_size,1))
		self.tiles_prob = np.zeros((self.preload_tiles, self.dataset_image_size, self.dataset_image_size, 2 * (self.max_degree + 1)))
		self.tiles_vector = np.zeros((self.preload_tiles, self.dataset_image_size, self.dataset_image_size, 2 * (self.max_degree)))

		self.tiles_gt_class = np.zeros((self.preload_tiles, self.dataset_image_size, self.dataset_image_size,1))
		
		self.tiles_prob[:,:,:,0::2] = 0
		self.tiles_prob[:,:,:,1::2] = 1


		for i in range(self.preload_tiles):
			filename = random.choice(self.filelist)

			img = tifffile.imread(filename+".tif")

			seg = img[:,:,5]
			seg = (seg >= 1)
			seg = seg.astype(np.float)

			seg = np.pad(seg, ((3,3),(3,3)), 'constant') # 256*256
			seg = seg - 0.5 

			#sat_img = img[:,:,0:5].astype(np.float)/(np.amax(img[:,:,0:5])+1.0) - 0.5 
			sat_img = img[:,:,0:5].astype(np.float)/(16384) - 0.5 

			sat_img = np.pad(sat_img, ((3,3),(3,3),(0,0)), 'constant') # 256*256
			
			neighbors = pickle.load(open(filename+".p"))

			road_class = img[:,:,5]
			road_class = np.pad(road_class, ((3,3),(3,3)), 'constant')


			angle = random.randint(0,3)*90 + random.random()*40-20

			if random.random() > 0.5:
				sat_img, seg, neighbors, road_class = rotate(sat_img, seg, neighbors, road_class, angle=angle)

			road_class = np.clip(road_class.astype(np.int32), 0, 4)

			self.tiles_input[i,:,:,:] = sat_img
			self.tiles_gt_seg[i,:,:,0] = seg
			self.tiles_gt_class[i,:,:,0] = road_class

			
			
			if self.transpose:
				neighbors = neighbor_transpos(neighbors)

			r = 1
			buffer = 3


			for loc, n_locs in neighbors.items():
				bx,by = 3, 3

				if loc[0]+bx < buffer or loc[1]+by < buffer or loc[0]+bx > self.dataset_image_size - buffer or loc[1]+by > self.dataset_image_size - buffer:
					continue

				

				self.tiles_prob[i,int(loc[0])+bx,int(loc[1])+by,0] = 1
				self.tiles_prob[i,int(loc[0])+bx,int(loc[1])+by,1] = 0

				for x in range(int(loc[0])-r, int(loc[0])+r+1):
					for y in range(int(loc[1])-r, int(loc[1])+r+1):

						self.tiles_prob[i,x+bx,y+by,0] = 1
						self.tiles_prob[i,x+bx,y+by,1] = 0


				for n_loc in n_locs:

					if n_loc[0]+bx < buffer or n_loc[1]+by < buffer or n_loc[0]+bx > self.dataset_image_size - buffer or n_loc[1]+by > self.dataset_image_size - buffer :
						continue

					d = math.atan2(n_loc[1] - loc[1], n_loc[0] - loc[0]) + math.pi 
					j = int(d/(math.pi/3.0)) % self.max_degree

					for x in range(int(loc[0])-r, int(loc[0])+r+1):
						for y in range(int(loc[1])-r, int(loc[1])+r+1):
							self.tiles_prob[i,x+bx,y+by,2+2*j] = 1
							self.tiles_prob[i,x+bx,y+by,2+2*j+1] = 0

							self.tiles_vector[i,x+bx,y+by,2*j] = (n_loc[0] - loc[0]) / vector_norm
							self.tiles_vector[i,x+bx,y+by,2*j+1] = (n_loc[1] - loc[1]) / vector_norm


			# random rotation augmentation 
			if self.testing == False:
				self.tiles_input[i,:,:,:] = self.tiles_input[i,:,:,:] * (0.8 + 0.2 * random.random()) - (random.random() * 0.4 - 0.2)
				self.tiles_input[i,:,:,:] = np.clip(self.tiles_input[i,:,:,:], -0.5, 0.5)

				self.tiles_input[i,:,:,0] = self.tiles_input[i,:,:,0] * (0.8 + 0.2 * random.random())
				self.tiles_input[i,:,:,1] = self.tiles_input[i,:,:,1] * (0.8 + 0.2 * random.random())
				self.tiles_input[i,:,:,2] = self.tiles_input[i,:,:,2] * (0.8 + 0.2 * random.random())
				self.tiles_input[i,:,:,3] = self.tiles_input[i,:,:,3] * (0.8 + 0.2 * random.random())
				self.tiles_input[i,:,:,4] = self.tiles_input[i,:,:,4] * (0.8 + 0.2 * random.random())


	def getBatch(self, batchsize = 4, st = None):
		
		image_size = self.image_size

		for i in range(batchsize):
			c = 0
			
			tile_id = random.randint(0,self.preload_tiles-1)

			self.input_sat[i,:,:,:] = self.tiles_input[tile_id, :, :, :]
			
			if random.randint(0,100) < 50 and self.random_mask==True:

				# add noise
				for it in range(random.randint(1,5)):
					xx = random.randint(0,image_size-16-1)
					yy = random.randint(0,image_size-16-1)

					self.input_sat[i,xx:xx+16,yy:yy+16,:] =  np.multiply(self.input_sat[i,xx:xx+16,yy:yy+16,:] + 0.5, self.noise_mask) - 0.5
				
				# add more noise 
				for it in range(random.randint(1,3)):
					xx = random.randint(0,image_size-16-1)
					yy = random.randint(0,image_size-16-1)

					self.input_sat[i,xx:xx+16,yy:yy+16,:] =  (self.noise_mask - 1.0) 
				
			self.target_prob[i,:,:,:] = self.tiles_prob[tile_id,:,:,:]
			self.target_vector[i,:,:,:] = self.tiles_vector[tile_id, :,:,:]
			self.gt_seg[i,:,:,:] = self.tiles_gt_seg[tile_id,:,:,:]
			self.gt_class[i,:,:,:] = self.tiles_gt_class[tile_id,:,:,:]
			

		st = 0

		return self.input_sat[st:st+batchsize,:,:,:], self.target_prob[st:st+batchsize,:,:,:], self.target_vector[st:st+batchsize,:,:,:], self.gt_seg[st:st+batchsize,:,:,:], self.gt_class[st:st+batchsize,:,:,:]
		