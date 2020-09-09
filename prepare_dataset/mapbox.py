import math 
import numpy as np 
import os 
import scipy.ndimage
from PIL import Image 
from subprocess import Popen 
from time import time, sleep 

def lonlat2mapboxTile(lonlat, zoom):
	n = np.exp2(zoom)
	x = int((lonlat[0] + 180)/360*n)
	y = int((1 - math.log(math.tan(lonlat[1] * math.pi / 180) + (1 / math.cos(lonlat[1] * math.pi / 180))) / math.pi) / 2 * n)

	return [x,y]

def lonlat2TilePos(lonlat, zoom):
	n = np.exp2(zoom)
	ix = int((lonlat[0] + 180)/360*n)
	iy = int((1 - math.log(math.tan(lonlat[1] * math.pi / 180) + (1 / math.cos(lonlat[1] * math.pi / 180))) / math.pi) / 2 * n)

	x = ((lonlat[0] + 180)/360*n)
	y = ((1 - math.log(math.tan(lonlat[1] * math.pi / 180) + (1 / math.cos(lonlat[1] * math.pi / 180))) / math.pi) / 2 * n)

	x = int((x - ix) * 512)
	y = int((y - iy) * 512)

	return x,y

def downloadMapBox(zoom, p, outputname):
	url = "https://c.tiles.mapbox.com/v4/mapbox.satellite/%d/%d/%d@2x.jpg?access_token=pk.eyJ1Ijoib3BlbnN0cmVldG1hcCIsImEiOiJjaml5MjVyb3MwMWV0M3hxYmUzdGdwbzE4In0.q548FjhsSJzvXsGlPsFxAQ" % (zoom, p[0], p[1])
	filename = "%d@2x.jpg?access_token=pk.eyJ1Ijoib3BlbnN0cmVldG1hcCIsImEiOiJjaml5MjVyb3MwMWV0M3hxYmUzdGdwbzE4In0.q548FjhsSJzvXsGlPsFxAQ" % (p[1])

	Succ = False

	print(outputname)
	retry_timeout = 10

	while Succ != True :
		Popen("gtimeout 30s wget "+url, shell = True).wait()
		Popen("timeout 30s wget "+url, shell = True).wait()
		Succ = os.path.isfile(filename) 
		Popen("mv \""+filename+"\" "+outputname, shell=True).wait()
		if Succ != True:
			sleep(retry_timeout)
			retry_timeout += 10
			if retry_timeout > 60:
				retry_timeout = 60

			print("Retry, timeout is ", retry_timeout)

	return Succ



def GetMapInRect(min_lat,min_lon, max_lat, max_lon , folder = "mapbox_cache/", start_lat = 42.1634, start_lon = -71.36, resolution = 1024, padding = 128, zoom = 19, scale = 2):
	mapbox1 = lonlat2mapboxTile([min_lon, min_lat], zoom)
	mapbox2 = lonlat2mapboxTile([max_lon, max_lat], zoom)

	ok = True

	print(mapbox1, mapbox2)

	print((mapbox2[0] - mapbox1[0])*(mapbox1[1] - mapbox2[1]))

	dimx = (mapbox2[0] - mapbox1[0]+1) * 512 # lon
	dimy = (mapbox1[1] - mapbox2[1]+1) * 512 # lat 

	img = np.zeros((dimy, dimx, 3), dtype = np.uint8)

	for i in range(mapbox2[0] - mapbox1[0]+1):
		if ok == False:
			break 

		for j in range(mapbox1[1] - mapbox2[1]+1):
			filename = folder + "/%d_%d_%d.jpg" % (zoom, i+mapbox1[0], j+mapbox2[1])
			Succ = os.path.isfile(filename)

			if Succ == True:
				try:
					subimg = scipy.ndimage.imread(filename).astype(np.uint8)
				except:
					print("image file is damaged, try to redownload it", filename)
					Succ = False

			if Succ == False:
				Succ = downloadMapBox(zoom, [i+mapbox1[0],j+mapbox2[1]], filename)

			if Succ:
				subimg = scipy.ndimage.imread(filename).astype(np.uint8)
				img[j*512:(j+1)*512, i*512:(i+1)*512,:] = subimg


			else:
				ok = False
				break


	x1,y1 = lonlat2TilePos([min_lon, max_lat], zoom)
	x2,y2 = lonlat2TilePos([max_lon, min_lat], zoom)

	x2 = x2 + dimx-512
	y2 = y2 + dimy-512

	img = img[y1:y2,x1:x2]

	return img, ok 

# img, ok = GetMapInRect(45.49066, -122.708558, 45.509092018432014, -122.68226506517134, start_lat = 45.49066, start_lon = -122.708558, zoom=16)

# Image.fromarray(img).save("mapboxtmp.png")



# https://c.tiles.mapbox.com/v4/mapbox.satellite/15/5264/11434@2x.jpg?access_token=pk.eyJ1Ijoib3BlbnN0cmVldG1hcCIsImEiOiJjaml5MjVyb3MwMWV0M3hxYmUzdGdwbzE4In0.q548FjhsSJzvXsGlPsFxAQ

