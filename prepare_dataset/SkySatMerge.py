import graph_ops 
import sys  
import os 
from subprocess import Popen
import tifffile
from PIL import Image 
import pickle 
import numpy as np 
import scipy.misc
import scipy.ndimage as nd 
import cv2 

skysat_folder = "../../data/SkySat_Sat2Graph/"
output_folder = "../../data/SkySatRegions/"


skysat_folder = sys.argv[1]
output_folder = sys.argv[2]

Popen("mkdir -p "+output_folder, shell=True).wait()

tilename = []
regionname = []


for filename in sorted(os.listdir(skysat_folder+"/")):
	if filename.endswith(".tif"):
		tilename.append(filename.replace(".tif",""))
        name = filename.replace(".tif","")[:-2]
        if name not in regionname:
            regionname.append(name)


for i in range(len(regionname)):
    print(i, regionname[i])
    regionimg = np.zeros((4096, 4096,3), dtype=np.uint8)
    gtimg = np.zeros((4096, 4096), dtype=np.uint8)
    try:
        regionimg[0:2048,0:2048] = tifffile.imread(skysat_folder+"/"+ regionname[i]+ "_0.tif")[:,:,0:3] 
        regionimg[2048:4096,0:2048] = tifffile.imread(skysat_folder+"/"+ regionname[i]+ "_1.tif")[:,:,0:3] 
        regionimg[0:2048,2048:4096] = tifffile.imread(skysat_folder+"/"+ regionname[i]+ "_2.tif")[:,:,0:3] 
        regionimg[2048:4096,2048:4096] = tifffile.imread(skysat_folder+"/"+ regionname[i]+ "_3.tif")[:,:,0:3] 
        
        # regionimg[0:2048,0:2048] = tifffile.imread(skysat_folder+"Sat2Graph_Satellite_Images/"+ regionname[i]+ "_0.tif")[:,:,0:3] 
        # regionimg[2048:4096,0:2048] = tifffile.imread(skysat_folder+"Sat2Graph_Satellite_Images/"+ regionname[i]+ "_1.tif")[:,:,0:3] 
        # regionimg[0:2048,2048:4096] = tifffile.imread(skysat_folder+"Sat2Graph_Satellite_Images/"+ regionname[i]+ "_2.tif")[:,:,0:3] 
        # regionimg[2048:4096,2048:4096] = tifffile.imread(skysat_folder+"Sat2Graph_Satellite_Images/"+ regionname[i]+ "_3.tif")[:,:,0:3] 
        
        # gtimg[0:2048,0:2048] = nd.imread(skysat_folder+"Sat2Graph_Graphs/"+ regionname[i]+ "_0_gt.png")[:,:] 
        # gtimg[2048:4096,0:2048] = nd.imread(skysat_folder+"Sat2Graph_Graphs/"+ regionname[i]+ "_1_gt.png")[:,:] 
        # gtimg[0:2048,2048:4096] = nd.imread(skysat_folder+"Sat2Graph_Graphs/"+ regionname[i]+ "_2_gt.png")[:,:] 
        # gtimg[2048:4096,2048:4096] = nd.imread(skysat_folder+"Sat2Graph_Graphs/"+ regionname[i]+ "_3_gt.png")[:,:] 
        
    except:
        print("wrong...", i, regionname[i] )

    # Image.fromarray(gtimg).save(output_folder+"/" + regionname[i]+ "_gt.png")

    # sat = nd.imread(output_folder+"/" + regionname[i]+ ".png").astype(np.float)
    # sat[:,:,0] += gtimg
    # sat[:,:,0] = np.clip(sat[:,:,0], 0, 255)
    # sat = sat.astype(np.uint8)

    # Image.fromarray(sat).save(output_folder+"/" + regionname[i]+ "_gt_sat.png")

    Image.fromarray(regionimg).save(output_folder+"/" + regionname[i]+ ".png")
    regionimg = scipy.misc.imresize(regionimg, (2458, 2458))
    Image.fromarray(regionimg).save(output_folder+"/" + regionname[i]+ "_1m.png")




