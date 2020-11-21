import graph_ops 
import sys  
import os 
from subprocess import Popen
import tifffile
from PIL import Image 
import pickle 

skysat_folder = "../../data/SkySat_Sat2Graph/"
sat2graph_folder = "../../data/dataset/"

Popen("mkdir -p "+sat2graph_folder, shell=True).wait()

regionname = []

for filename in sorted(os.listdir(skysat_folder+"Sat2Graph_Satellite_Images/")):
	if filename.endswith(".tif"):
		regionname.append(filename.replace(".tif",""))

print(regionname)

for rid in range(len(regionname)):
	if rid % 100 == 0:
		print(rid)
	# convert image to png 
	img = tifffile.imread(skysat_folder+"Sat2Graph_Satellite_Images/"+ regionname[rid]+".tif")
	rgb = img[:,:,0:3]
	Image.fromarray(rgb).save(sat2graph_folder+"/region_%d_sat.png" % rid)


	# interpolate graph to 20-pixel interval
	graph = pickle.load(open(skysat_folder+"Sat2Graph_Graphs/"+regionname[rid] + "_refine_gt_graph.p"))
	graph = graph_ops.graphDensify(graph, density=20, distFunc=graph_ops.PixelDistance)
	pickle.dump(graph, open(sat2graph_folder+"/region_%d_refine_gt_graph.p" % rid, "w"))


	# copy other files 
	Popen("cp "+skysat_folder+"Sat2Graph_Graphs/"+regionname[rid] + "_gt.png "+ sat2graph_folder+"/region_%d_refine_gt_graph.p" % rid, shell=True)
	Popen("cp "+skysat_folder+"Sat2Graph_Graphs/"+regionname[rid] + "_gt_graph_samplepoints.json "+ sat2graph_folder+"/region_%d_gt_graph_samplepoints.json" % rid, shell=True)

