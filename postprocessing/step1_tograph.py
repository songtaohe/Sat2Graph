import sys 
from subprocess import Popen 
import os  
from time import time, sleep 

in_folder = sys.argv[1]
out_folder = sys.argv[2]

tilenames = []
for file in os.listdir(in_folder):
	items = file.split("_")
	if len(items) < 4:
		continue

	name = items[0]+"_"+items[1]+"_"+items[2]+"_"+items[3]

	if name not in tilenames:
		tilenames.append(name)

print(tilenames)
print(len(tilenames))

# convert segmentation to graph 

pool = []
maxp = 32
cc = 0

for tilename in tilenames:
	print(cc, tilename)
	cmd = ""
	segfile = in_folder + "/" + tilename+ "__segmentation.png"
	classfile = in_folder + "/" + tilename+ "__class.png"
	outputgraph = in_folder + "/" + tilename+ "__seggraph.p"
	outputclass = in_folder + "/" + tilename+ "__segclass.p"
	outputclassvis = in_folder + "/" + tilename+ "__segclassvis.png"
	
	cmd += "python segmentation2graph.py "+segfile+" "+outputgraph+";"
	cmd += "python get_edge_class.py " + outputgraph+" "+classfile+" " + outputclass+";"
	cmd += "python vis_node_graph.py " + outputgraph+" "+outputclass+" " + outputclassvis+""

	while len(pool) > maxp:
		new_pool = []
		for p in pool:
			if p.poll() is None:
				new_pool.append(p)

		pool = new_pool
		sleep(1.0)

	pool.append(Popen(cmd, shell=True))

	cc += 1 

for p in pool:
	p.wait()





