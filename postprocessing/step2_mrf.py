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
#for tilename in ['4358815_2018-08-27_RE4_3A']:
	print(cc, tilename)
	cmd = ""
	
	graphfile = in_folder + "/" + tilename+ "__seggraph.p"
	classfile = in_folder + "/" + tilename+ "__segclass.p"
	outputclass = in_folder + "/" + tilename+ "__segclassMRF.p"
	outputclassvis = in_folder + "/" + tilename+ "__segclassMRFvis.png"
	
	
	cmd += "python mrf.py "+graphfile+" "+classfile+" " + outputclass + ";"
	cmd += "python vis_node_graph.py " + graphfile+" "+outputclass+" " + outputclassvis+""

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





