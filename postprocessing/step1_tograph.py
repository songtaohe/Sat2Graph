import sys 
from subprocess import Popen 
import os  

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

