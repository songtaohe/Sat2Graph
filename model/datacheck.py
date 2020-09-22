import json 
import os  
import sys    

datafiles = json.load(open(sys.argv[1]))['data']
basefolder = "/data/songtao/harvardDataset5m/"

trainfiles = []
validfiles = []
bad = 0
for item in datafiles:
	if item[-1] == 'train':
		filepath = basefolder+item[1]
		if os.path.isfile(filepath):
			bad+= 1

		trainfiles.append(basefolder+item[1].replace(".tif",""))
	elif item[-1] == 'valid':
		filepath = basefolder+item[1]
		if os.path.isfile(filepath):
			bad+= 1

		validfiles.append(basefolder+item[1].replace(".tif",""))

print("train size", len(trainfiles))
print("valid size", len(validfiles))
print("bad files", bad)