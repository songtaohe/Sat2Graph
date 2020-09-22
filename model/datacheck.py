import json 
import os  

datafiles = json.load(open("train_prep_RE_18_20_CHN_KZN_250.json"))['data']
basefolder = "/data/songtao/harvardDataset5m/"

trainfiles = []
validfiles = []
bad = 0
for item in datafiles:
	if item[-1] == 'train':
		filepath = basefolder+item[1]
		if os.isfile(filepath):
			bad+= 1

		trainfiles.append(basefolder+item[1].replace(".tif",""))
	elif item[-1] == 'valid':
		filepath = basefolder+item[1]
		if os.isfile(filepath):
			bad+= 1
			
		validfiles.append(basefolder+item[1].replace(".tif",""))

print("train size", len(trainfiles))
print("valid size", len(validfiles))
print("bad files", bad)