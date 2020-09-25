import json 
import os  
import sys    

datafiles = json.load(open(sys.argv[1]))['data']
basefolder = "/data/songtao/harvardDataset5m/"

trainfiles = []
validfiles = []
testfiles = []

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
	elif item[-1] == 'test':
		filepath = basefolder+item[1]
		if os.path.isfile(filepath):
			bad+= 1

		testfiles.append(basefolder+item[1].replace(".tif",""))

testfiles = sorted(testfiles)
#print(testfiles)

region_count = {}

for file in trainfiles:
	items = file.split("Analytic")
	if items[0] in region_count:
		region_count[items[0]].append(items[1])
	else:
		region_count[items[0]] = [items[1]]

for k in region_count.keys():
	print(k, len(region_count[k]))

	







print("train size", len(trainfiles))
print("valid size", len(validfiles))
print("bad files", bad)


