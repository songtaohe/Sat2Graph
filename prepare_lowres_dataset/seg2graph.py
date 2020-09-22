from PIL import Image 
import sys  
import tifffile 
import numpy as np 

img = tifffile.imread(sys.argv[1])
print(np.shape(img))

print(np.amax(img[:,:,5]))


im = np.zeros((250,250), dtype=np.uint8)

im[:,:] = img[:,:,5] * 60

for i in range(5):
	print(i, np.amax(img[:,:,i]))


Image.fromarray(im).save("tmp.png") 