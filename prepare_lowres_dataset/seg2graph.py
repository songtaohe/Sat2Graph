from PIL import Image 
import sys  
import tifffile 
import numpy as np 

img = tifffile.imread(sys.argv[1])
print(np.shape(img))
