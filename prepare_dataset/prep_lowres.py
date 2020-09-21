from subprocess import Popen 
import os 
import sys  
from time import time, sleep 

def run_cmds_parallel(cmds, max_processes = 8):
	pool = []
	cc = 0 
	t0 = time() 
	for cmd in cmds:
		cc += 1

		if cc % 100 == 0:
			t = time()-t0
			print(cc, len(cmds),t, t*(len(cmds)-cc)/100.0)
			t0 = time()


		while len(pool) == max_processes:
			sleep(0.01)
			new_pool = []
			for p in pool:
				if p.poll() is None:
					new_pool.append(p)

			pool = new_pool


		pool.append(Popen(cmd, shell=True))

	for p in pool:
		p.wait()


files = os.listdir(sys.argv[1])

cmds = []
for file in files:
	if file.endswith(".tif"):
		cmd = "python image2graph.py " + sys.argv[1]+"/"+file + " " + sys.argv[1]+"/"+file.replace(".tif",".p")
		cmds.append(cmd)

run_cmds_parallel(cmds)