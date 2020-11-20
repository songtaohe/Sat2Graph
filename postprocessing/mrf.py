import pickle
import numpy as np 
import cv2 
import sys 
import opengm
from common import neighbors_cos 

# 4358815_2018-08-27_RE4_3A



#python mrf.py /data/songtao/harvardDataset5mTestingResults2/4358815_2018-08-27_RE4_3A__


weight = 100.0 

nodeneighbor = pickle.load(open(sys.argv[1]))
edgeClass = pickle.load(open(sys.argv[2]))
output_fn = sys.argv[3]


edgeid = 0 
edges = {}
ids2edge = {}
edgelinks = {}

def getEdgeId(edge):
	global edgeid 
	if edge in edges:
		return edges[edge]
	elif (edge[1], edge[0]) in edges:
		return edges[(edge[1], edge[0])]
	else:
		edges[edge] = edgeid
		ids2edge[edgeid] = edge 
		edgeid += 1
		return edges[edge]

for nloc, neis in nodeneighbor.iteritems():
	for nei in neis:
		edge1 = (nloc, nei)
		for nei2 in neis:
			if nei2 != nei:
				edge2 = (nloc, nei2)

				eid1 = getEdgeId(edge1)
				eid2 = getEdgeId(edge2)
				# todo add angle constraint

				# if len(neis) > 2:
				# 	c = neighbors_cos(nodeneighbor, nloc, nei, nei2)
				# 	if abs(c) < 0.5:
				# 		continue 

				if eid1 not in edgelinks:
					edgelinks[eid1] = [eid2]
				else:
					edgelinks[eid1].append(eid2)

gm = opengm.gm([2]*(edgeid), operator='adder')
pairwise_dict = {}
unaries = np.zeros(((edgeid), 2))

for eid1, eids in edgelinks.iteritems():
	for eid2 in eids:

		pairwise_dict[(eid1, eid2)] = True 

for eid, edge in ids2edge.iteritems():
	if edge in edgeClass:
		probs = edgeClass[edge]
	else:
		probs = edgeClass[(edge[1], edge[0])]

	unaries[eid,0] = -np.log(max(probs[0], 0.000001))
	unaries[eid,1] = -np.log(max(probs[1], 0.000001))


for i in range(edgeid):
	fid =  gm.addFunction(unaries[i][:].reshape((2)).astype(opengm.value_type))
	gm.addFactor(fid, np.array([i]).astype(opengm.index_type))




for link in pairwise_dict.keys():
	n1 = link[0]
	n2 = link[1]

	if n1 == n2:
		print("n1==n2 should not happen")
		continue


	pf = np.zeros((2, 2))

	pf[0,0] = 0
	pf[1,1] = 0
	pf[0,1] = weight
	pf[1,0] = weight

	edge1 = ids2edge[n1]
	edge2 = ids2edge[n2]

	if edge1[0] == edge2[0] and edge1[1] == edge2[1]:
		print("edge1[0] == edge2[0] and edge1[1] == edge2[1] should not happen" )
		continue

	if edge1[0] == edge2[1] and edge1[1] == edge2[0]:
		print("edge1[0] == edge2[1] and edge1[1] == edge2[0] should not happen" )
		continue


	c = 1.0 

	if edge1[0] == edge2[0]:
		if len(nodeneighbor[edge1[0]]) > 2:
			c = neighbors_cos(nodeneighbor, edge1[0], edge1[1], edge2[1])

	if edge1[0] == edge2[1]:
		if len(nodeneighbor[edge1[0]]) > 2:
			c = neighbors_cos(nodeneighbor, edge1[0], edge1[1], edge2[0])

	if edge1[1] == edge2[0]:
		if len(nodeneighbor[edge1[1]]) > 2:
			c = neighbors_cos(nodeneighbor, edge1[1], edge1[0], edge2[1])

	if edge1[1] == edge2[1]:
		if len(nodeneighbor[edge1[1]]) > 2:
			c = neighbors_cos(nodeneighbor, edge1[1], edge1[0], edge2[0])

	c = abs(c)

	if c < 0.5:
		c = 0.0 
	else:
		c = (c-0.5)

	pf[0,1] *= c
	pf[1,0] *= c


	fid = gm.addFunction(pf.astype(opengm.value_type))
	if n1 <= n2:
		gm.addFactor(fid, np.array([n1,n2]).astype(opengm.index_type))
	else:
		gm.addFactor(fid, np.array([n2,n1]).astype(opengm.index_type))


class PyCallback(object):
	"""
	callback functor which will be passed to an inference
	visitor.
	In that way, pure python code can be injected into the c++ inference.
	This functor visualizes the labeling as an image during inference.
	Args :
		shape : shape of the image 
		numLabels : number of labels
	"""
	def __init__(self):
		self.step = 0
		pass 
		
	def begin(self,inference):
		"""
		this function is called from c++ when inference is started
		Args : 
			inference : python wrapped c++ solver which is passed from c++
		"""
		print("begin")

	def end(self,inference):
		"""
		this function is called from c++ when inference ends
		Args : 
			inference : python wrapped c++ solver which is passed from c++
		"""
		arg = inference.arg()
		gm  = inference.gm()
		print(self.step, "energy ",gm.evaluate(arg))
		print("end")

	def visit(self,inference):
		"""
		this function is called from c++ each time the visitor is called
		Args : 
			inference : python wrapped c++ solver which is passed from c++
		"""
		if self.step % 20 == 0:
			arg = inference.arg()
			gm  = inference.gm()
			print(self.step, "energy ",gm.evaluate(arg))
		self.step += 1


#inf=opengm.inference.BeliefPropagation(gm,parameter=opengm.InfParam(steps=200,damping=0.9,convergenceBound=0.0001))
inf = opengm.inference.GraphCut(gm)

callback = PyCallback() 
visitor=inf.pythonVisitor(callback,visitNth=1)
inf.infer(visitor)
#inf.infer()

arg=inf.arg()

print(np.shape(arg))
print(arg)

for eid, edge in ids2edge.iteritems():
	probs = [0,0]
	probs[arg[eid]] = 1.0 

	if edge in edgeClass:
		edgeClass[edge] = tuple(probs)
	else:
		edgeClass[(edge[1], edge[0])] = tuple(probs) 


pickle.dump(edgeClass, open(output_fn, "w"))	
	








