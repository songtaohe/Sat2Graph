# python 2
import graph 
import sys 
import json 
import pickle # python 2

infile = sys.argv[1]
outfile = sys.argv[2]

jsonedges = json.load(open(infile))
picklegraph = graph.link2graph(jsonedges)

pickle.dump(picklegraph, open(outfile, "wb"))
