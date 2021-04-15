def graphInsert(node_neighbor, n1key, n2key):
	if n1key != n2key:
		if n1key in node_neighbor:
			if n2key in node_neighbor[n1key]:
				pass 
			else:
				node_neighbor[n1key].append(n2key)
		else:
			node_neighbor[n1key] = [n2key]


		if n2key in node_neighbor:
			if n1key in node_neighbor[n2key]:
				pass 
			else:
				node_neighbor[n2key].append(n1key)
		else:
			node_neighbor[n2key] = [n1key]

	return node_neighbor

def link2graph(links):
    graph = {}
    for link in links:
        graph = graphInsert(graph, tuple(link[0]), tuple(link[1]))
    return graph 

def graph2link(graph):
    links = []
    for nid, nei in graph.items():
        for nn in nei:
            edge = (nid, nn)
            edge_ = (nn, nid)
            if edge not in links and edge_ not in links:
                links.append(edge)
    
    return links