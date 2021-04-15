# python 2
import requests
import json 
import argparse
import math 

def parseArgument():
    parser = argparse.ArgumentParser()

    parser.add_argument('-tile_size', action='store', dest='tile_size', type=int,
                        help='size of the tile in meters', required =False, default=300)
    
    parser.add_argument('-lat', action='store', dest='lat', type=float,
                        help='latitude (degree)', required =True)
    parser.add_argument('-lon', action='store', dest='lon', type=float,
                        help='longitude (degree)', required =True)

    parser.add_argument('-model_id', action='store', dest='model_id', type=int,
                        help='model id', required =False, default=4)
    
    parser.add_argument('-osm_only', action='store', dest='osm_only', type=int,
                        help='get ground truth graph from osm', required =False, default=0)
    
    parser.add_argument('-output', action='store', dest='output', type=str,
                        help='output graph (edges in json format)', required =False, default="out.json")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgument()

    msg = {}
    msg["lat"] = args.lat
    msg["lon"] = args.lon
    msg["v_thr"] = 0.05
    msg["e_thr"] = 0.01
    msg["snap_dist"] = 15
    msg["snap_w"] = 100
    msg["model_id"] = args.model_id

    msg["size"] = args.tile_size;
    n = int(math.ceil(args.tile_size / 176.0))
    msg["padding"] = (n * 176 - args.tile_size)//2;
    msg["stride"] = 176;
    msg["nPhase"] = 1;
    if args.model_id == 3:
        msg["nPhase"] = 5;
    
    if args.osm_only != 0:
        msg["osm"] = True

    url = "http://localhost:8011"

    x = requests.post(url, data = json.dumps(msg))
    graph = json.loads(x.text) 
    if graph["success"] == 'false':
        print("unknown error")
        exit()

    #print(graph)
    
    if args.osm_only != 0:
        json.dump(graph["osmgraph"], open(args.output, "w"), indent=2)
    else:
        json.dump(graph["graph"]["graph"][0], open(args.output, "w"), indent=2)
    
    tid = graph["taskid"]
    print("please check intermediate results at http://localhost:8010/t%d/" % tid)

