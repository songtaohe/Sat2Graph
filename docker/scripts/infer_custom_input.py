# python 2
import requests
import json 
import argparse
import math  
import base64
import graph

def parseArgument():
    parser = argparse.ArgumentParser()

    parser.add_argument('-input', action='store', dest='input', type=str,
                        help='input image file', required =True)
    
    parser.add_argument('-gsd', action='store', dest='gsd', type=float,
                        help='ground sample distance', required =False, default= 1)
    
    parser.add_argument('-model_id', action='store', dest='model_id', type=int,
                        help='model id', required =False, default=4)
    
    parser.add_argument('-output', action='store', dest='output', type=str,
                        help='output graph (edges in json format)', required =False, default="out.json")
    

    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgument()

    img_bin = open(args.input,"rb")
    img_base64 = base64.encodestring(img_bin.read())
    print(type(img_base64))
    msg = {}
    msg["inputtype"] = "base64"
    msg["imagebase64"] = img_base64
    msg["imagetype"] = args.input.split(".")[-1] 
    msg["imagegsd"] = args.gsd

    msg["v_thr"] = 0.05
    msg["e_thr"] = 0.01
    msg["snap_dist"] = 15
    msg["snap_w"] = 100
    msg["model_id"] = args.model_id

    msg["stride"] = 176;
    msg["nPhase"] = 1;
    if args.model_id == 3:
        msg["nPhase"] = 5;
    
    json.dump(msg, open("dbg.json","w"), indent=2)
    

    url = "http://localhost:8011"

    x = requests.post(url, data = json.dumps(msg))
    graph = json.loads(x.text) 
    if graph["success"] == 'false':
        print("unknown error")
        exit()

    #print(graph)
    json.dump(graph["graph"]["graph"][0], open(args.output, "w"), indent=2)
    
    tid = graph["taskid"]
    print("please check intermediate results at http://localhost:8010/t%d/" % tid)
    
    

