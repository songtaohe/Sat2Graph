# Sat2Graph Inference Server Docker Container
We containerize Sat2Graph inference server with several models (so far, 6) into one container image.   

The inference server supports two inference modes. (1) Given a lat/lon coordinate and the size of the tile, the inference server can automatically download MapBox images and run inference on it. (2) Run on custom input images as long as the ground sampling distance (e.g., 50 cm/pixel) is provided. 

We also add code into this container to make it easy to download OpenStreetMap graphs (often used as ground truth). Please check out the detailed instruction below.  

## Usage
### Start the Inference Server
Start the server with GPU support.
```bash
docker run -p8010:8000 -p8011:8001 --gpus all -it --rm songtaohe/sat2graph_inference_server_cpu:latest
```

Or just run it with CPU (the image is smaller).
```bash
docker run -p8010:8000 -p8011:8001 -it --rm songtaohe/sat2graph_inference_server_cpu:latest
```

You can find these two commands in run-cpu.sh and run-gpu.sh. 

### Inference on MapBox Images
You can use the Python scripts in the [scripts](\scripts) folder to start inference tasks.

For example, to run inference on MapBox images, use the following command,
```
cd scripts
python infer_mapbox_input.py -lat 47.601214 -lon -122.134466 -tile_size 500 -model_id 2 -output out.json
```
This script uses *meter* as the unit for the *tile_size*.
The *model_id* argument determines which model to use. We show the supported models below.

### Inference on Custom Images
To run inference on custom images (e.g., sample.png), use the following command,
```
cd scripts
time python infer_custom_input.py -input sample.png -gsd 0.5 -model_id 2 -output out.json
```
Here, the *gsd* argument indicates the ground sampling distance or the spatial resolution of the image. The unit is in meter. 

### Supported Models
Model ID | Note 
--------------------- | -------------
1  | Sat2Graph, 20-City US, 1 meter GSD
2  | Sat2Graph-V2, 20-City US, 50cm GSD
3  | Sat2Graph-V2, 80-City Global, 50cm GSD
4  | UNet segmentation (our implementation), 20-City US, 1 meter GSD
5  | [DeepRoadMapper](http://www.cs.toronto.edu/~wenjie/papers/iccv17/mattyus_etal_iccv17.pdf) segmentation (our implementation), 20-City US, 1 meter GSD
6  | [JointOrientation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Batra_Improved_Road_Connectivity_by_Joint_Learning_of_Orientation_and_Segmentation_CVPR_2019_paper.pdf) segmentation (our implementation), 20-City US, 1 meter GSD


### OpenStreetMap Graph
To get OSM graphs, set the *osm_only* flag to 1 when using *infer_mapbox_input.py*. For example, 
```
cd scripts
python infer_mapbox_input.py -lat 47.601214 -lon -122.134466 -tile_size 500 -osm_only 1 -output out.json
```
The *out.json* will be the road graph from OpenStreetMap.

### Inference Result
The inference result (graph) is stored in a json format. It is basically a list of edges. Each edge stores the coordinates of its two vertices. We add a simple script to visualize it (need opencv). 
```
python vis.py tile_size input_json output_image 
```

You can also convert this json format into the pickle format that is compatible with the Sat2Graph evaluation code (topo and apls).
```
convert.py input.json output.p
```



### Intermediate Results
The intermediate results (e.g., segmentation mask, the downloaded satellite imagery, etc.) of each inference task are stored at http://localhost:8010/t(task_id)/ (replace the *task_id* with the task id you get after each run.)











