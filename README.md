# FastMOT
[![HitCount](http://hits.dwyl.com/GeekAlexis/FastMOT.svg)](http://hits.dwyl.com/GeekAlexis/FastMOT) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![DOI](https://zenodo.org/badge/237143671.svg)](https://zenodo.org/badge/latestdoi/237143671)

<img src="assets/demo.gif" />

## News
  - (2020.11.28) Docker container is now supported on Ubuntu 18.04!

## Description
FastMOT is a custom multiple object tracker that implements:
  - YOLO detector
  - SSD detector
  - Deep SORT + OSNet ReID
  - KLT optical flow tracking
  - Camera motion compensation
  
Deep learning models are usually the bottleneck in Deep SORT, which makes Deep SORT unscalable for real-time applications. This repo significantly speeds up the entire system to run in **real-time** even on Jetson. It also provides enough flexibility to tune the speed-accuracy tradeoff without a lightweight model.

To achieve faster processing, the tracker only runs detector and feature extractor every *N* frames. Optical flow is then used to fill in the gaps. I swapped the feature extractor in Deep SORT for a better ReID model, OSNet. I also added a feature to re-identify targets that moved out of frame so that the tracker can keep the same IDs. I trained YOLOv4 on CrowdHuman while SSD's are pretrained COCO models from TensorFlow.

Both detector and feature extractor use the **TensorRT** backend and perform asynchronous inference. In addition, most algorithms, including Kalman filter, optical flow, and data association, are optimized using Numba. 

## Performance
| Sequence | Density | MOTA (SSD) | MOTA (YOLOv4) | MOTA (public) | FPS |
|:-------|:-------:|:-------:|:-------:|:-------:|:-----:|
| MOT17-13 | 5 - 30  | 19.8% | 45.6% | 41.3%  | 38 |
| MOT17-04 | 30 - 50  | 43.8% | 61.0% | 75.1% | 22 |
| MOT17-03 | 50 - 80  | - | - | - | 15 |

Performance is evaluated with the MOT17 dataset on Jetson Xavier NX using [py-motmetrics](https://github.com/cheind/py-motmetrics). When using public detections from MOT17, the MOTA scores are close to **state-of-the-art** trackers. Tracking speed can reach up to **38 FPS** depending on the number of objects. On a desktop CPU/GPU, FPS should be even higher. 

This means even though the tracker runs much faster, it is still highly accurate. More lightweight detector/feature extractor can potentially be used to obtain more speedup. Note that plain Deep SORT + YOLO struggles to run in real-time on most edge devices and desktop machines. 

## Requirements
- CUDA >= 10
- cuDNN >= 7
- TensorRT >= 7
- OpenCV >= 3.3
- PyCuda
- Numpy >= 1.15
- Scipy >= 1.5
- TensorFlow < 2.0 (for SSD support)
- Numba == 0.48
- cython-bbox

### Install for Jetson (TX2/Xavier NX/Xavier)
Make sure to have [JetPack 4.4](https://developer.nvidia.com/embedded/jetpack) installed and run the script:
  ```
  $ scripts/install_jetson.sh
  ```
### Install for Ubuntu 18.04
Make sure to have [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) installed. The image requires an NVIDIA Driver version >= 450. Build and run the docker image:
  ```
  $ docker build -t fastmot:latest .
  $ docker run --rm --gpus all -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY fastmot:latest
  ```
### Download models
This includes both pretrained OSNet, SSD, and my custom YOLOv4 ONNX model
  ```
  $ scripts/download_models.sh
  ```
### Build YOLOv4 TensorRT plugin
Modify `compute` [here](https://github.com/GeekAlexis/FastMOT/blob/2296fe414ca6a9515accb02ff88e8aa563ed2a05/fastmot/plugins/Makefile#L21) to match your [GPU compute capability](https://developer.nvidia.com/cuda-gpus#compute) for x86 PC
  ```
  $ cd fastmot/plugins
  $ make
  ```
### Download VOC dataset for INT8 calibration
Only required if you want to use SSD
  ```
  $ scripts/download_data.sh
  ```

## Usage
- USB Camera: 
  ```
  $ python3 app.py --input_uri /dev/video0 --mot
  ```
- CSI Camera: 
  ```
  $ python3 app.py --input_uri csi://0 --mot
  ```
- RTSP IP Camera: 
  ```
  $ python3 app.py --input_uri rtsp://<user>:<password>@<ip>:<port> --mot
  ```
- Video file: 
  ```
  $ python3 app.py --input_uri video.mp4 --mot
  ```
- Use `--gui` to visualize and `--output_uri` to save output
- To disable the GStreamer backend, set `WITH_GSTREAMER = False` [here](https://github.com/GeekAlexis/FastMOT/blob/3a4cad87743c226cf603a70b3f15961b9baf6873/fastmot/videoio.py#L11) 
- Note that the first run will be slow due to Numba compilation
- More options can be configured in `cfg/mot.json` 
  - Set `camera_size` and `camera_fps` to match your camera setting. List all settings for your camera:
    ```
    $ v4l2-ctl -d /dev/video0 --list-formats-ext
    ``` 
  - To change detector, modify `detector_type`. This can be either `YOLO` or `SSD`
  - To change classes, set `class_ids` under the correct detector. Default class is `1`, which corresponds to person
  - To swap model, modify `model` under a detector. For SSD, you can choose from `SSDInceptionV2`, `SSDMobileNetV1`, or `SSDMobileNetV2`
  - Note that with SSD, the detector splits a frame into tiles and processes them in batches for the best accuracy. Change `tiling_grid` to `[2, 2]`, `[2, 1]`, or `[1, 1]` if a smaller batch size is preferred
  - If more accuracy is desired and processing power is not an issue, reduce `detector_frame_skip`. Similarly, increase `detector_frame_skip` to speed up tracking at the cost of accuracy. You may also want to change `max_age` such that `max_age * detector_frame_skip` is around `30-40` 
 - Please star if you find this repo useful/interesting
  
 ## Track custom classes
This repo supports multi-class tracking and thus can be easily extended to custom classes (e.g. vehicle). You need to train both YOLO and a ReID model on your object classes. Check [Darknet](https://github.com/AlexeyAB/darknet) for training YOLO and [fast-reid](https://github.com/JDAI-CV/fast-reid) for training ReID. After training, convert the model to ONNX format and place it under `fastmot/models`. To convert YOLO to ONNX, [tensorrt_demos](https://github.com/jkjung-avt/tensorrt_demos) is a great reference.
### Add custom YOLOv3/v4
1. Subclass `YOLO` like here: https://github.com/GeekAlexis/FastMOT/blob/4e946b85381ad807d5456f2ad57d1274d0e72f3d/fastmot/models/yolo.py#L94
    ```
    ENGINE_PATH: path to TensorRT engine (converted at runtime)
    MODEL_PATH: path to ONNX model
    NUM_CLASSES: total number of classes
    INPUT_SHAPE: input size in the format "(channel, height, width)"
    LAYER_FACTORS: scale factors with respect to the input size for each yolo layer
                   For YOLOv3, change to [32, 16, 8]
                   For YOLOv3/v4-tiny, change to [32, 16]
    SCALES: scale_x_y parameter for each yolo layer
            For YOLOv3, change to [1., 1., 1.]
            For YOLOv3-tiny, change to [1., 1.]
            For YOLOv4-tiny, change to [1.05, 1.05]
    ANCHORS: anchors grouped by each yolo layer
    ```
    Note that anchors may not follow the same order in the Darknet cfg file. You need to mask out the anchors for each yolo layer using the indices in `mask` in Darknet cfg.
    Unlike YOLOv4, the anchors are usually in reverse for YOLOv3 and tiny
2. Change class labels [here](https://github.com/GeekAlexis/FastMOT/blob/master/fastmot/models/label.py) to your object classes
3. Modify `cfg/mot.json`: under `yolo_detector`, set `model` to the added Python class and set `class_ids` you want to detect. You may want to play with `conf_thresh` based on the accuracy of your model
### Add custom ReID
1. Subclass `ReID` like here: https://github.com/GeekAlexis/FastMOT/blob/aa707888e39d59540bb70799c7b97c58851662ee/fastmot/models/reid.py#L51
    ```
    ENGINE_PATH: path to TensorRT engine (converted at runtime)
    MODEL_PATH: path to ONNX model
    INPUT_SHAPE: input size in the format "(channel, height, width)"
    OUTPUT_LAYOUT: feature dimension output by the model (e.g. 512)
    METRIC: distance metric used to match features ('euclidean' or 'cosine')
    ```
2. Modify `cfg/mot.json`: under `feature_extractor`, set `model` to the added Python class. You may want to play with `max_feat_cost` and `max_reid_cost` - float values from `0` to `2`, based on the accuracy of your model
