# Fast MOT
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<img src="assets/demo.gif" />

## Description
Fast MOT is a multiple object tracker that implements:
  - YOLO detector
  - SSD detector
  - Deep SORT + OSNet ReID
  - KLT optical flow tracking
  - Camera motion compensation
  
Deep learning models are usually the bottleneck in Deep SORT, which makes Deep SORT unscalable for real-time applications. This repo significantly speeds up the entire system to run in **real-time** even on Jetson. It also provides enough flexibility to customize the speed-accuracy tradeoff without a lightweight model.

To achieve faster processing, the tracker only runs detector and feature extractor every *N* frames. Optical flow is then used to fill in the gaps. I swapped the feature extractor in Deep SORT for a better ReID model, OSNet. I also added a feature to re-identify targets that moved out of frame so that the tracker can keep the same IDs. I trained YOLOv4 on CrowdHuman while SSD's are pretrained COCO models from TensorFlow.

Both detector and feature extractor use the **TensorRT** backend and perform asynchronous inference. In addition, most algorithms, including Kalman filter, optical flow, and data association, are optimized using Numba. 

## Performance
| Sequence | Density | MOTA (SSD) | MOTA (YOLOv4) | MOTA (public) | FPS |
|:-------|:-------:|:-------:|:-------:|:-------:|:-----:|
| MOT17-13 | 5 - 30  | 19.8% | 45.6% | 41.3%  | 30 |
| MOT17-04 | 30 - 50  | 43.8% | 61.0% | 75.1% | 22 |
| MOT17-03 | 50 - 80  | - | - | - | 15 |

Performance is evaluated with the MOT17 dataset on Jetson Xavier NX using [py-motmetrics](https://github.com/cheind/py-motmetrics). When using public detections from MOT17, the MOTA scores are close to **state-of-the-art** trackers. The tracker can achieve **30 FPS** depending on the number of objects. On a desktop CPU/GPU, FPS should be even higher. 

This means even though the tracker runs much faster, it is still highly accurate. More lightweight detector/feature extractor can potentially be used to obtain more speedup. Note that plain Deep SORT + YOLO struggles to run in real-time on most edge devices and desktop machines. 

## Requirements
- CUDA >= 10
- cuDNN >= 7
- TensorRT >= 7 (SSD requires UFF converter)
- OpenCV >= 3.3 (with GStreamer)
- PyCuda
- Numpy >= 1.15
- Scipy >= 1.5
- TensorFlow <= 1.15.2 (for SSD)
- Numba >= 0.48
- cython-bbox

### Install for Jetson (TX2/Xavier NX/Xavier)
Install [JetPack 4.4](https://developer.nvidia.com/embedded/jetpack) and run the script
  ```
  $ scripts/install_jetson.sh
  ```
### Install for Ubuntu 18.04
Make sure to have [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html), [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html), and [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#downloading) (including Python API) installed. Follow the official guide if you want to install UFF with TensorRT for SSD support.

To only run YOLO, you can optionally use my script to install from scratch
  ```
  $ scripts/install_tensorrt.sh
  ```
Build OpenCV from source with GStreamer. Modify `ARCH_BIN=7.5` to match your [GPU compute capability](https://developer.nvidia.com/cuda-gpus#compute)

  ```
  $ scripts/install_opencv.sh
  ```

Install Python dependencies
  ```
  $ pip3 install -r requirements.txt
  ```
### Download models
This includes both pretrained OSNet, SSD, and my custom YOLOv4 ONNX model
  ```
  $ scripts/download_models.sh
  ```
### Build YOLOv4 TensorRT plugin
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
- Use `--gui` to visualize and `--output_uri out.mp4` to save output
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
This repo does not support training but multi-class tracking is supported. To track custom classes (e.g. vehicle), you need to train both YOLO and a ReID model. You can refer to [Darknet](https://github.com/AlexeyAB/darknet) for training YOLO and [fast-reid](https://github.com/JDAI-CV/fast-reid) for training ReID. Convert the model to ONNX format and place it under `fastmot/models`. You also need to change class labels [here](https://github.com/GeekAlexis/FastMOT/blob/master/fastmot/models/label.py). To convert YOLO to ONNX, [tensorrt_demos](https://github.com/jkjung-avt/tensorrt_demos) is a great reference.
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
    Note that anchors may not follow the same order in the Darknet cfg file. You need to mask out the anchors for each yolo layer using the indices in `mask`.
    For tiny and YOLOv3, the anchors are usually flipped.
2. Modify `cfg/mot.json`: under `yolo_detector`, set `model` to the added Python class and set `class_ids`
### Add custom ReID
1. Subclass `ReID` like here: https://github.com/GeekAlexis/FastMOT/blob/aa707888e39d59540bb70799c7b97c58851662ee/fastmot/models/reid.py#L51
    ```
    ENGINE_PATH: path to TensorRT engine (converted at runtime)
    MODEL_PATH: path to ONNX model
    INPUT_SHAPE: input size in the format "(channel, height, width)"
    OUTPUT_LAYOUT: feature dimension output by the model (e.g. 512)
    METRIC: distance metric used to match features (e.g. 'euclidean')
    ```
2. Modify `cfg/mot.json`: under `feature_extractor`, set `model` to the added Python class and set `class_ids`. You may want to play with `max_feat_cost` and `max_reid_cost` for your model
