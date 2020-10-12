# Fast MOT
High performance multiple object tracking in Python

<img src="assets/demo.gif" width="720" height="405" />

## Description
Fast MOT is a **real-time** tracker based on tracking by detection. The tracker implements:
  - YOLOv4 detector
  - SSD detector
  - Deep SORT + OSNet ReID
  - Optical flow tracking
  - Camera motion compensation
  
Unlike Deep SORT, the detector only runs every N frames to achieve faster processing. For this reason, optical flow is used to fill in the gaps. I swapped the feature extractor in Deep SORT to a better model, OSNet. The tracker is also able to re-identify previously lost targets and keep the same track IDs. Both detector and feature extractor use the TensorRT backend and perform asynchronous inference. In addition, most algorithms, including kalman filter, optical flow, and track association, are optimized using Numba. I trained YOLOv4 on CrowdHuman while SSD's are pretrained COCO models from TensorFlow. The tracker is currently designed for person tracking. 

## Performance
| Sequence | Density | MOTA (SSD) | MOTA (YOLOv4) | MOTA (public) | FPS |
|:-------|:-------:|:-------:|:-------:|:-------:|:-----:|
| MOT17-13 | 5 - 20  | 19.8% | 45.6% | 41.3%  | 30 |
| MOT17-04 | 20 - 50  | 43.8% | 61.0% | 75.1% | 24 |
| MOT17-03 | 40 - 80  | - | - | - | 16 |

Performance is evaluated with the MOT17 dataset on Jetson Xavier NX using [py-motmetrics](https://github.com/cheind/py-motmetrics). When using public detections from MOT17, the MOTA scores are close to **state-of-the-art** trackers. The tracker can achieve **30 FPS** depending on crowd density. On a desktop CPU/GPU, FPS will be even higher. This means even though the tracker runs much faster, it is still highly accurate. Note that plain Deep SORT cannot run in real-time on any edge device (or desktop). 

## Requirements
- CUDA >= 10
- cuDNN >= 7
- TensorRT >= 7 (SSD also requires UFF converter)
- OpenCV >= 3.3 (with GStreamer)
- PyCuda
- Numpy >= 1.15
- Scipy >= 1.5
- TensorFlow <= 1.15.2 (for SSD)
- Numba >= 0.48
- cython-bbox

### Install for Jetson (TX2/Xavier NX/Xavier)
Install OpenCV, CUDA, and TensorRT from [NVIDIA JetPack 4.4](https://developer.nvidia.com/embedded/jetpack) and run the script
  ```
  $ scripts/install_jetson.sh
  ```
### Install for Ubuntu 18.04
Make sure to have CUDA, cuDNN, TensorRT (Python API too) installed. You can optionally use my script to install from scratch
  ```
  $ scripts/install_tensorrt.sh
  ```
Build OpenCV from source with GStreamer. Modify `ARCH_BIN=7.5` to match your [GPU compute capability](https://developer.nvidia.com/cuda-gpus#compute). Then install Python dependencies

  ```
  $ scripts/install_opencv.sh
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
- For more flexibility, modify the config file `cfg/mot.json` 
  - Set `camera_size` and `camera_fps` to match your camera setting. List all settings for your camera:
    ```
    $ v4l2-ctl -d /dev/video0 --list-formats-ext
    ``` 
  - To change detector, modify `detector_type`. This can be either `YOLO` or `SSD`
  - To change classes, set `class_ids` under the correct detector. Default class is `1`, which corresponds to person
  - To swap model, modify `model` under a detector. For SSD, you can choose from `SSDInceptionV2`, `SSDMobileNetV1`, or `SSDMobileNetV2`
  - Note that with SSD, the detector splits a frame into tiles and processes them in batches for the best accuracy. Change `tiling_grid` to `[2, 2]` if a smaller batch size is preferred
  - If more accuracy is desired and processing power is not an issue, reduce `detector_frame_skip`. You may also want to increase `max_age` such that `max_age * detector_frame_skip` is around `30-40`. Similarly, increase `detector_frame_skip` to speed up tracking at the cost of accuracy
 - Please star if you find this repo useful/interesting.
  
 ## Track custom classes
This repo does not support training. To track custom classes (e.g. vehicle), you need to train both YOLOv4 and a ReID model. You can refer to [Darknet](https://github.com/AlexeyAB/darknet) for training YOLOv4 and [fast-reid](https://github.com/JDAI-CV/fast-reid) for training ReID. Convert the model to ONNX format and place it under `fastmot/models`. You also need to change the label names [here](https://github.com/GeekAlexis/FastMOT/blob/master/fastmot/models/label.py). To convert YOLOv4 to ONNX, [tensorrt_demos](https://github.com/jkjung-avt/tensorrt_demos) is a great reference. 
### Add custom YOLOv4
1. Subclass `YOLO` like here: https://github.com/GeekAlexis/FastMOT/blob/f7864e011699b355128d0cc25768c71d12ee6397/fastmot/models/yolo.py#L90
    ```
    ENGINE_PATH: path to TensorRT engine (converted at runtime)
    MODEL_PATH: path to ONNX model
    NUM_CLASSES: total number of classes
    INPUT_SHAPE: input size in the format (channel, height, width)
    LAYER_FACTORS: scale factors with respect to the input size for the three yolo layers. Change this to [32, 16] for YOLOv4-Tiny
    ANCHORS: anchors used to train the model
    ```
2. Modify `cfg/mot.json`. Under `yolo_detector`, set `model` to the added Python class and set `class_ids`
### Add custom ReID
1. Subclass `ReID` like here: https://github.com/GeekAlexis/FastMOT/blob/f7864e011699b355128d0cc25768c71d12ee6397/fastmot/models/reid.py#L49
    ```
    ENGINE_PATH: path to TensorRT engine (converted at runtime)
    MODEL_PATH: path to ONNX model
    INPUT_SHAPE: input size in the format (channel, height, width)
    OUTPUT_LAYOUT: feature dimension output by the model (e.g. 512)
    METRIC: distance metric used to match features (e.g. 'euclidean')
    ```
2. Modify `cfg/mot.json`. Under `feature_extractor`, set `model` to the added Python class and set `class_ids`. You may want to play with `max_feature_cost` and `max_reid_cost` for your model
