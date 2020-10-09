# Fast MOT
High performance multiple object tracking in Python

<img src="assets/demo.gif" width="720" height="405" />

## Description
Fast MOT is a real-time tracker based on tracking by detection. The tracker implements:
  - YOLOv4 detector
  - SSD detector
  - Deep SORT + OSNet ReID
  - Optical flow tracking
  - Camera motion compensation
  
Unlike Deep SORT, the detector only runs at every Nth frame to achieve faster processing. For this reason, I adapted Deep SORT significantly. Both detector and feature extractor use the TensorRT backend and perform inference in an asynchronous way. In addition, most algorithms, including kalman filter, optical flow, and track association, are optimized using Numba. Adding ReID enables Deep SORT to identify previously lost targets. I trained YOLOv4 on CrowdHuman while SSD's are pretrained COCO models from TensorFlow. The tracker is currently designed for pedestrian tracking. 

## Performance
| Sequence | Density | MOTA (SSD) | MOTA (YOLOv4) | MOTA (public) | FPS |
|:-------|:-------:|:-------:|:-------:|:-------:|:-----:|
| MOT17-13 | 5 - 20  | 19.8% | 45.6% | 41.3%  | 30 |
| MOT17-04 | 20 - 50  | 43.8% | 61.0% | 75.1% | 24 |
| MOT17-03 | 40 - 80  | - | - | - | 16 |

Performance is evaluated with the MOT17 dataset on Jetson Xavier NX using [py-motmetrics](https://github.com/cheind/py-motmetrics). When using public detections from MOT17, the MOTA scores are close to **state-of-the-art** trackers. The tracker can achieve **30 FPS** depending on crowd density. On a Desktop CPU/GPU, FPS will be even higher. This means even though the tracker runs much faster, it is still highly accurate. Note that plain Deep SORT cannot run in real-time on any edge device (or desktop). 

## Requirements
- CUDA >= 10
- CuDNN >= 7
- TensorRT >= 7 (UFF converter also required for SSD)
- OpenCV >= 3.3 (with GStreamer)
- TensorFlow <= 1.15.2 (for SSD support)
- PyCuda
- Numpy >= 1.15
- Scipy >= 1.5
- Numba
- cython-bbox

### Install for Jetson (TX2/Xavier NX/Xavier)
Install OpenCV, CUDA, and TensorRT from [NVIDIA JetPack 4.4](https://developer.nvidia.com/embedded/jetpack) and run the script
  ```
  $ scripts/install_jetson.sh
  ```
### Install for Ubuntu 18.04
Make sure to have CUDA, TensorRT, and its Python API installed. You can optionally use my script to install from scratch
  ```
  $ scripts/install_tensorrt.sh
  ```
Build OpenCV from source with GStreamer. Modify `ARCH_BIN=7.5` to match your GPU compute capability. Then install Python dependencies

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
Only if you want to use SSD
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
  - Set `camera_size` and `camera_fps` to match your camera setting. You can use `v4l2-ctl -d /dev/video0 --list-formats-ext` to list all settings for your camera
  - To change detector, modify `detector_type`. This can be either `YOLO` or `SSD`
  - To change target classes, please refer to the labels [here](https://github.com/GeekAlexis/FastMOT/blob/master/fastmot/models/label.py), and set `class_ids` under the correct detector. Default class is `1`, which corresponds to person
  - For SSD, a more lightweight backbone can be used by changing `model` to `SSDMobileNetV1` or `SSDMobileNetV2`
  - Note that with SSD, the detector splits a frame into tiles and processes them in batches for the best accuracy. Change `tiling_grid` to `[2, 2]` if a smaller batch size is preferred
  - If more accuracy is desired and processing power is not an issue, reduce `detector_frame_skip`. You may also want to reduce `max_age` to get rid of undetected lost tracks quickly
 - To track custom classes (e.g. vehicle), please refer to [fast-reid](https://github.com/JDAI-CV/fast-reid) and [darknet](https://github.com/AlexeyAB/darknet) to train your ReID model and YOLOv4, respectively. Convert the model to ONNX format and place it under `fastmot/models`. [tensorrt_demos](https://github.com/jkjung-avt/tensorrt_demos) is a great source if you want to convert a Darknet model to ONNX. Finally, follow the examples below to add additional models: 
https://github.com/GeekAlexis/FastMOT/blob/f7864e011699b355128d0cc25768c71d12ee6397/fastmot/models/reid.py#L49
https://github.com/GeekAlexis/FastMOT/blob/f7864e011699b355128d0cc25768c71d12ee6397/fastmot/models/yolo.py#L90
- Please star if you find this repo useful/interesting. It means a lot to me!
