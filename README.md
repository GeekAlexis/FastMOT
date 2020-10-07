# Fast MOT
High performance multiple object tracking in Python

<img src="assets/demo.gif" width="720" height="405" />

## Description
Fast MOT is a real-time tracker that includes both detection and tracking. The tracker implements:
  - YOLOv4 detector
  - SSD detector
  - Deep SORT + OSNet ReID
  - Optical flow tracking
  - Camera motion compensation
  
Unlike Deep SORT, detector is not run at every frame to achieve faster processing. Also, the tracker is optimized with TensorRT and Numba. YOLOv4 is trained on CrowdHuman while SSD's are pretrained TensorFlow models. Note that when using SSD, the detector will split a frame into tiles and process them in batches for the best accuracy. ReID enables Deep SORT to identify previously lost targets. The tracker is currently designed for pedestrian tracking. To track custom classes, please refer to [torchreid](https://github.com/KaiyangZhou/deep-person-reid) and [darknet](https://github.com/AlexeyAB/darknet) to train OSNet and YOLOv4 on your own classes. 

## Performance
| Sequence | Density | MOTA (SSD) | MOTA (YOLOv4) | MOTA (public) | FPS |
|:-------|:-------:|:-------:|:-------:|:-------:|:-----:|
| MOT17-13 | 5 - 20  | 19.8% | 41.3% | 45.6%  | 30 |
| MOT17-04 | 20 - 50  | 43.8% | 61.0% | 74.9% | 24 |
| MOT17-03 | 40 - 80  | - | - | - | 16 |

Tracking is evaluated with the MOT17 dataset on Jetson Xavier NX using [py-motmetrics](https://github.com/cheind/py-motmetrics). When using public detections from MOT17, the MOTA scores are close to state-of-the-art trackers. The tracker can achieve up to 30 FPS depending on crowd density. The speed on a Desktop CPU/GPU will be even higher. Note that plain Deep SORT cannot run in real-time on any edge device. 

## Requirements
- OpenCV (Gstreamer)
- TensorFlow (SSD)
- TensorRT 7+
- PyCuda
- Numpy
- Numba
- Scipy
- cython-bbox

### Install for Jetson (TX2/Xavier NX/Xavier)
Install OpenCV, CUDA, and TensorRT from [NVIDIA JetPack](https://developer.nvidia.com/embedded/jetpack) and run the script
  ```
  $ scripts/install_jetson.sh
  ```
### Install for Ubuntu 18.04
Make sure to have CUDA, TensorRT, and its Python API installed. You can optionally use my script
  ```
  $ scripts/install_tensorrt.sh
  ```
Build OpenCV from source with Gstreamer. Modify `ARCH_BIN=7.5` to match your GPU compute capability. Then install Python dependencies

  ```
  $ scripts/install_opencv.sh
  $ pip3 install -r requirements.txt
  ```
### Download models
This includes both pretrained OSNet, SSD and my custom YOLOv4 ONNX model
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
- Use `--gui` to visualize and `--output_uri` to save output
- To change detector, tracker parameters, or target classes, etc., modify `fastmot/configs/mot.json`
