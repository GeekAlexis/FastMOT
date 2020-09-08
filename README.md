# Fast MOT
High performance multiple object tracking in Python

<img src="assets/demo.gif" width="720" height="405" />

## Description
  - [x] TensorRT SSD
  - [x] Tiling for small object detection
  - [x] OSNet ReID model
  - [x] Optical flow tracking & camera motion compensation
  - [ ] TensorRT YOLO V4
  
Fast MOT is an end-to-end tracker that includes both detection and tracking. The tracker combines Deep SORT with optical flow and is optimized with TensorRT and Numba to run in real-time. It has an input size of 1280 x 720. Because of tiling, the tracker assumes medium/small targets and shouldn't be used to detect up close ones. I used a pretrained pedestrian OSNet model from [Torchreid](https://github.com/KaiyangZhou/deep-person-reid). Currently, tracking objects other than pedestrians will work but it is not recommended without further training the OSNet model on these classes. 

## Performance
| Dataset | Density | MOTA(SSD) | MOTA(public) | FPS |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| MOT17-13 | 5 - 20  | 19.8% | 38.5%  | 30 |
| MOT17-04 | 20 - 40  | 43.8% | 73.7% | 23 |
| MOT17-03 | 30 - 60  | - | - | 15 |

Tracking is evaluated on the MOT17 dataset with Jetson Xavier NX using [py-motmetrics](https://github.com/cheind/py-motmetrics). When using public detections from MOT17, the MOTA scores are close to state-of-the-art trackers. However, pretrained SSD models are not accurate enough for pedestrian detection and I will train a YOLOV4 model to replace SSD if I have time. The tracker can achieve up to 30 FPS depending on crowd density. The frame rate on a Desktop GPU will be even higher. Note that plain Deep SORT cannot run in real-time on any edge device. 

## Dependencies
- OpenCV (With Gstreamer)
- Numpy
- Numba
- Scipy
- PyCuda
- TensorRT (>=7)
- cython-bbox

### Install for Jetson (TX2/Xavier NX/Xavier)
Install OpenCV, CUDA, and TensorRT from [NVIDIA JetPack](https://developer.nvidia.com/embedded/jetpack)    
  ```
  $ sh install_jetson.sh
  ```
### Install for x86 Linux (Not tested)
Make sure to have CUDA and TensorRT installed and build OpenCV from source with Gstreamer
  ```
  $ pip3 install -r requirements.txt
  $ cd fastmot/models
  $ sh prepare_calib_data.sh
  ```

## Usage
- camera (/dev/video0): 
  ```
  $ python3 app.py --mot
  ```
- video: 
  ```
  $ python3 app.py --input video.mp4 --mot
  ```
- Use `--gui` to visualize and `--output` to save output
- For more flexibility, edit `fastmot/configs/mot.json` to configure parameters and target classes (COCO)
