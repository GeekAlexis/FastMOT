# Fast MOT
Fast and accurate multiple object tracking

<img src="assets/demo.gif" width="720" height="405" />

## Description
  - [x] TensorRT SSD
  - [x] Tiling for small object detection
  - [x] OSNet ReID model
  - [x] Optical flow tracking & camera motion compensation
  - [ ] TensorRT YOLO V4
  
The tracker follows an approach similiar to Deep Sort but runs a lot faster. It has an input size of 1280 x 720. Note that larger videos will be resized, which results in a slight drop in frame rate. Because of tiling, the tracker assumes medium/small targets and cannot detect up close targets properly. I used a pretrained OSNet from [Torchreid](https://github.com/KaiyangZhou/deep-person-reid). Currently, tracking targets other than pedestrians will work but it is not recommended without training the OSNet model on these classes. 

## Performance
Tracking is tested with the MOT17 dataset on Jetson Xavier NX. The tracker can achieve up to 30 FPS depending on crowd density. Other Deep Sort trackers cannot run in real-time on edge devices. The frame rate on a Desktop GPU will be even higher.

| # targets  | FPS on Xavier NX |
| ------------- | ------------- |
| 0 - 20  | 30  |
| 20 - 30  | 23  |
| 30 - 50  | 15  |

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
### Install for x86 (Not tested)
Make sure to have CUDA and TensorRT installed and build OpenCV from source with Gstreamer
  ```
  $ pip3 install -r requirements.txt
  $ cd fast_mot/models
  $ sh prepare_calib_data.sh
  ```

## Usage
- With camera (/dev/video0): 
  ```
  $ python3 app.py --mot
  ```
- Input video: 
  ```
  $ python3 app.py --input your_video.mp4 --mot
  ```
- Use `--gui` to visualize and `--output video_out.mp4` to save output
- For more flexibility, edit `fast_mot/configs/mot.json` to configure parameters and target classes (COCO)
