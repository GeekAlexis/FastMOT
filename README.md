<p align="center">
  <img src="assets/demo.gif" width="720" height="405" />
</p>

Real-time implementation of Deep Sort
  - [x] TensorRT optimized SSD detector
  - [x] Improve small object detection with tiling
  - [x] OSNet for accurate REID
  - [x] Optical flow tracking and camera motion compensation
  - [ ] Replace SSD with YOLO V4
  
The tracker has an input size of 1280 x 720. Note that larger videos will be resized, which results in a slight drop in frame rate. Because of tiling, the tracker assumes medium/small targets and cannot detect up close targets properly. I used a pretrained OSNet from [Torchreid](https://github.com/KaiyangZhou/deep-person-reid). Currently, tracking targets other than pedestrians will work but it is not recommended without further training. Tracking is tested with the MOT17 dataset on Jetson Xavier NX. The tracker can achieve up to 30 FPS depending on crowd density. The original Deep Sort cannot run in real-time on edge devices.

| # targets  | FPS on Xavier NX |
| ------------- | ------------- |
| 0 - 20  | 30  |
| 20 - 30  | 23  |
| 30 - 50  | 15  |

### Dependencies
- OpenCV (With Gstreamer)
- Numpy
- Numba
- Scipy
- PyCuda
- TensorRT (>=7)
- cython-bbox

#### Install for Jetson (TX2/Xavier NX/Xavier)
Install OpenCV, CUDA, and TensorRT from [NVIDIA JetPack](https://developer.nvidia.com/embedded/jetpack)    
  ```
  $ sh install_jetson.sh
  ```
#### Install for x86 (Not tested)
Make sure to have CUDA and TensorRT installed and build OpenCV from source with Gstreamer
  ```
  $ pip3 install -r requirements.txt
  $ cd fast_mot/models
  $ sh prepare_calib_data.sh
  ```

### Run tracking
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
