<p align="center">
  <img src="assets/demo.gif" width="720" height="405" />
</p>

- Real-time implementation of Deep Sort 
  - [x] SSD detector with frame tiling
  - [x] Deploy OSNet, a more accurate REID model
  - [x] Optical flow tracking and camera motion compensation
  - [ ] Replace SSD with YOLOV4
  
Fast MOT has an input size of 1280 x 720. Note that larger videos will be resized, which results in a drop in frame rate. It also assumes medium/small targets and struggles to detect up close targets properly due to frame tiling. Currently, tracking targets other than pedestrians will work but retraining the REID model on other classes can improve accuracy. Please refer to [Torchreid](https://github.com/KaiyangZhou/deep-person-reid) for retraining. Tracking is tested with the MOT17 dataset on Jetson Xavier NX. The frame rate can reach 15 - 35 FPS depending on crowd density.

### Dependencies
- OpenCV (With Gstreamer)
- Numpy
- Numba
- Scipy
- PyCuda
- TensorRT (>=6)
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
- Use `-h` to learn how to visualize and save output
- For more flexibility, edit `fast_mot/configs/mot.json` to configure parameters and target classes (COCO)
