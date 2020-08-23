<p align="center">
  <img src="demo.gif" width="720" height="405" />
</p>

- Real-time implementation of Deep Sort 
  - [x] Real-time SSD detector with frame tiling
  - [x] Deploy OSNet, a more accurate REID model
  - [x] Optical flow tracking and camera motion compensation
  - [ ] Replace SSD with YOLOV4
  
Fast MOT has an input size of 1280 x 720. Larger videos will be resized, which will slow down frame rate a bit. It also assumes medium/small targets and struggles to detect up close targets properly due to frame tiling. Currently, only pedestrian tracking is supported. Tracking is tested with the MOT17 dataset on Jetson Xavier NX. The frame rate can reach 15 - 35 FPS depending on crowd density.

### Dependencies
- OpenCV (Built with Gstreamer)
- Numpy
- Numba
- Scipy
- PyCuda
- TensorRT  
- cython-bbox

#### Install dependencies for Jetson platforms
- OpenCV, CUDA, and TensorRT can be installed from NVIDIA JetPack:    
https://developer.nvidia.com/embedded/jetpack
- `bash install_jetson.sh`

### Run tracking
- With camera: `python3 app.py --mot`
- Input video: `python3 app.py --input video.mp4 --mot`
- Use `-h` for detailed descriptions about other flags like saving output and visualization
- For more flexibility, edit fast_mot/configs/mot.json to configure parameters and object classes
