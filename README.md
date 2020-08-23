<p align="center">
  <img src="demo.gif" width="720" height="405" />
</p>

- Real-time implementation of Deep Sort 
  - [x] Real-time SSD detector with frame tiling
  - [x] Deploys a better REID model OSNet
  - [x] Optical flow tracking and camera motion compensation
  - [] Replace SSD with YOLOV4
  
- Input size: 1280 x 720
- Assumes medium/small targets (struggles with up close targets due to tiling)
- Currently only supports pedestrian tracking
- Speed on Jetson Xavier NX: 15 - 35 FPS (depends on crowd density)

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
- Edit fast_mot/configs/mot.json to configure parameters and change object classes
