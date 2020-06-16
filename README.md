<p align="center">
  <img src="demo.png" width="720" height="405" />
</p>

- [x] Real-time object detection and tracking for highly constrained embedded systems
  - Support all classes in the COCO dataset
  - Robust against moderate camera movement
  - Speed on Jetson Nano: 32 FPS

### Dependencies
- CUDA
- OpenCV (Built with Gstreamer)
- Numpy
- Scipy
- PyCuda
- TensorRT  

#### Install dependencies for Jetson platforms
- OpenCV, CUDA, and TensorRT can be installed from NVIDIA JetPack:    
https://developer.nvidia.com/embedded/jetpack
- `bash install_jetson.sh`

### Run tracking
- With camera: `python3 vision.py --analytics`
- Input video: `python3 vision.py --input video.mp4 --analytics`
- Use `-h` for detailed descriptions about other flags like saving output and visualization

### References
- SORT: https://arxiv.org/abs/1602.00763  
- Deep SORT: https://arxiv.org/pdf/1703.07402.pdf 
