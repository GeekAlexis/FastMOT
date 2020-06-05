<p align="center">
  <img src="demo.png" width="720" height="405" />
</p>

- [x] Real-time object detection and tracking for highly constrained embedded systems
  - Support all classes in the COCO dataset
  - Robust against moderate camera movement
  - Speed on Jetson Nano: 32 FPS
- [ ] Drone flight control for following targets using both GPS and vision


### Dependencies
#### Visual tracking
- OpenCV (Built with Gstreamer)
- Numpy
- Scipy
- PyCuda
- TensorRT  
#### Flight control
- DJI OSDK  

Note OpenCV, PyCuda, and TensorRT can be installed from NVIDIA JetPack using the SDK Manager:    
https://developer.nvidia.com/embedded/jetpack

### Run visual tracking
- With camera: `python3 vision.py --analytics`
- Input video: `python3 vision.py --input video.mp4 --analytics`
- Use `-h` for detailed descriptions about other flags like saving output and visualization

### References
- SORT: https://arxiv.org/abs/1602.00763  
- Deep SORT: https://arxiv.org/pdf/1703.07402.pdf  
- Tiling: https://arxiv.org/pdf/1911.06073.pdf  
