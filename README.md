# Guardian
An appliation for drones to autonomously "herd" elephants away in elephant-human conflicts common in Africa and Asia
<p align="center">
<img src="https://drive.google.com/uc?export=view&id=1J38g6nJbPlK3L8rlmR9Mt-0wpPOeYxrX" width="720">
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

### Run visual tracking only
- With camera: `python3 vision.py --analytics`
- Input video: `python3 vision.py --input video.mp4 --analytics`
- Use `-h` for detailed descriptions about other flags like saving output and visualization
### Run the whole systems
- Coming out soon

### References
SORT: https://arxiv.org/abs/1602.00763  
Deep SORT: https://arxiv.org/pdf/1703.07402.pdf  
Tiling: https://arxiv.org/pdf/1911.06073.pdf  
Kalman Filter with Homography: https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=1301&context=studentpub. 
