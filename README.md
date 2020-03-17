# Guardian
An appliation for drones to autonomously "herd" elephants away in elephant-human conflicts common in Africa and Asia
<p align="center">
<img src="https://drive.google.com/uc?export=view&id=1dER_83L4msWddD8ZS_Vx1uyFTJlT_zaa" width="500">
</p>

- [x] Real-time object detection and tracking for highly constrained embedded systems (Jetson Nano)
  - Support all classes in the COCO dataset
  - Multiple objects can be acquired but only a single target will be selected to track
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
- With camera: `python3 vision.py -a`
- Input video: `python3 vision.py -a -i video.mp4`
- Use `-h` for detailed descriptions about other flags like saving output and visualization
### Run the whole systems
- Coming out soon
