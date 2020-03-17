# Guardian
<img src="https://drive.google.com/uc?export=view&id=1dER_83L4msWddD8ZS_Vx1uyFTJlT_zaa" width="500">
Guardian is an appliation for drones to autonomously "herd" elephants away in elephant-human conflicts common in Africa and Asia
- [x] Real-time object detection and tracking for highly constrained systems (Jetson Nano)
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
