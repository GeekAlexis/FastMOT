# Guardian
- Real-time object detection and tracking for highly constrained systems (Jetson Nano)
- Flight control for drone that tracks targets using GPS signal and vision (WIP)
- Guardian is an appliation to "herd" elephants away in elephant-human conflicts common in Africa and Asia

### Dependencies
- OpenCV (Built with Gstreamer)
- Numpy
- Scipy
- PyCuda
- TensorRT
- DJI OSDK

### Example runs
- With camera: `python3 vision.py -a`
- Input video: `python3 vision.py -a -i video.mp4`
- Use `-h` for detailed descriptions about other flags like saving output and visualization
