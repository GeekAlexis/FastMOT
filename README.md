# Guardian
- Real-time object detection and tracking for highly constrained systems on drone (e.g. Jetson Nano)
- Guardian is an autonomous drone solution for human-elephant conflict common in Africa and Asia

### Dependencies
- OpenCV (Built with Gstreamer)
- Numpy
- Scipy
- PyCuda
- TensorRT
- tqdm

### Example runs
- With camera: `python3 main.py --analytics --gui`
- Input video: `python3 main.py --analytics --input video.mp4`
