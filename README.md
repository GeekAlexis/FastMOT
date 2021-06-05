# FastMOT
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FGeekAlexis%2FFastMOT&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![DOI](https://zenodo.org/badge/237143671.svg)](https://zenodo.org/badge/latestdoi/237143671)

<img src="assets/dense_demo.gif" width="400"/> <img src="assets/aerial_demo.gif" width="400"/>

## News
  - (2021.2.13) Support Scaled-YOLOv4 models
  - (2021.1.3) Add DIoU-NMS for YOLO (+1% MOTA)
  - (2020.11.28) Docker container provided for Ubuntu

## Description
FastMOT is a custom multiple object tracker that implements:
  - YOLO detector
  - SSD detector
  - Deep SORT + OSNet ReID
  - KLT optical flow tracking
  - Camera motion compensation

Deep learning models are usually the bottleneck in Deep SORT, making Deep SORT unusable for real-time applications. FastMOT significantly speeds up the entire system to run in **real-time** even on Jetson. It also allows tuning the speed-accuracy tradeoff without a lightweight model.

To achieve faster processing, FastMOT only runs the detector and feature extractor every N frames. KLT is used to fill in the gaps. YOLOv4 was trained on CrowdHuman (82% mAP@0.5) while SSD's are pretrained COCO models from TensorFlow. FastMOT also re-identifies objects that moved out of frame and will keep the same IDs.

Both detector and feature extractor use the **TensorRT** backend and perform asynchronous inference. In addition, most algorithms, including Kalman filter, optical flow, and data association, are optimized using Numba.

## Performance
### Results on MOT20 train set
| Detector Skip | MOTA | IDF1 | HOTA | MOTP | MT | ML |
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| N = 1 | 66.8% | 56.4% | 45.0% | 79.3% | 912 | 274 |
| N = 5 | 65.1% | 57.1% | 44.3% | 77.9% | 860 | 317 |

### FPS on MOT17 sequences
| Sequence | Density | FPS |
|:-------|:-------:|:-------:|
| MOT17-13 | 5 - 30  | 38 |
| MOT17-04 | 30 - 50  | 22 |
| MOT17-03 | 50 - 80  | 15 |

Performance is evaluated with YOLOv4 using [TrackEval](https://github.com/JonathonLuiten/TrackEval). Note that neither YOLOv4 nor OSNet was trained or finetuned on the MOT20 dataset, so train set results should generalize well. FPS results are obtained on Jetson Xavier NX.

FastMOT has MOTA scores close to **state-of-the-art** trackers from the MOT Challenge. Tracking speed can reach up to **38 FPS** depending on the number of objects. Lighter models (e.g. YOLOv4-tiny) are recommended for more constrained devices like Jetson Nano. FPS is expected to be in the range of **50 - 150** on desktop CPU/GPU.

## Requirements
- CUDA >= 10
- cuDNN >= 7
- TensorRT >= 7
- OpenCV >= 3.3
- PyCuda
- Numpy >= 1.15
- Scipy >= 1.5
- TensorFlow < 2.0 (for SSD support)
- Numba == 0.48
- cython-bbox

### Install for x86 Ubuntu
Make sure to have [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) installed. The image requires an NVIDIA Driver version >= 450 for Ubuntu 18.04 and >= 465.19.01 for Ubuntu 20.04. Build and run the docker image:
  ```bash
  # For Ubuntu 20.04, add --build-arg TRT_IMAGE_VERSION=21.05
  docker build -t fastmot:latest .
  
  # Run xhost + first if you have issues with display
  docker run --gpus all --rm -it -v $(pwd):/usr/src/app/FastMOT -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY -e TZ=$(cat /etc/timezone) fastmot:latest
  ```
### Install for Jetson Nano/TX2/Xavier NX/Xavier
Make sure to have [JetPack 4.4+](https://developer.nvidia.com/embedded/jetpack) installed and run the script:
  ```bash
  ./scripts/install_jetson.sh
  ```
### Download models
This includes both pretrained OSNet, SSD, and my custom YOLOv4 ONNX model
  ```bash
  ./scripts/download_models.sh
  ```
### Build YOLOv4 TensorRT plugin
  ```bash
  cd fastmot/plugins
  make
  ```
### Download VOC dataset for INT8 calibration
Only required for SSD (not supported on Ubuntu 20.04)
  ```bash
  ./scripts/download_data.sh
  ```

## Usage
- USB webcam:
  ```bash
  python3 app.py --input_uri /dev/video0 --mot
  ```
- MIPI CSI camera:
  ```bash
  python3 app.py --input_uri csi://0 --mot
  ```
- RTSP stream:
  ```bash
  python3 app.py --input_uri rtsp://<user>:<password>@<ip>:<port>/<path> --mot
  ```
- HTTP stream:
  ```bash
  python3 app.py --input_uri http://<user>:<password>@<ip>:<port>/<path> --mot
  ```
- Image sequence:
  ```bash
  python3 app.py --input_uri img_%06d.jpg --mot
  ```
- Video file:
  ```bash
  python3 app.py --input_uri video.mp4 --mot
  ```
- Use `--gui` to visualize and `--output_uri` to save output
- To disable the GStreamer backend, set `WITH_GSTREAMER = False` [here](https://github.com/GeekAlexis/FastMOT/blob/3a4cad87743c226cf603a70b3f15961b9baf6873/fastmot/videoio.py#L11)
- Note that the first run will be slow due to Numba compilation
<details>
<summary> More options can be configured in cfg/mot.json </summary>

  - Set `resolution` and `frame_rate` that corresponds to the source data or camera configuration (optional). They are required for image sequence, camera sources, and MOT Challenge evaluation. List all configurations for your USB/CSI camera:
    ```bash
    v4l2-ctl -d /dev/video0 --list-formats-ext
    ```
  - To change detector, modify `detector_type`. This can be either `YOLO` or `SSD`
  - To change classes, set `class_ids` under the correct detector. Default class is `1`, which corresponds to person
  - To swap model, modify `model` under a detector. For SSD, you can choose from `SSDInceptionV2`, `SSDMobileNetV1`, or `SSDMobileNetV2`
  - Note that with SSD, the detector splits a frame into tiles and processes them in batches for the best accuracy. Change `tiling_grid` to `[2, 2]`, `[2, 1]`, or `[1, 1]` if a smaller batch size is preferred
  - If more accuracy is desired and processing power is not an issue, reduce `detector_frame_skip`. Similarly, increase `detector_frame_skip` to speed up tracking at the cost of accuracy. You may also want to change `max_age` such that `max_age × detector_frame_skip ≈ 30`

</details>

 ## Track custom classes
FastMOT supports multi-class tracking and can be easily extended to custom classes (e.g. vehicle). You need to train both YOLO and a ReID model on your object classes. Check [Darknet](https://github.com/AlexeyAB/darknet) for training YOLO and [fast-reid](https://github.com/JDAI-CV/fast-reid) for training ReID. After training, convert the model to ONNX format and place it in fastmot/models. To convert YOLO to ONNX, use [tensorrt_demos](https://github.com/jkjung-avt/tensorrt_demos/blob/master/yolo/yolo_to_onnx.py) to be compatible with the TensorRT YOLO plugins.
### Add custom YOLOv3/v4
1. Subclass `YOLO` like here: https://github.com/GeekAlexis/FastMOT/blob/4e946b85381ad807d5456f2ad57d1274d0e72f3d/fastmot/models/yolo.py#L94
    ```
    ENGINE_PATH: path to TensorRT engine (converted at runtime)
    MODEL_PATH: path to ONNX model
    NUM_CLASSES: total number of classes
    LETTERBOX: keep aspect ratio when resizing
               For YOLOv4-csp/YOLOv4x-mish, set to True
    NEW_COORDS: new_coords parameter for each yolo layer
                For YOLOv4-csp/YOLOv4x-mish, set to True
    INPUT_SHAPE: input size in the format "(channel, height, width)"
    LAYER_FACTORS: scale factors with respect to the input size for each yolo layer
                   For YOLOv4/YOLOv4-csp/YOLOv4x-mish, set to [8, 16, 32]
                   For YOLOv3, set to [32, 16, 8]
                   For YOLOv4-tiny/YOLOv3-tiny, set to [32, 16]
    SCALES: scale_x_y parameter for each yolo layer
            For YOLOv4-csp/YOLOv4x-mish, set to [2.0, 2.0, 2.0]
            For YOLOv4, set to [1.2, 1.1, 1.05]
            For YOLOv4-tiny, set to [1.05, 1.05]
            For YOLOv3, set to [1., 1., 1.]
            For YOLOv3-tiny, set to [1., 1.]
    ANCHORS: anchors grouped by each yolo layer
    ```
    Note that anchors may not follow the same order in the Darknet cfg file. You need to mask out the anchors for each yolo layer using the indices in `mask` in Darknet cfg.
    Unlike YOLOv4, the anchors are usually in reverse for YOLOv3 and tiny
2. Change class labels [here](https://github.com/GeekAlexis/FastMOT/blob/master/fastmot/models/label.py) to your object classes
3. Modify cfg/mot.json: set `model` in `yolo_detector` to the added Python class and set `class_ids` you want to detect. You may want to play with `conf_thresh` based on the accuracy of your model
### Add custom ReID
1. Subclass `ReID` like here: https://github.com/GeekAlexis/FastMOT/blob/aa707888e39d59540bb70799c7b97c58851662ee/fastmot/models/reid.py#L51
    ```
    ENGINE_PATH: path to TensorRT engine (converted at runtime)
    MODEL_PATH: path to ONNX model
    INPUT_SHAPE: input size in the format "(channel, height, width)"
    OUTPUT_LAYOUT: feature dimension output by the model (e.g. 512)
    METRIC: distance metric used to match features ('euclidean' or 'cosine')
    ```
2. Modify cfg/mot.json: set `model` in `feature_extractor` to the added Python class. You may want to play with `max_feat_cost` and `max_reid_cost` - float values from `0` to `2`, based on the accuracy of your model

 ## Citation
 If you find this repo useful in your project or research, please star and consider citing it:
 ```bibtex
@software{yukai_yang_2020_4294717,
  author       = {Yukai Yang},
  title        = {{FastMOT: High-Performance Multiple Object Tracking Based on Deep SORT and KLT}},
  month        = nov,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.4294717},
  url          = {https://doi.org/10.5281/zenodo.4294717}
}
```
