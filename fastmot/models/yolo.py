from pathlib import Path
import tensorrt as trt
import logging


class Yolo:
    PATH = None
    INPUT_SHAPE = None
    OUTPUT_NAME = None
    OUTPUT_LAYOUT = None

    @classmethod
    def add_plugin(cls, network):
        raise NotImplementedError

    @classmethod
    def build_engine(cls, trt_logger, batch_size):
        raise NotImplementedError


class YoloV4(Yolo):
    PATH = Path(__file__).parent / 'yolov4-512.trt'
    ONNX_PATH = None
    INPUT_SHAPE = (3, 512, 512)
    OUTPUT_NAME = None
    OUTPUT_LAYOUT = None