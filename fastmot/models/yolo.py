import logging
from pathlib import Path
import tensorrt as trt


class YOLO:
    PLUGIN_PATH = Path(__file__).parents[1] / 'plugins' / 'libyolo_layer.so'
    ENGINE_PATH = None
    MODEL_PATH = None
    INPUT_SHAPE = None
    OUTPUT_NAME = None
    OUTPUT_LAYOUT = None

    @classmethod
    def add_plugin(cls, network):
        raise NotImplementedError

    @classmethod
    def build_engine(cls, trt_logger, batch_size):
        raise NotImplementedError


class YOLOV4(YOLO):
    ENGINE_PATH = Path(__file__).parent / 'yolov4-512.trt'
    MODEL_PATH = None
    INPUT_SHAPE = (3, 512, 512)
    OUTPUT_NAME = None
    OUTPUT_LAYOUT = None