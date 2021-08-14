from pathlib import Path
import logging
import numpy as np
import tensorrt as trt


EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
LOGGER = logging.getLogger(__name__)


class YOLO:
    """Base class for YOLO models.

    Attributes
    ----------
    PLUGIN_PATH : Path, optional
        Path to TensorRT plugin.
    ENGINE_PATH : Path
        Path to TensorRT engine.
        If not found, TensorRT engine will be converted from the ONNX model
        at runtime and cached for later use.
    MODEL_PATH : Path
        Path to ONNX model.
    NUM_CLASSES : int
        Total number of trained classes.
    LETTERBOX : bool
        Keep aspect ratio when resizing.
    NEW_COORDS : bool
        new_coords Darknet parameter for each yolo layer.
    INPUT_SHAPE : tuple
        Input size in the format `(channel, height, width)`.
    LAYER_FACTORS : List[int]
        Scale factors with respect to the input size for each yolo layer.
    SCALES : List[float]
        scale_x_y Darknet parameter for each yolo layer.
    ANCHORS : List[List[int]]
        Anchors grouped by each yolo layer.
    """
    __registry = {}

    PLUGIN_PATH = Path(__file__).parents[1] / 'plugins' / 'libyolo_layer.so'
    ENGINE_PATH = None
    MODEL_PATH = None
    NUM_CLASSES = None
    LETTERBOX = False
    NEW_COORDS = False
    INPUT_SHAPE = None
    LAYER_FACTORS = None
    SCALES = None
    ANCHORS = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__registry[cls.__name__] = cls

    @classmethod
    def get_model(cls, name):
        return cls.__registry[name]

    @classmethod
    def add_plugin(cls, network):
        def get_plugin_creator(plugin_name):
            plugin_creators = trt.get_plugin_registry().plugin_creator_list
            for plugin_creator in plugin_creators:
                if plugin_creator.name == plugin_name:
                    return plugin_creator
            return None

        assert len(cls.LAYER_FACTORS) == network.num_outputs
        assert len(cls.SCALES) == network.num_outputs
        assert len(cls.ANCHORS) == network.num_outputs
        assert all(s >= 1.0 for s in cls.SCALES)

        plugin_creator = get_plugin_creator('YoloLayer_TRT')
        if not plugin_creator:
            raise RuntimeError('Failed to get YoloLayer_TRT plugin creator')

        old_tensors = [network.get_output(i) for i in range(network.num_outputs)]
        new_tensors = []
        for i, old_tensor in enumerate(old_tensors):
            yolo_width = cls.INPUT_SHAPE[2] // cls.LAYER_FACTORS[i]
            yolo_height = cls.INPUT_SHAPE[1] // cls.LAYER_FACTORS[i]
            num_anchors = len(cls.ANCHORS[i]) // 2
            plugin = network.add_plugin_v2(
                [old_tensor],
                plugin_creator.create_plugin('YoloLayer_TRT', trt.PluginFieldCollection([
                    trt.PluginField("yoloWidth", np.array(yolo_width, dtype=np.int32), trt.PluginFieldType.INT32),
                    trt.PluginField("yoloHeight", np.array(yolo_height, dtype=np.int32), trt.PluginFieldType.INT32),
                    trt.PluginField("inputMultiplier", np.array(cls.LAYER_FACTORS[i], dtype=np.int32), trt.PluginFieldType.INT32),
                    trt.PluginField("newCoords", np.array(cls.NEW_COORDS, dtype=np.int32), trt.PluginFieldType.INT32),
                    trt.PluginField("numClasses", np.array(cls.NUM_CLASSES, dtype=np.int32), trt.PluginFieldType.INT32),
                    trt.PluginField("numAnchors", np.array(num_anchors, dtype=np.int32), trt.PluginFieldType.INT32),
                    trt.PluginField("anchors", np.array(cls.ANCHORS[i], dtype=np.float32), trt.PluginFieldType.FLOAT32),
                    trt.PluginField("scaleXY", np.array(cls.SCALES[i], dtype=np.float32), trt.PluginFieldType.FLOAT32),
                ]))
            )
            new_tensors.append(plugin.get_output(0))

        for new_tensor in new_tensors:
            network.mark_output(new_tensor)
        for old_tensor in old_tensors:
            network.unmark_output(old_tensor)
        return network

    @classmethod
    def build_engine(cls, trt_logger, batch_size):
        with trt.Builder(trt_logger) as builder, builder.create_network(EXPLICIT_BATCH) as network, \
            trt.OnnxParser(network, trt_logger) as parser:

            builder.max_batch_size = batch_size
            LOGGER.info('Building engine with batch size: %d', batch_size)
            LOGGER.info('This may take a while...')

            # parse model file
            with open(cls.MODEL_PATH, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    LOGGER.critical('Failed to parse the ONNX file')
                    for err in range(parser.num_errors):
                        LOGGER.error(parser.get_error(err))
                    return None

            # yolo*.onnx is generated with batch size 64
            # reshape input to the right batch size
            net_input = network.get_input(0)
            assert cls.INPUT_SHAPE == net_input.shape[1:]
            net_input.shape = (batch_size, *cls.INPUT_SHAPE)

            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)

            profile = builder.create_optimization_profile()
            profile.set_shape(
                net_input.name,                  # input tensor name
                (batch_size, *cls.INPUT_SHAPE),  # min shape
                (batch_size, *cls.INPUT_SHAPE),  # opt shape
                (batch_size, *cls.INPUT_SHAPE)   # max shape
            )
            config.add_optimization_profile(profile)

            network = cls.add_plugin(network)
            engine = builder.build_engine(network, config)
            if engine is None:
                LOGGER.critical('Failed to build engine')
                return None

            LOGGER.info("Completed creating engine")
            with open(cls.ENGINE_PATH, 'wb') as engine_file:
                engine_file.write(engine.serialize())
            return engine


class YOLOv4(YOLO):
    ENGINE_PATH = Path(__file__).parent / 'yolov4_crowdhuman.trt'
    MODEL_PATH = Path(__file__).parent /  'yolov4_crowdhuman.onnx'
    NUM_CLASSES = 2
    INPUT_SHAPE = (3, 512, 512)
    LAYER_FACTORS = [8, 16, 32]
    SCALES = [1.2, 1.1, 1.05]
    ANCHORS = [[11,22, 24,60, 37,116],
               [54,186, 69,268, 89,369],
               [126,491, 194,314, 278,520]]


"""
The following models are supported but not provided.
Modify paths, # classes, input shape, and anchors according to your Darknet cfg for custom model.
"""

class YOLOv4CSP(YOLO):
    ENGINE_PATH = Path(__file__).parent / 'yolov4-csp.trt'
    MODEL_PATH = Path(__file__).parent /  'yolov4-csp.onnx'
    NUM_CLASSES = 1
    LETTERBOX = True
    NEW_COORDS = True
    INPUT_SHAPE = (3, 512, 512)
    LAYER_FACTORS = [8, 16, 32]
    SCALES = [2.0, 2.0, 2.0]
    ANCHORS = [[12,16, 19,36, 40,28],
               [36,75, 76,55, 72,146],
               [142,110, 192,243, 459,401]]


class YOLOv4xMish(YOLO):
    ENGINE_PATH = Path(__file__).parent / 'yolov4x-mish.trt'
    MODEL_PATH = Path(__file__).parent /  'yolov4x-mish.onnx'
    NUM_CLASSES = 1
    LETTERBOX = True
    NEW_COORDS = True
    INPUT_SHAPE = (3, 640, 640)
    LAYER_FACTORS = [8, 16, 32]
    SCALES = [2.0, 2.0, 2.0]
    ANCHORS = [[12,16, 19,36, 40,28],
               [36,75, 76,55, 72,146],
               [142,110, 192,243, 459,401]]


class YOLOv4P5(YOLO):
    ENGINE_PATH = Path(__file__).parent / 'yolov4-p5.trt'
    MODEL_PATH = Path(__file__).parent /  'yolov4-p5.onnx'
    NUM_CLASSES = 1
    LETTERBOX = True
    NEW_COORDS = True
    INPUT_SHAPE = (3, 896, 896)
    LAYER_FACTORS = [8, 16, 32]
    SCALES = [2.0, 2.0, 2.0]
    ANCHORS = [[13,17, 31,25, 24,51, 61,45],
               [48,102, 119,96, 97,189, 217,184],
               [171,384, 324,451, 616,618, 800,800]]


class YOLOv4P6(YOLO):
    ENGINE_PATH = Path(__file__).parent / 'yolov4-p6.trt'
    MODEL_PATH = Path(__file__).parent /  'yolov4-p6.onnx'
    NUM_CLASSES = 1
    LETTERBOX = True
    NEW_COORDS = True
    INPUT_SHAPE = (3, 1280, 1280)
    LAYER_FACTORS = [8, 16, 32, 64]
    SCALES = [2.0, 2.0, 2.0, 2.0]
    ANCHORS = [[13,17,  31,25,  24,51, 61,45],
               [61,45,  48,102,  119,96,  97,189],
               [97,189,  217,184,  171,384,  324,451],
               [324,451, 545,357, 616,618, 1024,1024]]


class YOLOv4Tiny(YOLO):
    ENGINE_PATH = Path(__file__).parent / 'yolov4-tiny.trt'
    MODEL_PATH = Path(__file__).parent /  'yolov4-tiny.onnx'
    NUM_CLASSES = 1
    INPUT_SHAPE = (3, 416, 416)
    LAYER_FACTORS = [32, 16]
    SCALES = [1.05, 1.05]
    ANCHORS = [[81,82, 135,169, 344,319],
               [23,27, 37,58, 81,82]]


class YOLOv3(YOLO):
    ENGINE_PATH = Path(__file__).parent / 'yolov3.trt'
    MODEL_PATH = Path(__file__).parent /  'yolov3.onnx'
    NUM_CLASSES = 1
    INPUT_SHAPE = (3, 416, 416)
    LAYER_FACTORS = [32, 16, 8]
    SCALES = [1., 1.]
    ANCHORS = [[116,90, 156,198, 373,326],
               [30,61, 62,45, 59,119],
               [10,13, 16,30, 33,23]]


class YOLOv3SPP(YOLO):
    ENGINE_PATH = Path(__file__).parent / 'yolov3-spp.trt'
    MODEL_PATH = Path(__file__).parent /  'yolov3-spp.onnx'
    NUM_CLASSES = 1
    INPUT_SHAPE = (3, 608, 608)
    LAYER_FACTORS = [32, 16, 8]
    SCALES = [1., 1.]
    ANCHORS = [[116,90, 156,198, 373,326],
               [30,61, 62,45, 59,119],
               [10,13, 16,30, 33,23]]


class YOLOv3Tiny(YOLO):
    ENGINE_PATH = Path(__file__).parent / 'yolov3-tiny.trt'
    MODEL_PATH = Path(__file__).parent /  'yolov3-tiny.onnx'
    NUM_CLASSES = 1
    INPUT_SHAPE = (3, 416, 416)
    LAYER_FACTORS = [32, 16]
    SCALES = [1., 1.]
    ANCHORS = [[81,82, 135,169, 344,319],
               [10,14, 23,27, 37,58]]
