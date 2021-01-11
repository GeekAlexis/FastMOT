from pathlib import Path
import logging
import numpy as np
import tensorrt as trt


EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
LOGGER = logging.getLogger(__name__)


class YOLO:
    PLUGIN_PATH = Path(__file__).parents[1] / 'plugins' / 'libyolo_layer.so'
    ENGINE_PATH = None
    MODEL_PATH = None
    NUM_CLASSES = None
    LETTER_BOX = False
    NEW_COORDS = False
    INPUT_SHAPE = ()
    LAYER_FACTORS = []
    SCALES = []
    ANCHORS = []

    @classmethod
    def add_plugin(cls, network):
        """
        Adapted from https://github.com/jkjung-avt/tensorrt_demos
        """
        def get_plugin_creator(plugin_name):
            plugin_creators = trt.get_plugin_registry().plugin_creator_list
            for plugin_creator in plugin_creators:
                if plugin_creator.name == plugin_name:
                    return plugin_creator
            return None

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

            builder.max_workspace_size = 1 << 30
            builder.max_batch_size = batch_size
            LOGGER.info('Building engine with batch size: %d', batch_size)
            LOGGER.info('This may take a while...')

            if builder.platform_has_fast_fp16:
                builder.fp16_mode = True

            # parse model file
            with open(cls.MODEL_PATH, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    LOGGER.critical('Failed to parse the ONNX file')
                    for err in range(parser.num_errors):
                        LOGGER.error(parser.get_error(err))
                    return None

            # yolo*.onnx is generated with batch size 64
            # reshape input to the right batch size
            network.get_input(0).shape = [batch_size, *cls.INPUT_SHAPE]

            network = cls.add_plugin(network)
            engine = builder.build_cuda_engine(network)
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
    ANCHORS = [[11, 22, 24, 60, 37, 116],
               [54, 186, 69, 268, 89, 369],
               [126, 491, 194, 314, 278, 520]]


class YOLOv4CSP(YOLO):
    ENGINE_PATH = Path(__file__).parent / 'yolov4_csp_crowdhuman.trt'
    MODEL_PATH = Path(__file__).parent /  'yolov4_csp_crowdhuman.onnx'
    NUM_CLASSES = 1
    LETTER_BOX = True
    NEW_COORDS = True
    INPUT_SHAPE = (3, 512, 512)
    LAYER_FACTORS = [8, 16, 32]
    SCALES = [2.0, 2.0, 2.0]
    ANCHORS = [[10, 24, 23, 55, 37, 92],
               [53, 135, 67, 188, 98, 251],
               [138, 168, 138, 341, 241, 283]]
