from pathlib import Path
import logging
import tensorrt as trt


EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
LOGGER = logging.getLogger(__name__)


class ReID:
    """Base class for ReID models.

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
    INPUT_SHAPE : tuple
        Input size in the format `(channel, height, width)`.
    OUTPUT_LAYOUT : int
        Feature dimension output by the model.
    METRIC : {'euclidean', 'cosine'}
        Distance metric used to match features.
    """
    __registry = {}

    PLUGIN_PATH = None
    ENGINE_PATH = None
    MODEL_PATH = None
    INPUT_SHAPE = None
    OUTPUT_LAYOUT = None
    METRIC = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__registry[cls.__name__] = cls

    @classmethod
    def get_model(cls, name):
        return cls.__registry[name]

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

            # engine = builder.build_cuda_engine(network)
            engine = builder.build_engine(network, config)
            if engine is None:
                LOGGER.critical('Failed to build engine')
                return None

            LOGGER.info("Completed creating engine")
            with open(cls.ENGINE_PATH, 'wb') as engine_file:
                engine_file.write(engine.serialize())
            return engine


class OSNet025(ReID):
    ENGINE_PATH = Path(__file__).parent / 'osnet_x0_25_msmt17.trt'
    MODEL_PATH = Path(__file__).parent / 'osnet_x0_25_msmt17.onnx'
    INPUT_SHAPE = (3, 256, 128)
    OUTPUT_LAYOUT = 512
    METRIC = 'euclidean'


class OSNet10(ReID):
    """Multi-source model trained on MSMT17, DukeMTMC, and CUHK03, not provided."""
    ENGINE_PATH = Path(__file__).parent / 'osnet_x1_0_msdc.trt'
    MODEL_PATH = Path(__file__).parent / 'osnet_x1_0_msdc.onnx'
    INPUT_SHAPE = (3, 256, 128)
    OUTPUT_LAYOUT = 512
    METRIC = 'cosine'
