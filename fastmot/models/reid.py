from pathlib import Path
import logging
import tensorrt as trt


class ReID:
    PLUGIN_PATH = None
    ENGINE_PATH = None
    MODEL_PATH = None
    INPUT_SHAPE = ()

    @classmethod
    def build_engine(cls, trt_logger, batch_size):
        # if isinstance(batch_size, int):
        #     dynamic_batch_opts = [batch_size]
        # else:
        #     dynamic_batch_opts = batch_size

        EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(trt_logger) as builder, builder.create_network(EXPLICIT_BATCH) as network, \
            trt.OnnxParser(network, trt_logger) as parser:

            # logging.info('Building engine with batch options: %s', dynamic_batch_opts)
            # logging.info('This may take a while...')

            # # Parse model file
            # with open(cls.MODEL_PATH, 'rb') as model_file:
            #     parser.parse(model_file.read())

            # # Create optimization profiles for the batch options
            # config = builder.create_builder_config()
            # config.max_workspace_size = 1 << 30
            # if builder.platform_has_fast_fp16:
            #     config.flags = 1 << int(trt.BuilderFlag.FP16)
            # for batch_size in dynamic_batch_opts:
            #     profile = builder.create_optimization_profile()
            #     batch_shape = (batch_size, *cls.INPUT_SHAPE)
            #     profile.set_shape(network.get_input(0).name, batch_shape, batch_shape, batch_shape) 
            #     config.add_optimization_profile(profile)
            
            # # Reshape input batch size for static model
            # if len(dynamic_batch_opts) == 1 and network.get_input(0).shape[0] != -1:
            #     network.get_input(0).shape = [batch_size, *cls.INPUT_SHAPE]

            # engine = builder.build_engine(network, config)
            # if engine is None:
            #     return None
            # logging.info("Completed creating Engine")
            # with open(cls.ENGINE_PATH, "wb") as engine_file:
            #     engine_file.write(engine.serialize())
            # return engine

            builder.max_workspace_size = 1 << 30
            builder.max_batch_size = batch_size
            logging.info('Building engine with batch size: %d', batch_size)
            logging.info('This may take a while...')
            
            if builder.platform_has_fast_fp16:
                builder.fp16_mode = True

            # Parse model file
            with open(cls.MODEL_PATH, 'rb') as model_file:
                parser.parse(model_file.read())

            # Reshape input to the right batch_size
            network.get_input(0).shape = [batch_size, *cls.INPUT_SHAPE]
            engine = builder.build_cuda_engine(network)
            if engine is None:
                return None
            logging.info("Completed creating Engine")
            with open(cls.ENGINE_PATH, "wb") as engine_file:
                engine_file.write(engine.serialize())
            return engine


class OSNetAIN(ReID):
    ENGINE_PATH = Path(__file__).parent / 'osnet_ain_x1_0_msmt17.trt'
    MODEL_PATH = Path(__file__).parent / 'osnet_ain_x1_0_msmt17' / 'osnet_ain_x1_0_32.onnx'
    INPUT_SHAPE = (3, 256, 128)
    OUTPUT_LAYOUT = 512
    METRIC = 'cosine'


class OSNet025(ReID):
    ENGINE_PATH = Path(__file__).parent / 'osnet_x0_25_msmt17.trt'
    MODEL_PATH = Path(__file__).parent / 'osnet_x0_25_msmt17' / 'osnet_x0_25_dynamic.onnx'
    INPUT_SHAPE = (3, 256, 128)
    OUTPUT_LAYOUT = 512
    METRIC = 'euclidean'
