from pathlib import Path
import tensorrt as trt


class ReID:
    PATH = None
    ONNX_PATH = None
    INPUT_SHAPE = None
    OUTPUT_LAYOUT = None

    @classmethod
    def build_engine(cls, trt_logger, batch_size=32):
        def round_up(n):
            return n if n & (n - 1) == 0 else 1 << int.bit_length(n)
    
        # /usr/src/tensorrt/bin/trtexec --onnx=osnet_x0_5_market/osnet_x0_5.onnx --fp16 --explicitBatch --saveEngine=TRT_osnet_x0_5_market.bin

        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(trt_logger) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, trt_logger) as parser:
            builder.max_workspace_size = 1 << 30
            batch_size = round_up(batch_size)
            builder.max_batch_size = batch_size
            print('Building engine with batch size:', builder.max_batch_size)
            print('This may take a while...')
            
            if builder.platform_has_fast_fp16:
                builder.fp16_mode = True

            # Parse model file
            with open(cls.ONNX_PATH, 'rb') as model_file:
                parser.parse(model_file.read())

            # Reshape input to the correct batch_size
            network.get_input(0).shape = [batch_size, *cls.INPUT_SHAPE]
            # Mark last layer as output
            print(network.get_input(0).name)
            last_layer = network.get_layer(network.num_layers - 1)
            if not last_layer.get_output(0):
                network.mark_output(last_layer.get_output(0))
            engine = builder.build_cuda_engine(network)
            if engine is None:
                return None
            print("Completed creating Engine")
            with open(cls.PATH, "wb") as f:
                f.write(engine.serialize())
            return engine


class OSNetAIN(ReID):
    PATH = Path(__file__).parent / 'TRT_osnet_ain_x1_0_msmt17.bin'
    ONNX_PATH = Path(__file__).parent / 'osnet_ain_x1_0_msmt17' / 'osnet_ain_x1_0.onnx'
    INPUT_SHAPE = (3, 256, 128)
    OUTPUT_LAYOUT = 512
    METRIC = 'cosine'


class OSNet05(ReID):
    PATH = Path(__file__).parent / 'TRT_osnet_x0_5_msmt17.bin'
    ONNX_PATH = Path(__file__).parent / 'osnet_x0_5_msmt17' / 'osnet_x0_5.onnx'
    INPUT_SHAPE = (3, 256, 128)
    OUTPUT_LAYOUT = 512
    METRIC = 'euclidean'


class OSNet025(ReID):
    PATH = Path(__file__).parent / 'TRT_osnet_x0_25_msmt17.bin'
    ONNX_PATH = Path(__file__).parent / 'osnet_x0_25_msmt17' / 'osnet_x0_25.onnx'
    INPUT_SHAPE = (3, 256, 128)
    OUTPUT_LAYOUT = 512
    METRIC = 'euclidean'


class OSNet025FP32(ReID):
    PATH = Path(__file__).parent / 'TRT_osnet_x0_25_msmt17_fp32.bin'
    ONNX_PATH = Path(__file__).parent / 'osnet_x0_25_msmt17' / 'osnet_x0_25.onnx'
    INPUT_SHAPE = (3, 256, 128)
    OUTPUT_LAYOUT = 512
    METRIC = 'euclidean'


# trt_logger = trt.Logger(trt.Logger.INFO)
# trt.init_libnvinfer_plugins(trt_logger, '')
# OSNetAIN.build_engine(trt_logger, batch_size=32)