from pathlib import Path
import tensorrt as trt
import os

from . import calibrator


class ReID:
    PATH = None
    ONNX_PATH = None
    INPUT_SHAPE = None
    OUTPUT_LAYOUT = None

    @classmethod
    def build_engine(cls, trt_logger, batch_size=1, calib_dataset=Path(__file__).parent / 'VOCdevkit' / 'VOC2007' / 'JPEGImages'):
        assert batch_size > 0

        def compute_max_batch_size(n):
            return n if n & (n - 1) == 0 else 1 << int.bit_length(n)
    
        # os.system(f'/usr/src/tensorrt/bin/trtexec --onnx={cls.ONNX_PATH} --fp16 --maxBatch={compute_max_batch_size(batch_size)} --verbose --saveEngine={cls.PATH}')

        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(trt_logger) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, trt_logger) as parser:
            builder.max_workspace_size = 1 << 24
            builder.max_batch_size = compute_max_batch_size(batch_size)
            print('Building engine with batch size:', builder.max_batch_size)
            print('This may take a while...')
            
            if builder.platform_has_fast_fp16:
                builder.fp16_mode = True
            # if builder.platform_has_fast_int8:
            #     builder.int8_mode = True
            #     builder.int8_calibrator = calibrator.SSDEntropyCalibrator(cls.INPUT_SHAPE, data_dir=calib_dataset, cache_file=Path(__file__).parent / 'INT8CacheFile')

            # Parse model file
            with open(cls.ONNX_PATH, 'rb') as model_file:
                parser.parse(model_file.read())
            # Mark last layer as output
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


class OSNet(ReID):
    PATH = Path(__file__).parent / 'TRT_osnet_x1_0_market.bin'
    ONNX_PATH = Path(__file__).parent / 'osnet_x1_0_market' / 'osnet_x1_0.onnx'
    INPUT_SHAPE = (3, 256, 128)
    OUTPUT_LAYOUT = 512

# trt_logger = trt.Logger(trt.Logger.VERBOSE)
# trt.init_libnvinfer_plugins(trt_logger, '')
# OSNet.build_engine(trt_logger, batch_size=32)