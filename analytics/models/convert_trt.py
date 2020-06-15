import os
import pycuda.autoinit
import pycuda.driver as cuda

import uff
import tensorrt as trt
import graphsurgeon as gs
from ssd import InceptionV2 as model
# from ssd import MobileNetV1 as model
# from ssd import MobileNetV2 as model
# from configs import model_ssd_inception_v2_coco_2018_01_28 as model
# from configs import model_ssd_mobilenet_v1_coco_2018_01_28 as model
# from configs import model_ssd_mobilenet_v2_coco_2018_03_29 as model

# initialize
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
runtime = trt.Runtime(TRT_LOGGER)

# compile model into TensorRT
dynamic_graph = model.add_plugin(gs.DynamicGraph(str(model.TF_PATH)))
uff_model = uff.from_tensorflow(dynamic_graph.as_graph_def(), model.OUTPUT_NAME, output_filename='tmp.uff')

with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
    builder.max_workspace_size = 1 << 28
    builder.max_batch_size = 1
    builder.fp16_mode = True
    # builder.int8_mode = True

    parser.register_input('Input', model.INPUT_SHAPE)
    parser.register_output('MarkOutput_0')
    parser.parse('tmp.uff', network)
    engine = builder.build_cuda_engine(network)

    buf = engine.serialize()
    with open(model.PATH, 'wb') as f:
        f.write(buf)
