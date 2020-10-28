from pathlib import Path
import logging
import tensorrt as trt


LOGGER = logging.getLogger(__name__)


class SSD:
    PLUGIN_PATH = None
    ENGINE_PATH = None
    MODEL_PATH = None
    INPUT_SHAPE = ()
    OUTPUT_NAME = None

    @classmethod
    def add_plugin(cls, graph):
        raise NotImplementedError

    @classmethod
    def build_engine(cls, trt_logger, batch_size, calib_dataset=Path.home() / 'VOCdevkit' / 'VOC2007' / 'JPEGImages'):
        import graphsurgeon as gs
        import uff
        from . import calibrator

        # convert TensorFlow graph into UFF
        dynamic_graph = gs.DynamicGraph(str(cls.MODEL_PATH))
        dynamic_graph = cls.add_plugin(dynamic_graph)
        uff_model = uff.from_tensorflow(dynamic_graph.as_graph_def(), [cls.OUTPUT_NAME], quiet=True)

        with trt.Builder(trt_logger) as builder, builder.create_network() as network, trt.UffParser() as parser:
            builder.max_workspace_size = 1 << 30
            builder.max_batch_size = batch_size
            LOGGER.info('Building engine with batch size: %d', batch_size)
            LOGGER.info('This may take a while...')

            if builder.platform_has_fast_fp16:
                builder.fp16_mode = True
            if builder.platform_has_fast_int8:
                builder.int8_mode = True
                builder.int8_calibrator = calibrator.SSDEntropyCalibrator(cls.INPUT_SHAPE,
                                                                          data_dir=calib_dataset,
                                                                          cache_file=Path(__file__).parent /
                                                                          f'{cls.__name__}_calib_cache')

            parser.register_input('Input', cls.INPUT_SHAPE)
            parser.register_output('MarkOutput_0')
            parser.parse_buffer(uff_model, network)
            engine = builder.build_cuda_engine(network)
            if engine is None:
                LOGGER.critical('Failed to build engine')
                return None

            LOGGER.info("Completed creating engine")
            with open(cls.ENGINE_PATH, 'wb') as engine_file:
                engine_file.write(engine.serialize())
            return engine


class SSDMobileNetV1(SSD):
    ENGINE_PATH = Path(__file__).parent / 'ssd_mobilenet_v1_coco.trt'
    MODEL_PATH = Path(__file__).parent / 'ssd_mobilenet_v1_coco.pb'
    INPUT_SHAPE = (3, 300, 300)
    OUTPUT_NAME = 'NMS'
    NMS_THRESH = 0.5
    TOPK = 100

    @classmethod
    def add_plugin(cls, graph):
        import tensorflow as tf
        import graphsurgeon as gs

        all_assert_nodes = graph.find_nodes_by_op("Assert")
        graph.remove(all_assert_nodes, remove_exclusive_dependencies=True)
        all_identity_nodes = graph.find_nodes_by_op("Identity")
        graph.forward_inputs(all_identity_nodes)

        Input = gs.create_plugin_node(
            name="Input",
            op="Placeholder",
            dtype=tf.float32,
            shape=[1, *cls.INPUT_SHAPE]
        )

        PriorBox = gs.create_plugin_node(
            name="MultipleGridAnchorGenerator",
            op="GridAnchor_TRT",
            minSize=0.2,
            maxSize=0.95,
            aspectRatios=[1.0, 2.0, 0.5, 3.0, 0.33],
            variance=[0.1, 0.1, 0.2, 0.2],
            featureMapShapes=[19, 10, 5, 3, 2, 1],
            numLayers=6
        )

        NMS = gs.create_plugin_node(
            name="NMS",
            op="NMS_TRT",
            shareLocation=1,
            varianceEncodedInTarget=0,
            backgroundLabelId=0,
            confidenceThreshold=1e-8,
            nmsThreshold=cls.NMS_THRESH,
            topK=100,
            keepTopK=100,
            numClasses=91,
            inputOrder=[0, 2, 1],
            confSigmoid=1,
            isNormalized=1
        )

        concat_priorbox = gs.create_node(
            "concat_priorbox",
            op="ConcatV2",
            dtype=tf.float32,
            axis=2
        )

        concat_box_loc = gs.create_plugin_node(
            "concat_box_loc",
            op="FlattenConcat_TRT",
            dtype=tf.float32,
            axis=1,
            ignoreBatch=0
        )

        concat_box_conf = gs.create_plugin_node(
            "concat_box_conf",
            op="FlattenConcat_TRT",
            dtype=tf.float32,
            axis=1,
            ignoreBatch=0
        )

        namespace_plugin_map = {
            "MultipleGridAnchorGenerator": PriorBox,
            "Postprocessor": NMS,
            "Preprocessor": Input,
            "ToFloat": Input,
            "image_tensor": Input,
            "MultipleGridAnchorGenerator/Concatenate": concat_priorbox,
            "concat": concat_box_loc,
            "concat_1": concat_box_conf
        }

        # Create a new graph by collapsing namespaces
        graph.collapse_namespaces(namespace_plugin_map)
        # Remove the outputs, so we just have a single output node (NMS).
        # If remove_exclusive_dependencies is True, the whole graph will be removed!
        graph.remove(graph.graph_outputs, remove_exclusive_dependencies=False)
        graph.find_nodes_by_op("NMS_TRT")[0].input.remove("Input")
        graph.find_nodes_by_name("Input")[0].input.remove("image_tensor:0")

        return graph


class SSDMobileNetV2(SSD):
    ENGINE_PATH = Path(__file__).parent / 'ssd_mobilenet_v2_coco.trt'
    MODEL_PATH = Path(__file__).parent / 'ssd_mobilenet_v2_coco.pb'
    INPUT_SHAPE = (3, 300, 300)
    OUTPUT_NAME = 'NMS'
    NMS_THRESH = 0.5
    TOPK = 100

    @classmethod
    def add_plugin(cls, graph):
        import tensorflow as tf
        import graphsurgeon as gs

        all_assert_nodes = graph.find_nodes_by_op("Assert")
        graph.remove(all_assert_nodes, remove_exclusive_dependencies=True)
        all_identity_nodes = graph.find_nodes_by_op("Identity")
        graph.forward_inputs(all_identity_nodes)

        Input = gs.create_plugin_node(
            name="Input",
            op="Placeholder",
            dtype=tf.float32,
            shape=[1, *cls.INPUT_SHAPE]
        )

        PriorBox = gs.create_plugin_node(
            name="GridAnchor",
            op="GridAnchor_TRT",
            minSize=0.2,
            maxSize=0.95,
            aspectRatios=[1.0, 2.0, 0.5, 3.0, 0.33],
            variance=[0.1, 0.1, 0.2, 0.2],
            featureMapShapes=[19, 10, 5, 3, 2, 1],
            numLayers=6
        )

        NMS = gs.create_plugin_node(
            name="NMS",
            op="NMS_TRT",
            shareLocation=1,
            varianceEncodedInTarget=0,
            backgroundLabelId=0,
            confidenceThreshold=1e-8,
            nmsThreshold=cls.NMS_THRESH,
            topK=100,
            keepTopK=100, 
            numClasses=91,
            inputOrder=[1, 0, 2],
            confSigmoid=1,
            isNormalized=1
        )

        concat_priorbox = gs.create_node(
            "concat_priorbox",
            op="ConcatV2",
            dtype=tf.float32,
            axis=2
        )

        concat_box_loc = gs.create_plugin_node(
            "concat_box_loc",
            op="FlattenConcat_TRT",
            dtype=tf.float32,
            axis=1,
            ignoreBatch=0
        )

        concat_box_conf = gs.create_plugin_node(
            "concat_box_conf",
            op="FlattenConcat_TRT",
            dtype=tf.float32,
            axis=1,
            ignoreBatch=0
        )

        namespace_plugin_map = {
            "MultipleGridAnchorGenerator": PriorBox,
            "Postprocessor": NMS,
            "Preprocessor": Input,
            "ToFloat": Input,
            "image_tensor": Input,
            "Concatenate": concat_priorbox,
            "concat": concat_box_loc,
            "concat_1": concat_box_conf
        }

        # Create a new graph by collapsing namespaces
        graph.collapse_namespaces(namespace_plugin_map)
        # Remove the outputs, so we just have a single output node (NMS).
        # If remove_exclusive_dependencies is True, the whole graph will be removed!
        graph.remove(graph.graph_outputs, remove_exclusive_dependencies=False)
        graph.find_nodes_by_op("NMS_TRT")[0].input.remove("Input")

        return graph


class SSDInceptionV2(SSD):
    ENGINE_PATH = Path(__file__).parent / 'ssd_inception_v2_coco.trt'
    MODEL_PATH = Path(__file__).parent / 'ssd_inception_v2_coco.pb'
    INPUT_SHAPE = (3, 300, 300)
    OUTPUT_NAME = 'NMS'
    NMS_THRESH = 0.5
    TOPK = 100

    @classmethod
    def add_plugin(cls, graph):
        import tensorflow as tf
        import graphsurgeon as gs

        all_assert_nodes = graph.find_nodes_by_op("Assert")
        graph.remove(all_assert_nodes, remove_exclusive_dependencies=True)
        all_identity_nodes = graph.find_nodes_by_op("Identity")
        graph.forward_inputs(all_identity_nodes)

        # Create TRT plugin nodes to replace unsupported ops in Tensorflow graph
        Input = gs.create_plugin_node(
            name="Input",
            op="Placeholder",
            dtype=tf.float32,
            shape=[1, *cls.INPUT_SHAPE]
        )

        PriorBox = gs.create_plugin_node(
            name="GridAnchor",
            op="GridAnchor_TRT",
            minSize=0.2,
            maxSize=0.95,
            aspectRatios=[1.0, 2.0, 0.5, 3.0, 0.33],
            variance=[0.1, 0.1, 0.2, 0.2],
            featureMapShapes=[19, 10, 5, 3, 2, 1],
            numLayers=6
        )

        NMS = gs.create_plugin_node(
            name="NMS",
            op="NMS_TRT",
            shareLocation=1,
            varianceEncodedInTarget=0,
            backgroundLabelId=0,
            confidenceThreshold=1e-8,
            nmsThreshold=cls.NMS_THRESH,
            topK=100,
            keepTopK=100,
            numClasses=91,
            inputOrder=[0, 2, 1],
            confSigmoid=1,
            isNormalized=1
        )

        concat_priorbox = gs.create_node(
            "concat_priorbox",
            op="ConcatV2",
            dtype=tf.float32,
            axis=2
        )

        concat_box_loc = gs.create_plugin_node(
            "concat_box_loc",
            op="FlattenConcat_TRT",
            dtype=tf.float32,
            axis=1,
            ignoreBatch=0
        )

        concat_box_conf = gs.create_plugin_node(
            "concat_box_conf",
            op="FlattenConcat_TRT",
            dtype=tf.float32,
            axis=1,
            ignoreBatch=0
        )

        # Create a mapping of namespace names -> plugin nodes.
        namespace_plugin_map = {
            "MultipleGridAnchorGenerator": PriorBox,
            "Postprocessor": NMS,
            "Preprocessor": Input,
            "ToFloat": Input,
            "image_tensor": Input,
            "MultipleGridAnchorGenerator/Concatenate": concat_priorbox,
            "concat": concat_box_loc,
            "concat_1": concat_box_conf
        }

        # Create a new graph by collapsing namespaces
        graph.collapse_namespaces(namespace_plugin_map)
        # Remove the outputs, so we just have a single output node (NMS).
        # If remove_exclusive_dependencies is True, the whole graph will be removed!
        graph.remove(graph.graph_outputs, remove_exclusive_dependencies=False)
        return graph
