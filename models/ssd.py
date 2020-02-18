class MobileNetV1:
    PATH = 'models/TRT_ssd_mobilenet_v1_coco_2018_01_28.bin'
    OUTPUT_NAME = ['NMS']
    TOPK = 100
    INPUT_SHAPE = (3, 300, 300)
    OUTPUT_LAYOUT = 7


class MobileNetV2:
    PATH = 'models/TRT_ssd_mobilenet_v2_coco_2018_03_29.bin'
    OUTPUT_NAME = ['NMS']
    TOPK = 100
    INPUT_SHAPE = (3, 300, 300)
    OUTPUT_LAYOUT = 7


class InceptionV2:
    PATH = 'models/TRT_ssd_inception_v2_coco_2017_11_17.bin'
    OUTPUT_NAME = ['NMS']
    TOPK = 100
    INPUT_SHAPE = (3, 300, 300)
    OUTPUT_LAYOUT = 7
