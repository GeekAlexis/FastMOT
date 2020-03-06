from pathlib import Path

class MobileNetV1:
    PATH = Path(__file__).parent / 'TRT_ssd_mobilenet_v1_coco_2018_01_28.bin'
    OUTPUT_NAME = ['NMS']
    NMS_THRESH = 0.5
    TOPK = 100
    INPUT_SHAPE = (3, 300, 300)
    OUTPUT_LAYOUT = 7


class MobileNetV2:
    PATH = Path(__file__).parent / 'TRT_ssd_mobilenet_v2_coco_2018_03_29.bin'
    OUTPUT_NAME = ['NMS']
    NMS_THRESH = 0.5
    TOPK = 100
    INPUT_SHAPE = (3, 300, 300)
    OUTPUT_LAYOUT = 7


class InceptionV2:
    PATH = Path(__file__).parent / 'TRT_ssd_inception_v2_coco_2017_11_17.bin'
    OUTPUT_NAME = ['NMS']
    NMS_THRESH = 0.5
    TOPK = 100
    INPUT_SHAPE = (3, 300, 300)
    OUTPUT_LAYOUT = 7


COCO_LABELS = [
    'unlabeled',
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'street sign',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'hat',
    'backpack',
    'umbrella',
    'shoe',
    'eye glasses',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'plate',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'mirror',
    'dining table',
    'window',
    'desk',
    'toilet',
    'door',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'blender',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
]