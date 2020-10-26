#!/bin/bash

BASEDIR=$(dirname "$0")
DIR=$BASEDIR/../fastmot/models

set -e

sudo -H pip3 install gdown

gdown https://drive.google.com/uc?id=19APITvaqv_Ts5j3zkbo6MWCTvwTmka_T -O $DIR/osnet_ain_x1_0_msmt17.onnx
gdown https://drive.google.com/uc?id=1hC3dFrjPM_olJ5utDCUkkMUqXIMIybKq -O $DIR/osnet_x0_25_msmt17.onnx
gdown https://drive.google.com/uc?id=1-Cqk2P72P4feYLJGtJFPcCxN5JttzTfX -O $DIR/ssd_inception_v2_coco.pb
gdown https://drive.google.com/uc?id=1IfSveiXaub-L6PO9mqne5pk2EByzb25z -O $DIR/ssd_mobilenet_v1_coco.pb
gdown https://drive.google.com/uc?id=1ste0fQevAjF4UqD3JsCtu1rUAwCTmETN -O $DIR/ssd_mobilenet_v2_coco.pb
gdown https://drive.google.com/uc?id=1-kXZpA6y8pNbDMMD7N--IWIjwqqnAIGZ -O $DIR/yolov4_crowdhuman.onnx
