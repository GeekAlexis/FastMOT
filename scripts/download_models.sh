#!/bin/bash

BASEDIR=$(dirname "$0")
DIR=$BASEDIR/../fastmot/models

set -e

pip3 install gdown

gdown https://drive.google.com/uc?id=1MLC2lKnQvAQgBKZP1EXB6UdmqujY9qVd -O $DIR/osnet_x0_25_msmt17.onnx
gdown https://drive.google.com/uc?id=1-Cqk2P72P4feYLJGtJFPcCxN5JttzTfX -O $DIR/ssd_inception_v2_coco.pb
gdown https://drive.google.com/uc?id=1IfSveiXaub-L6PO9mqne5pk2EByzb25z -O $DIR/ssd_mobilenet_v1_coco.pb
gdown https://drive.google.com/uc?id=1ste0fQevAjF4UqD3JsCtu1rUAwCTmETN -O $DIR/ssd_mobilenet_v2_coco.pb
gdown https://drive.google.com/uc?id=1-kXZpA6y8pNbDMMD7N--IWIjwqqnAIGZ -O $DIR/yolov4_crowdhuman.onnx
gdown https://drive.google.com/uc?id=1Nyxj_muAKwQOrdk6ftdcq0e0di51nTi- -O $DIR/veriwild_r50ibn.onnx
gdown https://drive.google.com/uc?id=1nk5GqReWBZ4xWoUgNdjqp8fusD8wT9ch -O $DIR/vehicleid_r50ibn.onnx