if [ -z "$1" ]
then
    TEMP_DIR=./
else
    TEMP_DIR=$1/
fi

PASCAL_VOC2007_DATASET=$TEMP_DIR/VOCtrainval_06-Nov-2007.tar
wget -O $PASCAL_VOC2007_DATASET http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
tar -xf $PASCAL_VOC2007_DATASET -C $TEMP_DIR
rm $PASCAL_VOC2007_DATASET