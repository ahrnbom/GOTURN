./build.sh

DEPLOY=nets/tracker.prototxt
CAFFE_MODEL=nets/models/pretrained_model/tracker.caffemodel

build/test_on_stuff $DEPLOY $CAFFE_MODEL $1
