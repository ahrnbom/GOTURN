./build.sh
sleep 5

DEPLOY=nets/tracker.prototxt
CAFFE_MODEL=nets/models/pretrained_model/tracker.caffemodel

build/test_on_stuff $DEPLOY $CAFFE_MODEL
