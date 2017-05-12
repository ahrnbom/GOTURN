if [ -d build ] ; then
echo "Build path already exists."
else
mkdir build
fi
cd build
cmake -D Caffe_DIR=/caffe/ ..
make
cd ..
