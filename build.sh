rm -rf build
mkdir build && cd build
cmake -DBUILD_WITH_CUDA=ON -DARCH=sm_80 ..
make -j
