# Remove the previous 'build' directory and create a new one
rm -rf build
mkdir build && cd build

## CUDA Platform ###########
# cmake -DBUILD_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="90" -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda -DARCH=sm_90 ..
# make -j
## HIP Platform ############
# Support on MI210
# cmake -DBUILD_WITH_GFX90=ON ..
#
# Support on RX7900
# cmake -DBUILD_WITH_GFX1100=ON ..

## MX Platform ############
#

cmake -DBUILD_WITH_GFX90=ON ..
make -j
