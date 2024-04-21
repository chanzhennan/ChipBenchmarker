# Remove the previous 'build' directory and create a new one
rm -rf build
mkdir build && cd build

## CUDA Platform ###########
# cmake -DBUILD_WITH_CUDA=ON -DARCH=sm_80 ..

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
