rm -rf build
mkdir build && cd build
cmake -DBUILD_WITH_MI210=ON ..
make -j
