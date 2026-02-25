#!/bin/bash

NDK_PATH=${ANDROID_NDK_HOME:-${ANDROID_NDK:-$HOME/android-ndk-r29}}
API_LEVEL=${ANDROID_API:-24}
ABI=${ANDROID_ABI:-arm64-v8a}
BUILD_DIR=build-android-${ABI}

if [ ! -d "$NDK_PATH" ]; then
    echo "Error: Android NDK not found at $NDK_PATH"
    echo "Set ANDROID_NDK_HOME or ANDROID_NDK environment variable."
    exit 1
fi

TOOLCHAIN=$NDK_PATH/build/cmake/android.toolchain.cmake
if [ ! -f "$TOOLCHAIN" ]; then
    echo "Error: Toolchain file not found: $TOOLCHAIN"
    exit 1
fi

echo "NDK:       $NDK_PATH"
echo "ABI:       $ABI"
echo "API Level: $API_LEVEL"
echo "Build Dir: $BUILD_DIR"
echo ""

mkdir -p $BUILD_DIR && cd $BUILD_DIR

cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN \
    -DANDROID_ABI=$ABI \
    -DANDROID_NATIVE_API_LEVEL=$API_LEVEL \
    -DANDROID_STL=c++_static \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA=OFF \
    -DUSE_ROCM=OFF \
    -DUSE_TFACC=OFF \
    -DPY_API=OFF

make -j$(nproc) main

echo ""
if [ -f main ]; then
    echo "Build succeeded: $BUILD_DIR/main"
    file main
else
    echo "Build failed."
    exit 1
fi
