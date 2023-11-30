#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
 
# please set env SDK_ROOT, NDK_ROOT, ANDROID_ABI, MINSDKVERSION
 
# export SDK_ROOT=/root/cmdline-tools/sdk/
# export NDK_ROOT=/root/cmdline-tools/sdk/ndk/24.0.8215888/
 
mkdir -p ${SCRIPT_DIR}/build
cd ${SCRIPT_DIR}/build
 
if [ -z ${ANDROID_ABI} ] ; then
    ANDROID_ABI=arm64-v8a
fi
if [ -z ${MINSDKVERSION} ] ; then
    MINSDKVERSION=21
fi
 
echo "SDK_ROOT:" $SDK_ROOT
echo "NDK_ROOT:" $NDK_ROOT
echo "ANDROID_ABI:" $ANDROID_ABI
echo "MINSDKVERSION:" $MINSDKVERSION
 
${SDK_ROOT}/cmake/3.18.1/bin/cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=${NDK_ROOT}/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=${ANDROID_ABI} \
    -DANDROID_NDK=${NDK_ROOT} \
    -DANDROID_PLATFORM=android-${MINSDKVERSION} \
    -DCMAKE_ANDROID_ARCH_ABI=${ANDROID_ABI} \
    -DCMAKE_ANDROID_NDK=${NDK_ROOT} \
    -DCMAKE_MAKE_PROGRAM=${SDK_ROOT}/cmake/3.18.1/bin/ninja \
    -DCMAKE_SYSTEM_NAME=Android \
    -DCMAKE_SYSTEM_VERSION=${MINSDKVERSION} \
    -DANDROID_STL=c++_static \
    -GNinja \
    ..
if [ $? -ne 0 ]; then
    echo "ERROR: cmake $TARGET_NAME failed"
    exit 1
fi
 
${SDK_ROOT}/cmake/3.18.1/bin/ninja
if [ $? -ne 0 ]; then
    echo "ERROR: build $TARGET_NAME failed"
    exit 1
fi

export DEV_DIR=/data/local/tmp

adb push gemm_bib_nn $DEV_DIR
adb push gemm_iib_nn $DEV_DIR
adb push gemm_iii_nn $DEV_DIR

adb push gemm_iib_tn $DEV_DIR
