#!/bin/bash

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

mkdir -p $CUDA_MPS_PIPE_DIRECTORY
chmod 1777 $CUDA_MPS_PIPE_DIRECTORY
mkdir -p $CUDA_MPS_LOG_DIRECTORY
chmod 1777 $CUDA_MPS_LOG_DIRECTORY

ps aux | grep nvidia-cuda-mps | grep -v grep > /dev/null
if [ $? -ne 0 ]; then
    echo "Starting MPS Server"
    CUDA_VISIBLE_DEVICES=0 \
    CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY \
    CUDA_MPS_LOG_DIRECTORY=$CUDA_MPS_LOG_DIRECTORY \
    nvidia-cuda-mps-control -d 
    USERID=`id -u $USER`
    echo set_default_active_thread_percentage 30.0 | nvidia-cuda-mps-control
    echo start_server -uid $USERID | nvidia-cuda-mps-control
fi
