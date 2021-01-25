#!/bin/bash
# Test everything works

CHECK=/mnt/ilcompf9d1/user/mgharbi/code/karima/check
BROKEN=/mnt/ilcompf9d1/user/mgharbi/code/karima/check/broken
WORKING=/mnt/ilcompf9d1/user/mgharbi/code/karima/check/working
mkdir -p $WORKING
mkdir -p $BROKEN

echo "GPU info"
cat /proc/driver/nvidia/gpus/*/information

echo "Host NVIDIA driver"
cat /proc/driver/nvidia/version

echo "Testing GPU configuration..."
if ! nvidia-smi; then
    echo $(hostname) "nvidia-smi failed, aborting."
    touch $BROKEN/$(hostname)
    exit
fi
echo $(hostname) "nvidia-smi works"

echo "Testing PyTorch configuration..."
if ! python -c "import torch as th; print(th.zeros(3, 3).cuda())" ; then
    echo $(hostname) "PyTorch CUDA failed, aborting."
    touch $BROKEN/$(hostname)
    exit
fi
echo $(hostname) "PyTorch CUDA works"

echo "$(hostname) works fine"
touch $WORKING/$(hostname)

echo "Memory"
free -h

# TODO: check if data is already there
freespace=$(df | grep ssd | awk '{print $4}')
echo "Disk space" $freespace
if (( $freespace < 90000000 )); then
    echo "not enough disk space to copy demosaicking dataset"
    echo $(hostname) "not enough space for dataset"
    touch $BROKEN/$(hostname)
fi
echo "Sufficient free space for data" $freespace

exit

echo "SHM space"
df -h | grep shm
