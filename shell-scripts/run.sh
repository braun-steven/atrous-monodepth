umask 0007

#!/bin/bash
nohup ./env/bin/python main.py \
    --model deeplab \
    --output-stride 16 \
    --encoder-dilations 1 1 1 2 \
    --atrous-rates 1 3 5 7 \
    --learning-rate 2e-4 \
    --epochs 50 \
    --batch-size 16 \
    --cuda-device-ids 1 \
    --data-dir /visinf/projects_students/monolab/data/kitti \
    --filenames-file resources/filenames/kitti_train_files.txt \
    --val-filenames-file resources/filenames/kitti_val_files.txt \
    --test-filenames-file resources/filenames/kitti_stereo_2015_test_files.txt \
    --eval kitti-gt \
    --log-level info \
    --tag stride16_atrous-1-3-5-7 \
    --notify your@email.address &
echo $! > run.pid
