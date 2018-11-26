umask 0007

#!/bin/bash
./env/bin/python main.py \
    --model resnet50_md \
    --output-dir /visinf/projects_students/monolab/results \
    --epochs 50 \
    --batch-size 32 \
    --device cuda:0 \
    --data-dir /visinf/projects_students/monolab/data/kitti \
    --filenames-file resources/filenames/kitti_train_files.txt \
    --val-filenames-file resources/filenames/kitti_val_files.txt \
    --use-multiple-gpu \
    --log-level info \
    --tag resnet50_md_slang 

echo $! > run.pid
