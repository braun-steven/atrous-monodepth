umask 0007

#!/bin/bash
nohup ./env/bin/python test.py \
    --model deeplab \
    --output-stride 32 \
    --encoder-dilations 1 1 1 1 \
    --atrous-rates 1 1 1 1 \
    --device cuda:0 \
    --data-dir /visinf/projects_students/monolab/data/kitti \
    --model-path /visinf/projects_students/monolab/results/run_19-01-28_15h:11m_deeplab_stride32_encoder_1-1-1-1_aspp_1-1-1-1_skips/checkpoints/last-model.pth \
    --output-dir /visinf/projects_students/monolab/results/test-vkitti/deeplab_stride32_aspp1111 \
    --test-filenames-file resources/filenames/vkitti_test_files.txt \
    --eval vkitti \
    --log-level info &
echo $! > run.pid
