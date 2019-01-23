#!/bin/bash
# maintainer: zhaoyafei (https://github.com/walkoncross, zhaoyafei0210@gmail.com)

python main_zyf.py --cifar-dir ./data \
    --net resnet20_cifar \
    --gpu-ids 2 \
    --lr-scheduler cosine \
    --lr 0.1 \
    --num-epochs 320 \
    --mom 0.9 \
    --wd 0.0001 \
    --batch-size 256 \
    --data-workers 4 \
    --test-bs 200 \
    --test-dw 4 \
    --model-prefix res20-cifar \
    --save-dir checkpoints-res20-cifar-coslr-320ep
    # --no-progress-bar \
    # --resume \
    # --resume-checkpoints checkpoints/ckpt.t7 \
