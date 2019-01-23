#!/bin/bash
# maintainer: zhaoyafei (https://github.com/walkoncross, zhaoyafei0210@gmail.com)

python train_large_margin_zyf.py --cifar-dir ./data \
    --net resnet20_cifar \
    --gpu-ids 1 \
    --lr-scheduler cosine \
    --lr 0.1 \
    --num-epochs 200 \
    --mom 0.9 \
    --wd 0.0001 \
    --batch-size 256 \
    --data-workers 4 \
    --test-bs 200 \
    --test-dw 4 \
    --loss-type cosine \
    --loss-scale 32 \
    --loss-m 0.5 \
    --loss-b 0 \
    --model-prefix res20-cifar \
    --save-dir checkpoints-res20-cifar-lmcos-coslr-200
    # --no-progress-bar \
    # --resume \
    # --resume-checkpoints checkpoints/ckpt.t7 \
