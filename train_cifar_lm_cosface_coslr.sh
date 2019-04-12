#!/bin/bash
# maintainer: zhaoyafei (https://github.com/walkoncross, zhaoyafei0210@gmail.com)

s=16
m=0.5
n=0
b=0
lambda=0

gpu=0
epoch=200
loss_type=cosface

python train_large_margin_zyf.py --cifar-dir ./data \
    --net resnet20_cifar10_nofc \
    --gpu-ids ${gpu} \
    --lr-scheduler cosine \
    --lr 0.1 \
    --num-epochs ${epoch} \
    --mom 0.9 \
    --wd 0.0001 \
    --batch-size 256 \
    --data-workers 4 \
    --test-bs 200 \
    --test-dw 4 \
    --loss-type ${loss_type} \
    --loss-scale ${s} \
    --loss-m ${m} \
    --loss-n ${n} \
    --loss-b ${b} \
    --loss-lambda ${lambda} \
    --model-prefix res20-cifar \
    --save-dir checkpoints-res20-cifar-coslr-${epoch}ep-${loss_type}-s${s}-m${m}-n${n}-b${b}-l${lambda}
    # --save-dir checkpoints-res20-cifar-coslr-200ep-${loss_type}-s32-coslr-200-m0.35-test
    # --no-progress-bar \
    # --resume \
    # --resume-checkpoints checkpoints/ckpt.t7 \
