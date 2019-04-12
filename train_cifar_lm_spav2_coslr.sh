#!/bin/bash
# maintainer: zhaoyafei (https://github.com/walkoncross, zhaoyafei0210@gmail.com)

s=16
m=3
n=2
b=0
gpu=1
epoch=200

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
    --loss-type spav2 \
    --loss-scale ${s} \
    --loss-m ${m} \
    --loss-n ${n} \
    --loss-b ${b} \
    --model-prefix res20-cifar-coslr-200ep \
    --save-dir checkpoints-res20-cifar-coslr-200ep-spav2-s${s}-coslr-${epoch}ep-m${m}-n${n}-b${b}
    # --save-dir checkpoints-res20-cifar-coslr-200ep-spa-s32-coslr-200-m0.35-test
    # --no-progress-bar \
    # --resume \
    # --resume-checkpoints checkpoints/ckpt.t7 \
