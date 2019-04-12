#!/bin/bash
# maintainer: zhaoyafei (https://github.com/walkoncross, zhaoyafei0210@gmail.com)

s=32
m=1
l=0
gpu=0
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
    --loss-type asoftmax \
    --loss-scale ${s} \
    --loss-m ${m} \
    --loss-lambda ${l} \
    --model-prefix res20-cifar-coslr-200ep \
    --save-dir checkpoints-res20-cifar-coslr-${epoch}ep-sasoftmax-s${s}-m${m}-l${l}
    # --no-progress-bar \
    # --resume \
    # --resume-checkpoints checkpoints/ckpt.t7 \
