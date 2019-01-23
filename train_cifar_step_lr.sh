#!/bin/bash
# maintainer: zhaoyafei (https://github.com/walkoncross, zhaoyafei0210@gmail.com)

python main_zyf.py --cifar-dir ./data \
    --net resnet20_cifar \
    --lr-scheduler step \
    --lr 0.1 \
    --num-epochs 320 \
    --lr-factor 0.1 \
    --lr-step-epochs '160,240' \
    --mom 0.9 \
    --wd 0.0001 \
    --batch-size 256 \
    --model-prefix res20-cifar \
    # --no-progress-bar \    
    --save-dir checkpoints-res20-cifar \
