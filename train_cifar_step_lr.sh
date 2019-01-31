#!/bin/bash
# maintainer: zhaoyafei (https://github.com/walkoncross, zhaoyafei0210@gmail.com)

python main_zyf.py --cifar-coslr-200ep-dir ./data \
    --net resnet20_cifar-coslr-200ep \
    --gpu-ids 0 \
    --lr-scheduler step \
    --lr 0.1 \
    --num-epochs 320 \
    --lr-factor 0.1 \
    --lr-step-epochs '160,240' \
    --mom 0.9 \
    --wd 0.0001 \
    --batch-size 256 \
    --data-workers 4 \
    --test-bs 200 \
    --test-dw 4 \
    --model-prefix res20-cifar-coslr-200ep \
    --save-dir checkpoints-res20-cifar-coslr-200ep
    # --no-progress-bar \
    # --resume \
    # --resume-checkpoints checkpoints/ckpt.t7 \
