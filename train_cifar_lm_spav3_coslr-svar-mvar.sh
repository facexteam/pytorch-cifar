#!/bin/bash
# maintainer: zhaoyafei (https://github.com/walkoncross, zhaoyafei0210@gmail.com)

overwrite=1

#m_list=(0.05	0.075	0.05	0.075	0.05    0.075)	
#s_list=(64	64	32	32	4	4)
m_list=(0.05	0.075	0.05	0.075)
s_list=(64	64	32	32   )

#m=0.025
n=0
b=0
#b=5
#m_list=(0.1	0.15	0.2	0.25	0.45	0.5	0.55	0.6	0.65	0.7	0.75	0.8	0.85	0.9	0.95	1.0)
len_m=${#m_list[@]}

#s_list=(64	32	16	8	4	2	1)
len_s=${#s_list[@]}
#s=4

#gpu_list=(0	0	0	0	1	1	1	1	2	2	2	2	3	3	3	3)
#gpu_list=(0	1	2	3	0	1	2	3	0	1	2	3	0	1	2	3)	
#gpu_list=(0	1	2	3	4	5	6	7	0	1	2	3	4	5	6	7)	
#gpu_list=(0	1	2	3	1	2	3)	
#gpu_list=(4	5	6	7	4     5       6      )
gpu_list=(4	5	4	5)
len_g=${#gpu_list[@]}

echo 'len_m=' $len_m, 'len_s=' $len_s, 'len_gpu=' $len_g

if [[ $len_m != $len_s || $len_m != $len_g ]]; then
#if [[ $len_m != $len_g ]]; then
#if [[ $len_s != $len_g ]]; then
	echo '===> wrong settings'
else
	for (( i=0; i<$len_g; i++ )); 
	do
		m=${m_list[$i]}
		s=${s_list[$i]}
		gpu=${gpu_list[$i]}
		echo 'm=' $m, 's=' $s, 'gpu=' $gpu

                save_dir=checkpoints-res20-cifar-coslr-200ep-spav3-s${s}-m${m}-n${n}-b${b}
                if [[ -f ${save_dir}/train-loss.txt ]] && [[ ${overwrite} != 1 ]]; then
                        echo ${save_dir}/train-loss.txt ' already exists, skip this setting'
                else
                        echo ${save_dir}/train-loss.txt ' does not exist, start training'

                        nohup python train_large_margin_zyf.py --cifar-dir ./data \
                            --net resnet20_cifar10_nofc \
                            --gpu-ids ${gpu} \
                            --lr-scheduler cosine \
                            --lr 0.1 \
                            --num-epochs 200 \
                            --mom 0.9 \
                            --wd 0.0001 \
                            --batch-size 256 \
                            --data-workers 4 \
                            --test-bs 200 \
                            --test-dw 4 \
                            --loss-type spav3 \
                            --loss-scale ${s}\
                            --loss-m ${m} \
                            --loss-n ${n} \
                            --loss-b ${b} \
                            --model-prefix res20-cifar-coslr-200ep \
                            --no-progress-bar \
                            --save-dir ${save_dir} > train-log-spav3-s${s}-m${m}-n${n}-b${b}.txt &
                            #--save-dir checkpoints-res20-cifar-coslr-200ep-lmcos-s${s}-m${m} > train-log-s${s}-m${m}.txt &
                            # --no-progress-bar \
                            # --resume \
                            # --resume-checkpoints checkpoints/ckpt.t7 \
                fi
	done
fi
