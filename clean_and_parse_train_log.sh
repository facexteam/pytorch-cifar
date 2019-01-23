#!/bin/bash
# @author: zhaoyafei

if [[ ! -f $1.bk ]]; then
sed -i.bk \
        -e 's/^M/\n/g' \
        -e 's/^H\+/ /g' \
        -e 's/\[[>=\.]\+\]//g' \
        $1
else
echo "$1.bk already exists"
fi

train_iters_per_epoch=`grep -m 1 'Step' $1 | cut -d '/' -f3`
echo "train_iters_per_epoch" ${train_iters_per_epoch}
# sed -n "/${train_iters_per_epoch}\/${train_iters_per_epoch}/p" $1 > $1.train-acc.txt

find_str="/${train_iters_per_epoch}\/${train_iters_per_epoch}/p"
find_str=${find_str// /} # replace the spaces
echo 'find_str' $find_str
sed -n $find_str $1 > $1.train-acc.txt

test_iters_per_epoch=`grep -m 1 '/10000' $1 | cut -d '/' -f3`
echo "test_iters_per_epoch" ${test_iters_per_epoch}
# sed -n "/${test_iters_per_epoch}\/${test_iters_per_epoch}/p" $1 > $1.test-acc.txt

find_str="/${test_iters_per_epoch}\/${test_iters_per_epoch}/p"
find_str=${find_str// /} # replace the spaces
echo 'find_str' $find_str
sed -n $find_str $1 > $1.test-acc.txt
