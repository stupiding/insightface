#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=1
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice
export CUDA_VISIBLE_DEVICES='8,9'

DATA_DIR=../datasets/gy50_ohem/pos.rec,../datasets/gy50_ohem/neg.rec

NETWORK=fresnet,100
MODELDIR="../models/gy50_ohem-r100"
mkdir -p "$MODELDIR"
cp train_pair.sh "$MODELDIR/train_pair.sh"

PREFIX="$MODELDIR/model"
LRSTEPS='100000,140000,160000'
LOGFILE="$MODELDIR/log_arcface0.5"
python  train_pairwise.py --lr 0.00001 --fc7-lr-mult 1 --fc7-wd-mult 10 --lr-steps "$LRSTEPS" --data-dir $DATA_DIR --network "$NETWORK" --width-mult 1 --loss-type "arcface" --margin-m 0.5 --prefix "$PREFIX" --per-batch-size 64 --verbose 1000 --ckpt 2 --max-steps  200000 --pretrained "../models/r100-arcface-combined_all-sentec005/model,12" --version-output "Ep" --target "9374"

LRSTEPS='160000'
LOGFILE="$MODELDIR/log_arcface0.55"
#python -u train_softmax.py --lr 0.001 --fc7-lr-mult 1 --fc7-wd-mult 10 --lr-steps "$LRSTEPS" --data-dir $DATA_DIR --network "$NETWORK" --width-mult 1 --loss-type 4 --margin-m 0.55 --prefix "$PREFIX" --per-batch-size 32 --verbose 20000 --ckpt 2 --max-steps  200001 --pretrained "$PREFIX,40" --version-output "E"  > "$LOGFILE" 2>&1 &
