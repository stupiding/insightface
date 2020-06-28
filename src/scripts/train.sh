#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=1
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
#export CUDA_VISIBLE_DEVICES='0'

#DATA_DIR=/opt/jiaguo/faces_vgg_112x112
#DATA_DIR=/data/face/MS_Celeb_1M/images/ArcFace/faces_ms1m_112x112
DATA_DIR=../datasets/faces_emore,../datasets/glint_cn,../datasets/gy50_final
#DATA_DIR=/data/face/MS_Celeb_1M/images/ArcFace/faces_emore
#DATA_DIR=../datasets/bjz30w+grid2

NETWORK=fresnet,100
JOB=new_code
#MODELDIR="../models/emore_glint-model-resnest101-$JOB"
MODELDIR="../models/emore_glint_gy50-model-resnet100-$JOB"
mkdir -p "$MODELDIR"
cp train.sh "$MODELDIR/train.sh"

PREFIX="$MODELDIR/model"
#LOGFILE="$MODELDIR/log_softmax"
LRSTEPS='100000,140000,160000'
#python -u train_softmax.py --lr 0.1 --fc7-lr-mult 1 --fc7-wd-mult 10 --lr-steps "$LRSTEPS" --data-dir $DATA_DIR --network "$NETWORK" --width-mult 1 --loss-type "softmax" --margin-m 0.5 --prefix "$PREFIX" --per-batch-size 32 --verbose 20000 --ckpt 2 --max-steps  200001 --version-bn "bn" --version-output "E" #--parallel #> "$LOGFILE" 2>&1 &
LOGFILE="$MODELDIR/log_arcface0.35"
#python -u train_softmax.py --lr 0.1 --fc7-lr-mult 1 --fc7-wd-mult 10 --lr-steps "$LRSTEPS" --data-dir $DATA_DIR --network "$NETWORK" --width-mult 1 --loss-type "arcface" --margin-m 0.35 --prefix "$PREFIX" --per-batch-size 32 --verbose 20000 --ckpt 2 --max-steps  200001 --pretrained "$PREFIX,10" --version-output "E"  > "$LOGFILE" 2>&1 

LRSTEPS='400000,600000'
LOGFILE="$MODELDIR/log_arcface0.5_l"
python -u train_softmax.py --lr 0.01 --fc7-lr-mult 1 --fc7-wd-mult 10 --lr-steps "$LRSTEPS" --data-dir $DATA_DIR --network "$NETWORK" --width-mult 1 --loss-type "arcface" --margin-m 0.5 --prefix "$PREFIX" --per-batch-size 48 --verbose 20000 --ckpt 2 --max-steps  800001 --pretrained "$PREFIX,10" --version-output "E"  --target "9374" > "$LOGFILE" 2>&1 

#LRSTEPS='160000'
#LOGFILE="$MODELDIR/log_arcface0.55"
#python -u train_softmax.py --lr 0.001 --fc7-lr-mult 1 --fc7-wd-mult 10 --lr-steps "$LRSTEPS" --data-dir $DATA_DIR --network "$NETWORK" --width-mult 1 --loss-type 4 --margin-m 0.55 --prefix "$PREFIX" --per-batch-size 32 --verbose 20000 --ckpt 2 --max-steps  200001 --pretrained "$PREFIX,40" --version-output "E"  > "$LOGFILE" 2>&1 &
