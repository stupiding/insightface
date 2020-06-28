UDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_parall.py --loss softmax --models-root ../models/ --pretrained ../models/gy50_ohem-r100-sentec003/model,50 --max-steps 10000 --per-batch-size 48

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_parall.py --loss arcface --models-root ../models/ --pretrained ../models/r100-softmax-combined_all/model,1 --max-steps 140000

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_parall.py --loss arcface --models-root ../models/ --pretrained ../models/r100-arcface-combined_all/model,14 --max-steps 140000
