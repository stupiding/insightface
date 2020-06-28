#CUDA_VISIBLE_DEVICES=9 python predictor.py --pretrained ../models/r100-arcface-bjz30W_20W/model,35 --network fresnet,100 --version-output E
#CUDA_VISIBLE_DEVICES=0 python predictor.py --pretrained ../models/emore-model-r100-softmax1e3/model,38 --network fresnet,100 --version-output E
#CUDA_VISIBLE_DEVICES=0 python predictor.py --pretrained ../models/glint_cn-model-r100-softmax1e3-test/model,0 --network fresnet,100 --version-output E
#CUDA_VISIBLE_DEVICES=0 python predictor.py --pretrained ../models/emore_glint-model-resnet100-new_code/model,20 --network fresnet,100 --version-output E
CUDA_VISIBLE_DEVICES=0 python predictor.py --pretrained ../models/glint_emore_gy50-r100-short_sentec-005/model,50 --network fresnet,100 --version-output E
#CUDA_VISIBLE_DEVICES=0 python predictor.py --pretrained ../models/new/r100-arcface-bjz30W_20W_GY-backup/model,7 --network fresnet,100 --version-output E
#CUDA_VISIBLE_DEVICES=0 python predictor.py --pretrained ../models/emore_glint_gy50-model-resnet100-new_code/model,16 --network fresnet,100 --version-output E

#CUDA_VISIBLE_DEVICES=9 python predictor.py --pretrained ../models/new/r100-curricular-bjz30W_20W_GY/model,1 --network fresnet,100 --version-output E
#CUDA_VISIBLE_DEVICES=9 python predictor.py --pretrained ~/mx_converted/pretrained_new,1 --network fmobilefacenet,72 --version-output GDC

