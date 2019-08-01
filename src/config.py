import numpy as np
import os
from easydict import EasyDict as edict

config = edict()

config.bn_mom = 0.9
config.workspace = 256
config.emb_size = 512
config.ckpt_embedding = True
config.net_se = 0
config.net_act = 'prelu'
config.net_unit = 3
config.net_input = 1
config.net_output = 'E'
config.net_multiplier = 1.0
config.val_targets = ['lfw', 'cfp_fp', 'agedb_30']
config.ce_loss = True
config.fc7_lr_mult = 1000.0
config.fc7_wd_mult = 1.0
config.fc7_no_bias = False
config.max_steps = 280000
config.data_rand_mirror = True
config.data_cutoff = False
config.data_color = 0
config.data_images_filter = 0
config.downsample_back = 0.0
config.motion_blur = 0.0


# network settings
network = edict()

network.r100 = edict()
network.r100.net_name = 'fresnet'
network.r100.num_layers = 100
network.r100.emb_size = 512
network.r100.shake_drop = False
network.r100.version_se = False
network.r100.version_input = 1
network.r100.version_output = 'E'
network.r100.version_unit = 3
network.r100.version_act = 'prelu'
network.r100.width_mult = 1
network.r100.version_bn = 'bn'


network.r50 = edict()
network.r50.net_name = 'fresnet'
network.r50.num_layers = 50

network.r50v1 = edict()
network.r50v1.net_name = 'fresnet'
network.r50v1.num_layers = 50
network.r50v1.net_unit = 1

network.d169 = edict()
network.d169.net_name = 'fdensenet'
network.d169.num_layers = 169
network.d169.per_batch_size = 64
network.d169.densenet_dropout = 0.0

network.d201 = edict()
network.d201.net_name = 'fdensenet'
network.d201.num_layers = 201
network.d201.per_batch_size = 64
network.d201.densenet_dropout = 0.0

network.y1 = edict()
network.y1.net_name = 'fmobilefacenet'
network.y1.emb_size = 128
network.y1.net_output = 'GDC'

network.z1 = edict()
network.z1.net_name = 'fsmall'
network.z1.emb_size = 512
network.z1.net_output = 'E'
network.z1.num_layers = 11

network.m1 = edict()
network.m1.net_name = 'fmobilenet'
network.m1.emb_size = 256
network.m1.net_output = 'GDC'
network.m1.net_multiplier = 1.0

network.m05 = edict()
network.m05.net_name = 'fmobilenet'
network.m05.emb_size = 256
network.m05.net_output = 'GDC'
network.m05.net_multiplier = 0.5

network.mnas = edict()
network.mnas.net_name = 'fmnasnet'
network.mnas.emb_size = 256
network.mnas.net_output = 'GDC'
network.mnas.net_multiplier = 1.0

network.mnas05 = edict()
network.mnas05.net_name = 'fmnasnet'
network.mnas05.emb_size = 256
network.mnas05.net_output = 'GDC'
network.mnas05.net_multiplier = 0.5

network.mnas025 = edict()
network.mnas025.net_name = 'fmnasnet'
network.mnas025.emb_size = 256
network.mnas025.net_output = 'GDC'
network.mnas025.net_multiplier = 0.25

# dataset settings
dataset = edict()

dataset.emore = edict()
dataset.emore.dataset = 'emore'
dataset.emore.dataset_path = '../datasets/faces_emore'
dataset.emore.num_classes = 85742
dataset.emore.image_shape = (112,112,3)
dataset.emore.val_targets = ['lfw', 'cfp_fp', 'agedb_30']

dataset.left_eye = edict()
dataset.left_eye.dataset = 'left_eye'
dataset.left_eye.dataset_path = '../datasets/left_eye'
dataset.left_eye.num_classes = 85742
dataset.left_eye.image_shape = (112,112,3)
dataset.left_eye.val_targets = []

dataset.glint_cn = edict()
dataset.glint_cn = edict()
dataset.glint_cn.dataset = 'glint_cn'
dataset.glint_cn.dataset_path = '../datasets/glint_cn'
dataset.glint_cn.num_classes = 93979
dataset.glint_cn.image_shape = (112,112,3)
dataset.glint_cn.val_targets = ['lfw', 'cfp_fp', 'agedb_30']

dataset.bjz_20W = edict()
dataset.bjz_20W.dataset = 'bjz_20W'
dataset.bjz_20W.dataset_path = '../datasets/bjz+20W+gridRemoval'
dataset.bjz_20W.num_classes = 1460480
dataset.bjz_20W.image_shape = (112,112,3)
dataset.bjz_20W.val_targets = ['lfw', 'cfp_fp', 'agedb_30']

dataset.bjz30W_20W = edict()
dataset.bjz30W_20W.dataset = 'bjz30w_grid2'
dataset.bjz30W_20W.dataset_path = '../datasets/bjz30w+grid2'
dataset.bjz30W_20W.num_classes = 560480
dataset.bjz30W_20W.image_shape = (112,112,3)
dataset.bjz30W_20W.val_targets = ['lfw', 'cfp_fp', 'agedb_30']

loss = edict()
loss = edict()
loss = edict()
loss.softmax = edict()
loss.softmax.loss_name = 'softmax'

loss.nsoftmax = edict()
loss.nsoftmax.loss_name = 'margin_softmax'
loss.nsoftmax.loss_s = 64.0
loss.nsoftmax.loss_m1 = 1.0
loss.nsoftmax.loss_m2 = 0.0
loss.nsoftmax.loss_m3 = 0.0

loss.arcface = edict()
loss.arcface.loss_name = 'margin_softmax'
loss.arcface.loss_s = 60.0
loss.arcface.loss_m1 = 1.0
loss.arcface.loss_m2 = 0.35
loss.arcface.loss_m3 = 0.0

loss.cosface = edict()
loss.cosface.loss_name = 'margin_softmax'
loss.cosface.loss_s = 64.0
loss.cosface.loss_m1 = 1.0
loss.cosface.loss_m2 = 0.0
loss.cosface.loss_m3 = 0.35

loss.combined = edict()
loss.combined.loss_name = 'margin_softmax'
loss.combined.loss_s = 64.0
loss.combined.loss_m1 = 1.0
loss.combined.loss_m2 = 0.3
loss.combined.loss_m3 = 0.2

loss.triplet = edict()
loss.triplet.loss_name = 'triplet'
loss.triplet.images_per_identity = 5
loss.triplet.triplet_alpha = 0.3
loss.triplet.triplet_bag_size = 7200
loss.triplet.triplet_max_ap = 0.0
loss.triplet.per_batch_size = 60
loss.triplet.lr = 0.05

loss.atriplet = edict()
loss.atriplet.loss_name = 'atriplet'
loss.atriplet.images_per_identity = 5
loss.atriplet.triplet_alpha = 0.35
loss.atriplet.triplet_bag_size = 7200
loss.atriplet.triplet_max_ap = 0.0
loss.atriplet.per_batch_size = 60
loss.atriplet.lr = 0.05

# default settings
default = edict()

# default network
default.network = 'r100'
default.pretrained = '' #'../models/r100-arcface-bjz_20W/model,0'
# default dataset
#default.dataset = 'bjz_20W'
default.dataset = 'bjz30W_20W'
#default.dataset = 'emore'
default.loss = 'arcface'
default.frequent = 200
default.verbose = 20000
default.kvstore = 'device'

default.end_epoch = 100
default.lr = 0.00001
default.wd = 0.0005
default.mom = 0.9
default.per_batch_size = 8
default.ckpt = 2
default.lr_steps = '220000,260000,280000'
default.models_root = '../models'


def generate_config(_network, _dataset, _loss):
    for k, v in loss[_loss].items():
      config[k] = v
      if k in default:
        default[k] = v
    for k, v in network[_network].items():
      config[k] = v
      if k in default:
        default[k] = v
    for k, v in dataset[_dataset].items():
      config[k] = v
      if k in default:
        default[k] = v
    config.loss = _loss
    config.network = _network
    config.dataset = _dataset
    config.num_workers = 1
    if 'DMLC_NUM_WORKER' in os.environ:
      config.num_workers = int(os.environ['DMLC_NUM_WORKER'])

