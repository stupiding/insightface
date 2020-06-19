from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb

import os
import sys
import math
import random
import logging
import pickle
import numpy as np
from image_iter import FaceImageIter
from image_iter import FaceImageIterList
import mxnet as mx
from mxnet import ndarray as nd
import argparse
import mxnet.optimizer as optimizer
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
import face_image
sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'symbols'))
from utils.adabound import AdaBound
from utils.parser import parse_args
from networks import *
from symbol_fc7 import *
import verification
import sklearn
#sys.path.append(os.path.join(os.path.dirname(__file__), 'losses'))
#import center_loss

logger = logging.getLogger()
logger.setLevel(logging.INFO)

args = None
check_layer = None #'bn0'
def get_module(args, data_shapes, label_shapes):
  network, num_layers = args.network.split(',')
  num_layers = int(num_layers)
  data_shape = (3, 112, 112)
  image_shape = ",".join([str(x) for x in data_shape])
  margin_symbols = []

  with mx.AttrScope(ctx_group='dev0'):
    embedding = eval(network).get_symbol(args.emb_size, num_layers, shake_drop=args.shake_drop,
        version_se=args.version_se, version_input=args.version_input, 
        version_output=args.version_output, version_unit=args.version_unit,
        version_act=args.version_act, width_mult = args.width_mult, version_bn=args.version_bn, 
        bn_mom = args.bn_mom)

  if network=='fspherenet':
    data_shape_dict = {'data' : (args.per_batch_size,)+data_shape}
    fspherenet.init_weights(sym, data_shape_dict, int(num_layers))

  #all_label = mx.symbol.Variable('softmax_label')
  args.num_classes = [560480]
  #if len(args.num_classes) > 1:
  #  all_label = mx.symbol.split(data=all_label, axis=1, num_outputs=len(args.num_classes))

  for i in range(len(args.num_classes)):
    #gt_label = all_label[i].reshape([-1, ])
    extra_loss = None

    cvd, name = None, 'fc7_dt%d' % i
    classes_each_ctx = args.num_classes[i]

    if i == 0:
      out_list = [mx.symbol.BlockGrad(embedding)]
    #softmax = mx.symbol.SoftmaxOutput(data=fc7, label = gt_label, name='softmax_%d' % i, normalization='valid', use_ignore=True)
    #softmax = mx.symbol.SoftmaxOutput(data=fc7, name='softmax_%d' % i, normalization='valid', use_ignore=True)
    #out_list.append(softmax)
  sym = mx.symbol.Group(out_list)

  ctx = [mx.gpu(0)]
  all_layers = sym.get_internals()
  #print(all_layers.list_outputs()[:30])
  #if check_layer is not None:
  #  sym = all_layers['%s_output' % check_layer]
  #sym3 = all_layers['flatten0_output']
  model = mx.mod.Module(
      context       = ctx,
      symbol        = sym,
      data_names    = ['data'] if args.loss_type != 6 else ['data', 'margin'],
      label_names   = None #['softmax_label']
  )

  model.bind(for_training=False, data_shapes=data_shapes) #, label_shapes=label_shapes)

  if args.pretrained != '':
    vec = args.pretrained.split(',')
    _, arg_params, aux_params = mx.model.load_checkpoint(vec[0], int(vec[1]))
    #model.set_params(arg_params, aux_params, allow_missing=True) #, allow_extra=True)
    model.set_params(arg_params, aux_params, allow_extra=True, allow_missing=True) 

  return model

def get_data(path, batch_size, use_bgr=False):
    image_size = (112, 112)
    data_set = verification.load_bin(path, image_size)
    
    data_list, issame_list = data_set
    _label = nd.ones( (batch_size, 1))
    for i in range(1): #len(data_list)):
      data = data_list[i]
      print('datasize: ', len(data))
      embeddings = None
      ba = 0
      while ba<data.shape[0]:
        bb = min(ba+batch_size, data.shape[0])
        count = bb-ba
        _data = nd.slice_axis(data, axis=0, begin=bb-batch_size, end=bb)
        if use_bgr:
          _data = _data[:, ::-1, :, :]
        #print(_data.shape, _label.shape)
        db = mx.io.DataBatch(data=(_data,)) #, label=(_label,))
        ba = bb
        yield (db, count)

def predict(args):
    args.batch_size = 8 #args.per_batch_size
    image_size = (112, 112)
    args.image_channel = 3
    data_shapes = [('data', (args.batch_size, args.image_channel, image_size[0], image_size[1]))]
    label_shapes = [] #[('softmax_label', (args.batch_size, 1))]

    print('Called with argument:', args)
    mean = None

    dataset = get_data('../datasets/9374.bin', args.batch_size, use_bgr=False)
    #dataset = get_data('../datasets/bjz_maskVScard.bin', args.batch_size)
    #dataset = get_data('../datasets/test.bin', args.batch_size)
    model = get_module(args, data_shapes, label_shapes)
    i = 0
    batch_start = 0
    embeddings = None
    if check_layer is not None:
      caffe_blob = pickle.load(open('/home/guojinma/%s.pkl' % check_layer, 'rb'), encoding='latin1')[check_layer]
    #caffe_blob = pickle.load(open('/home/guojinma/data.pkl', 'rb'), encoding='latin1')['data']
    embeddings = []
    for batch_data, batch_count in dataset:
      batch_end = batch_start + batch_count
      model.forward(batch_data, is_train=False)
      
      net_out = model.get_outputs()
      _embeddings = net_out[0].asnumpy()

      if check_layer is not None:
        print(np.abs(_embeddings).sum())
        print(np.abs(caffe_blob).sum())
        print(np.abs(_embeddings - caffe_blob).sum())

      #pd = np.argmax(_embeddings, 1)
      #for idx in range(8): 
      #  print('%d %f' % (pd[idx], _embeddings[idx, pd[idx]]))
      #break      

      embeddings.append(_embeddings[args.batch_size-batch_count:, :])
      batch_start = batch_end
      i += 1
      if i % 1000 == 0:
        print(i)
    embeddings = np.concatenate(embeddings, 0)
    print(embeddings.shape)
    scores = verification.calc_cos(embeddings[0::2, :], embeddings[1::2, :])
    
    fp_rates, fp_dict, thred_dict, recall_dict = verification.calc_pr(scores)
    for k in fp_rates:
      print("TPR at FPR %.2e[%.2e: %.4f]:\t%.5f" %(k, fp_dict[k], thred_dict[k], recall_dict[k]))

def main():
    global args
    args = parse_args()
    predict(args)

if __name__ == '__main__':
    main()

