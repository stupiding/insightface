# THIS FILE IS FOR EXPERIMENTS, USE train_softmax.py FOR NORMAL TRAINING.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb

import os, sys
import math, random
import logging
import pickle
import sklearn
import numpy as np
from easydict import EasyDict as edict

import mxnet as mx
from mxnet import ndarray as nd
import mxnet.optimizer as optimizer

sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'symbols'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_loader'))

import face_image
import verification
from pair_image_iter import FaceImageIter

from networks import *
from symbol_fc7 import *

from utils.adabound import AdaBound
from utils.parser import parse_args

#sys.path.append(os.path.join(os.path.dirname(__file__), 'losses'))
#import center_loss

logger = logging.getLogger()
logger.setLevel(logging.INFO)


args = None


class LossValueMetric(mx.metric.EvalMetric):
  def __init__(self):
    self.axis = 1
    super(LossValueMetric, self).__init__(
        'lossvalue', axis=self.axis,
        output_names=None, label_names=None)
    self.losses = []

  def update(self, labels, preds):
    loss = preds[-1].asnumpy()[0]
    self.sum_metric += loss
    self.num_inst += 1.0
    gt_label = preds[-2].asnumpy()
    #print(gt_label)

def get_symbol(network, num_layers, args, arg_params, aux_params):
  data_shape = (args.image_channel,args.image_h,args.image_w)
  image_shape = ",".join([str(x) for x in data_shape])
  margin_symbols = []

  print('init %s, num_layers: %d' % (network, num_layers))
  with mx.AttrScope(ctx_group='dev0'):
    embedding = eval(network).get_symbol(args.emb_size, num_layers, shake_drop=args.shake_drop,
        version_se=args.version_se, version_input=args.version_input, 
        version_output=args.version_output, version_unit=args.version_unit,
        version_act=args.version_act, width_mult = args.width_mult, version_bn=args.version_bn, 
        bn_mom = args.bn_mom, use_global_stats=True)

  if network=='fspherenet':
    data_shape_dict = {'data' : (args.per_batch_size,)+data_shape}
    fspherenet.init_weights(sym, data_shape_dict, int(num_layers))

  gt_label = mx.symbol.Variable('softmax_label')
  label = mx.sym.slice_axis(gt_label, axis=0, begin=0, end=args.per_batch_size // 2)
  
  nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')
  nembedding1 = mx.symbol.slice_axis(nembedding, axis=0, begin=0, end=args.per_batch_size//2)
  nembedding2 = mx.symbol.slice_axis(nembedding, axis=0, begin=args.per_batch_size//2, end=2*args.per_batch_size//2)

  cos_simi = mx.sym.sum(nembedding1 * nembedding2, axis=1, keepdims=1)
  target = mx.sym.where(label, 0.64 - cos_simi, cos_simi - 0.36)
  pairwise_loss = mx.symbol.clip(target, 0, 1)
  pairwise_loss = mx.symbol.mean(pairwise_loss)
  pairwise_loss = mx.symbol.MakeLoss(pairwise_loss)

  out_list = [mx.symbol.BlockGrad(embedding)]
  out_list.append(mx.sym.BlockGrad(gt_label))
  out_list.append(mx.sym.BlockGrad(cos_simi))
  out_list.append(mx.sym.BlockGrad(target))
  out_list.append(pairwise_loss)
  out = mx.symbol.Group(out_list)
  return (out, arg_params, aux_params)

def get_module(args, data_shapes):
  network, num_layers = args.network.split(',')
  num_layers = int(num_layers)
  data_shape = (3, 112, 112)
  image_shape = ",".join([str(x) for x in data_shape])

  with mx.AttrScope(ctx_group='dev0'):
    embedding = eval(network).get_symbol(args.emb_size, num_layers, shake_drop=args.shake_drop,
        version_se=args.version_se, version_input=args.version_input, 
        version_output=args.version_output, version_unit=args.version_unit,
        version_act=args.version_act, width_mult = args.width_mult, version_bn=args.version_bn, 
        bn_mom = args.bn_mom)


  ctx = [mx.gpu(0)]
  #all_layers = sym.get_internals()
  all_layers = embedding.get_internals()
  out_list = [all_layers[layer] for layer in['fc1_output']]
  sym = mx.symbol.Group(out_list)
  
  model = mx.mod.Module(
      context       = ctx,
      symbol        = sym,
      data_names    = ['data'],
      label_names   = None
  )

  model.bind(for_training=False, data_shapes=data_shapes)

  if args.pretrained != '':
    print(args.pretrained)
    vec = args.pretrained.split(',')
    _, arg_params, aux_params = mx.model.load_checkpoint(vec[0], int(vec[1]))
    model.set_params(arg_params, aux_params, allow_extra=True, allow_missing=True) 

  return model

def train_net(args):
    ctx = []
    cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
    if len(cvd)>0:
      for i in range(len(cvd.split(','))):
        ctx.append(mx.gpu(i))
    if len(ctx)==0:
      ctx = [mx.cpu()]
      print('use cpu')
    else:
      print('gpu num:', len(ctx))
    prefix = args.prefix
    prefix_dir = os.path.dirname(prefix)
    if not os.path.exists(prefix_dir):
      os.makedirs(prefix_dir)
    end_epoch = args.end_epoch
    args.ctx_num = len(ctx)
    network, num_layers = args.network.split(',')
    print('num_layers', num_layers)
    if args.per_batch_size==0:
      args.per_batch_size = 128
    args.batch_size = args.per_batch_size*args.ctx_num
    args.image_channel = 3

    data_dir_list = args.data_dir.split(',')
    path_imgrecs = []
    path_imglist = None
    for data_idx, data_dir in enumerate(data_dir_list):
      image_size = (112, 112)
      if data_idx == 0:
        args.image_h = image_size[0]
        args.image_w = image_size[1]
      else:
        args.image_h = min(args.image_h, image_size[0]) 
        args.image_w = min(args.image_w, image_size[1])
      print('image_size', image_size)
      path_imgrecs.append(data_dir)

    args.use_val = False
    val_rec = None

    print('Called with argument:', args)

    data_shape = (args.image_channel,image_size[0],image_size[1])
    mean = None

    begin_epoch = 0
    base_lr = args.lr
    base_wd = args.wd
    base_mom = args.mom
    if len(args.pretrained)==0:
      arg_params = None
      aux_params = None
      sym, arg_params, aux_params = get_symbol(network, int(num_layers), args, arg_params, aux_params)
    else:
      vec = args.pretrained.split(',')
      print('loading', vec)
      _, arg_params, aux_params = mx.model.load_checkpoint(vec[0], int(vec[1]))
      sym, arg_params, aux_params = get_symbol(network, int(num_layers), args, arg_params, aux_params)
    if args.network[0]=='s':
      data_shape_dict = {'data' : (args.per_batch_size,)+data_shape}
      spherenet.init_weights(sym, data_shape_dict, args.num_layers)

    data_extra = None
    hard_mining = False
    model = mx.mod.Module(
        context       = ctx,
        symbol        = sym,
        #data_names = ('data',),
        #label_names = None,
        #label_names = ('softmax_label',),
    )
    label_shape = (args.batch_size,)

    val_dataiter = None

    from config import crop
    from config import cutout

    train_dataiter = FaceImageIter(
        batch_size           = args.batch_size,
        data_shape           = data_shape,
        path_imgrecs         = path_imgrecs,
        shuffle              = True,
        rand_mirror          = args.rand_mirror,
        mean                 = mean,
        cutout               = None, #cutout,
        crop                 = crop, 
        downsample_back      = args.downsample_back,
        motion_blur          = args.motion_blur,
        mx_model             = model,
        ctx_num              = args.ctx_num,
    )

    _metric = LossValueMetric()
    eval_metrics = [mx.metric.create(_metric)]

    if args.network[0]=='r':
      initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2) #resnet style
    elif args.network[0]=='i' or args.network[0]=='x':
      initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2) #inception
    else:
      initializer = mx.init.Xavier(rnd_type='uniform', factor_type="in", magnitude=2)
    _rescale = 1.0/args.ctx_num
    opt = optimizer.SGD(learning_rate=base_lr, momentum=base_mom, wd=base_wd, rescale_grad=_rescale)
    som = 200
    _cb = mx.callback.Speedometer(args.batch_size, som)

    ver_list = []
    ver_name_list = []
    """
    for name in args.target.split(','):
      path = os.path.join(os.path.dirname(data_dir),name+".bin")
      if os.path.exists(path):
        data_set = verification.load_bin(path, image_size)
        ver_list.append(data_set)
        ver_name_list.append(name)
        print('ver', name)
    """



    def ver_test(nbatch):
      results = []
      for i in range(len(ver_list)):
        _, issame_list = ver_list[i]
        if all(issame_list):
          fp_rates, fp_dict, thred_dict, recall_dict = verification.test(ver_list[i], model, args.batch_size, label_shape = (args.batch_size, len(path_imgrecs)))
          for k in fp_rates:
            print("[%s] TPR at FPR %.2e[%.2e: %.4f]:\t%.5f" %(ver_name_list[i], k, fp_dict[k], thred_dict[k], recall_dict[k]))
        else:
          acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(ver_list[i], model, args.batch_size, 10, None, label_shape = (args.batch_size, len(path_imgrecs)))
          print('[%s][%d]XNorm: %f' % (ver_name_list[i], nbatch, xnorm))
          #print('[%s][%d]Accuracy: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc1, std1))
          print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc2, std2))
          results.append(acc2)
      return results

    highest_acc = [0.0, 0.0]  #lfw and target
    #for i in range(len(ver_list)):
    #  highest_acc.append(0.0)
    global_step = [0]
    save_step = [0]
    if len(args.lr_steps)==0:
      lr_steps = [40000, 60000, 80000]
      if args.loss_type>=1 and args.loss_type<=7:
        lr_steps = [100000, 140000, 160000]
      p = 512.0/args.batch_size
      for l in range(len(lr_steps)):
        lr_steps[l] = int(lr_steps[l]*p)
    else:
      lr_steps = [int(x) for x in args.lr_steps.split(',')]
    print('lr_steps', lr_steps)
    def _batch_callback(param):
      #global global_step
      global_step[0]+=1
      mbatch = global_step[0]
      for _lr in lr_steps:
        if mbatch==_lr:
          opt.lr *= 0.1
          print('lr change to', opt.lr)
          break

      _cb(param)
      if mbatch%1000==0:
        print('lr-batch-epoch:',opt.lr,param.nbatch,param.epoch)

      if mbatch>=0 and mbatch%args.verbose==0:
        acc_list = ver_test(mbatch)
        save_step[0]+=1
        msave = save_step[0]
        do_save = False
        if len(acc_list)>0:
          lfw_score = acc_list[0]
          if lfw_score>highest_acc[0]:
            highest_acc[0] = lfw_score
            if lfw_score>=0.998:
              do_save = True
          if acc_list[-1]>=highest_acc[-1]:
            highest_acc[-1] = acc_list[-1]
            if lfw_score>=0.99:
              do_save = True
        if args.ckpt==0:
          do_save = False
        elif args.ckpt>1:
          do_save = True
        #for i in range(len(acc_list)):
        #  acc = acc_list[i]
        #  if acc>=highest_acc[i]:
        #    highest_acc[i] = acc
        #    if lfw_score>=0.99:
        #      do_save = True
        #if args.loss_type==1 and mbatch>lr_steps[-1] and mbatch%10000==0:
        #  do_save = True
        if do_save:
          print('saving', msave)
          if val_dataiter is not None:
            val_test()
          arg, aux = model.get_params()
          mx.model.save_checkpoint(prefix, msave, model.symbol, arg, aux)
        print('[%d]Accuracy-Highest: %1.5f'%(mbatch, highest_acc[-1]))
      if args.max_steps>0 and mbatch>args.max_steps:
        sys.exit(0)

    #epoch_cb = mx.callback.do_checkpoint(prefix, 1)
    epoch_cb = None

    model.fit(train_dataiter,
        begin_epoch        = begin_epoch,
        num_epoch          = end_epoch,
        eval_data          = val_dataiter,
        eval_metric        = eval_metrics,
        kvstore            = 'device',
        optimizer          = opt,
        #optimizer_params   = optimizer_params,
        initializer        = initializer,
        arg_params         = arg_params,
        aux_params         = aux_params,
        allow_missing      = True,
        batch_end_callback = _batch_callback,
        epoch_end_callback = epoch_cb )

def main():
    #time.sleep(3600*6.5)
    global args
    args = parse_args()
    train_net(args)

if __name__ == '__main__':
    main()

