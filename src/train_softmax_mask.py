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
from mask_image_iter import FaceImageIter

from metrics import *
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
  def __init__(self, loss_name, loss_idx):
    self.loss_idx = loss_idx
    self.loss_name = loss_name
    super(LossValueMetric, self).__init__(loss_name)
    self.losses = []

  def update(self, labels, preds):
    if 'softmax' in self.loss_name:
      label_gt = labels[0][:, 0].asnumpy()

      keep_inds = np.where(label_gt != -1)[0]
      label_gt = label_gt[keep_inds].astype('int32')
      cls_prob = preds[self.loss_idx].asnumpy()[keep_inds, label_gt]
      #print(cls_prob, label_gt)

      cls_prob += 1e-14
      cls_loss = -1 * np.log(cls_prob)
      cls_loss = np.sum(cls_loss)
      self.sum_metric += cls_loss
      self.num_inst += label_gt.shape[0]
    else:
      feat_orig = preds[0].asnumpy()
      feat_mask = preds[1].asnumpy()
      #print(feat_orig[:, 0], feat_mask[:, 0])
      dist = np.sum((feat_orig - feat_mask) ** 2, 1) / 2000
      self.sum_metric += np.sum(dist)
      self.num_inst += feat_orig.shape[0]

def get_symbol(args, arg_params, aux_params):
  data_shape = (args.image_channel,args.image_h,args.image_w)
  image_shape = ",".join([str(x) for x in data_shape])
  margin_symbols = []

  cvd, name = None, 'fc7_%d' % 0
  classes_each_ctx = args.num_classes[0]
  if args.parallel:
    cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip().split(',')
    classes_each_ctx = (args.num_classes[i] + len(cvd) - 1) // len(cvd)
  else:
    name = 'fc7'
  args.ctx_num_classes = classes_each_ctx

  bn1_orig = mx.symbol.Variable('bn1_orig')
  bn1_mask = mx.symbol.Variable('bn1_mask')
  embedding_orig = mx.symbol.Variable('fc1_orig')
  embedding_mask = mx.symbol.Variable('fc1_mask')

  all_label = mx.symbol.Variable('softmax_label')

  gt_label = all_label.reshape([-1, ])

  def get_mask(data):
    global_pool = mx.sym.Pooling(data=data, global_pool=True, kernel=(7, 7), pool_type='avg', name='bn1_global_pooling')
  
    mask_fc1 = mx.sym.FullyConnected(data=global_pool, num_hidden=512 // 2, name='mask_fc1')
    mask_fc1 = mx.sym.BatchNorm(data=mask_fc1, fix_gamma=True, eps=2e-5, momentum=args.bn_mom, name='mask_fc1_bn')
    mask_fc1 = mx.sym.Activation(data=mask_fc1, act_type='relu', name='mask_fc1_relu')
  
    mask_fc2 = mx.sym.FullyConnected(data=mask_fc1, num_hidden=512 // 2, name='mask_fc2')
    mask_fc2 = mx.sym.BatchNorm(data=mask_fc2, fix_gamma=True, eps=2e-5, momentum=args.bn_mom, name='mask_fc2_bn')
    mask_fc2 = mx.sym.Activation(data=mask_fc2, act_type='relu', name='mask_fc2_relu')
  
    mask_fc3 = mx.sym.FullyConnected(data=mask_fc2, num_hidden=512, name='mask_fc3')

    return mask_fc3

  bn1 = mx.sym.Concat(bn1_orig, bn1_mask, dim=0, name='bn1_concat')
  embedding = mx.sym.Concat(embedding_orig, embedding_mask, dim=0, name='fc1_concat')
  labels = mx.sym.Concat(gt_label, gt_label, dim=0, name='label_concat')

  residual = get_mask(bn1)
  feats = embedding + residual

  feats_orig, feats_mask = mx.sym.split(data=feats, axis=0, num_outputs=2)
  diff = feats_orig - feats_mask
  dist = mx.sym.sum(diff * diff, axis=1) / 2000
  diff_square = mx.sym.mean(dist) 
  distill_loss = mx.sym.MakeLoss(diff_square)

  if args.loss_type==4:
    fc7 = ArcFace(feats, labels, name, args, cvd)
  elif args.loss_type==5:
    fc7 = CombineFace(feats, labels, name, args, cvd)

  fc7_orig, fc7_mask = mx.sym.split(data=fc7, axis=0, num_outputs=2)
  
  softmax_orig = mx.symbol.SoftmaxOutput(data=fc7_orig, label = gt_label, name='softmax_orig', normalization='valid', use_ignore=True)
  softmax_mask = mx.symbol.SoftmaxOutput(data=fc7_mask, label = gt_label, name='softmax_mask', normalization='valid', use_ignore=True)

  out_list = [mx.symbol.BlockGrad(fc7_orig), mx.sym.BlockGrad(fc7_mask)]
  out_list += [distill_loss, softmax_orig, softmax_mask]

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

  #out_list = [mx.symbol.BlockGrad(embedding)]
  #sym = mx.symbol.Group(out_list)

  ctx = [mx.gpu(0)]
  #all_layers = sym.get_internals()
  all_layers = embedding.get_internals()
  out_list = [all_layers[layer] for layer in['bn1_output', 'fc1_output']]
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
    args.rescale_threshold = 0
    args.image_channel = 3

    os.environ['BETA'] = str(args.beta)
    data_dir_list = args.data_dir.split(',')
    path_imgrecs = []
    path_imglist = None
    args.num_classes = []
    for data_dir in data_dir_list:
      prop = face_image.load_property(data_dir)
      args.num_classes.append(prop.num_classes)
      image_size = prop.image_size
      args.image_h = image_size[0]
      args.image_w = image_size[1]
      print('image_size', image_size)
      assert(args.num_classes[-1]>0)
      print('num_classes', args.num_classes)
      path_imgrecs.append(os.path.join(data_dir, "train.rec"))

    if args.loss_type==1 and args.num_classes>20000:
      args.beta_freeze = 5000
      args.gamma = 0.06

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
      _, arg_params, aux_params = mx.model.load_checkpoint(vec[0], int(vec[1]))
      sym, arg_params, aux_params = get_symbol(args, arg_params, aux_params)

    #label_name = 'softmax_label'
    #label_shape = (args.batch_size,)
    ctx_group = dict(zip(['dev%d' % (i+1) for i in range(len(ctx))], ctx))
    ctx_group['dev0'] = ctx
    model = mx.mod.Module(
        context       = ctx,
        symbol        = sym,
        data_names    = ['bn1_orig', 'bn1_mask', 'fc1_orig', 'fc1_mask'],
        label_names   = ['softmax_label'],
        group2ctxs    = ctx_group
    )
    val_dataiter = None

    from config import cutout

    data_shapes = [('data', (args.batch_size, args.image_channel, image_size[0], image_size[1]))]
    pretrain_model = get_module(args, data_shapes)

    train_dataiter = FaceImageIter(
        batch_size           = args.batch_size,
        data_shape           = data_shape,
        path_imgrecs         = path_imgrecs,
        shuffle              = True,
        rand_mirror          = args.rand_mirror,
        mean                 = mean,
        cutout               = cutout,
        loss_type            = args.loss_type,
        #margin_m             = args.margin_m,
        #margin_policy        = args.margin_policy,
        #max_steps            = args.max_steps,
        data_names           = ['bn1', 'fc1'],
        downsample_back      = args.downsample_back,
        motion_blur          = args.motion_blur,
        mx_model             = pretrain_model,
    )

    #if args.loss_type<10:
    #  _metric = AccMetric()
    #else:
    #_metric = LossValueMetric()
    eval_metrics = mx.metric.CompositeEvalMetric()
    for loss_name, loss_idx in zip(['distill_loss', 'softmax_orig', 'softmax_mask'], [2, 3, 4]):
      eval_metrics.add(LossValueMetric(loss_name, loss_idx))
    #eval_metrics = None

    if args.network[1]=='r' or args.network[1]=='y':
      initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2) #resnet style
    _rescale = 1.0/args.ctx_num
    #opt = AdaBound()
    #opt = AdaBound(lr=base_lr, wd=base_wd, gamma = 2. / args.max_steps)

    lr_steps = [int(x) for x in args.lr_steps.split(',')]
    lr_scheduler =  mx.lr_scheduler.MultiFactorScheduler(lr_steps, factor=0.1, base_lr=base_lr)
    optimizer_params = {'learning_rate':base_lr,
                        'momentum':base_mom,
                        'wd':base_wd,
                        'rescale_grad':_rescale, 
                        'lr_scheduler': lr_scheduler}

    opt = optimizer.SGD(learning_rate=base_lr, momentum=base_mom, wd=base_wd, rescale_grad=_rescale)
    som = 200
    _cb = mx.callback.Speedometer(args.batch_size, som)

    ver_list = []
    ver_name_list = []

    def ver_test(nbatch):
      return [0]

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
      #for _lr in lr_steps:
      #  if mbatch==args.beta_freeze+_lr:
      #    opt.lr *= 0.1
      #    print('lr change to', opt.lr)
      #    break

      _cb(param)
      if mbatch%10000==0:
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
        if do_save:
          print('saving', msave)
          arg, aux = model.get_params()
          mx.model.save_checkpoint(prefix, msave, model.symbol, arg, aux)
        print('[%d]Accuracy-Highest: %1.5f'%(mbatch, highest_acc[-1]))
      if mbatch<=args.beta_freeze:
        _beta = args.beta
      else:
        move = max(0, mbatch-args.beta_freeze)
        _beta = max(args.beta_min, args.beta*math.pow(1+args.gamma*move, -1.0*args.power))
      #print('beta', _beta)
      os.environ['BETA'] = str(_beta)
      if args.max_steps>0 and mbatch>args.max_steps:
        sys.exit(0)

    epoch_cb = None
    train_dataiter = mx.io.PrefetchingIter(train_dataiter)

    model.fit(train_dataiter,
        begin_epoch        = begin_epoch,
        num_epoch          = end_epoch,
        eval_data          = val_dataiter,
        eval_metric        = eval_metrics,
        kvstore            = 'device',
        #optimizer          = opt,
        optimizer_params   = optimizer_params,
        initializer        = initializer,
        arg_params         = arg_params,
        aux_params         = aux_params,
        allow_missing      = True,
        #allow_extra        = True,
        batch_end_callback = _batch_callback,
        epoch_end_callback = epoch_cb )

def main():
    global args
    args = parse_args()
    train_net(args)

if __name__ == '__main__':
    main()

