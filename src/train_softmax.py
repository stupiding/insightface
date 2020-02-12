from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
from easydict import EasyDict as edict
#sys.path.append(os.path.join(os.path.dirname(__file__), 'losses'))
#import center_loss

logger = logging.getLogger()
logger.setLevel(logging.INFO)

args = None

class AccMetric(mx.metric.EvalMetric):
  def __init__(self):
    self.axis = 1
    super(AccMetric, self).__init__(
        'acc', axis=self.axis,
        output_names=None, label_names=None)
    self.losses = []
    self.count = 0

  def update(self, labels_ls, preds_ls):
    self.count+=1
    labels = [labels_ls[0][:, i] for i in range(len(preds_ls) - 1)] if len(preds_ls) > 2 else labels_ls
    for label, pred_label in zip(labels, preds_ls[1:]):
        if pred_label.shape != label.shape:
            pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)
        pred_label = pred_label.asnumpy().astype('int32').flatten()
        label = label.asnumpy()
        if label.ndim==2:
            label = label[:,0]
        label = label.astype('int32').flatten()
        assert label.shape==pred_label.shape
        pred_label, label = pred_label.flat, label.flat
        #valid_ids = np.argwhere(label.asnumpy() != -1)
        self.sum_metric += (pred_label == label).sum()
        self.num_inst += len(pred_label)

class LossValueMetric(mx.metric.EvalMetric):
  def __init__(self):
    self.axis = 1
    super(LossValueMetric, self).__init__(
        'lossvalue', axis=self.axis,
        output_names=None, label_names=None)
    self.losses = []

  def update(self, labels, preds):
    print(labels[0].shape, preds[0].shape)
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
        bn_mom = args.bn_mom)

  if network=='fspherenet':
    data_shape_dict = {'data' : (args.per_batch_size,)+data_shape}
    fspherenet.init_weights(sym, data_shape_dict, int(num_layers))

  all_label = mx.symbol.Variable('softmax_label')
  if len(args.num_classes) > 1:
    all_label = mx.symbol.split(data=all_label, axis=1, num_outputs=len(args.num_classes))

  for i in range(len(args.num_classes)):
    gt_label = all_label[i].reshape([-1, ])
    extra_loss = None

    cvd, name = None, 'fc7_dt%d' % i
    classes_each_ctx = args.num_classes[i]
    print('use Parallel or not: {}'.format(args.parallel))
    if args.parallel:
      name += '_sub%d'
      cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip().split(',')
      classes_each_ctx = (args.num_classes[i] + len(cvd) - 1) // len(cvd)

    if args.loss_type==0: #softmax
      fc7 = Softmax(embedding, gt_label, classes_each_ctx, name, args, cvd)
    elif args.loss_type==1: #sphere
      _weight = mx.symbol.L2Normalization(_weight, mode='instance')
      fc7 = mx.sym.LSoftmax(data=embedding, label=gt_label, num_hidden=args.num_classes[i],
                            weight = _weight,
                            beta=args.beta, margin=args.margin, scale=args.scale,
                            beta_min=args.beta_min, verbose=1000, name='fc7_%d' % i)
    elif args.loss_type==2:
      fc7 = CosFace(embedding, gt_label, classes_each_ctx, name, args, cvd)
    elif args.loss_type==4:
      fc7 = ArcFace(embedding, gt_label, classes_each_ctx, name, args, cvd)
    elif args.loss_type==5:
      fc7 = CombineFace(embedding, gt_label, classes_each_ctx, name, args, cvd)
    elif args.loss_type==6: # linear margin m
      fc7 = LarcFace(embedding, gt_label, classes_each_ctx, name, args, cvd)

    if i == 0:
      out_list = [mx.symbol.BlockGrad(embedding)]
    softmax = mx.symbol.SoftmaxOutput(data=fc7, label = gt_label, name='softmax_%d' % i, normalization='valid', use_ignore=True)
    out_list.append(softmax)
  out = mx.symbol.Group(out_list)
  return (out, arg_params, aux_params)

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
      print('loading', vec)
      _, arg_params, aux_params = mx.model.load_checkpoint(vec[0], int(vec[1]))
      sym, arg_params, aux_params = get_symbol(network, int(num_layers), args, arg_params, aux_params)

    #label_name = 'softmax_label'
    #label_shape = (args.batch_size,)
    ctx_group = dict(zip(['dev%d' % (i+1) for i in range(len(ctx))], ctx))
    ctx_group['dev0'] = ctx
    model = mx.mod.Module(
        context       = ctx,
        symbol        = sym,
        data_names    = ['data'] if args.loss_type != 6 else ['data', 'margin'],
        group2ctxs    = ctx_group
    )
    val_dataiter = None

    cutoff = edict()
    cutoff.ratio = 0.3
    cutoff.size = 32
    cutoff.mode = 'fixed' # 'uniform'
    cutoff.filler = 127.5

    train_dataiter = FaceImageIter(
        batch_size           = args.batch_size,
        data_shape           = data_shape,
        path_imgrecs         = path_imgrecs,
        shuffle              = True,
        rand_mirror          = args.rand_mirror,
        mean                 = mean,
        cutoff               = cutoff,
        loss_type            = args.loss_type,
        #margin_m             = args.margin_m,
        #margin_policy        = args.margin_policy,
        #max_steps            = args.max_steps,
        #data_names           = ['data', 'margin'],
        downsample_back      = args.downsample_back,
        motion_blur          = args.motion_blur,
    )

    if args.loss_type<10:
      _metric = AccMetric()
    else:
      _metric = LossValueMetric()
    eval_metrics = [mx.metric.create(_metric)]

    if args.network[0]=='r' or args.network[0]=='y':
      initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2) #resnet style
    elif args.network[0]=='i' or args.network[0]=='x':
      initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2) #inception
    else:
      initializer = mx.init.Xavier(rnd_type='uniform', factor_type="in", magnitude=2)
    _rescale = 1.0/args.ctx_num
    #opt = AdaBound()
    #opt = AdaBound(lr=base_lr, wd=base_wd, gamma = 2. / args.max_steps)
    opt = optimizer.SGD(learning_rate=base_lr, momentum=base_mom, wd=base_wd, rescale_grad=_rescale)
    som = 200
    _cb = mx.callback.Speedometer(args.batch_size, som)

    ver_list = []
    ver_name_list = []
    """
    for name in args.target.split(','):
      path = os.path.join(data_dir,name+".bin")
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
          fp_rates, fp_dict, thred_dict, recall_dict = verification.test(ver_list[i], model, args.batch_size)
          for k in fp_rates:
            print("[%s] TPR at FPR %.2e[%.2e: %.4f]:\t%.5f" %(ver_names_list[i], k, fp_dict[k], thred_dict[k], recall_dict[k]))
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
        if mbatch==args.beta_freeze+_lr:
          opt.lr *= 0.1
          print('lr change to', opt.lr)
          break

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

