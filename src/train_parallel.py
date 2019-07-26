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
from networks import *
import verification
import sklearn
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

def parse_args():
  parser = argparse.ArgumentParser(description='Train face network')
  # general
  parser.add_argument('--data-dir', default='', help='training set directory')
  parser.add_argument('--prefix', default='../model/model', help='directory to save model.')
  parser.add_argument('--pretrained', default='', help='pretrained model to load')
  parser.add_argument('--ckpt', type=int, default=1, help='checkpoint saving option. 0: discard saving. 1: save when necessary. 2: always save')
  parser.add_argument('--loss-type', type=int, default=4, help='loss type')
  parser.add_argument('--verbose', type=int, default=2000, help='do verification testing and model saving every verbose batches')
  parser.add_argument('--max-steps', type=int, default=0, help='max training batches')
  parser.add_argument('--end-epoch', type=int, default=100000, help='training epoch size.')
  parser.add_argument('--network', default='fresnet,100', help='specify network')
  parser.add_argument('--width-mult', type=float, default=1, help="width-mult")
  parser.add_argument("--shake-drop", default=False, action="store_true" , help="whether use ShakeDrop")
  parser.add_argument('--version-se', type=int, default=0, help='whether to use se in network')
  parser.add_argument('--version-ibn', type=int, default=0, help='whether to use IBN in resnet')
  parser.add_argument('--version-input', type=int, default=1, help='network input config')
  parser.add_argument('--version-output', type=str, default='E', help='network embedding output config')
  parser.add_argument('--version-unit', type=int, default=3, help='resnet unit config')
  parser.add_argument('--version-act', type=str, default='prelu', help='network activation config')
  parser.add_argument('--version-bn', default='bn', help='version of bn: bn, row, col')
  parser.add_argument('--pyramid-alpha', type=int, default=0, help='0 for resnet, otherwise for pyramid alpha')
  parser.add_argument('--use-deformable', type=int, default=0, help='use deformable cnn in network')
  parser.add_argument('--lr', type=float, default=0.1, help='start learning rate')
  parser.add_argument('--lr-steps', type=str, default='', help='steps of lr changing')
  parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
  parser.add_argument('--fc7-wd-mult', type=float, default=1.0, help='weight decay mult for fc7')
  parser.add_argument('--fc7-lr-mult', type=float, default=1.0, help='lr mult for fc7')
  parser.add_argument("--fc7-no-bias", default=False, action="store_true" , help="fc7 no bias flag")
  parser.add_argument('--bn-mom', type=float, default=0.9, help='bn mom')
  parser.add_argument('--mom', type=float, default=0.9, help='momentum')
  parser.add_argument('--emb-size', type=int, default=512, help='embedding length')
  parser.add_argument('--per-batch-size', type=int, default=128, help='batch size in each context')
  parser.add_argument('--margin-m', type=float, default=0.5, help='margin for loss')
  parser.add_argument('--margin-policy', type=str, default='fixed', help='margin_m policy [fixed, step, linear]')
  parser.add_argument('--margin-s', type=float, default=64.0, help='scale for feature')
  parser.add_argument('--margin-a', type=float, default=1.0, help='')
  parser.add_argument('--margin-b', type=float, default=0.0, help='')
  parser.add_argument('--easy-margin', type=int, default=0, help='')
  parser.add_argument('--margin', type=int, default=4, help='margin for sphere')
  parser.add_argument('--beta', type=float, default=1000., help='param for sphere')
  parser.add_argument('--beta-min', type=float, default=5., help='param for sphere')
  parser.add_argument('--beta-freeze', type=int, default=0, help='param for sphere')
  parser.add_argument('--gamma', type=float, default=0.12, help='param for sphere')
  parser.add_argument('--power', type=float, default=1.0, help='param for sphere')
  parser.add_argument('--scale', type=float, default=0.9993, help='param for sphere')
  parser.add_argument('--rand-mirror', type=int, default=1, help='if do random mirror in training')
  parser.add_argument('--cutoff', type=int, default=0, help='cut off aug')
  parser.add_argument('--downsample-back', type=float, default=0.0, help='use downsample data augmentation')
  parser.add_argument('--motion-blur', type=float, default=0.0, help='motion blur aug')
  parser.add_argument('--target', type=str, default='lfw,cfp_fp,agedb_30', help='verification targets')
  args = parser.parse_args()
  return args


def get_symbol(network, num_layers, args, arg_params, aux_params):
  data_shape = (args.image_channel,args.image_h,args.image_w)
  image_shape = ",".join([str(x) for x in data_shape])
  margin_symbols = []
  print('init %s, num_layers: %d' % (network, num_layers))
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

  if args.loss_type == 6:
    m = mx.symbol.Variable(name='margin') 

  cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip().split(',')
  for i in range(len(args.num_classes)):
    gt_label = all_label[i].reshape([-1, ])
    extra_loss = None
    #_weight = mx.symbol.Variable("fc7_%d_weight" % i, shape=(args.num_classes[i], args.emb_size), lr_mult=args.fc7_lr_mult, wd_mult=args.fc7_wd_mult)
    if args.loss_type==0: #softmax
      fc7_subs = []
      classes_each_ctx = (args.num_classes[i] + len(cvd) - 1) // len(cvd)
      for ctx_id in range(len(cvd)):
        with mx.AttrScope(ctx_group='dev%d' % ctx_id):
          #_weight = mx.symbol.Variable("fc7_%d_%d_weight" % (i, ctx_id), shape=(classes_each_ctx, args.emb_size), lr_mult=args.fc7_lr_mult, wd_mult=args.fc7_wd_mult)
          _weight = mx.symbol.Variable("fc7_%d_weight" % (ctx_id), shape=(classes_each_ctx, args.emb_size), lr_mult=args.fc7_lr_mult, wd_mult=args.fc7_wd_mult)
          if args.fc7_no_bias:
            fc7_sub = mx.sym.FullyConnected(data=embedding, weight = _weight, no_bias = True, num_hidden=classes_each_ctx, name='fc7_%d' % (ctx_id))
          else:
            #_bias = mx.symbol.Variable('fc7_%d_%d_bias' % (i, ctx_id), lr_mult=2.0, wd_mult=0.0)
            _bias = mx.symbol.Variable('fc7_%d_bias' % (ctx_id), lr_mult=2.0, wd_mult=0.0)
            fc7_sub = mx.sym.FullyConnected(data=embedding, weight = _weight, bias = _bias, num_hidden=classes_each_ctx, name='fc7_%d' % (ctx_id))
          fc7_subs.append(fc7_sub)
      fc7 = mx.sym.concat(*fc7_subs, dim=1, name='fc7_subs_concat')
    elif args.loss_type==2:
      s = args.margin_s
      m = args.margin_m
      assert(s>0.0)
      assert(m>0.0)
      
      nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n_%d' % i)*s
      classes_each_ctx = (args.num_classes[i] + len(cvd) - 1) // len(cvd)
      fc7_subs = []
      for ctx_id in range(1, len(cvd)+1):
        with mx.AttrScope(ctx_group='dev%d' % ctx_id):
          _weight = mx.symbol.Variable("fc7_%d_%d_weight" % (i, ctx_id), shape=(classes_each_ctx, args.emb_size), lr_mult=args.fc7_lr_mult, wd_mult=args.fc7_wd_mult)
          _weight = mx.symbol.L2Normalization(_weight, mode='instance')
          fc7_sub = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=classes_each_ctx, name='fc7_%d_%d' % (i, ctx_id))
          fc7_subs.append(fc7_sub)
      fc7 = mx.sym.concat(*fc7_subs, dim=1, name='fc7_subs_concat')
      
      s_m = s*m
      gt_one_hot = mx.sym.one_hot(gt_label, depth = args.num_classes[i], on_value = s_m, off_value = 0.0)
      fc7 = fc7-gt_one_hot
    elif args.loss_type==4:
      s = args.margin_s
      m = args.margin_m
      assert s>0.0
      assert m>=0.0
      assert m<(math.pi/2)
      #_weight = mx.symbol.L2Normalization(_weight, mode='instance')
      nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n_%d' % i)*s
      #fc7 = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=args.num_classes[i], name='fc7_%d' % i)
      classes_each_ctx = (args.num_classes[i] + len(cvd) - 1) // len(cvd)
      fc7_subs = []
      for ctx_id in range(1, len(cvd)+1):
        with mx.AttrScope(ctx_group='dev%d' % ctx_id):
          _weight = mx.symbol.Variable("fc7_%d_%d_weight" % (i, ctx_id), shape=(classes_each_ctx, args.emb_size), lr_mult=args.fc7_lr_mult, wd_mult=args.fc7_wd_mult)
          _weight = mx.symbol.L2Normalization(_weight, mode='instance')
          fc7_sub = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=classes_each_ctx, name='fc7_%d_%d' % (i, ctx_id))
          fc7_subs.append(fc7_sub)
      fc7 = mx.sym.concat(*fc7_subs, dim=1, name='fc7_subs_concat')

      zy = mx.sym.pick(fc7, gt_label, axis=1)
      cos_t = zy/s
      cos_m = math.cos(m)
      sin_m = math.sin(m)
      mm = math.sin(math.pi-m)*m
      #threshold = 0.0
      threshold = math.cos(math.pi-m)
      if args.easy_margin:
        cond = mx.symbol.Activation(data=cos_t, act_type='relu')
      else:
        cond_v = cos_t - threshold
        cond = mx.symbol.Activation(data=cond_v, act_type='relu')
      body = cos_t*cos_t
      body = 1.0-body
      sin_t = mx.sym.sqrt(body)
      new_zy = cos_t*cos_m
      b = sin_t*sin_m
      new_zy = new_zy - b
      new_zy = new_zy*s
      if args.easy_margin:
        zy_keep = zy
      else:
        zy_keep = zy - s*mm
      new_zy = mx.sym.where(cond, new_zy, zy_keep)
  
      diff = new_zy - zy
      diff = mx.sym.expand_dims(diff, 1)
      gt_one_hot = mx.sym.one_hot(gt_label, depth = args.num_classes[i], on_value = 1.0, off_value = 0.0)
      body = mx.sym.broadcast_mul(gt_one_hot, diff)
      fc7 = fc7+body
    elif args.loss_type==5:
      s = args.margin_s
      m = args.margin_m
      assert s>0.0
      _weight = mx.symbol.L2Normalization(_weight, mode='instance')
      nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n_%d' % i)*s
      fc7 = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=args.num_classes[i], name='fc7_%d' % i)
      if args.margin_a!=1.0 or args.margin_m!=0.0 or args.margin_b!=0.0:
        if args.margin_a==1.0 and args.margin_m==0.0:
          s_m = s*args.margin_b
          gt_one_hot = mx.sym.one_hot(gt_label, depth = args.num_classes[i], on_value = s_m, off_value = 0.0)
          fc7 = fc7-gt_one_hot
        else:
          zy = mx.sym.pick(fc7, gt_label, axis=1)
          cos_t = zy/s
          t = mx.sym.arccos(cos_t)
          if args.margin_a!=1.0:
            t = t*args.margin_a
          if args.margin_m>0.0:
            t = t+args.margin_m
          body = mx.sym.cos(t)
          if args.margin_b>0.0:
            body = body - args.margin_b
          new_zy = body*s
          diff = new_zy - zy
          diff = mx.sym.expand_dims(diff, 1)
          gt_one_hot = mx.sym.one_hot(gt_label, depth = args.num_classes[i], on_value = 1.0, off_value = 0.0)
          body = mx.sym.broadcast_mul(gt_one_hot, diff)
          fc7 = fc7+body
    elif args.loss_type==6: # linear margin m
      s = args.margin_s
      assert s>0.0
      _weight = mx.symbol.L2Normalization(_weight, mode='instance')
      nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n_%d' % i)*s
      fc7 = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=args.num_classes[i], name='fc7_%d' % i)
      zy = mx.sym.pick(fc7, gt_label, axis=1)
      cos_t = zy/s
      cos_m = mx.symbol.cos(m)
      sin_m = mx.symbol.sin(m)
      mm = mx.symbol.sin(math.pi-m)*m
      #threshold = 0.0
      threshold = mx.symbol.cos(math.pi-m)
      if args.easy_margin:
        cond = mx.symbol.Activation(data=cos_t, act_type='relu')
      else:
        cond_v = cos_t - threshold
        cond = mx.symbol.Activation(data=cond_v, act_type='relu')
      body = cos_t*cos_t
      body = 1.0-body
      sin_t = mx.sym.sqrt(body)
      new_zy = cos_t*cos_m
      b = sin_t*sin_m
      new_zy = new_zy - b
      new_zy = new_zy*s
      if args.easy_margin:
        zy_keep = zy
      else:
        zy_keep = zy - s*mm
      new_zy = mx.sym.where(cond, new_zy, zy_keep)
  
      diff = new_zy - zy
      diff = mx.sym.expand_dims(diff, 1)
      gt_one_hot = mx.sym.one_hot(gt_label, depth = args.num_classes[i], on_value = 1.0, off_value = 0.0)
      body = mx.sym.broadcast_mul(gt_one_hot, diff)
      fc7 = fc7+body
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
      print('num_classes', args.num_classes[-1])
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
    network, num_layers = args.network.split(',')
    if len(args.pretrained)==0:
      arg_params = None
      aux_params = None
      sym, arg_params, aux_params = get_symbol(network, int(num_layers), args, arg_params, aux_params)
    else:
      vec = args.pretrained.split(',')
      print('loading', vec)
      _, arg_params, aux_params = mx.model.load_checkpoint(vec[0], int(vec[1]))
      sym, arg_params, aux_params = get_symbol(network, int(num_layers), args, arg_params, aux_params)
      #print(sym.collect_params()[0].keys())
      print(dir(sym)) #.collect_params()[0].keys())

    #label_name = 'softmax_label'
    #label_shape = (args.batch_size,)
    model = mx.mod.Module(
        context       = ctx,
        symbol        = sym,
        data_names    = ['data'] if args.loss_type != 6 else ['data', 'margin'],
        group2ctxs    = dict(zip(['dev%d' % i for i in range(len(ctx))], ctx))
    )
    val_dataiter = None

    train_dataiter = FaceImageIter(
        batch_size           = args.batch_size,
        data_shape           = data_shape,
        path_imgrecs         = path_imgrecs,
        shuffle              = True,
        rand_mirror          = args.rand_mirror,
        mean                 = mean,
        cutoff               = args.cutoff,
        loss_type            = args.loss_type,
        margin_m             = args.margin_m,
        margin_policy        = args.margin_policy,
        max_steps            = args.max_steps,
        data_names           = ['data', 'margin'],
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
    _rescale = 1.0/ args.batch_size # args.ctx_num 
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

