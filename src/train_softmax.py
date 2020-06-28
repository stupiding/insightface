from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
from image_iter import FaceImageIter

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
        bn_mom = args.bn_mom, avg_down=args.avg_down)

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
    args.ctx_num_classes = classes_each_ctx

    losses = {'softmax'   : Softmax,
              'cosface'   : CosFace,
              'adacos'    : AdaCos,
              'arcface'   : ArcFace,
              'circle'    : CircleLoss,
              'combine'   : CombineFace,
              'curricular': CurricularLoss
             }

    if args.loss_type=="sphere": #sphere
      _weight = mx.symbol.L2Normalization(_weight, mode='instance')
      fc7 = mx.sym.LSoftmax(data=embedding, label=gt_label, num_hidden=args.num_classes[i],
                            weight = _weight,
                            beta=args.beta, margin=args.margin, scale=args.scale,
                            beta_min=args.beta_min, verbose=1000, name='fc7_%d' % i)
    elif args.loss_type in losses:
      fc7 = losses[args.loss_type](embedding, gt_label, name, args, cvd)
    else:
      raise ValueError("Loss [%s] is not implemented" % args.loss_type)

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
    for data_idx, data_dir in enumerate(data_dir_list):
      prop = face_image.load_property(data_dir)
      args.num_classes.append(prop.num_classes)
      image_size = prop.image_size
      if data_idx == 0:
        args.image_h = image_size[0]
        args.image_w = image_size[1]
      else:
        args.image_h = min(args.image_h, image_size[0]) 
        args.image_w = min(args.image_w, image_size[1])
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

    from config import crop
    from config import cutout

    train_dataiter = FaceImageIter(
        batch_size           = args.batch_size,
        data_shape           = data_shape,
        path_imgrecs         = path_imgrecs,
        shuffle              = True,
        rand_mirror          = args.rand_mirror,
        mean                 = mean,
        cutout               = cutout,
        crop                 = crop, 
        loss_type            = args.loss_type,
        #margin_m             = args.margin_m,
        #margin_policy        = args.margin_policy,
        #max_steps            = args.max_steps,
        #data_names           = ['data', 'margin'],
        downsample_back      = args.downsample_back,
        motion_blur          = args.motion_blur,
    )

    _metric = AccMetric()
    #_metric = LossValueMetric()
    eval_metrics = [mx.metric.create(_metric)]

    if args.network[0]=='r' or args.network[0]=='y':
      initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2) #resnet style
    elif args.network[0]=='i' or args.network[0]=='x':
      initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2) #inception
    else:
      initializer = mx.init.Xavier(rnd_type='uniform', factor_type="in", magnitude=2)
    _rescale = 1.0/args.ctx_num

    if len(args.lr_steps)==0:
      print('Error: lr_steps is not seted')
      sys.exit(0)
    else:
      lr_steps = [int(x) for x in args.lr_steps.split(',')]
    print('lr_steps', lr_steps)

    lr_scheduler =  mx.lr_scheduler.MultiFactorScheduler(lr_steps, factor=0.1, base_lr=base_lr)
    optimizer_params = {'learning_rate':base_lr,
                        'momentum':base_mom,
                        'wd':base_wd,
                        'rescale_grad':_rescale,
                        'lr_scheduler': lr_scheduler}

    #opt = AdaBound()
    #opt = AdaBound(lr=base_lr, wd=base_wd, gamma = 2. / args.max_steps)
    opt = optimizer.SGD(**optimizer_params)

    som = 2000
    _cb = mx.callback.Speedometer(args.batch_size, som)

    ver_list = []
    ver_name_list = []
    for name in args.target.split(','):
      path = os.path.join(data_dir,name+".bin")
      if os.path.exists(path):
        data_set = verification.load_bin(path, image_size)
        ver_list.append(data_set)
        ver_name_list.append(name)
        print('ver', name)

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


    def _batch_callback(param):
      #global global_step
      global_step[0]+=1
      mbatch = global_step[0]

      _cb(param)
      if mbatch%10000==0:
        print('lr-batch-epoch:',opt.learning_rate,param.nbatch,param.epoch)

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
        optimizer_params   = optimizer_params,
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

