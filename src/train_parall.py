'''
@author: insightface
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pair_image_iter import FaceImageIter

import pdb
import os, sys
import math, random
import logging
import pickle
import sklearn
import argparse
import numpy as np
import mxnet as mx
from mxnet import ndarray as nd
import mxnet.optimizer as optimizer

sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'symbols'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

import face_image
from config import config, default, generate_config
import verification
import init_random
import fresnet
import fmobilefacenet
import fmobilenet
import fmnasnet
import fdensenet
import fsmall


logger = logging.getLogger()
logger.setLevel(logging.INFO)


args = None



def parse_args():
  parser = argparse.ArgumentParser(description='Train parall face network')
  # general
  parser.add_argument('--dataset', default=default.dataset, help='dataset config')
  parser.add_argument('--network', default=default.network, help='network config')
  parser.add_argument('--loss', default=default.loss, help='loss config')
  args, rest = parser.parse_known_args()
  generate_config(args.network, args.dataset, args.loss)
  parser.add_argument('--models-root', default=default.models_root, help='root directory to save model.')
  parser.add_argument('--pretrained', default=default.pretrained, help='pretrained model to load')
  parser.add_argument('--ckpt', type=int, default=default.ckpt, help='checkpoint saving option. 0: discard saving. 1: save when necessary. 2: always save')
  parser.add_argument('--verbose', type=int, default=default.verbose, help='do verification testing and model saving every verbose batches')
  parser.add_argument('--max-steps', type=int, default=config.max_steps, help='max training batches')
  parser.add_argument('--lr', type=float, default=default.lr, help='start learning rate')
  parser.add_argument('--lr-steps', type=str, default=default.lr_steps, help='steps of lr changing')
  parser.add_argument('--wd', type=float, default=default.wd, help='weight decay')
  parser.add_argument('--mom', type=float, default=default.mom, help='momentum')
  parser.add_argument('--frequent', type=int, default=default.frequent, help='')
  parser.add_argument('--per-batch-size', type=int, default=default.per_batch_size, help='batch size in each context')
  parser.add_argument('--kvstore', type=str, default=default.kvstore, help='kvstore setting')
  parser.add_argument('--worker-id', type=int, default=0, help='worker id for dist training, starts from 0')
  parser.add_argument('--margin-policy', type=str, default='fixed', help='margin_m policy [fixed, step, linear]')
  args = parser.parse_args()
  return args


def get_symbol_embedding():
  tmp_config = {}
  for k,v in config.items():
    if 'num_classes' == k or 'num_layers' == k:
      continue
    tmp_config[k] = v
  embedding = eval(config.net_name).get_symbol(config.emb_size, config.num_layers, **tmp_config)

  all_label = mx.symbol.Variable('softmax_label')
  #embedding = mx.symbol.BlockGrad(embedding)
  all_label = mx.symbol.BlockGrad(all_label)
  out_list = [embedding, all_label]
  out = mx.symbol.Group(out_list)
  return out

def get_symbol_arcface(args):
  embedding = mx.symbol.Variable('data')
  all_label = mx.symbol.Variable('softmax_label')
  gt_label = all_label
  is_softmax = True
  #print('call get_sym_arcface with', args, config)
  _weight = mx.symbol.Variable("fc7_%d_weight"%args._ctxid, shape=(args.ctx_num_classes, config.emb_size)) #,wd_mult=config.fc7_wd_mult, lr_mult=config.fc7_lr_mult)
  if config.loss_name=='softmax': #softmax
    fc7 = mx.sym.FullyConnected(data=embedding, weight = _weight, no_bias = True, num_hidden=args.ctx_num_classes, name='fc7_%d'%args._ctxid)
  elif config.loss_name=='margin_softmax':
    #_weight = mx.symbol.L2Normalization(_weight, mode='instance')
    #_weight_norm = mx.symbol.norm(_weight, ord=2, axis=1).reshape([-1, 1]) + 1e-10
    #_weight_norm = mx.symbol.BlockGrad(_weight_norm)
    #_weight = mx.symbol.broadcast_div(_weight, _weight_norm, axis=1)
    nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n_%d'%args._ctxid)
    fc7 = mx.sym.FullyConnected(data=nembedding, weight = _weight, num_hidden=args.ctx_num_classes,
                                no_bias = True, normalize=True, name='fc7_%d'%args._ctxid)

    if config.loss_m1!=1.0 or config.loss_m2!=0.0 or config.loss_m3!=0.0:
      gt_one_hot = mx.sym.one_hot(gt_label, depth = args.ctx_num_classes, on_value = 1.0, off_value = 0.0)
      if config.loss_m1==1.0 and config.loss_m2==0.0:
        _one_hot = gt_one_hot*args.margin_b
        fc7 = fc7-_one_hot
      else:
        fc7_onehot = fc7 * gt_one_hot
        cos_t = fc7_onehot
        t = mx.sym.arccos(cos_t)
        if config.loss_m1!=1.0:
          t = t*config.loss_m1
        if config.loss_m2!=0.0:
          t = t+config.loss_m2
        margin_cos = mx.sym.cos(t)
        if config.loss_m3!=0.0:
          margin_cos = margin_cos - config.loss_m3
        margin_fc7 = margin_cos
        margin_fc7_onehot = margin_fc7 * gt_one_hot
        diff = margin_fc7_onehot - fc7_onehot
        fc7 = fc7+diff
    fc7 = fc7*config.loss_s
  elif config.loss_name=='circle_loss':
    nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n_%d'%args._ctxid)
    fc7 = mx.sym.FullyConnected(data=nembedding, weight = _weight, num_hidden=args.ctx_num_classes,
                                no_bias = True, normalize=True, name='fc7_%d'%args._ctxid)

    gt_one_hot = mx.sym.one_hot(gt_label, depth = args.ctx_num_classes, on_value = 1.0, off_value = 0.0)
    gt_reverse = mx.sym.one_hot(gt_label, depth = args.ctx_num_classes, on_value = -1.0, off_value = 1.0)

    # an (ap) detached
    fc7_detached = mx.sym.BlockGrad(fc7)
    fc7_detached = fc_detached7 - gt_one_hot
    scale_factor = fc_detached7 * gt_reverse + config.margin
    # clip min to 0
    scale_factor = mx.symbol.Activation(data=scale_factor, act_type='relu')

    delta = gt_one_hot + config.margin * gt_reverse
    margin_fc7 = fc7 - delta
    fc7 = scale_factor * margin_fc7

    # an (ap) not detached
    """    
    fc7 = fc7 - gt_one_hot
    fc7 = mx.sym.square(fc7) - config.margin ** 2
    fc7 = (fc7 * gt_reverse) * config.gamma
    """

  out_list = []
  out_list.append(fc7)
  if config.loss_name=='softmax': #softmax
    out_list.append(gt_label)
  out = mx.symbol.Group(out_list)
  return out

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
    prefix = os.path.join(args.models_root, '%s-%s-%s'%(args.network, args.loss, args.dataset), 'model')
    prefix_dir = os.path.dirname(prefix)
    print('prefix', prefix)
    if not os.path.exists(prefix_dir):
      os.makedirs(prefix_dir)
    args.ctx_num = len(ctx)
    args.num_layers = int(args.network[1:])
    print('num_layers', args.num_layers)
    if args.per_batch_size==0:
      args.per_batch_size = 128
    args.batch_size = args.per_batch_size*args.ctx_num
    args.rescale_threshold = 0
    args.image_channel = config.image_shape[2]
    data_dir = config.dataset_path
    path_imgrecs = None
    path_imglist = None
    image_size = config.image_shape[0:2]
    assert len(image_size)==2
    assert image_size[0]==image_size[1]
    print('image_size', image_size)
    print('num_classes', config.num_classes)
    path_imgrecs = [os.path.join(data_dir, "train.rec")]

    data_shape = (args.image_channel,image_size[0],image_size[1])

    num_workers = config.num_workers
    global_num_ctx = num_workers * args.ctx_num
    if config.num_classes%global_num_ctx==0:
      args.ctx_num_classes = config.num_classes//global_num_ctx
    else:
      args.ctx_num_classes = config.num_classes//global_num_ctx+1
    print(config.num_classes, global_num_ctx, args.ctx_num_classes)
    args.local_num_classes = args.ctx_num_classes * args.ctx_num
    args.local_class_start = args.local_num_classes * args.worker_id

    #if len(args.partial)==0:
    #  local_classes_range = (0, args.num_classes)
    #else:
    #  _vec = args.partial.split(',')
    #  local_classes_range = (int(_vec[0]), int(_vec[1]))

    #args.partial_num_classes = local_classes_range[1] - local_classes_range[0]
    #args.partial_start = local_classes_range[0]

    print('Called with argument:', args, config)
    mean = None

    begin_epoch = 0
    base_lr = args.lr
    base_wd = args.wd
    base_mom = args.mom
    arg_params = None
    aux_params = None
    esym = get_symbol_embedding()
    asym = get_symbol_arcface
    if config.num_workers==1:
      sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
      from parall_module_local_v1 import ParallModule
    else:
      from parall_module_dist import ParallModule

    model = ParallModule(
        context       = ctx,
        symbol        = esym,
        data_names    = ['data'],
        label_names    = ['softmax_label'],
        asymbol       = asym,
        args = args,
    )

    val_dataiter = None
    train_dataiter = FaceImageIter(
        batch_size           = args.batch_size,
        data_shape           = data_shape,
        path_imgrecs         = path_imgrecs,
        shuffle              = True,
        rand_mirror          = config.data_rand_mirror,
        mean                 = mean,
        cutoff               = default.cutoff if config.data_cutoff else None,
        crop                 = default.crop if config.data_crop else None,
        mask                 = default.mask if config.data_mask else None,
        gridmask             = default.gridmask if config.data_grid else None,
        #color_jittering      = config.data_color,
        #images_filter        = config.data_images_filter,
        loss_type            = args.loss,
        #margin_m             = config.loss_m2,
        data_names           = ['data'],
        downsample_back      = config.downsample_back,
        motion_blur          = config.motion_blur,
        use_bgr              = config.use_bgr
    )


    
    if config.net_name=='fresnet' or config.net_name=='fmobilefacenet':
      initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2) #resnet style
    else:
      initializer = mx.init.Xavier(rnd_type='uniform', factor_type="in", magnitude=2)

    _rescale = 1.0 / 8 #/ args.batch_size
    print(base_lr, base_mom, base_wd, args.batch_size)

    lr_steps = [int(x) for x in args.lr_steps.split(',')]
    lr_scheduler =  mx.lr_scheduler.MultiFactorScheduler(lr_steps, factor=0.1, base_lr=base_lr)
    optimizer_params = {'learning_rate':base_lr,
                        'momentum':base_mom,
                        'wd':base_wd,
                        'rescale_grad':_rescale, 
                        'lr_scheduler': lr_scheduler}

    opt = optimizer.SGD(learning_rate=base_lr, momentum=base_mom, wd=base_wd, rescale_grad=_rescale)

    _cb = mx.callback.Speedometer(args.batch_size, args.frequent)


    ver_list = []
    ver_name_list = []
    for name in config.val_targets:
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
          fp_rates, fp_dict, thred_dict, recall_dict = verification.test(ver_list[i], model, args.batch_size, use_bgr=config.use_bgr)
          for k in fp_rates:
            print("[%s] TPR at FPR %.2e[%.2e: %.4f]:\t%.5f" %(ver_name_list[i], k, fp_dict[k], thred_dict[k], recall_dict[k]))

        else:
          acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(ver_list[i], model, args.batch_size, 10, None, 
              label_shape = (args.batch_size, len(path_imgrecs)), use_bgr=config.use_bgr)
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
    lr_steps = [int(x) for x in args.lr_steps.split(',')]
    print('lr_steps', lr_steps)
    def _batch_callback(param):
      #global global_step
      global_step[0]+=1
      mbatch = global_step[0]

      #for step in lr_steps:
      #  if mbatch==step:
      #    opt.lr *= 0.1
      #    print('lr change to', opt.lr)
      #    break

      _cb(param)
      if mbatch%1000==0:
        #print('lr-batch-epoch:',opt.lr,param.nbatch,param.epoch)
        print('batch-epoch:',param.nbatch,param.epoch)

      if mbatch>=0 and mbatch%args.verbose==0:
        acc_list = ver_test(mbatch)
        save_step[0]+=1
        msave = save_step[0]
        do_save = False
        is_highest = False
        if len(acc_list)>0:
          #lfw_score = acc_list[0]
          #if lfw_score>highest_acc[0]:
          #  highest_acc[0] = lfw_score
          #  if lfw_score>=0.998:
          #    do_save = True
          score = sum(acc_list)
          if acc_list[-1]>=highest_acc[-1]:
            if acc_list[-1]>highest_acc[-1]:
              is_highest = True
            else:
              if score>=highest_acc[0]:
                is_highest = True
                highest_acc[0] = score
            highest_acc[-1] = acc_list[-1]
            #if lfw_score>=0.99:
            #  do_save = True
        if is_highest:
          do_save = True
        if args.ckpt==0:
          do_save = False
        elif args.ckpt==2:
          do_save = True
        elif args.ckpt==3:
          msave = 1

        if do_save:
          print('saving', msave)
          arg, aux = model.get_params() #get_export_params()
          all_layers = model.symbol.get_internals()
          _sym = model.symbol #all_layers['fc1_output']
          mx.model.save_checkpoint(prefix, msave, _sym, arg, aux)
        print('[%d]Accuracy-Highest: %1.5f'%(mbatch, highest_acc[-1]))
      if args.max_steps>0 and mbatch>args.max_steps:
        sys.exit(0)

    epoch_cb = None
    train_dataiter = mx.io.PrefetchingIter(train_dataiter)

    if len(args.pretrained) !=0 :
      model_prefix, epoch = args.pretrained.split(',')
      begin_epoch = int(epoch)
      _, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, begin_epoch)
      #model.set_params(arg_params, aux_params)

    model.fit(train_dataiter,
        begin_epoch        = 0, #begin_epoch,
        num_epoch          = default.end_epoch,
        eval_data          = val_dataiter,
        #eval_metric        = eval_metrics,
        kvstore            = args.kvstore,
        #optimizer          = opt,
        optimizer_params   = optimizer_params,
        initializer        = initializer,
        arg_params         = arg_params,
        aux_params         = aux_params,
        allow_missing      = True,
        batch_end_callback = _batch_callback,
        epoch_end_callback = epoch_cb )

def main():
    global args
    args = parse_args()
    train_net(args)

if __name__ == '__main__':
    main()

