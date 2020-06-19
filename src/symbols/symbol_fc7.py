import math
import mxnet as mx

def get_fc7(embedding, name, args, cvd=None):
  nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name=name + '_norm')
  if cvd is None:
    _weight = mx.symbol.Variable(name + "_weight", shape=(args.ctx_num_classes, args.emb_size), 
        attr = {'lr_mult':str(args.fc7_lr_mult), 'wd_mult':str(args.fc7_wd_mult)}, init=mx.init.Normal(0.01))
    fc7 = mx.sym.FullyConnected(data=nembedding, weight = _weight, num_hidden=args.ctx_num_classes,
                              no_bias = True, normalize=True, name=name)
  else:  
    fc7_subs = []
    for ctx_id in range(len(cvd)):
      with mx.AttrScope(ctx_group='dev%d' % (ctx_id+1)):
        _weight = mx.symbol.Variable(name % ctx_id + '_weight', shape=(args.ctx_num_classes, args.emb_size))
        fc7_sub = mx.sym.FullyConnected(data=nembedding, weight = _weight, num_hidden=args.ctx_num_classes,
                              no_bias = True, normalize=True, name=name % ctx_id)
        fc7_subs.append(fc7_sub)
    fc7 = mx.sym.concat(*fc7_subs, dim=1, name=name + '_concat')
  return fc7

def Softmax(embedding, gt_label, name, args, cvd=None):
  if cvd is None:
    _weight = mx.symbol.Variable(name + "_weight", shape=(args.ctx_num_classes, args.emb_size), 
        lr_mult=args.fc7_lr_mult, wd_mult=args.fc7_wd_mult, init=mx.init.Normal(0.01))
    if args.fc7_no_bias:
      fc7 = mx.sym.FullyConnected(data=embedding, weight = _weight, no_bias = True, num_hidden=args.ctx_num_classes, name=name)
    else:
      _bias = mx.symbol.Variable(name + '_bias', lr_mult=2.0, wd_mult=0.0)
      fc7 = mx.sym.FullyConnected(data=embedding, weight = _weight, bias = _bias, num_hidden=args.ctx_num_classes, name=name)
    return fc7
  else:
    fc7_subs = []
    for ctx_id in range(len(cvd)):
      with mx.AttrScope(ctx_group='dev%d' % (ctx_id+1)):
        _weight = mx.symbol.Variable(name % ctx_id + "_weight", shape=(args.ctx_num_classes, args.emb_size), 
            lr_mult=args.fc7_lr_mult, wd_mult=args.fc7_wd_mult, init=mx.init.Normal(0.01))
        if args.fc7_no_bias:
          fc7_sub = mx.sym.FullyConnected(data=embedding, weight = _weight, no_bias = True, num_hidden=args.ctx_num_classes, name=name % ctx_id)
        else:
          _bias = mx.symbol.Variable(name % ctx_id + '_bias', lr_mult=2.0, wd_mult=0.0)
          fc7_sub = mx.sym.FullyConnected(data=embedding, weight = _weight, bias = _bias, num_hidden=args.ctx_num_classes, name=name % ctx_id)
        fc7_subs.append(fc7_sub)
    fc7 = mx.sym.concat(*fc7_subs, dim=1, name=name + '_concat')
    return fc7

def CosFace(embedding, gt_label, name, args, cvd=None):
  s = args.margin_s
  m = args.margin_m
  assert(s>0.0)
  assert(m>0.0)
  
  fc7 = get_fc7(embedding, name, args, cvd)
    
  gt_one_hot = mx.sym.one_hot(gt_label, depth = args.ctx_num_classes, on_value = m, off_value = 0.0)
  fc7 = (fc7-gt_one_hot) * s
  return fc7

def ArcFace(embedding, gt_label, name, args, cvd=None):
  s = args.margin_s
  m = args.margin_m
  assert s>0.0
  assert m>=0.0
  assert m<(math.pi/2)
  
  fc7 = get_fc7(embedding, name, args, cvd)

  zy = mx.sym.pick(fc7, gt_label, axis=1)

  cos_t = zy
  sin_t = mx.sym.sqrt(1 - cos_t*cos_t)

  cos_m = math.cos(m)
  sin_m = math.sin(m)
  new_zy = cos_t*cos_m - sin_t*sin_m
  new_zy = new_zy

  #threshold = 0.0
  threshold = math.cos(math.pi-m)
  if args.easy_margin:
    cond = mx.symbol.Activation(data=cos_t, act_type='relu')
    zy_keep = zy
  else:
    cond_v = cos_t - threshold
    cond = mx.symbol.Activation(data=cond_v, act_type='relu')
    mm = math.sin(math.pi-m)*m
    zy_keep = zy - mm
  new_zy = mx.sym.where(cond, new_zy, zy_keep)

  diff = new_zy - zy
  diff = mx.sym.expand_dims(diff, 1)
  gt_one_hot = mx.sym.one_hot(gt_label, depth = args.ctx_num_classes, on_value = 1.0, off_value = 0.0)
  body = mx.sym.broadcast_mul(gt_one_hot, diff)

  fc7 = (fc7 + body) * s
  return fc7

def AdaCos(embedding, gt_label, name, args, cvd=None):
  fc7 = get_fc7(embedding, name, args, cvd)

  from gluon_modules import AdaCosModule
  mod = AdaCosModule(args.ctx_num_classes)
  fc7 = mod(fc7)
  return fc7

def CombineFace(embedding, gt_label, name, args, cvd=None):
  s = args.margin_s
  m = args.margin_m
  assert s>0.0

  fc7 = get_fc7(embedding, name, args, cvd)

  if args.margin_a!=1.0 or args.margin_m!=0.0 or args.margin_b!=0.0:
    if args.margin_a==1.0 and args.margin_m==0.0:
      s_m = args.margin_b
      gt_one_hot = mx.sym.one_hot(gt_label, depth = args.ctx_num_classes, on_value = s_m, off_value = 0.0)
      fc7 = (fc7-gt_one_hot) * s
    else:
      zy = mx.sym.pick(fc7, gt_label, axis=1)
      cos_t = zy
      t = mx.sym.arccos(cos_t)
      if args.margin_a!=1.0:
        t = t*args.margin_a
      if args.margin_m>0.0:
        t = t+args.margin_m
      body = mx.sym.cos(t)
      if args.margin_b>0.0:
        body = body - args.margin_b
      new_zy = body
      diff = new_zy - zy
      diff = mx.sym.expand_dims(diff, 1)
      gt_one_hot = mx.sym.one_hot(gt_label, depth = args.ctx_num_classes, on_value = 1.0, off_value = 0.0)
      body = mx.sym.broadcast_mul(gt_one_hot, diff)
      fc7 = (fc7+body) * s
  return fc7

def LarcFace(embedding, gt_label, name, args, cvd=None):
  s = args.margin_s
  assert s>0.0

  m = mx.symbol.Variable(name='margin') 

  fc7 = get_fc7(embedding, name, args, cvd)

  zy = mx.sym.pick(fc7, gt_label, axis=1)
  cos_t = zy
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
  sin_t = mx.sym.sqrt(1 - cos_t * cos_t)
  new_zy = cos_t*cos_m - sin_t * sin_m
  if args.easy_margin:
    zy_keep = zy
  else:
    zy_keep = zy - mm
  new_zy = mx.sym.where(cond, new_zy, zy_keep)

  diff = new_zy - zy
  diff = mx.sym.expand_dims(diff, 1)
  gt_one_hot = mx.sym.one_hot(gt_label, depth = args.ctx_num_classes, on_value = 1.0, off_value = 0.0)
  body = mx.sym.broadcast_mul(gt_one_hot, diff)
  fc7 = (fc7+body)*s
  return fc7

def CircleLoss(embedding, gt_label, name, args, cvd=None):
  fc7 = get_fc7(embedding, name, args, cvd)

  gt_one_hot = mx.sym.one_hot(gt_label, depth = args.ctx_num_classes, on_value = 1.0, off_value = 0.0)
  gt_reverse = mx.sym.one_hot(gt_label, depth = args.ctx_num_classes, on_value = -1.0, off_value = 1.0)

  # an (ap) detached
  fc7_detached = mx.sym.BlockGrad(fc7)
  alpha = fc7_detached * gt_reverse + gt_one_hot + config.margin
  alpha_relu = mx.symbol.Activation(data=alpha, act_type='relu')

  # calculate delta
  delta = gt_one_hot + config.margin * gt_reverse

  fc7 = config.gamma * alpha_relu * (fc7 - delta)
  return fc7

def CurricularLoss(embedding, gt_label, name, args, cvd=None):
  fc7 = get_fc7(embedding, name, args, cvd)

  from gluon_modules import CurricularModule
  mod = CurricularModule(args.ctx_num_classes, args.loss_m, args.loss_s, args._ctxid)
  fc7 = mod(fc7)
  return fc7

def MarginDistillation(embedding, gt_label, name, args, cvd=None):
  s = args.margin_s
  assert s>0.0

  fc7 = get_fc7(embedding, args, cvd)
  
  margin = mx.symbol.Variable(name='margin')

  """
  gt_label_hardlabel = mx.symbol.slice_axis(gt_label, axis=-1, begin=-1, end=args.emb_size + 1)
  gt_label_hardlabel = mx.symbol.reshape(gt_label_hardlabel, shape=(args.per_batch_size))
  gt_label_softlabel = mx.symbol.slice_axis(gt_label, axis=-1, begin=0, end=-1)
  
  gt_one_hot = mx.sym.one_hot(gt_label_hardlabel, depth = args.ctx_num_classes, on_value = 1.0, off_value = 0.0)
  teacher_centers = mx.sym.linalg.gemm2(gt_one_hot, _weight)
  
  gt_label_softlabel = mx.symbol.L2Normalization(gt_label_softlabel, mode='instance')
  el = gt_label_softlabel * teacher_centers
  cos_loss = mx.symbol.sum(el, axis=1)
  
  Dmax = mx.symbol.max(cos_loss)
  margin = mx.symbol.broadcast_mul((args.Mmax - config.Mmin) / Dmax, cos_loss) + config.Mmin
  """
  loss_m2 = margin
  
  zy = mx.sym.pick(fc7, gt_label_hardlabel, axis=1)
  cos_t = zy
  t = mx.sym.arccos(cos_t)
  t = t+loss_m2
  body = mx.sym.cos(t)
  new_zy = body
  diff = new_zy - zy
  diff = mx.sym.expand_dims(diff, 1)
  gt_one_hot = mx.sym.one_hot(gt_label_hardlabel, depth = args.ctx_num_classes, on_value = 1.0, off_value = 0.0)
  body = mx.sym.broadcast_mul(gt_one_hot, diff)
  fc7 = (fc7+body)*s
  return fc7
