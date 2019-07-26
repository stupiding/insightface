import mxnet as mx

def Softmax(embedding, gt_label, classes_each_ctx, name, args, cvd=None):
  if cvd is None:
    _weight = mx.symbol.Variable(name + "_weight", shape=(classes_each_ctx, args.emb_size), lr_mult=args.fc7_lr_mult, wd_mult=args.fc7_wd_mult)
    if args.fc7_no_bias:
      fc7 = mx.sym.FullyConnected(data=embedding, weight = _weight, no_bias = True, num_hidden=classes_each_ctx, name=name)
    else:
      _bias = mx.symbol.Variable(name + '_bias', lr_mult=2.0, wd_mult=0.0)
      fc7 = mx.sym.FullyConnected(data=embedding, weight = _weight, bias = _bias, num_hidden=classes_each_ctx, name=name)
  else:
    fc7_subs = []
    for ctx_id in range(len(cvd)):
      with mx.AttrScope(ctx_group='dev%d' % ctx_id):
        _weight = mx.symbol.Variable(name % ctx_id + "weight", shape=(classes_each_ctx, args.emb_size), lr_mult=args.fc7_lr_mult, wd_mult=args.fc7_wd_mult)
        if args.fc7_no_bias:
          fc7_sub = mx.sym.FullyConnected(data=embedding, weight = _weight, no_bias = True, num_hidden=classes_each_ctx, name=name % ctx_id)
        else:
          _bias = mx.symbol.Variable(name % ctx_id + '_bias', lr_mult=2.0, wd_mult=0.0)
          fc7_sub = mx.sym.FullyConnected(data=embedding, weight = _weight, bias = _bias, num_hidden=classes_each_ctx, name=name % ctx_id)
        fc7_subs.append(fc7_sub)
    fc7 = mx.sym.concat(*fc7_subs, dim=1, name=name + '_concat')
  return fc7

def CosFace(embedding, gt_label, classes_each_ctx, args, cvd=None):
  s = args.margin_s
  m = args.margin_m
  assert(s>0.0)
  assert(m>0.0)
  
  nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name=name + '_norm')*s
  if cvd is None:
    _weight = mx.symbol.Variable(name + "_weight", shape=(classes_each_ctx, args.emb_size), lr_mult=args.fc7_lr_mult, wd_mult=args.fc7_wd_mult)
    _weight = mx.symbol.L2Normalization(_weight, mode='instance')
    fc7 = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=classes_each_ctx, name=name)
  else:  
    fc7_subs = []
    for ctx_id in range(len(cvd)):
      with mx.AttrScope(ctx_group='dev%d' % ctx_id):
        _weight = mx.symbol.Variable(name % ctx_id + "weight", shape=(classes_each_ctx, args.emb_size), lr_mult=args.fc7_lr_mult, wd_mult=args.fc7_wd_mult)
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        fc7_sub = mx.sym.FullyConnected(data=embedding, weight = _weight, no_bias = True, num_hidden=classes_each_ctx, name=name % ctx_id)
        fc7_subs.append(fc7_sub)
    fc7 = mx.sym.concat(*fc7_subs, dim=1, name=name + '_concat')
    
  s_m = s*m
  gt_one_hot = mx.sym.one_hot(gt_label, depth = args.num_classes[i], on_value = s_m, off_value = 0.0)
  fc7 = fc7-gt_one_hot
  return fc7

def ArcFace(embedding, gt_label, classes_each_ctx, name, args, cvd=None):
  s = args.margin_s
  m = args.margin_m
  assert s>0.0
  assert m>=0.0
  assert m<(math.pi/2)
  
  nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name=name + '_norm')*s
  if cvd is None:
    _weight = mx.symbol.Variable(name + '_weight', shape=(classes_each_ctx, args.emb_size), lr_mult=args.fc7_lr_mult, wd_mult=args.fc7_wd_mult)
    _weight = mx.symbol.L2Normalization(_weight, mode='instance')
    fc7 = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=classes_each_ctx, name=name)
  else:
    fc7_subs = []
    for ctx_id in range(len(cvd)):
      with mx.AttrScope(ctx_group='dev%d' % ctx_id):
        _weight = mx.symbol.Variable(name % ctx_id + "_weight", shape=(classes_each_ctx, args.emb_size), lr_mult=args.fc7_lr_mult, wd_mult=args.fc7_wd_mult)
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        fc7_sub = mx.sym.FullyConnected(data=embedding, weight = _weight, no_bias = True, num_hidden=classes_each_ctx, name=name % ctx_id)
        fc7_subs.append(fc7_sub)
    fc7 = mx.sym.concat(*fc7_subs, dim=1, name=name + '_concat')

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
  return fc7

def CombineFace(embedding, gt_label, classes_each_ctx, name, args, cvd=None):
  s = args.margin_s
  m = args.margin_m
  assert s>0.0

  nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name=name + '_norm')*s
  if cvd is None:
    _weight = mx.symbol.Variable(name + '_weight', shape=(classes_each_ctx, args.emb_size), lr_mult=args.fc7_lr_mult, wd_mult=args.fc7_wd_mult)
    _weight = mx.symbol.L2Normalization(_weight, mode='instance')
    fc7 = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=classes_each_ctx, name=name)
  else:
    fc7_subs = []
    for ctx_id in range(len(cvd)):
      with mx.AttrScope(ctx_group='dev%d' % ctx_id):
        _weight = mx.symbol.Variable(name % ctx_id + "_weight", shape=(classes_each_ctx, args.emb_size), lr_mult=args.fc7_lr_mult, wd_mult=args.fc7_wd_mult)
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        fc7_sub = mx.sym.FullyConnected(data=embedding, weight = _weight, no_bias = True, num_hidden=classes_each_ctx, name=name % ctx_id)
        fc7_subs.append(fc7_sub)
    fc7 = mx.sym.concat(*fc7_subs, dim=1, name=name + '_concat')

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
  return fc7

def LarcFace(embedding, gt_label, classes_each_ctx, name, args, cvd=None):
  s = args.margin_s
  assert s>0.0

  m = mx.symbol.Variable(name='margin') 

  if cvd is None:
    _weight = mx.symbol.Variable(name + '_weight', shape=(classes_each_ctx, args.emb_size), lr_mult=args.fc7_lr_mult, wd_mult=args.fc7_wd_mult)
    _weight = mx.symbol.L2Normalization(_weight, mode='instance')
    fc7 = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=classes_each_ctx, name=name)
  else:
    fc7_subs = []
    for ctx_id in range(len(cvd)):
      with mx.AttrScope(ctx_group='dev%d' % ctx_id):
        _weight = mx.symbol.Variable(name % ctx_id + "_weight", shape=(classes_each_ctx, args.emb_size), lr_mult=args.fc7_lr_mult, wd_mult=args.fc7_wd_mult)
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        fc7_sub = mx.sym.FullyConnected(data=embedding, weight = _weight, no_bias = True, num_hidden=classes_each_ctx, name=name % ctx_id)
        fc7_subs.append(fc7_sub)
    fc7 = mx.sym.concat(*fc7_subs, dim=1, name=name + '_concat')

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
  return fc7
