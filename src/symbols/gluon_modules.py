import math
import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn
import mxnet.autograd as ag
import symbol_utils

class AdaCosModule(nn.HybridBlock):
  def __init__(self, num_classes, **kwargs):
    super(AdaCosModule, self).__init__(**kwargs)
    self.n_classes = num_classes
    self.s = math.sqrt(2) * math.log(num_classes - 1)
    self.min_cos = math.cos(math.pi / 4)

  def hybrid_forward(self, F, x, gt_label):
    logits = F.BlockGrad(x)
    gt_logits = F.pick(logtis, gt_label, axis=1)

    one_hot = F.one_hot(gt_label, depth = self.num_classes, on_value = 1.0, off_value = 0.0)

    # compute B_avg
    B_avg = F.where(one_hot < 1, F.exp(self.s * logits), F.zeros_like(logits))
    B_avg = F.sum(B_avg) / x.shape[0]

    # compute theta_med (USING MEAN)
    cos_med = F.min(gt_logits)

    cos_rectified = F.where(cos_med < self.min_cos, self.min_cos * F.ones_like(cos_med), cos_med)
    self.s = F.log(B_avg) / cos_rectified
    self.s = F.BlockGrad(self.s)

    output = self.s * x
    return output

class CurricularModule(nn.HybridBlock):
  def __init__(self, num_classes, loss_m, loss_s, ctxid, **kwargs):
    super(CurricularModule, self).__init__(**kwargs)
    self.num_classes = num_classes
    self.loss_m, self.loss_s = loss_m, loss_s

    self.threshold = math.cos(math.pi - self.loss_m)
    self.mm = math.sin(math.pi - self.loss_m) * self.loss_m

    with self.name_scope():
      self.curricular_t = self.params.get('t', grad_req='null',
                                           shape=(1,),
                                           init=mx.init.Zero(),
                                           allow_deferred_init=True,
                                           differentiable=False)

    #self.curricular_t = mx.symbol.Variable("curricular_t_%d"%ctxid, init=mx.init.Zero())

  def hybrid_forward(self, F, x, gt_label, curricular_t):
    fc7 = x

    cos_theta = F.pick(fc7, gt_label, axis=1)
    gt_one_hot = F.one_hot(gt_label, depth = self.num_classes, 
                           on_value = 1.0, off_value = 0.0)

    # update curricular_t
    curricular_t = F.mean(cos_theta) * 0.01 + (1 - 0.01) * curricular_t

    # get T(t, cos(theta_yi))
    theta = F.arccos(cos_theta)
    T_cos = F.cos(theta + self.loss_m)

    # get N(t, cos(theta_j))
    T_cos_2d = F.expand_dims(T_cos, 1)
    N_cos = F.broadcast_add(fc7, curricular_t) * fc7

    # handle conditions where theta > pi - self.m
    final_T_cos = F.where(cos_theta > self.threshold, T_cos, cos_theta - self.mm)
    final_T_cos = F.expand_dims(final_T_cos, 1)
    final_T_cos = F.broadcast_mul(gt_one_hot, final_T_cos)

    # set T_cos and N_cos
    fc7 = F.where(F.broadcast_greater(fc7, T_cos_2d), fc7, N_cos)
    fc7 = F.where(gt_one_hot, final_T_cos, fc7)
    output = fc7 * self.loss_s
    return output
