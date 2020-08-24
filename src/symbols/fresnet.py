# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

'''
Adapted from https://github.com/tornadomeet/ResNet/blob/master/symbol_resnet.py
Original author Wei Wu

Implemented the following paper:

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Identity Mappings in Deep Residual Networks"
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import mxnet as mx
import numpy as np
import symbol_utils
import sklearn
from shake_drop import *
#from attention_block import BottleneckV2, AttentionBlock

def _update_input_size(input_size, stride):
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    ih, iw = (input_size, input_size) if isinstance(input_size, int) else input_size
    oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
    input_size = (oh, ow)
    return input_size

def calc_prob(curr_layer, total_layers, p_l):
  """Calculates drop prob depending on the current layer."""
  return 1 - (float(curr_layer) / total_layers) * p_l

def bn_block(data, fix_gamma, eps=2e-5, momentum=0.9, name='bn', method='bn', use_global_stats=False):
    if method == 'bn':
        out = mx.sym.BatchNorm(data=data,fix_gamma=fix_gamma, eps=eps, momentum=momentum, name=name + '/bn', use_global_stats=use_global_stats)
    elif method == 'sbn':
        out = mx.contrib.sym.SyncBatchNorm(data=data, fix_gamma=fix_gamma, eps=eps, momentum=momentum, name=name + '/bn', key = name + '/bn')
    elif method == 'in':
        out = mx.sym.InstanceNorm(data=data, eps=eps, name=name + '/in')
    elif method == 'row':
        row_data = mx.sym.transpose(data, axes=(0, 2, 1, 3))
        bn_out = mx.sym.BatchNorm(data=row_data,fix_gamma=fix_gamma, eps=eps, momentum=momentum, name=name + '/rowbn')
        out = mx.sym.transpose(bn_out, axes=(0, 2, 1, 3))
    elif method == 'col':
        row_data = mx.sym.transpose(data, axes=(0, 3, 2, 1))
        bn_out = mx.sym.BatchNorm(data=col_data,fix_gamma=fix_gamma, eps=eps, momentum=momentum, name=name + '/colbn')
        out = mx.sym.transpose(bn_out, axes=(0, 3, 2, 1))
    elif method == 'ibn':
        split = mx.symbol.split(data=data, axis=1, num_outputs=2)
        out1 = mx.symbol.InstanceNorm(data=split[0], eps=eps, name=name + '_ibn/in')
        out2 = mx.sym.BatchNorm(data=split[1],fix_gamma=False, eps=eps, momentum=bn_mom, name=name + '_ibn/bn')
        out = mx.symbol.Concat(out1, out2, dim=1, name=name + '_ibn')
    elif method == 'rbn':
        split = mx.symbol.split(data=data, axis=1, num_outputs=2)
        row_data = mx.sym.transpose(split[0], axes=(0, 2, 1, 3))
        bn_out = mx.sym.BatchNorm(data=row_data,fix_gamma=fix_gamma, eps=eps, momentum=momentum, name=name + '_rbn/rowbn')
        out1 = mx.sym.transpose(bn_out, axes=(0, 2, 1, 3))
        out2 = mx.sym.BatchNorm(data=split[1],fix_gamma=False, eps=eps, momentum=bn_mom, name=name + '_rbn/bn')
        out = mx.symbol.Concat(out1, out2, dim=1, name=name + '_rbn')
    return out

def ibn_block(data, name, eps=2e-5, bn_mom=0.9):
    split = mx.symbol.split(data=data, axis=1, num_outputs=2)
    # import pdb
    # pdb.set_trace()
    out1 = mx.symbol.InstanceNorm(data=split[0], eps=eps, name=name + '_in1')
    out2 = mx.sym.BatchNorm(data=split[1],fix_gamma=False, eps=eps, momentum=bn_mom, name=name + '_bn1')
    out = mx.symbol.Concat(out1, out2, dim=1, name=name + '_ibn1')
    return out

def Conv(**kwargs):
    #name = kwargs.get('name')
    #_weight = mx.symbol.Variable(name+'_weight')
    #_bias = mx.symbol.Variable(name+'_bias', lr_mult=2.0, wd_mult=0.0)
    #body = mx.sym.Convolution(weight = _weight, bias = _bias, **kwargs)
    body = mx.sym.Convolution(**kwargs)
    return body

def Act(data, act_type, name, num_filter=0, workspace=0):
    if act_type=='frelu' and num_filter>0:
      conv_frelu = mx.sym.Convolution(data=data, num_filter=num_filter, num_group=num_filter, 
                              kernel=(3,3), stride=(1,1), pad=(1,1),
                              name=name+"_frelu_conv", workspace=workspace)
      bn_frelu   = mx.sym.BatchNorm(data)
      body = mx.sym.maximum(conv_frelu, bn_frelu)
    elif act_type=='prelu':
      body = mx.sym.LeakyReLU(data = data, act_type='prelu', name = name)
    else:
      body = mx.symbol.Activation(data=data, act_type=act_type, name=name)
    return body

def se_block(data, num_filter, act_type, name, workspace):
    #se begin
    body = mx.sym.Pooling(data=data, global_pool=True, kernel=(7, 7), pool_type='avg', name=name+'_se_pool1')
    body = Conv(data=body, num_filter=num_filter//16, kernel=(1,1), stride=(1,1), pad=(0,0),
                              name=name+"_se_conv1", workspace=workspace)
    body = Act(data=body, act_type=act_type, name=name+'_se_relu1', num_filter=num_filter//16, workspace=workspace)
    body = Conv(data=body, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
                              name=name+"_se_conv2", workspace=workspace)
    body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name+"_se_sigmoid")
    bn3 = mx.symbol.broadcast_mul(bn3, body)
    #se end
    return bn3 

def block_head(shortcut, trunk, dim_match, num_filter, stride, name, resnet_v2, **kwargs):
    use_se = kwargs.get('use_se', False)
    memonger = kwargs.get('memonger', False)
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    act_type = kwargs.get('version_act', 'prelu')
    avg_down = kwargs.get('avg_down', False)
    shake_drop = kwargs.get('shake_drop', False)

    # sequeze excitation
    if use_se:
        trunk = se_block(trunk, num_filter, act_type, name, workspace)

    #shake_drop
    if shake_drop:
        prob = kwargs.get('prob', 1)
        shakedrop_module = ShakeDrop(prob=prob, alpha=(-1, 1), beta=(0, 1))
        trunk = shakedrop_module(trunk)
        #trunk = mx.sym.Custom(data=trunk, prob=prob, alpha=[-1, 1], beta=[0, 1], name=name+'_shakedrop', op_type='shakedrop')
    else:
        trunk = trunk

    if avg_down:
        shortcut = mx.sym.Pooling(data=shortcut, kernel=(3,3), stride=stride, pad=(1,1), pool_type='avg')
        trunk = mx.sym.Pooling(data=trunk, kernel=(3,3), stride=stride, pad=(1,1), pool_type='avg')
        stride = (1,1)

    # shorcut
    if dim_match is not True:
        if resnet_v2:
            shortcut = Conv(data=shortcut, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        else:
            conv1sc = Conv(data=shortcut, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_conv1sc')
            shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_sc')
    if memonger:
        shortcut._set_attr(mirror_stage='True')

    return shortcut + trunk

def residual_unit_v1(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    use_se = kwargs.get('version_se', 1)
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    memonger = kwargs.get('memonger', False)
    act_type = kwargs.get('version_act', 'prelu')
    version_bn = kwargs.get('version_bn', 'bn')
    #print('in unit1')
    if bottle_neck:
        conv1 = Conv(data=data, num_filter=int(num_filter*0.25), kernel=(1,1), stride=stride, pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1', num_filter=int(num_filter*0.25), workspace=workspace)
        conv2 = Conv(data=act1, num_filter=int(num_filter*0.25), kernel=(3,3), stride=(1,1), pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = Act(data=bn2, act_type=act_type, name=name + '_relu2', num_filter=int(num_filter*0.25), workspace=workspace)
        conv3 = Conv(data=act2, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        trunk = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
    else:
        conv1 = Conv(data=data, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = bn_block(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1', method=version_bn)
        act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1', num_filter=num_filter, workspace=workspace)
        conv2 = Conv(data=act1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        trunk = bn_block(data=conv2, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2', method=version_bn)

    merged = block_head(data, trunk, dim_match, num_filter, stride, name, False, **kwargs)
    return Act(data=merged, act_type=act_type, name=name + '_relu3', num_filter=num_filter, workspace=workspace)

def residual_unit_v1_L(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    act_type = kwargs.get('version_act', 'prelu')
    #print('in unit1')
    if bottle_neck:
        conv1 = Conv(data=data, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1', num_filter=int(num_filter*0.25), workspace=workspace)
        conv2 = Conv(data=act1, num_filter=int(num_filter*0.25), kernel=(3,3), stride=(1,1), pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = Act(data=bn2, act_type=act_type, name=name + '_relu2', num_filter=int(num_filter*0.25), workspace=workspace)
        conv3 = Conv(data=act2, num_filter=num_filter, kernel=(1,1), stride=stride, pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        trunk = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
    else:
        conv1 = Conv(data=data, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1', num_filter=num_filter, workspace=workspace)
        conv2 = Conv(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        trunk = mx.sym.BatchNorm(data=conv2, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')

    merged = block_head(data, trunk, dim_match, num_filter, stride, name, False, **kwargs)
    return Act(data=merged, act_type=act_type, name=name + '_relu3', num_filter=num_filter, workspace=workspace)

def residual_unit_v2(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    memonger = kwargs.get('memonger', False)
    act_type = kwargs.get('version_act', 'prelu')
    #print('in unit2')

    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
    act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1')
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        conv1 = Conv(data=act1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = Act(data=bn2, act_type=act_type, name=name + '_relu2')
        conv2 = Conv(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=stride, pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        act3 = Act(data=bn3, act_type=act_type, name=name + '_relu3')
        trunk = Conv(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
    else:
        conv1 = Conv(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        act2 = Act(data=bn2, act_type=act_type, name=name + '_relu2')
        trunk = Conv(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')

    if dim_match:
        merged = block_head(data, trunk, dim_match, num_filter, stride, name, True, **kwargs)
    else:
        merged = block_head(act1, trunk, dim_match, num_filter, stride, name, True, **kwargs)
    return merged
    #return Act(data=merged, act_type=act_type, name=name + '_relu3')

def residual_unit_v3(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
    
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    use_ibn = kwargs.get('version_ibn', 1)
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    act_type = kwargs.get('version_act', 'prelu')
    avg_down = kwargs.get('avg_down', False)
    version_bn = kwargs.get('version_bn', 'bn')
    use_global_stats = kwargs.get('use_global_stats', False)

    #print('in unit3')
    if bottle_neck:
        if use_ibn and num_filter != 2048:
            bn1 = ibn_block(data=data, name=name)
        else:
            bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1', use_global_stats=use_global_stats)

        #self.dropblock1 = DropBlock(dropblock_prob, 3, group_width, *input_size)

        conv1 = Conv(data=bn1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2', use_global_stats=use_global_stats)
        act1 = Act(data=bn2, act_type=act_type, name=name + '_relu1', num_filter=int(num_filter*0.25), workspace=workspace)
        conv2 = Conv(data=act1, num_filter=int(num_filter*0.25), kernel=(3,3), stride=(1,1), pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3', use_global_stats=use_global_stats)
        act2 = Act(data=bn3, act_type=act_type, name=name + '_relu2', num_filter=int(num_filter*0.25), workspace=workspace)
        if avg_down:
            conv3 = Conv(data=act2, num_filter=num_filter, kernel=(1,1), stride=(1,), pad=(0,0), no_bias=True,
                                       workspace=workspace, name=name + '_conv3')
        else:
            conv3 = Conv(data=act2, num_filter=num_filter, kernel=(1,1), stride=stride, pad=(0,0), no_bias=True,
                                       workspace=workspace, name=name + '_conv3')
        trunk = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn4', use_global_stats=use_global_stats)
    else:
        bn1 = bn_block(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1', method=version_bn, use_global_stats=use_global_stats)
        conv1 = Conv(data=bn1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = bn_block(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2', method=version_bn, use_global_stats=use_global_stats)
        act1 = Act(data=bn2, act_type=act_type, name=name + '_relu1', num_filter=num_filter, workspace=workspace)
        if avg_down:
            conv2 = Conv(data=act1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                          no_bias=True, workspace=workspace, name=name + '_conv2')
        else:
            conv2 = Conv(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                          no_bias=True, workspace=workspace, name=name + '_conv2')
        trunk = bn_block(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3', method=version_bn, use_global_stats=use_global_stats)

    merged = block_head(data, trunk, dim_match, num_filter, stride, name, False, **kwargs)
    return merged

def residual_unit_v3_x(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
    
    """Return ResNeXt Unit symbol for building ResNeXt
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    assert(bottle_neck)
    use_se = kwargs.get('version_se', 1)
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    memonger = kwargs.get('memonger', False)
    act_type = kwargs.get('version_act', 'prelu')
    num_group = 32
    #print('in unit3')
    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
    conv1 = Conv(data=bn1, num_group=num_group, num_filter=int(num_filter*0.5), kernel=(1,1), stride=(1,1), pad=(0,0),
                               no_bias=True, workspace=workspace, name=name + '_conv1')
    bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
    act1 = Act(data=bn2, act_type=act_type, name=name + '_relu1', num_filter=int(num_filter*0.25), workspace=workspace)
    conv2 = Conv(data=act1, num_group=num_group, num_filter=int(num_filter*0.5), kernel=(3,3), stride=(1,1), pad=(1,1),
                               no_bias=True, workspace=workspace, name=name + '_conv2')
    bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
    act2 = Act(data=bn3, act_type=act_type, name=name + '_relu2', num_filter=int(num_filter*0.25), workspace=workspace)
    conv3 = Conv(data=act2, num_filter=num_filter, kernel=(1,1), stride=stride, pad=(0,0), no_bias=True,
                               workspace=workspace, name=name + '_conv3')
    bn4 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn4')

    merged = block_head(data, bn4, dim_match, num_filter, stride, name, False, **kwargs)
    return merged


def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
  uv = kwargs.get('version_unit', 3)
  version_input = kwargs.get('version_input', 1)
  if uv==1:
    if version_input==0:
      return residual_unit_v1(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)
    else:
      return residual_unit_v1_L(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)
  elif uv==2:
    return residual_unit_v2(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)
  elif uv==4:
    return residual_unit_v4(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)
  else:
    if version_input<=1:
      return residual_unit_v3(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)
    else:
      return residual_unit_v3_x(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)

def resnet(units, num_stages, filter_list, num_classes, bottle_neck, **kwargs):
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    """Return ResNet symbol of
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stages : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_classes : int
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    """
    version_se = kwargs.get('version_se', 1)
    version_input = kwargs.get('version_input', 1)
    assert version_input>=0
    version_output = kwargs.get('version_output', 'E')
    fc_type = version_output
    version_unit = kwargs.get('version_unit', 3)
    act_type = kwargs.get('version_act', 'prelu')
    pyramid_alpha = kwargs.get('pyramid_alpha', 0)
    stride_in_res = kwargs.get('stride_in_res', True)
    use_global_stats = kwargs.get('use_global_stats', False)
    kwargs['use_global_stats'] = use_global_stats
    use_attention = kwargs.get('use_attention', False)
    use_dropblock = kwargs.get('use_dropblock', False)
    input_size = kwargs.get('input_size', 112)
    print('use_global_stats: {}'.format(use_global_stats))
    print(version_se, version_input, version_output, version_unit, act_type)

    num_unit = len(units)
    assert(num_unit == num_stages)
    data = mx.sym.Variable(name='data')
    if version_input==0:
      data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data', use_global_stats=use_global_stats)
      body = Conv(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                no_bias=True, name="conv0", workspace=workspace)
      body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0', use_global_stats=use_global_stats)
      body = Act(data=body, act_type='prelu', name='relu0')
      body = mx.sym.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    elif version_input==2:
      data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data', use_global_stats=use_global_stats)
      body = Conv(data=data, num_filter=filter_list[0], kernel=(3,3), stride=(1,1), pad=(1,1),
                                no_bias=True, name="conv0", workspace=workspace)
      body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0', use_global_stats=use_global_stats)
      body = Act(data=body, act_type='prelu', name='relu0')
    else:
      data = mx.sym.identity(data=data, name='id')
      data = (data-127.5) * 0.0078125
      body = Conv(data=data, num_filter=filter_list[0], kernel=(3,3), stride=(1,1), pad=(1, 1),
                                no_bias=True, name="conv0", workspace=workspace)
      body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0', use_global_stats=use_global_stats)
      body = Act(data=body, act_type='prelu', name='relu0')

    #input_size = _update_input_size(input_size, 1)

    layer_num, p_l = 1, 0.5
    total_layers = sum(units)
    for i in range(num_stages):
      if stride_in_res:
        print(layer_num)
        kwargs['prob'] = calc_prob(layer_num, total_layers, p_l)
        if version_input==0:
          body = residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,
                               name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, **kwargs)
        else:
          body = residual_unit(body, filter_list[i+1], (2, 2), False,
                               name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, **kwargs)
        layer_num += 1
        for j in range(1, units[i]):
          kwargs['prob'] = calc_prob(layer_num, total_layers, p_l)
          body = residual_unit(body, filter_list[i+1] + pyramid_alpha * j, (1,1), True, name='stage%d_unit%d' % (i+1, j+1),
            bottle_neck=bottle_neck, **kwargs)
          layer_num += 1
      else:
        stride = (1, 1) if (version_input == 0 and i==0) else (2, 2)
        body = Conv(body, num_filter=filter_list[i+1], kernel=(3,3), stride=stride, pad=(1,1),
                    no_bias=True, name='stage%d_unit%d' % (i + 1, 0), workspace=workspac)
        for j in range(0, units[i]):
          kwargs['prob'] = calc_prob(layer_num, total_layers, p_l)
          body = residual_unit(body, filter_list[i+1] + pyramid_alpha * j, (1,1), True, name='stage%d_unit%d' % (i+1, j+1),
            bottle_neck=bottle_neck, **kwargs)
          layer_num += 1
      if use_attention:
        attention = Conv(data=body, num_filter=filter_list[i+1]//16, kernel=(3,3), stride=(1,1), pad=(1,1),
                                    name="stage%d_attention_conv1" % (i+1), workspace=workspace)
        attention = Act(data=attention, act_type=act_type, name='stage%d_attention_relu1' % (i+1))
        attention = Conv(data=attention, num_filter=1, kernel=(3,3), stride=(1,1), pad=(1,1),
                                no_bias=True, name="stage%d_attention_conv2" % (i+1), workspace=workspace)
        attention = mx.symbol.Activation(data=attention, act_type='sigmoid', name='stage%d_attention_sigmoid' % (i+1)) + 1
        body = mx.sym.broadcast_mul(body, attention)
    assert layer_num - 1 == total_layers, 'layer_num = %d, total_layers = %d' % (layer_num, total_layers)

    fc1 = symbol_utils.get_fc1(body, num_classes, fc_type, use_global_stats=use_global_stats)
    return fc1

def get_symbol(num_classes, num_layers, **kwargs):
    """
    Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
    Original author Wei Wu
    """
    if num_layers >= 101:
        filter_list = [64, 256, 512, 1024, 2048]
        bottle_neck = True
    else:
        filter_list = [64, 64, 128, 256, 512]
        bottle_neck = False
    num_stages = 4
    if num_layers == 18:
        units = [2, 2, 2, 2]
    elif num_layers == 20:
        units = [1, 2, 4, 1]
    elif num_layers == 34:
        units = [3, 4, 6, 3]
    elif num_layers == 36:
        units = [2, 4, 8, 2]
    elif num_layers == 49:
        units = [3, 4, 14, 3]
    elif num_layers == 50:
        units = [3, 4, 14, 3]
    elif num_layers == 64:
        units = [3, 8, 16, 3]
    elif num_layers == 74:
        units = [3, 6, 24, 3]
    elif num_layers == 90:
        units = [3, 8, 30, 3]
    elif num_layers == 100:
        units = [3, 13, 30, 3]
    elif num_layers == 101:
        units = [3, 4, 23, 3]
    elif num_layers == 152:
        units = [3, 8, 36, 3]
    elif num_layers == 200:
        units = [3, 24, 36, 3]
    elif num_layers == 269:
        units = [3, 30, 48, 8]
    else:
        raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))

    if num_layers in [20, 36, 64]:
        kwargs['stride_in_res'] = False
    kwargs['total_layers'] = sum(units)
    width_mult = kwargs.get('width_mult', 1)
    filter_list = [int(c * width_mult) for c in filter_list]
    shake_drop = kwargs.get('shake_drop', False)
    print('use shake_drop: ', shake_drop)

    pyramid_alpha = kwargs.get('pyramid_alpha', 0)
    if pyramid_alpha > 0:
        first_stage = 64
        pyramid_alpha = int(np.ceil(pyramid_alpha // sum(units) / 32.) * 32)
        kwags['pyramid_alpha'] = pyramid_alpha
        filter_list = [first_stage] + [first_stage + sum(units[:stage] + 1) * pyramid_alpha for stage in range(4)]

    return resnet(units       = units,
                  num_stages  = num_stages,
                  filter_list = filter_list,
                  num_classes = num_classes,
                  bottle_neck = bottle_neck,
                  **kwargs)

