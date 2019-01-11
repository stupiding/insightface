import mxnet as mx
import symbol_utils

def Conv(**kwargs):
    #name = kwargs.get('name')
    #_weight = mx.symbol.Variable(name+'_weight')
    #_bias = mx.symbol.Variable(name+'_bias', lr_mult=2.0, wd_mult=0.0)
    #body = mx.sym.Convolution(weight = _weight, bias = _bias, **kwargs)
    body = mx.sym.Convolution(**kwargs)
    return body


def Act(data, act_type, name):
    if act_type=='prelu':
      body = mx.sym.LeakyReLU(data = data, act_type='prelu', name = name + '/prelu')
    elif act_type == 'maxout':
      branch1, branch2 = mx.symbol.split(data=data, axis=1, num_outputs=2, name = name + '/split')
      #branch1, branch2 = mx.nd.NDArray(branch1), mx.nd.NDArray(branch2)
      #body = mx.ndarray.where(branch1 > branch2, branch1, branch2, name = name + '/maxout')
      body = mx.symbol.broadcast_maximum(branch1, branch2, name = name + '/maxout')
    else:
      body = mx.symbol.Activation(data=data, act_type=act_type, name=name + '/' + act_type)
    return body

def light_unit(data, num_filter, kernel, name, **kwargs):

    """Return ResNeXt Unit symbol for building ResNeXt
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    kernel : int
        Convolution kernel size
    name : str
        Base name of the operators
    """
    use_bn = kwargs.get('use_bn', True)
    bn_mom = kwargs.get('bn_mom', 0.9) 
    workspace = kwargs.get('workspace', 256)
    if use_bn:
      bn = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '/bn')
      conv = Conv(data=bn, num_filter=num_filter, kernel=kernel, stride=(1,1), pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '/conv')
    else:
      conv = Conv(data=data, num_filter=num_filter, kernel=kernel, stride=(1,1), pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '/conv')
    act = Act(data=conv, act_type='maxout', name=name)
    return act

def get_before_pool(data, **kwargs):
  conv1 = light_unit(data, num_filter=64, kernel=(3,3), name='block1', **kwargs)
  pool1 = mx.sym.Pooling(data=conv1, kernel=(2, 2), stride=(2,2), pool_type='max', name='block1/pool')
  
  conv2a = light_unit(pool1, num_filter=96, kernel=(1,1), name='block2a', **kwargs)
  conv2b = light_unit(conv2a, num_filter=192, kernel=(3,3), name='block2b', **kwargs)
  pool2 = mx.sym.Pooling(data=conv2b, kernel=(2, 2), stride=(2,2), pool_type='max', name='block2/pool')

  conv3a = light_unit(pool2, num_filter=192, kernel=(1,1), name='block3a', **kwargs)
  conv3b = light_unit(conv3a, num_filter=384, kernel=(3,3), name='block3b', **kwargs)
  pool3 = mx.sym.Pooling(data=conv3b, kernel=(2, 2), stride=(2,2), pool_type='max', name='block3/pool')

  conv4a = light_unit(pool3, num_filter=384, kernel=(1,1), name='block4a', **kwargs)
  conv4b = light_unit(conv4a, num_filter=256, kernel=(3,3), name='block4b', **kwargs)
  #pool4 = mx.sym.Pooling(data=conv4b, kernel=(2, 2), stride=(2,2), pool_type='max', name='block4/pool')

  conv5a = light_unit(conv4b, num_filter=256, kernel=(1,1), name='block5a', **kwargs)
  conv5b = light_unit(conv5a, num_filter=256, kernel=(3,3), name='block5b', **kwargs)
  pool5 = mx.sym.Pooling(data=conv5b, kernel=(2, 2), stride=(2,2), pool_type='max', name='block5/pool')

  return pool5
  

def get_symbol(num_classes, num_layers, fc_type = 'E', **kwargs):
  data = mx.sym.Variable(name='data')
  data = data-127.5
  data = data*0.0078125
  body = get_before_pool(data)
  fc1 = symbol_utils.get_fc1(body, num_classes, fc_type)
  return fc1
