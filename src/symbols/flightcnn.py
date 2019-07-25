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
      body = mx.sym.LeakyReLU(data = data, act_type='prelu', name = name + '_prelu')
    elif act_type == 'maxout':
      branch1, branch2 = mx.symbol.split(data=data, axis=1, num_outputs=2, name = name + '_split')
      body = mx.symbol.broadcast_maximum(branch1, branch2, name = name + '_maxout')
    else:
      body = mx.symbol.Activation(data=data, act_type=act_type, name=name + '_' + act_type)
    return body

def light_unit(data, num_filter, kernel, name, **kwargs):

    """Return LightCNN Unit symbol for building LightCNN
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
    act_type = kwargs.get('act_type', 'maxout')
    if act_type == 'maxout':
      num_filter *= 2

    if use_bn:
      bn = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn')
      conv = Conv(data=bn, num_filter=num_filter, kernel=kernel, stride=(1,1), pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv')
    else:
      conv = Conv(data=data, num_filter=num_filter, kernel=kernel, stride=(1,1), pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv')
    act = Act(data=conv, act_type=act_type, name=name)
    return act

def get_before_pool(data, **kwargs):
  conv1 = light_unit(data, num_filter=48, kernel=(3,3), name='block1', **kwargs)
  pool1 = mx.sym.Pooling(data=conv1, kernel=(2, 2), stride=(2,2), pool_type='max', name='block1_pool')
  
  conv2a = light_unit(pool1, num_filter=48, kernel=(1,1), name='block2a', **kwargs)
  conv2b = light_unit(conv2a, num_filter=96, kernel=(3,3), name='block2b', **kwargs)
  pool2 = mx.sym.Pooling(data=conv2b, kernel=(2, 2), stride=(2,2), pool_type='max', name='block2_pool')

  conv3a = light_unit(pool2, num_filter=96, kernel=(1,1), name='block3a', **kwargs)
  conv3b = light_unit(conv3a, num_filter=192, kernel=(3,3), name='block3b', **kwargs)
  pool3 = mx.sym.Pooling(data=conv3b, kernel=(2, 2), stride=(2,2), pool_type='max', name='block3_pool')

  conv4a = light_unit(pool3, num_filter=192, kernel=(1,1), name='block4a', **kwargs)
  conv4b = light_unit(conv4a, num_filter=128, kernel=(3,3), name='block4b', **kwargs)
  #pool4 = mx.sym.Pooling(data=conv4b, kernel=(2, 2), stride=(2,2), pool_type='max', name='block4_pool')

  conv5a = light_unit(conv4b, num_filter=128, kernel=(1,1), name='block5a', **kwargs)
  conv5b = light_unit(conv5a, num_filter=128, kernel=(3,3), name='block5b', **kwargs)
  pool5 = mx.sym.Pooling(data=conv5b, kernel=(2, 2), stride=(2,2), pool_type='max', name='block5_pool')

  return pool5
  

def get_symbol(num_classes, num_layers,  **kwargs):
  filter_list = [48, 96, 192, 128, 128]
  if num_layers == 9:
    units = [1, 1, 1, 1, 1]
  #elif num_layers == 19:
  #  units = [
  data = mx.sym.Variable(name='data')
  data = data-127.5
  data = data*0.0078125
  body = get_before_pool(data)
  version_output = kwargs.get('version_output', 'E')
  fc_type = version_output
  fc1 = symbol_utils.get_fc1(body, num_classes, fc_type)
  return fc1
