import mxnet as mx
import symbol_utils
from symbol_advanced import *

k_sec = {  2:  3, \
      3:  6, \
      4: 18, \
      5:  3  }
R  = 32
bw1  = 256
k_D  = [128, 176]

def get_before_pool(data):
  data = mx.symbol.Variable(name="data")

  #conv1_x_x  = Conv(data=data,  num_filter=64,  kernel=(7, 7), name='conv1_x_1', pad=(3,3), stride=(2,2))
  conv1_x_x  = Conv(data=data,  num_filter=64,  kernel=(3, 3), name='conv1_x_1', pad=(1,1), stride=(1,1))
  conv1_x_x  = BN_AC(conv1_x_x, name='conv1_x_1__relu-sp')
  conv1_x_x  = mx.symbol.Pooling(data=conv1_x_x, pool_type="max", kernel=(3, 3), pad=(1,1), stride=(2,2), name="pool1")

  bw = bw1
  D  = k_D[0]*(bw/bw1)
  conv2_x_x  = ResidualFactory(   conv1_x_x,   D,   D,   bw,   R,   'conv2_x__1',      None, 'proj'  )
  for i_ly in range(2, k_sec[2]+1):
    conv2_x_x  = ResidualFactory( conv2_x_x,   D,   D,   bw,   R,  ('conv2_x__%d'% i_ly),  None, 'normal')


  bw = 2*bw
  D  = k_D[1]*(bw/bw1)
  weights  = DeclearWeights(D, bw, 'conv3_x__x')
  conv3_x_x  = CRUFactory(      conv2_x_x,   D,   D,   bw,   D,   'conv3_x__1',     weights, 'down'  )
  for i_ly in range(2, k_sec[3]+1):
    conv3_x_x  = CRUFactory(    conv3_x_x,   D,   D,   bw,   D,  ('conv3_x__%d'% i_ly), weights, 'normal')


  bw = 2*bw
  D  = k_D[1]*(bw/bw1)
  weights  = DeclearWeights(D, bw, 'conv4_x__(1)')
  conv4_x_x  = CRUFactory(      conv3_x_x,   D,   D,   bw,   D,   'conv4_x__1',     weights, 'down',   True)
  for i_ly in range(2, k_sec[4]+1):
    if (i_ly%6) == 1:
      weights = DeclearWeights(D, bw, 'conv4_x__(%d)'%(int(i_ly/6)+1))
    conv4_x_x  = CRUFactory(    conv4_x_x,   D,   D,   bw,   D,  ('conv4_x__%d'% i_ly), weights, 'normal', True)


  bw = 2*bw
  D  = k_D[0]*(bw/bw1)
  conv5_x_x  = ResidualFactory(   conv4_x_x,   D,   D,   bw,   R,   'conv5_x__1',      None, 'down'  )
  for i_ly in range(2, k_sec[5]+1):
    conv5_x_x  = ResidualFactory( conv5_x_x,   D,   D,   bw,   R,  ('conv5_x__%d'% i_ly),  None, 'normal')


  conv5_x_x = BN_AC(conv5_x_x, name='conv5_x_x__relu-sp')
  return conv5_x_x
  

def get_symbol(num_classes = 1000, fc_type):
  data = mx.sym.Variable(name='data')
  data = data-127.5
  data = data*0.0078125
  body = get_before_pool(data)
  fc1 = symbol_utils.get_fc1(body, num_classes, fc_type)
  return fc1



