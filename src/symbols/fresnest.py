import mxnet as mx
import symbol_utils
from resnest.resnet import ResNet, Bottleneck

def get_symbol(num_classes, num_layers, **kwargs):
  version_output = kwargs.get('version_output', 'E')
  fc_type = version_output

  data = mx.sym.Variable(name='data')
  data = data-127.5
  data = data*0.0078125

  if num_layers == 50:
    units = [3, 4, 6, 3]
    stem_with, final_drop = 32, 0.0
  elif num_layers == 101:
    units = [3, 4, 23, 3]
    stem_with, final_drop = 64, 0.0
  elif num_layers == 200:
    units = [3, 24, 36, 3]
    stem_with, final_drop = 64, 0.2
  elif num_layers == 269:
    units = [3, 30, 48, 8]
    stem_with, final_drop = 64, 0.2

  net = ResNet(Bottleneck, units,  deep_stem=True, avg_down=True, stem_width=stem_with,
                      avd=True, avd_first=False, use_splat=True, dropblock_prob=0.1, final_drop=final_drop,
                      name_prefix='resnest_', face_recog=True, input_size=112, **kwargs)

  body = net(data)
  fc1 = symbol_utils.get_fc1(body, num_classes, fc_type)
  return fc1

