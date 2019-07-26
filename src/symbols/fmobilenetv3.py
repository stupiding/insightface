import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn
import mxnet.autograd as ag
import symbol_utils

class HSwish(nn.HybridBlock):
    """HSwish used in MobileNetV3."""
    def __init__(self, **kwargs):
        super(HSwish, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.clip(x + 3, 0, 6, name="relu6") / 6 * x

def ConvBlock(channels, kernel_size, strides, nlin_layer=None):
    out = nn.HybridSequential()
    padding = kernel_size // 2
    out.add(
        nn.Conv2D(channels, kernel_size, strides=strides, padding=padding, use_bias=False),
        nn.BatchNorm(scale=True)
    )
    if nlin_layer is not None:
        out.add(nlin_layer)
    return out

class SEModule(nn.HybridBlock):
    def __init__(self, channels, reduction=4, **kwargs):
        super(SEModule, self).__init__(**kwargs)
        with self.name_scope():
            self.se = nn.HybridSequential(prefix='')
            self.se.add(nn.Dense(channels // reduction, use_bias=False))
            self.se.add(nn.Activation('relu'))
            self.se.add(nn.Dense(channels, use_bias=False))
            self.se.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x):
        w = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
        w = self.se(w)
        x = F.broadcast_mul(x, w.expand_dims(axis=2).expand_dims(axis=2))
        return x

class Identity(nn.HybridBlock):
    def __init__(self, channels, **kwargs):
        super(Identity, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return x

def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)

class MobileBottleneck(nn.HybridBlock):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE', **kwargs):
        super(MobileBottleneck, self).__init__(**kwargs)
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        if nl == 'RE':
            nlin_layer = nn.Activation('relu')
        elif nl == 'HS':
            nlin_layer = HSwish()
        else:
            raise NotImplementedError

        if se:
            SELayer = SEModule
        else:
            SELayer = Identity  
        
        with self.name_scope():
            self.bottleneck = nn.HybridSequential(prefix='')
            self.bottleneck.add(
                # point wise
                nn.Conv2D(exp, 1, padding=0, use_bias=False),
                nn.BatchNorm(scale=True), nlin_layer,
                # depth wise
                nn.Conv2D(exp, 3, strides=stride, padding=1, groups=exp, use_bias=False),
                nn.BatchNorm(scale=True), SELayer(exp), nlin_layer, 
                #pw-linear
                nn.Conv2D(oup, 1, padding=0, use_bias=False),
                nn.BatchNorm(scale=True),
            )

    def hybrid_forward(self, F, x):
        out = self.bottleneck(x)
        if self.use_res_connect:
            out = F.elemwise_add(out, x)
        return out

class MobilenetV3(nn.HybridBlock):
    def __init__(self, num_classes=1000, width_mult=1.0, mode='small', **kwargs):
        super(MobilenetV3, self).__init__(**kwargs)
        input_channel = 16
        last_channel = 1280
        if mode == 'large':
            # refer to Table 1 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  False, 'RE', 1],
                [3, 64,  24,  False, 'RE', 2],
                [3, 72,  24,  False, 'RE', 1],
                [5, 72,  40,  True,  'RE', 2],
                [5, 120, 40,  True,  'RE', 1],
                [5, 120, 40,  True,  'RE', 1],
                [3, 240, 80,  False, 'HS', 2],
                [3, 200, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 480, 112, True,  'HS', 1],
                [3, 672, 112, True,  'HS', 1],
                [5, 672, 112, True,  'HS', 1],  # c = 112, paper set it to 160 by error
                [5, 672, 160, True,  'HS', 2],
                [5, 960, 160, True,  'HS', 1],
            ]
        elif mode == 'small':
            # refer to Table 2 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  True,  'RE', 2],
                [3, 72,  24,  False, 'RE', 2],
                [3, 88,  24,  False, 'RE', 1],
                [5, 96,  40,  True,  'HS', 2],  # stride = 2, paper set it to 1 by error
                [5, 240, 40,  True,  'HS', 1],
                [5, 240, 40,  True,  'HS', 1],
                [5, 120, 48,  True,  'HS', 1],
                [5, 144, 48,  True,  'HS', 1],
                [5, 288, 96,  True,  'HS', 2],
                [5, 576, 96,  True,  'HS', 1],
                [5, 576, 96,  True,  'HS', 1],
            ]
        else:
            raise NotImplementedError        

        # building first layer
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel

        with self.name_scope():
            self.features = nn.HybridSequential()
            with self.features.name_scope():
                self.features.add(ConvBlock(input_channel, 3, 1, nlin_layer=HSwish()))

                # building mobile blocks
                for k, exp, c, se, nl, s in mobile_setting:
                    output_channel = make_divisible(c * width_mult)
                    exp_channel = make_divisible(exp * width_mult)
                    self.features.add(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
                    input_channel = output_channel

                if mode == 'large':
                    last_conv = make_divisible(960 * width_mult)
                    self.features.add(ConvBlock(last_conv, 1, 1, nlin_layer=HSwish()))
                elif mode == 'small':
                    last_conv = make_divisible(576 * width_mult)
                    self.features.add(ConvBlock(last_conv, 1, 1, nlin_layer=HSwish()))
                    self.features.add(SEModule(last_conv))  # refer to paper Table2
                else:
                    raise NotImplementedError

            self.output = nn.HybridSequential()
            with self.output.name_scope():
                # building last several layers
                if mode == 'large':
                    self.output.add(nn.GlobalAvgPool2D(), HSwish(),
                                    nn.Conv2D(last_channel, 1, padding=0, use_bias=False), HSwish(),
                                    nn.Conv2D(num_classes, 1, padding=0, use_bias=False)
                                   )
                elif mode == 'small':
                    self.output.add(nn.GlobalAvgPool2D(), HSwish(),
                                    ConvBlock(last_channel, 1, 1, nlin_layer=HSwish()),
                                    ConvBlock(num_classes, 1, 1, nlin_layer=HSwish())
                                   )
                else:
                    raise NotImplementedError
                
    def hybrid_forward(self, F, x):
        x = self.features(x)
        #x = self.output(x)
        return x

def get_symbol(num_classes, **kwargs):
  version_output = kwargs.get('version_output', 'E')
  fc_type = version_output
  net = MobilenetV3(num_classes, 1, 'large')
  data = mx.sym.Variable(name='data')
  data = data-127.5
  data = data*0.0078125
  body = net(data)
  fc1 = symbol_utils.get_fc1(body, num_classes, fc_type)
  return fc1

