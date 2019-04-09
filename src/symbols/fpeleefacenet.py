
import mxnet as mx
import symbol_utils

eps = 2e-5
bn_mom = 0.9
#bn_mom = 0.9997

def Act(data, act_type, name):
    #ignore param act_type, set it in this function 
    body = mx.sym.LeakyReLU(data = data, act_type='prelu', name = name)
    #body = mx.sym.Activation(data=data, act_type='relu', name=name)
    return body

def ConvBlock(data, num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), act_type='relu', name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel,
               stride=stride, pad=pad, no_bias=True, name='%s_%s_conv' % (name, suffix))
    bn = mx.sym.BatchNorm(data=conv, fix_gamma=False, momentum=bn_mom, eps=eps, name='%s_%s_batchnorm' % (name, suffix))
    if act_type is not None:
        act = Act(data=bn, act_type=act_type, name='%s%s_relu' % (name, suffix))
    else:
        act = bn
    return act

def DenseBlock(data, num_layers, growth_rate, name, bottleneck_width=4):
    growth_rate = int(growth_rate/2)
    for i in range(num_layers):
        base_name = '{}_{}'.format(name, i+1)
        inter_channels = int(growth_rate * bottleneck_width / 4) * 4
        branch_1a = ConvBlock(data, num_filter=inter_channels, kernel=(1,1), 
                                  stride=(1,1), pad=(0,0), name=base_name, suffix='branch1a')
        branch_1b = ConvBlock(branch_1a, num_filter=growth_rate, kernel=(3,3),
                                  stride=(1,1), pad=(1,1), name=base_name, suffix='branch1b')

        branch_2a = ConvBlock(data, num_filter=inter_channels, kernel=(1,1), 
                                  stride=(1,1), pad=(0,0), name=base_name, suffix='branch2a')
        branch_2b = ConvBlock(branch_2a, num_filter=growth_rate, kernel=(3,3),
                                  stride=(1,1), pad=(1,1), name=base_name, suffix='branch2b')
        branch_2c = ConvBlock(branch_2b, num_filter=growth_rate, kernel=(3,3),
                                  stride=(1,1), pad=(1,1), name=base_name, suffix='branch2c')

        data = mx.symbol.Concat(*[data, branch_1b, branch_2c], dim=1, name=base_name + '_concat')
    return data

def TransitionBlock(data, num_filter, name, with_pooling=True):
    conv = ConvBlock(data, num_filter=num_filter, kernel=(1,1), 
                         stride=(1,1), pad=(0,0), name=name)
    if with_pooling:
        out = mx.sym.Pooling(data=conv, kernel=(2,2), stride=(2,2),
                                 pad=(0,0), pool_type='avg', name=name+'_pool')
    else:
        out = conv
    return out

def StemBlock(data, num_init_features):
    stem1 = ConvBlock(data, num_filter=num_init_features, kernel=(3,3),
                          stride=(1,1), pad=(1,1), name='stem1')
    stem2a = ConvBlock(stem1, num_filter=int(num_init_features/2), kernel=(1,1), 
                          stride=(1,1), pad=(0,0), name='stem2a')
    stem2b = ConvBlock(stem2a, num_filter=num_init_features, kernel=(3,3), 
                          stride=(2,2), pad=(1,1), name='stem2b')
    stem1_pool = mx.sym.Pooling(data=stem1, kernel=(2,2), stride=(2,2), 
                                    pad=(0,0), pool_type='max', name='stem1_pool')

    concat = mx.sym.Concat(*[stem1_pool, stem2b], dim=1, name='stem_concat')
    
    stem3 = ConvBlock(data=concat, num_filter=num_init_features, kernel=(1,1),
                          stride=(1,1), pad=(0,0), name='stem3')
    return stem3

def get_symbol(num_classes, num_layers, **kwargs):
    global bn_mom
    bn_mom = kwargs.get('bn_mom', 0.9)
    wd_mult = kwargs.get('wd_mult', 1.)
    version_output = kwargs.get('version_output', 'GNAP')
    #assert version_output=='GDC' or version_output=='GNAP'
    fc_type = version_output

    growth_rate = kwargs.get('growth_rate', 32)
    block_config = [3, 4, 8, 6]
    bottleneck_width = [1, 2, 4, 4]
    num_init_features = 32
    init_kernel_size = 3
    use_stem_block = True
    
    data = mx.symbol.Variable(name="data")
    data = data-127.5
    data = data*0.0078125

    if use_stem_block:
        net = StemBlock(data, num_init_features)
    else:
        padding_size = init_kernel_size / 2
        net = ConvBlock(data, num_filter=num_init_features, kernel=(init_kernel_size, init_kernel_size),
                            stride=(2, 2), pad=(padding_size, padding_size), name='conv1')
        net = mx.sym.Pooling(net, kernel=(2,2), stride=(2,2), pad=(0,0), pool_type='max', name='pool1')

    total_filter = num_init_features
    if type(bottleneck_width) is list:
        bottleneck_widths = bottleneck_width
    else:
        bottleneck_widths = [bottleneck_width] * 4

    for idx, num_layers in enumerate(block_config):
        net = DenseBlock(net, num_layers, growth_rate, bottleneck_width=bottleneck_widths[idx], name='stage{}'.format(idx+1))
        total_filter += growth_rate * num_layers

        if idx == len(block_config) - 1:
            with_pooling = False
        else:
            with_pooling = True
        net = TransitionBlock(net, total_filter, with_pooling=with_pooling, name='stage{}_tb'.format(idx+1))

    fc1 = symbol_utils.get_fc1(net, num_classes, fc_type)

    return fc1

