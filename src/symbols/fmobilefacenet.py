
import mxnet as mx
import symbol_utils

bn_mom = 0.9
#bn_mom = 0.9997

def Act(data, act_type, name):
    #ignore param act_type, set it in this function 
    body = mx.sym.LeakyReLU(data = data, act_type='prelu', name = name)
    #body = mx.sym.Activation(data=data, act_type='relu', name=name)
    return body

def Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix='', eps=1e-3, use_global_stats=False):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=False, eps=eps, momentum=bn_mom, use_global_stats=use_global_stats)
    act = Act(data=bn, act_type='relu', name='%s%s_relu' %(name, suffix))
    return act
    
def Linear(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix='', eps=1e-3, use_global_stats=False):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=False,eps=eps, momentum=bn_mom, use_global_stats=use_global_stats)    
    return bn

def ConvOnly(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    return conv    

    
def DResidual_v1(data, num_out=1, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=1, name=None, suffix='', use_global_stats=False):
    conv = Conv(data=data, num_filter=num_group, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name='%s%s_conv_sep' %(name, suffix))
    conv_dw = Conv(data=conv, num_filter=num_group, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name='%s%s_conv_dw' %(name, suffix))
    proj = Linear(data=conv_dw, num_filter=num_out, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name='%s%s_conv_proj' %(name, suffix), use_global_stats=use_global_stats)
    return proj
    
def DResidual_v3(data, num_out=1, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=1, name=None, suffix='', eps=1e-3, use_global_stats=False):
    bn = mx.sym.BatchNorm(data=data, name='%s%s_batchnorm' %(name, suffix), fix_gamma=False, peps=1e-3, momentum=bn_mom, use_global_stats=use_global_stats)
    conv = Conv(data=bn, num_filter=num_group, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name='%s%s_conv_sep' %(name, suffix), use_global_stats=use_global_stats)
    conv_dw = Conv(data=conv, num_filter=num_group, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name='%s%s_conv_dw' %(name, suffix), use_global_stats=use_global_stats)
    proj = Linear(data=conv_dw, num_filter=num_out, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name='%s%s_conv_proj' %(name, suffix), use_global_stats=use_global_stats)
    return proj
    
def Residual(data, num_block=1, num_out=1, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=1, name=None, suffix='', use_global_stats=False):
    identity=data
    for i in range(num_block):
    	shortcut=identity
    	conv=DResidual_v1(data=identity, num_out=num_out, kernel=kernel, stride=stride, pad=pad, num_group=num_group, name='%s%s_block' %(name, suffix), suffix='%d'%i, use_global_stats=use_global_stats)
    	identity=conv+shortcut
    return identity
        

def get_symbol(num_classes, num_layers, **kwargs):
    global bn_mom
    bn_mom = kwargs.get('bn_mom', 0.9)
    wd_mult = kwargs.get('wd_mult', 1.)
    version_output = kwargs.get('version_output', 'GNAP')
    use_global_stats = kwargs.get('use_global_stats', False)
    assert version_output=='GDC' or version_output=='GNAP'
    fc_type = version_output
    data = mx.symbol.Variable(name="data")
    data = data-127.5
    data = data*0.0078125

    filter_list = [64, 64, 128, 256, 512]
    if num_layers == 49:
        units = [0, 4, 6, 2]
    elif num_layers == 72:
        units = [2, 6, 10, 2]
    elif num_layers == 108:
        units = [4, 8, 16, 4]
    width_mult = 0.5
  
    """
    conv_1 = Conv(data, num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_1")
    conv_2_dw = Conv(conv_1, num_group=64, num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_2_dw")
    conv_23 = DResidul_v1(conv_2_dw, num_out=64, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=128, name="dconv_23")
    conv_3 = Residual(conv_23, num_block=4, num_out=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=128, name="res_3")
    conv_34 = DResidul_v1(conv_3, num_out=128, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=256, name="dconv_34")
    conv_4 = Residual(conv_34, num_block=6, num_out=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=256, name="res_4")
    conv_45 = DResidul_v1(conv_4, num_out=128, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=512, name="dconv_45")
    conv_5 = Residual(conv_45, num_block=2, num_out=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=256, name="res_5")
    conv_6_sep = Conv(conv_5, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_6sep")
    """
    conv_1 = Conv(data, num_filter=filter_list[0], kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_1", use_global_stats=use_global_stats)
    if units[0] == 0:
        conv_2 = Conv(conv_1, num_group=filter_list[1], num_filter=filter_list[1], kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_2_dw", use_global_stats=use_global_stats)
    else:
        conv_2 = Residual(conv_1, num_block=units[0], num_out=filter_list[1], kernel=(3, 3), stride=(1, 1), pad=(1, 1), 
                              num_group=filter_list[1], name="res_2", use_global_stats=use_global_stats)    

    conv_23 = DResidual_v1(conv_2, num_out=filter_list[2], kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=filter_list[2], name="dconv_23", use_global_stats=use_global_stats)
    conv_3 = Residual(conv_23, num_block=units[1], num_out=filter_list[2], kernel=(3, 3), stride=(1, 1), pad=(1, 1), 
                          num_group=filter_list[2], name="res_3", use_global_stats=use_global_stats)

    conv_34 = DResidual_v1(conv_3, num_out=filter_list[3], kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=filter_list[3], name="dconv_34", use_global_stats=use_global_stats)
    conv_4 = Residual(conv_34, num_block=units[2], num_out=filter_list[3], kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                          num_group=filter_list[3], name="res_4", use_global_stats=use_global_stats)

    conv_45 = DResidual_v1(conv_4, num_out=filter_list[4], kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=filter_list[4], name="dconv_45", use_global_stats=use_global_stats)
    conv_5 = Residual(conv_45, num_block=units[3], num_out=filter_list[4], kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                          num_group=filter_list[4], name="res_5", use_global_stats=use_global_stats)
    conv_6_sep = Conv(conv_5, num_filter=filter_list[4], kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_6sep", use_global_stats=use_global_stats)

    fc1 = symbol_utils.get_fc1(conv_6_sep, num_classes, fc_type, use_global_stats=use_global_stats)
    return fc1

