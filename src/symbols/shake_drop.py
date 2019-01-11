import mxnet as mx

class ShakeDrop(mx.operator.CustomOp):
    def __init__(self, ctx, prob, alpha, beta):
        self.prob = prob
        self.alpha = alpha
        self.beta = beta
        self.ctx = mx.Context(ctx)
        
    def forward(self, is_train, req, in_data, out_data, aux):
        if is_train:
            x = in_data[0]
            drop_shape = [x.shape[0], 1, 1, 1]
            shake_shape = [x.shape[0], 1, 1, 1]

            random_tensor = self.prob + mx.ndarray.random.uniform(low=0, high=1,shape=drop_shape, dtype='float32', ctx = self.ctx)
            self.binary_tensor = mx.ndarray.floor(random_tensor)
 
            alpha_values = mx.ndarray.random.uniform(
                low=self.alpha[0], high=self.alpha[1],
                shape=shake_shape, dtype='float32', ctx=self.ctx)

            rand_forward = self.binary_tensor + alpha_values - self.binary_tensor * alpha_values
            self.assign(out_data[0], req[0], x * rand_forward)
        else:
            x = in_data[0]
            expected_alpha = (self.alpha[0] + self.alpha[1]) / 2
            expected_prob = (self.prob + expected_alpha - self.prob * expected_alpha)
            self.assign(out_data[0], req[0], x * expected_prob)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        beta_values = mx.ndarray.random.uniform(
            low=self.beta[0], high=self.beta[1], 
            shape=shake_shape, dtype='float32', ctx=self.ctx)
        rand_backward = self.binary_tensor + beta_values - self.binary_tensor * beta_values
        self.assign(in_grad[0], req[0], out_grad[0] * rand_backward)

@mx.operator.register("shakedrop")
class ShakeDropProp(mx.operator.CustomOpProp):
    def __init__(self, prob="0.5", alpha="[-1, 1]", beta="[0, 1]"):
        super(ShakeDropProp, self).__init__(need_top_grad=True)
        self.prob, self.alpha, self.beta = eval(prob), eval(alpha), eval(beta)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        output_shape = in_shape[0]
        return [data_shape], [output_shape], []

    def infer_type(self, in_type):
        return in_type, [in_type[0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return ShakeDrop(ctx=ctx, prob=self.prob, alpha=self.alpha, beta=self.beta)

if __name__ == '__main__':
    import mxnet.symbol as ms
    x = ms.Variable('x', shape=(4,4,1,1))
    y = ms.Custom(data=x, prob=0.4, alpha=[-1, 1], beta=[0, 1], name='shakedrop', op_type='shakedrop')
    #y = ms.Custom(data=x, name='shakedrop', op_type='shakedrop')
    print(y.list_arguments())
    
    ex = y.bind(ctx=mx.gpu(0), args={'x' : mx.nd.ones([4,4,1,1], ctx=mx.gpu(0))})
    
    ex.forward(is_train=True)
    ex.backward(is_train=True)
    print('number of outputs = %d\nthe first output = \n%s' % (
               len(ex.outputs), ex.outputs[0].asnumpy()))
    
