"""
    ShakeDrop-ResNet for CIFAR/SVHN, implemented in Gluon.
    Original paper: 'ShakeDrop Regularization for Deep Residual Learning,' https://arxiv.org/abs/1802.02375.
"""

import os
import numpy as np
import mxnet as mx
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock


class ShakeDrop(mx.autograd.Function):
    """
    ShakeDrop function.

    Parameters:
    ----------
    p : float
        ShakeDrop specific probability (of life) for Bernoulli random variable.
    """
    def __init__(self, prob, alpha=(-1,1), beta=(0,1)):
        super(ShakeDrop, self).__init__()
        assert(isinstance(alpha, tuple))
        assert(isinstance(beta, tuple))
        self.prob = prob
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if mx.autograd.is_training():
            b = np.random.binomial(n=1, p=self.prob)
            alpha = mx.nd.random.uniform_like(x.slice(begin=(None, 0, 0, 0), end=(None, 1, 1, 1)), low=self.alpha[0], high=self.alpha[1])
            y = mx.nd.broadcast_mul(b + alpha - b * alpha, x)
            self.save_for_backward(b)
        else:
            expected_alpha = (self.alpha[0] + self.alpha[1]) / 2
            expected_prob = (self.prob + expected_alpha - self.prob * expected_alpha)
            y = expected_prob * x
        return y

    def backward(self, dy):
        b, = self.saved_tensors
        beta = mx.nd.random.uniform_like(dy.slice(begin=(None, 0, 0, 0), end=(None, 1, 1, 1)), low=self.beta[0], high=self.beta[1])
        return mx.nd.broadcast_mul(b + beta - b * beta, dy)


