#!/usr/bin/env python3
"""
Gradients for inner product.
"""

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
inner_product_grad_module = tf.load_op_library('build/libinner_product_grad.so')

@ops.RegisterGradient("InnerProduct")
def _inner_product_grad_cc(op, grad):
    """
    The gradient for `inner_product` using the operation implemented in C++.

    :param op: `inner_product` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
    :param grad: gradient with respect to the output of the `inner_product` op.
    :return: gradients with respect to the input of `inner_product`.
    """

    return inner_product_grad_module.inner_product_grad(grad, op.inputs[0], op.inputs[1])