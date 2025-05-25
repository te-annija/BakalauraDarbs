# Uncertainty Weighing implementation adapted and modified from yaringal/multi-task-learning-example
# Original: https://github.com/yaringal/multi-task-learning-example (MIT Licensed)
# Copyright (c) 2017 

from keras.layers import Layer
from keras.initializers import Constant
import tensorflow as tf

class CustomMultiLossLayer(Layer):
    def __init__(self, loss_fns, nb_outputs=2, **kwargs):
        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        self.loss_fns = loss_fns
        super(CustomMultiLossLayer, self).__init__(**kwargs)
        
    def build(self, input_shape=None):
        self.log_vars = []
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(0.), trainable=True)]
        super(CustomMultiLossLayer, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0
        for y_true, y_pred, log_var, loss_fn in zip(ys_true, ys_pred, self.log_vars, self.loss_fns):
            log_var_val = tf.math.abs(log_var[0])
            precision = tf.exp(-log_var_val)
            loss += precision * loss_fn(y_true, y_pred) + log_var_val
        return tf.reduce_mean(loss)

    def call(self, inputs):
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss)
        return tf.concat(inputs, axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0]
