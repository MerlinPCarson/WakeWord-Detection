import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from scipy.io.wavfile import read, write
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, \
    Input, Lambda, Activation, Multiply, Add, GlobalAveragePooling1D, \
    BatchNormalization


def weighted_L1_loss(y_true, y_pred):
    eps = (.001 / 128)
    abs_difference = tf.abs(y_true - y_pred)
    return tf.reduce_mean(abs_difference, axis=-1) * tf.reduce_mean((y_true + tf.reduce_mean(y_true) + eps))  # Note the `axis=-1`

def wavenet_block(num_filters, filter_size, dilation_rate, layer_num):
    def f(input_):
        residual = input_
        input_ = BatchNormalization()(input_)
        tanh_out = Conv1D(num_filters, filter_size, name=f'Tanh_{layer_num}_Dilation_{dilation_rate}',
                                        dilation_rate=dilation_rate,
                                        padding='causal',
                                        activation='tanh')(input_)
        sigmoid_out = Conv1D(num_filters, filter_size, name=f'Sigmoid_{layer_num}_Dilation_{dilation_rate}',
                                        dilation_rate=dilation_rate,
                                        padding='causal',
                                        activation='sigmoid')(input_)
        merged = Multiply(name=f'Merge_{layer_num}')([tanh_out, sigmoid_out])
        res_out = Conv1D(num_filters, 1, activation='relu', dilation_rate=dilation_rate, padding='causal', name=f'Residual_{layer_num}')(merged)
        skip_out = Conv1D(num_filters*2, 1, activation='relu', dilation_rate=dilation_rate, padding='causal', name=f'Skip_{layer_num}')(merged)
        out = Add(name=f'ResidualOut_{layer_num}')([res_out, residual])
        return out, skip_out
    return f


def build_wavenet_model(num_timesteps, num_features):
    input_ = Input(shape=(num_timesteps, num_features))
    net = Conv1D(16, 1, activation='relu', name='Features', padding='causal')(input_)

    skip_connections = []
    layer_num = 0
    for block_num in range(6):
        for i in range(4):
            layer_num += 1
            net, skip = wavenet_block(16, 3, 2**i, layer_num)(net)
            skip_connections.append(skip)
    net = Add()(skip_connections)
    net = Activation('relu', name='SkipOut_ReLU')(net)
    net = Conv1D(32, 1, activation='relu', name='SkipOut_Conv1D_1')(net)
    net = Conv1D(1, 1, name='SkipOut_Conv1D_2', activation='sigmoid')(net)
    #net = Activation('sigmoid', name='SkipOut_Softmax')(net)
    net = GlobalAveragePooling1D(name='Output')(net)
    model = Model(inputs=input_, outputs=net)
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model
