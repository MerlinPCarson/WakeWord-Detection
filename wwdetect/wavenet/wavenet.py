import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from scipy.io.wavfile import read, write
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, \
    Input, Lambda, Activation, Multiply, Add



def wavenet_block(num_filters, filter_size, dilation_rate, layer_num):
    def f(input_):
        residual = input_
        tanh_out = Conv1D(num_filters, filter_size, name=f'Tanh_{layer_num}_Dilation_{dilation_rate}',
                                        dilation_rate=dilation_rate,
                                        padding='same',
                                        activation='tanh')(input_)
        sigmoid_out = Conv1D(num_filters, filter_size, name=f'Sigmoid_{layer_num}_Dilation_{dilation_rate}',
                                        dilation_rate=dilation_rate,
                                        padding='same',
                                        activation='sigmoid')(input_)
        merged = Multiply(name=f'Merge_{layer_num}')([tanh_out, sigmoid_out])
        skip_out = Conv1D(num_filters, 1, activation='relu', padding='same', name=f'Merged_{layer_num}')(merged)
        out = Add(name=f'Residual_{layer_num}')([skip_out, residual])
        return out, skip_out
    return f


def build_wavenet_model(timesteps, num_mels):
    input_ = Input(shape=(timesteps, num_mels))
    net = Conv1D(16, 3, activation='relu', name='Features')(input_)
    #A, B = wavenetBlock(64, 2, 2)(input_)
    #skip_connections = [B]
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
    net = Conv1D(32, 1, name='SkipOut_Conv1D_2')(net)
    net = Activation('softmax', name='SkipOut_Softmax')(net)
    #net = Flatten()(net)
    #net = Dense(input_size, activation='softmax')(net)
    model = Model(inputs=input_, outputs=net)
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model
