import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv1D, \
    Input, Activation, Multiply, Add, \
    GlobalMaxPooling1D, BatchNormalization


def weighted_L1_loss(y_true, y_pred):
    eps = (.001 / 128)
    abs_difference = tf.abs(y_true - y_pred)
    return tf.reduce_mean(abs_difference, axis=-1) * tf.reduce_mean((y_true + tf.reduce_mean(y_true) + eps))  # Note the `axis=-1`

def wavenet_block(num_filters, filter_size, dilation_rate, layer_num, args, initializer='glorot_normal'):
    
    def f(input_):

        residual = input_

        input_ = BatchNormalization(momentum=0.9)(input_)

        tanh_out = Conv1D(num_filters, filter_size, name=f'Tanh_{layer_num}_Dilation_{dilation_rate}',
                                        dilation_rate=dilation_rate,
                                        padding='causal',
                                        activation='tanh',
                                        kernel_initializer=initializer,
                                        kernel_regularizer=l2(args.l2),
                                        bias_regularizer=l2(args.l2))(input_)

        sigmoid_out = Conv1D(num_filters, filter_size, name=f'Sigmoid_{layer_num}_Dilation_{dilation_rate}',
                                        dilation_rate=dilation_rate,
                                        padding='causal',
                                        activation='sigmoid',
                                        kernel_initializer=initializer,
                                        kernel_regularizer=l2(args.l2),
                                        bias_regularizer=l2(args.l2))(input_)

        merged = Multiply(name=f'Merge_{layer_num}')([tanh_out, sigmoid_out])

        res_out = Conv1D(num_filters, 1, name=f'Residual_{layer_num}',
                         dilation_rate=dilation_rate, 
                         padding='causal', 
                         activation='relu',
                         kernel_initializer=initializer,
                         kernel_regularizer=l2(args.l2),
                         bias_regularizer=l2(args.l2))(merged)

        skip_out = Conv1D(num_filters*2, 1, name=f'Skip_{layer_num}', 
                          dilation_rate=dilation_rate, 
                          padding='causal', 
                          activation='relu', 
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(args.l2),
                          bias_regularizer=l2(args.l2))(merged)

        out = Add(name=f'ResidualOut_{layer_num}')([res_out, residual])

        return out, skip_out

    return f

def build_wavenet_model(args, initializer='glorot_normal'):
    input_ = Input(shape=(args.timesteps, args.num_features))

    net = Conv1D(16, 1, activation='relu', name='Features', 
                 padding='causal', 
                 kernel_initializer=initializer,
                 kernel_regularizer=l2(args.l2),
                 bias_regularizer=l2(args.l2))(input_)

    skip_connections = []
    layer_num = 0
    for block_num in range(6):
        for i in range(4):
            layer_num += 1
            net, skip = wavenet_block(16, 3, 2**i, layer_num, args)(net)
            skip_connections.append(skip)

    net = Add()(skip_connections)
    net = Activation('relu', name='SkipOut_ReLU')(net)
    net = Conv1D(32, 1, activation='relu', name='SkipOut_Conv1D_1', kernel_initializer=initializer, 
                 kernel_regularizer=l2(args.l2), bias_regularizer=l2(args.l2))(net)
    net = Conv1D(2, 1, name='SkipOut_Conv1D_2', kernel_initializer=initializer)(net)
    net = GlobalMaxPooling1D(name='Output')(net)
    net = Activation('softmax', name='Softmax')(net)
    model = Model(inputs=input_, outputs=net)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=args.lr),
                  metrics=['accuracy'])
    model.summary()
    return model


if __name__ == "__main__":
    import argparse
    from train_wavenet import parse_args
    print('parsing args')
    args = parse_args()
    print(args)
    model = build_wavenet_model(args)

