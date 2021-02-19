import os
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv1D, \
    Input, Activation, Multiply, Add, \
    GlobalMaxPooling1D, BatchNormalization


class wavenet_block(tf.keras.layers.Layer):

    def __init__(self, num_filters, filter_size, dilation_rate, layer_num, args, initializer='glorot_normal'):
        super(wavenet_block, self).__init__()

        self.initializer = initializer

        self.bn = BatchNormalization(momentum=0.9)

        self.tanh_gate = Conv1D(num_filters, filter_size, name=f'Tanh_{layer_num}_Dilation_{dilation_rate}',
                                        dilation_rate=dilation_rate,
                                        padding='causal',
                                        activation='tanh',
                                        kernel_initializer=initializer,
                                        kernel_regularizer=l2(args.l2),
                                        bias_regularizer=l2(args.l2))

        self.sigmoid_gate = Conv1D(num_filters, filter_size, name=f'Sigmoid_{layer_num}_Dilation_{dilation_rate}',
                                        dilation_rate=dilation_rate,
                                        padding='causal',
                                        activation='sigmoid',
                                        kernel_initializer=initializer,
                                        kernel_regularizer=l2(args.l2),
                                        bias_regularizer=l2(args.l2))

        self.res_conv = Conv1D(num_filters, 1, name=f'Residual_{layer_num}',
                         dilation_rate=dilation_rate, 
                         padding='causal', 
                         activation='relu',
                         kernel_initializer=initializer,
                         kernel_regularizer=l2(args.l2),
                         bias_regularizer=l2(args.l2))

        self.skip_conv = Conv1D(num_filters*2, 1, name=f'Skip_{layer_num}', 
                          dilation_rate=dilation_rate, 
                          padding='causal', 
                          activation='relu', 
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(args.l2),
                          bias_regularizer=l2(args.l2))


    def call(self, inputs):

        residual = inputs

        outputs = self.bn(inputs)

        merged = Multiply()([self.tanh_gate(outputs), self.sigmoid_gate(outputs)])

        res_out = self.res_conv(merged)

        skip_out = self.skip_conv(merged)

        out = Add()([res_out, residual])

        return out, skip_out

    
class wavenet(tf.keras.Model):

    def __init__(self, args, initializer='glorot_normal'):
        super(wavenet, self).__init__()

        self.initializer = initializer
        self.timesteps = args.timesteps
        self.num_features = args.num_features
        self.l2 = args.l2

        self.input_conv = Conv1D(16, 1, activation='relu', name='Features', 
                                 padding='causal', 
                                 kernel_initializer=self.initializer,
                                 kernel_regularizer=l2(args.l2),
                                 bias_regularizer=l2(args.l2))

        self.encoder = self.build_encoder(args)
        self.detect = self.build_detect(args)

    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.detect(x)
        return x
    
    def build_encoder(self, args):

        inputs = Input(shape=(args.timesteps, args.num_features))
        outputs = self.input_conv(inputs)
        
        skip_connections = []

        layer_num = 0
        for block_num in range(6):
            for i in range(4):
                layer_num += 1
                outputs, skip = wavenet_block(16, 3, 2**i, layer_num, args)(outputs)
                skip_connections.append(skip)

        outputs = Add()(skip_connections)

        model = Model(inputs=inputs, outputs=outputs, name='Encoder')
        model.summary()

        return model

    def build_detect(self, args):

        inputs = Input(shape=(args.timesteps, 32))

        outputs = Activation('relu', name='SkipOut_ReLU')(inputs)
        outputs = Conv1D(32, 1, activation='relu', name='SkipOut_Conv1D_1', kernel_initializer=self.initializer, 
                     kernel_regularizer=l2(args.l2), bias_regularizer=l2(args.l2))(outputs)
        outputs = Conv1D(2, 1, name='SkipOut_Conv1D_2', kernel_initializer=self.initializer)(outputs)
        outputs = GlobalMaxPooling1D(name='Output')(outputs)
        outputs = Activation('softmax', name='Softmax')(outputs)
        model = Model(inputs=inputs, outputs=outputs, name='Detect')
        model.summary()

        return model

    def save(self, out_dir):

        self.encoder.save(os.path.join(out_dir, 'encode.h5'))
        self.detect.save(os.path.join(out_dir, 'detect.h5'))

    def save_to_tflite(self, out_dir):
        encode_converter = tf.lite.TFLiteConverter.from_keras_model(self.encoder)
        detect_converter = tf.lite.TFLiteConverter.from_keras_model(self.detect)
        tflite_encode_model = encode_converter.convert()
        tflite_detect_model = detect_converter.convert()

        # Save the model.
        with open(os.path.join(out_dir, 'encode.tflite'), 'wb') as f:
            f.write(tflite_encode_model)

        with open(os.path.join(out_dir, 'detect.tflite'), 'wb') as f:
            f.write(tflite_detect_model)

def build_wavenet_model(args):
    model = wavenet(args)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=args.lr),
            metrics=['accuracy'])
    model.build(input_shape=(None, args.timesteps, args.num_features))
    model.summary()

    return model

if __name__ == "__main__":
    from tensorflow.keras.models import load_model
    import argparse
    from train_wavenet import parse_args
    print('parsing args')
    args = parse_args()
    print(args)
    model = build_wavenet_model(args)
    model.load_weights(os.path.join(args.model_dir, 'wavenet_model'))
    #model.encoder.load_weights('models2/wavenet_model.data-00000-of-00002')
    #model.detect.load_weights('models2/wavenet_model.data-00001-of-00002')

