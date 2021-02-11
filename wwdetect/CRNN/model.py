import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers, losses, metrics


class Arik_CRNN(tf.keras.Model):
    def __init__(self, input_features, input_frames,
                       n_c, l_t, l_f, s_t, s_f, r, n_r,
                       n_f, rnn_type="gru", dropout=0.0,
                       activation='relu'):
        super(Arik_CRNN, self).__init__()
        # Input shape is (features, timestep/frames).
        # Stride and kernel shape transposed from paper to match input shape.
        #   kernel: (feature_dim, time_dim)
        #   stride: (feature_dim, time_dim)
        self.pathname = "input_" + str(input_features) + "_" + str(input_frames) + "_" \
                        + "conv_filt_" + str(n_c) + "_filt_size_" + str(l_f) + "_" + str(l_t) \
                        + "_stride_" + str(s_f) + "_" + str(s_t) + "_rnn_layers_" + str(r) \
                        + "_rnn_units_" + str(n_r) + "_rnn_type_" + rnn_type + "_dropout_" \
                        + str(dropout) + "_activation_" + activation
        self.encoder = models.Sequential(name="encoder")
        self.encoder.add(layers.Conv2D(filters=n_c,
                                       kernel_size=(l_f, l_t),
                                       strides=(s_f, s_t),
                                       activation=activation,
                                       input_shape=(input_features, input_frames, 1),
                                       data_format="channels_last",
                                       padding="same"))
        # "Then a sequence of feature vectors is extracted from the feature maps produced by the
        # component of convolutional layers, which is the input for the recurrent layers. Specifically,
        # each feature vector of a feature sequence is generated from left to right on the feature maps
        # by column. This means the i-th feature vector is the concatenation of the i-th columns of all the
        # maps."
        #           -- Shi et al. 2015 CRNN paper.
        # Reshaping process assisted by:
        # https://github.com/ZainNasrullah/music-artist-classification-crnn/blob/master/src/models.py
        self.encoder.add(layers.Permute((2, 1, 3)))
        resize_shape = self.encoder.output_shape[2] * self.encoder.output_shape[3]
        self.encoder.add(layers.Reshape((self.encoder.output_shape[1], resize_shape)))

        # Intermediate layers, return sequences.
        # Input shape: [batch, timesteps, feature]
        for r in range(r - 1):
            if rnn_type == 'gru':
                self.encoder.add(layers.Bidirectional(layers.GRU(units=n_r, activation='tanh', return_sequences=True)))
            elif rnn_type == 'lstm':
                self.encoder.add(layers.Bidirectional(layers.LSTM(units=n_r, activation='tanh', return_sequences=True)))
        # Final layer, do not return sequences.
        if rnn_type == 'gru':
            self.encoder.add(layers.Bidirectional(layers.GRU(units=n_r, activation='tanh', return_sequences=False)))
        elif rnn_type == 'lstm':
            self.encoder.add(layers.Bidirectional(layers.LSTM(units=n_r, activation='tanh', return_sequences=False)))
        self.detect = models.Sequential(name="detector")
        self.detect.add(layers.Dense(units=n_f, activation='relu', input_shape=self.encoder.output_shape[1:]))
        self.detect.add(layers.Dropout(dropout))
        self.detect.add(layers.Dense(units=2, activation='softmax'))
        print("pause")

    def call(self, inputs, training=False):
        x = self.encoder(inputs)
        x = self.detect(x)
        return x

    def save(self):
        self.encoder.save("models/" + self.pathname + "_encode.h5")
        self.detect.save("models/" + self.pathname +"_detect.h5")

    def save_to_tflite(self):
        encode_converter = tf.lite.TFLiteConverter.from_keras_model(self.encoder)
        detect_converter = tf.lite.TFLiteConverter.from_keras_model(self.detect)
        tflite_encode_model = encode_converter.convert()
        tflite_detect_model = detect_converter.convert()

        # Save the model.
        with open('models/' + self.pathname + 'encode.tflite', 'wb') as f:
            f.write(tflite_encode_model)

        with open('models/' + self.pathname + 'detect.tflite', 'wb') as f:
            f.write(tflite_detect_model)
