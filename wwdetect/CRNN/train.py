'''
Module for training CRNN model on wakeword.
'''

import os
import sys
import argparse

import tensorflow as tf
from tensorflow.random import set_seed
from tensorflow.keras import optimizers, metrics, losses, backend
from tensorflow.keras.callbacks import EarlyStopping, \
                                       ReduceLROnPlateau, \
                                       ModelCheckpoint
from kerastuner.tuners import Hyperband

from model import Arik_CRNN, Arik_CRNN_CTC
from dataloader import HeySnipsPreprocessed

set_seed(42)

# Parameters as defined in Arik et al (set to best performing in paper).
# Note: preprocessed files using filter_dataset.py use frame widths of 20ms,
#       with a 10ms stride (50% overlap). Given the overlap, 151 frames should
#       still correspond to 1.5 seconds, so using the same dimensions as Arik
#       et al.
#
#       TODO: 149 frames is actually 1.5 seconds, not 151. Could fix, not
#             a huge deal.

# Convolutional layer parameters:
N_C = 32                # Number of filters in convolutional layer.
L_T = 20                # Filter size (time dimension).
L_F = 5                 # Filter size (frequency dimension).
S_T = 8                 # Stride (time dimension).
S_F = 2                 # Stride (frequency dimension).

# Recurrent layer(s) parameters:
R = 2                   # Number of recurrent layers.
N_R = 32                 # Number of hidden units.
RNN_TYPE = "gru"        # Use GRU or LSTM for recurrent units.

# Dense layer parameters:
N_F = 64                # Number of hidden units.

# Training hyperparameters:
OPTIMIZER = optimizers.Adam()
BATCH_SIZE = 64
INITIAL_LR = 0.001
DROPPED_LR = 0.0003

# Input details:
INPUT_SHAPE_FEATURES = 40
INPUT_SHAPE_FRAMES = 151

# Metrics used for non-CTC model.
METRICS = [
      metrics.TruePositives(name='tp'),
      metrics.FalsePositives(name='fp'),
      metrics.TrueNegatives(name='tn'),
      metrics.FalseNegatives(name='fn'),
      "accuracy",
      metrics.Precision(name='precision'),
      metrics.Recall(name='recall')
]


def data_prep(data_path, ctc=False, use_enhanced=False):
    '''
    Prep data for training.

    :param data_path: Path to preprocessed .h5 files containing data.
    :param ctc: Using CTC loss or not.
    :return: Training generator and development generator.
    '''
    dev_generator = HeySnipsPreprocessed([os.path.join(data_path, "dev.h5")],
                                         batch_size=BATCH_SIZE,
                                         frame_num=INPUT_SHAPE_FRAMES,
                                         feature_num=INPUT_SHAPE_FEATURES,
                                         ctc=ctc)
    if use_enhanced:
        train_paths = [os.path.join(data_path, "train.h5"),
                       os.path.join(data_path, "train_enhanced.h5")]
    else:
        train_paths = [os.path.join(data_path, "train.h5")]

    training_generator = HeySnipsPreprocessed(train_paths,
                                              batch_size=BATCH_SIZE,
                                              frame_num=INPUT_SHAPE_FRAMES,
                                              feature_num=INPUT_SHAPE_FEATURES,
                                              ctc=ctc)

    return training_generator, dev_generator


def train_hypermodel(training_generator, dev_generator, early_stopping=False):
    '''
    Use TF HyperModel to identify best hyperparameters for model,
    then train using winning set of HP's. Currently only set up for
    sigmoid/binary-crossentropy.

    TODO: Set up to use CTC-loss based model.

    :param training_generator: Generator which outputs preprocessed training dataset.
    :param dev_generator: Generator which outputs preprocessed development dataset.
    :param early_stopping: Use early stopping or not.
    :return: Final trained model object.
    '''

    callbacks = [ReduceLROnPlateau(monitor="val_loss", patience=3,
                                   factor=0.03, min_lr=DROPPED_LR),
                 ModelCheckpoint(filepath="best", save_best_only=True,
                                 monitor="val_loss", mode="min")]

    if early_stopping:
        callbacks = [EarlyStopping(monitor="val_loss", patience=6)] + callbacks

    def build_model(hp):
        model = Arik_CRNN(input_features=INPUT_SHAPE_FEATURES,
                          input_frames=INPUT_SHAPE_FRAMES,
                          n_c=N_C,
                          l_t=L_T,
                          l_f=L_F,
                          s_t=S_T,
                          s_f=S_F,
                          r=R,
                          n_r=hp.Int("rnn_units", 16, 64, step=16),
                          n_f=N_F,
                          dropout=0.0,
                          activation=hp.Choice("activation", values=["relu"]),
                          rnn_type=hp.Choice("rnn_type", values=["gru","lstm"]))  
        model.compile(optimizer=OPTIMIZER,
                      loss=losses.CategoricalCrossentropy(),
                      metrics=["acc"])
        model.build(input_shape=(BATCH_SIZE, INPUT_SHAPE_FEATURES, INPUT_SHAPE_FRAMES, 1))
        return model

    tuner = Hyperband(
        build_model,
        max_epochs=10,
        factor=2,
        objective="val_loss",
        directory='CRNN_tuning',
        project_name='hyperparam_search')

    tuner.search(x=training_generator,
                 epochs=10,
                 verbose=1,
                 validation_data=dev_generator,
                 use_multiprocessing=False)

    best_params = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(best_params)

    model = tuner.hypermodel.build(best_params)
    model.fit(x=training_generator,
              epochs=100,
              verbose=1,
              validation_data=dev_generator,
              use_multiprocessing=False)
    model.save()
    model.save_to_tflite()


def train_basic(training_generator, dev_generator, early_stopping=False, ctc=False,
                positive_percentage=1.0):
    '''
    Train a CRNN model without hyperparameter search. Uses HP
    settings from global variables at top of file. Outputs .h5 and
    .tflite model files.

    :param training_generator: Generator which outputs preprocessed training dataset.
    :param dev_generator: Generator which outputs preprocessed development dataset.
    :param early_stopping: Use early stopping or not.
    :param ctc: Use CTC-based model training or not.
    :return: Trained model object.
    '''

    callbacks = [ReduceLROnPlateau(monitor="val_loss", patience=3,
                                     factor=0.03, min_lr=DROPPED_LR),
                   ModelCheckpoint(filepath="best", save_best_only=True,
                                   monitor="val_loss", mode="min")]

    if early_stopping:
      callbacks = [EarlyStopping(monitor="val_loss", patience=6)] + callbacks


    # If using CTC loss.
    if ctc:
        def ctc_loss(y_true, y_pred):
            input_length = tf.fill((BATCH_SIZE,1),19)
            label_length = tf.fill((BATCH_SIZE,1),3)
            return backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

        model = Arik_CRNN_CTC(INPUT_SHAPE_FEATURES, INPUT_SHAPE_FRAMES,
                          N_C, L_T, L_F, S_T, S_F, R, N_R, N_F,
                          activation='relu', positive_percentage=positive_percentage)
        model.compile(optimizer=OPTIMIZER, loss=ctc_loss)
    else:
        model = Arik_CRNN(INPUT_SHAPE_FEATURES, INPUT_SHAPE_FRAMES,
                          N_C, L_T, L_F, S_T, S_F, R, N_R, N_F,
                          activation='relu', positive_percentage=positive_percentage)
        model.compile(optimizer=OPTIMIZER, loss=losses.CategoricalCrossentropy(), metrics="accuracy")

    model.build(input_shape=(BATCH_SIZE, INPUT_SHAPE_FEATURES, INPUT_SHAPE_FRAMES, 1))

    model.fit(x=training_generator,
              epochs=100,
              verbose=1,
              validation_data=dev_generator,
              use_multiprocessing=False,
              callbacks = [callbacks])

    model.load_weights("best")
    model.save()
    model.save_to_tflite()
    return model


def parse_args():
    '''
    Parse commandline arguments.

    :return: Arguments dict.
    '''
    parser = argparse.ArgumentParser(description='Trains CRNN, outputs model files.')
    parser.add_argument('--data_dir', type=str, default='/data_enhanced_silero', help='Directory where training data is stored.')
    parser.add_argument('--hyperparameter_search', type=bool, default=False)
    parser.add_argument('--early_stopping', type=bool, default=True)
    parser.add_argument('--use_augmented_train', type=bool, default=True)
    parser.add_argument('--ctc', type=bool, default=False)
    args = parser.parse_args()
    return args


def main(args):
    '''
    Main training function.

    :param args: Arguments dict.
    :return: 0
    '''
    train, dev = data_prep(args.data_dir, args.ctc, use_enhanced=args.use_augmented_train)
    if args.hyperparameter_search:
        model = train_hypermodel(train, dev, early_stopping=True)
    else:
        model = train_basic(train, dev, early_stopping=args.early_stopping, ctc=args.ctc)
    return 0


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
