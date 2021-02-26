import sys
import glob
import json
import argparse
from itertools import chain

from tensorflow.random import set_seed
from tensorflow.keras import optimizers, metrics, losses
from tensorflow.keras.callbacks import EarlyStopping, \
                                       ReduceLROnPlateau, \
                                       ModelCheckpoint
from kerastuner.tuners import Hyperband
from sklearn.metrics import balanced_accuracy_score
import numpy as np

from model import Arik_CRNN
from dataloader import HeySnipsPreprocessed

set_seed(42)

# Parameters as defined in Arik et al (set to best performing in paper).
# Note: preprocessed files using filter_dataset.py use frame widths of 20ms,
#       with a 10ms stride (50% overlap). Given the overlap, 151 frames should
#       still correspond to 1.5 seconds, so using the same dimensions as Arik
#       et al.

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

METRICS = [
      metrics.TruePositives(name='tp'),
      metrics.FalsePositives(name='fp'),
      metrics.TrueNegatives(name='tn'),
      metrics.FalseNegatives(name='fn'),
      "accuracy",
      metrics.Precision(name='precision'),
      metrics.Recall(name='recall')
]


def eval_basics(model, test_generator):
    X, y = test_generator[0]

    metrics = model.evaluate(X,y)
    predictions = model.predict(X, verbose=0)
    class_predictions = np.where(predictions < 0.5, 0, 1)
    bal_acc = balanced_accuracy_score(y, class_predictions)

    for label, value in zip(model.metrics_names, metrics):
        print(f"{label}: {value}")
    print(f"balanced accuracy: {bal_acc}")


def data_prep(data_path):

    dev_generator = HeySnipsPreprocessed([data_path + "dev.h5"], batch_size=BATCH_SIZE,
                                              frame_num=INPUT_SHAPE_FRAMES, feature_num=INPUT_SHAPE_FEATURES)
    test_generator = HeySnipsPreprocessed([data_path + "test.h5"], batch_size=0,
                                              frame_num=INPUT_SHAPE_FRAMES, feature_num=INPUT_SHAPE_FEATURES)
    training_generator = HeySnipsPreprocessed([data_path + "train.h5", data_path + "train_enhanced.h5"],
                                              batch_size=BATCH_SIZE, frame_num=INPUT_SHAPE_FRAMES,
                                                                     feature_num=INPUT_SHAPE_FEATURES)

    return training_generator, dev_generator, test_generator


def training_hypermodel(training_generator, dev_generator, early_stopping=False):
    if early_stopping:
      callbacks = EarlyStopping(monitor="val_loss", patience=6)
    else:
      callbacks = None

    def build_model(hp):
        model = Arik_CRNN(input_features=INPUT_SHAPE_FEATURES,
                          input_frames=INPUT_SHAPE_FRAMES,
                          n_c=N_C,
                          l_t=L_T,
                          l_f=L_F,
                          s_t=S_T,
                          s_f=S_F,
                          r=hp.Int("rnn_layers", 1, 4),
                          n_r=hp.Int("rnn_units", 16, 64, step=8),
                          n_f=hp.Int("dense_units", 16, 128, step=16),
                          dropout=hp.Float("dense_dropout", 0.0, 0.5, step=0.1),
                          activation=hp.Choice("activation", values=["relu"]))  # Perhaps try other activations?
        model.compile(optimizer=OPTIMIZER,
                      loss=losses.BinaryCrossentropy(),
                      metrics=METRICS)
        model.build(input_shape=(BATCH_SIZE, INPUT_SHAPE_FEATURES, INPUT_SHAPE_FRAMES, 1))
        return model

    tuner = Hyperband(
        build_model,
        max_epochs=10,
        factor=3,
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

def training_basic(training_generator, dev_generator, early_stopping=False):
    print(len(training_generator))
    print(len(dev_generator))
    if early_stopping:
      callbacks = [EarlyStopping(monitor="val_loss", patience=6),
                   ReduceLROnPlateau(monitor="val_loss", patience=3,
                                     factor=0.03, min_lr=DROPPED_LR),
                   ModelCheckpoint(filepath="best", save_best_only=True,
                                   monitor="val_loss", mode="min")]
    else:
      callbacks = None

    model = Arik_CRNN(INPUT_SHAPE_FEATURES, INPUT_SHAPE_FRAMES,
                      N_C, L_T, L_F, S_T, S_F, R, N_R, N_F,
                      activation='relu')
    model.compile(optimizer=OPTIMIZER, loss=losses.BinaryCrossentropy(), metrics=METRICS)
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
    parser = argparse.ArgumentParser(description='Trains CRNN, outputs model files.')
    parser.add_argument('--data_dir', type=str, default='/data_enhanced/', help='Directory where training data is stored.')
    args = parser.parse_args()
    return args

def main(args):
    train, dev, test = data_prep(args.data_dir)
    model = training_basic(train, dev, early_stopping=True)
    eval_basics(model, test)

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
