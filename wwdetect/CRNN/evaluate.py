'''
Module to evaluate and compare CRNN model(s).
'''

import sys
import os
import argparse
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import metrics, models
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import numpy as np
from matplotlib import pyplot as plt

from dataloader import HeySnipsPreprocessed

from tensorflow_tflite import TFLiteModel

# Metrics to calculate for each model.
METRICS = [
      metrics.TruePositives(name='tp'),
      metrics.FalsePositives(name='fp'),
      metrics.TrueNegatives(name='tn'),
      metrics.FalseNegatives(name='fn'),
      metrics.Precision(name='precision'),
      metrics.Recall(name='recall')
]


def prep_test_data(data_path, input_frames, input_features, ctc):
    '''
    Small function to load test data.

    :param data_path: Path to .h5 file containing data.
    :param input_frames: Number of input frames model expects.
    :param input_features: Number of input features model expects.
    :param ctc: Boolean, if model CTC-based.
    :return: Loaded test generator.
    '''
    test_generator = HeySnipsPreprocessed([os.path.join(data_path, "test.h5")],
                                           batch_size=0,
                                           frame_num=input_frames,
                                           feature_num=input_features,
                                           ctc=ctc)
    return test_generator


def eval_basics(encoder, decoder, test_generator):
    '''
    Evaluation for model trained with cross-entropy loss
    (two output nodes, first is not wakeword, second is
    wakeword).

    :param encoder: Preloaded encoder model.
    :param decoder: Preloaded decoder model.
    :param test_generator: Generator for test data.
    :return: Dict with all statistics.
    '''
    X, y = test_generator[0]
    keys = test_generator.list_IDs_temp
    FAs = []
    FRs = []
    X = X.astype(np.float32)
    y = y[:,1]
    y_pred = []
    index = 0
    for sample in tqdm(X):
        sample = np.expand_dims(sample, axis=0)
        y_pred_sample = encoder(sample)[0]
        y_pred_sample = np.squeeze(decoder(y_pred_sample)[0])
        y_pred.append(float(y_pred_sample[1]))
        index += 1
    y_pred = np.array(y_pred)
    y_pred_class = np.where(y_pred >= 0.5,1,0)

    for index, (prediction, actual) in enumerate(zip(y_pred_class, y)):
        if prediction == 1 and actual == 0:
            FAs.append(keys[index])
        if prediction == 0 and actual == 1:
            FRs.append(keys[index])

    print("False accept files:", FAs)
    print("False reject files:", FRs)
    acc = accuracy_score(y, y_pred_class)
    bal_acc = balanced_accuracy_score(y, y_pred_class)

    stats = {}
    for metric in METRICS:
        metric.reset_states()
        metric.update_state(y,y_pred_class)
        print(f"{metric.name}: {metric.result().numpy()}")
        stats[metric.name] = metric.result().numpy()
    print(f"accuracy: {acc}")
    print(f"balanced accuracy: {bal_acc}")
    stats["accuracy"] = acc
    stats["balanced accuracy"] = bal_acc
    return stats

def eval_CTC(encoder, decoder, test_generator):
    '''
    Evaluation for model trained with CTC loss.

    :param encoder: Preloaded encoder tflite model.
    :param decoder: Preloaded decoder tflite model.
    :param test_generator: Generator containing test data.
    :return: None.
    '''
    X, y = test_generator[0]
    X = X.astype(np.float32)
    y_seq = [tf.strings.reduce_join(test_generator.num2char(y_i)).numpy().decode("utf-8") for y_i in y]
    y_pred = []
    index = 0

    # Iterate through all test samples and decode output of model.
    for sample in tqdm(X):
        sample = np.expand_dims(sample, axis=0)
        y_encode_sample = encoder(sample)[0]
        y_decode_sample = decoder(y_encode_sample)[0]
        input_len = np.ones(y_decode_sample.shape[0]) * y_decode_sample.shape[1]

        # Final index must change based on how long the max target sequence length is.
        # Current setup is two, sequences are '[HEY][SNIPS]' and '[OTHER]'.
        result = tf.keras.backend.ctc_decode(y_decode_sample, input_length=input_len, greedy=True)[0][0][:, :2]
        result_labels = "".join(test_generator.num2char(result[0].numpy()))

        # Toggle to plot posterior trajectories.
        if not "plot_posteriors" and y_seq[index] == "[HEY][SNIPS]":
            labels = list(test_generator.char2num_dict.keys())[1:] + ['[BLANK]']
            target = "".join([test_generator.num2char_dict[num] for num in y[index] if num >= 0])
            plt.imshow(np.squeeze(y_decode_sample).T, cmap='Greys')
            plt.yticks(ticks=list(range(len(labels))), labels=labels)
            plt.xticks(ticks=list(range(int(input_len[0]))))
            plt.xlabel("timepoint")
            plt.title("Target sequence: " + str(target))
            plt.show()
        y_pred += [result_labels]
        index += 1

    # Convert sequences to classes.
    y_pred_class = [1 if y_pred_i == "[HEY][SNIPS]" else 0 for y_pred_i in y_pred]
    y_true_class = [1 if y_i == "[HEY][SNIPS]" else 0 for y_i in y_seq]

    # Calculate and output metrics.
    bal_acc = balanced_accuracy_score(y_true_class, y_pred_class)

    for metric in METRICS:
        metric.update_state(y_true_class,y_pred_class)
        print(f"{metric.name}: {metric.result().numpy()}")
    print(f"balanced accuracy: {bal_acc}")

def parse_args():
    '''
    Parse commandline arguments.

    :return: Arguments dict.
    '''
    parser = argparse.ArgumentParser(description='Evaluates CRNN model(s).')
    parser.add_argument('--data_dir', type=str, default='/Users/amie/Desktop/OHSU/CS606 - Deep Learning II/FinalProject/spokestack-python/data_speech_isolated/silero', help='Directory where test data is stored.')
    parser.add_argument('--model_dir', type=str, default='models/Arik_CRNN_data_nosilence_enhanced')
    args = parser.parse_args()
    return args

def load_model(encode_path, detect_path):
    '''
    Helper function to load tflite model.

    :param encode_path: Path to encoder model.
    :param detect_path: Path to detect model.
    :return: Loaded models.
    '''
    encode_model: TFLiteModel = TFLiteModel(
        model_path=encode_path
    )
    detect_model: TFLiteModel = TFLiteModel(
        model_path=detect_path
    )

    return encode_model, detect_model

def main(args):

    encode_model, detect_model = load_model(os.path.join(args.model_dir, "encode.tflite"),
                                            os.path.join(args.model_dir, "detect.tflite"))

    if "CTC" in args.model_dir:
        test = prep_test_data(args.data_dir, ctc=True, input_features=40, input_frames=151)
        eval_CTC(encode_model, detect_model, test)
    else:
        test = prep_test_data(args.data_dir, ctc=False, input_features=40, input_frames=151)
        eval_basics(encode_model, detect_model, test)


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(args))