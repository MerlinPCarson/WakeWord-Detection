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

METRICS = [
      metrics.TruePositives(name='tp'),
      metrics.FalsePositives(name='fp'),
      metrics.TrueNegatives(name='tn'),
      metrics.FalseNegatives(name='fn'),
      metrics.Precision(name='precision'),
      metrics.Recall(name='recall')
]


def prep_test_data(data_path, input_frames, input_features, ctc):
    test_generator = HeySnipsPreprocessed([os.path.join(data_path, "test.h5")],
                                           batch_size=0,
                                           frame_num=input_frames,
                                           feature_num=input_features,
                                           ctc=ctc)
    return test_generator


def eval_basics(encoder, decoder, test_generator):
    X, y = test_generator[0]
    X = X.astype(np.float32)
    y = y[:,1]
    y_pred = []
    for sample in tqdm(X):
        sample = np.expand_dims(sample, axis=0)
        y_pred_sample = encoder(sample)[0]
        y_pred_sample = np.squeeze(decoder(y_pred_sample)[0])
        y_pred.append(float(y_pred_sample[1]))
    y_pred = np.array(y_pred)
    y_pred_class = np.where(y_pred >= 0.5,1,0)
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
    X, y = test_generator[0]
    X = X.astype(np.float32)
    y_seq = [tf.strings.reduce_join(test_generator.num2char(y_i)).numpy().decode("utf-8") for y_i in y]
    y_pred = []
    for sample in tqdm(X):
        sample = np.expand_dims(sample, axis=0)
        y_encode_sample = encoder(sample)[0]
        y_decode_sample = decoder(y_encode_sample)[0]
        input_len = np.ones(y_decode_sample.shape[0]) * y_decode_sample.shape[1]
        result = tf.keras.backend.ctc_decode(y_decode_sample, input_length=input_len, greedy=True)[0][0][:, :4]
        result_labels = "".join(test_generator.num2char(result[0].numpy()))
        y_pred += [result_labels]

    y_pred_class = [1 if y_pred_i == "[SILENCE]heysnips[SILENCE]" else 0 for y_pred_i in y_pred]
    y_true_class = [1 if y_i == "[SILENCE]heysnips[SILENCE]" else 0 for y_i in y_seq]
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
    parser.add_argument('--model_dir', type=str, default='CTC_models/keras_4_seq_silence_better_labeling/for_inference')
    args = parser.parse_args()
    return args

def load_model(encode_path, detect_path):
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