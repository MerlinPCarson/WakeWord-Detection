import os
import sys
import time
import h5py
import pickle
import argparse

import numpy as np 
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, balanced_accuracy_score

from tf_lite.tf_lite import TFLiteModel


def load_tf_models(models_dir, quant=False):

    if not quant:
        encode_model: TFLiteModel = TFLiteModel(
            model_path=os.path.join(models_dir, "encode.tflite")
        )
        detect_model: TFLiteModel = TFLiteModel(
            model_path=os.path.join(models_dir, "detect.tflite")
        )
    else:
        encode_model: TFLiteModel = TFLiteModel(
            model_path=os.path.join(models_dir, "encode-quant.tflite")
        )
        detect_model: TFLiteModel = TFLiteModel(
            model_path=os.path.join(models_dir, "detect-quant.tflite")
    )

    return encode_model, detect_model

def load_data(data_file, timesteps, num_features):
    labels = []
    print(f'pre-loading dataset from file {data_file}')
    with h5py.File(data_file, 'r') as h5:
        keys = list(h5.keys())
        X = np.zeros((len(keys), timesteps, num_features), dtype=np.float32)
        for i, key in enumerate(tqdm(keys)):
            labels.append(h5[key].attrs['is_hotword'])
            features = h5[key][()][:timesteps]
            features = np.expand_dims(features, 0)
            X[i, :features.shape[1],:features.shape[2]] = features 

    return X, np.array(labels, dtype=np.uint8)

def models_predict(encode_model, detect_model, X, model_type, threshold=0.5):

    posteriors = []
    for x in tqdm(X):
        # inference for CRNN model
        if model_type == 'CRNN':
            x = np.expand_dims(x.T, 0)
            x = np.expand_dims(x, -1)
            x = np.array(encode_model(x)).squeeze(0)
            posteriors.append(detect_model(x)[0][0][1])

        # inference for Wavenet model
        elif model_type == 'Wavenet':
            x = np.expand_dims(x, 0)
            x = np.array(encode_model(x)).squeeze(0)
            posteriors.append(detect_model(x)[0][0][1])

    # threshold posteriors
    preds = [1 if posterior >= threshold else 0 for posterior in posteriors]

    return preds 

def metrics(preds, targets):

    # get confusion matrix
    cf = confusion_matrix(targets, preds)
    print(f'\nConfusion matrix:\n {cf}')

    # get metrics
    tn, fp, fn, tp = cf.ravel()
    results = {'true_negative': tn, 'false_positive': fp, 
               'true_positive': tp, 'false_negative': fn,
               'recall': tp/(tp+fp), 'precision': tp/(tp+fn),
               'accuracy': balanced_accuracy_score(targets, preds)}

    # print all metrics in results
    for key, val in results.items():
        print(f'{key}: {val}')

    return results

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation script for TF-Lite models.')
    parser.add_argument('--tf_models_dir', type=str, default='CRNN_tf_model', help='Directory to saved TF-Lite models')
    parser.add_argument('--dataset_dir', type=str, default='data_speech_isolated/silero', help='Directory with testing vectors in H5 format')
    parser.add_argument('--testset', type=str, default='test.h5', help='Filename for testing vectors in H5 format')
    parser.add_argument('--timesteps', type=int, default=151, help='Number of timesteps used as input to models')
    parser.add_argument('--num_features', type=int, default=40, help='Number of features per timestep used as input to models')
    parser.add_argument('--model_type', type=str, default='CRNN', choices=['CRNN', 'Wavenet'],
                        help='Model type being evaluated.')

    return parser.parse_args()


def main(args):

    start = time.time()

    # load non-quantized and quantized version of TF-Lite models
    encode_model, detect_model = load_tf_models(args.tf_models_dir)
    encode_model_quant, detect_model_quant = load_tf_models(args.tf_models_dir, quant=True)

    # load test data
    X, y = load_data(os.path.join(args.dataset_dir, args.testset), args.timesteps, args.num_features)


    results = {}

    # get predictions for float 32 bit model    
    print(f'Testing {args.model_type} TF-Lite models with 32-bit floats')
    preds = models_predict(encode_model, detect_model, X, args.model_type)
    results['float32'] = metrics(preds, y)

    # get predictions from quantized float 16 bit model
    print(f'Testing {args.model_type} TF-Lite models with 16-bit floats')
    preds = models_predict(encode_model_quant, detect_model_quant, X, args.model_type)
    results['float16'] = metrics(preds, y)

    pickle.dump(results, open(os.path.join(args.tf_models_dir, 'tf_lite_results.npy'), 'wb'))

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))

