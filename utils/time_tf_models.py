from json import load
import os 
import sys
import time
import argparse
import numpy as np

from tqdm import tqdm 

from tensorflow.keras.models import load_model
import tensorflow.lite as tflite


def time_tf_model(model, num_timesteps, num_features, model_type, num_runs):

    # define model input shape
    if model_type == 'Wavenet':
        X_shape = (1, num_timesteps, num_features)
    elif model_type == 'CRNN':
        X_shape = (1, num_features, num_timesteps, 1)

    # create random input vector
    X = np.array(np.random.random_sample(X_shape), dtype=np.float32)

    # prime the Tensorflow graph, first run is slow
    model.predict(X)

    total_time = 0.0
    for _ in tqdm(range(num_runs)):
        start = time.perf_counter() 
        model.predict(X)
        total_time += time.perf_counter() - start

    return total_time/num_runs


def time_tf_lite_models(encode, detect, num_runs):

    # allocate input and output tensors for models
    encode.allocate_tensors()
    detect.allocate_tensors()

    # Get input and output information for models.
    encode_input_details = encode.get_input_details()
    encode_output_details = encode.get_output_details()
    detect_input_details = detect.get_input_details()

    # Create random input vector
    X_shape = encode_input_details[0]['shape']
    X = np.array(np.random.random_sample(X_shape), dtype=np.float32)

    # prime TF-Lite models, first time is slow
    encode.set_tensor(encode_input_details[0]['index'], X)
    encode.invoke()
    encoded = encode.get_tensor(encode_output_details[0]['index'])
    detect.set_tensor(detect_input_details[0]['index'], encoded)

    # get average inference time
    total_time = 0.0
    for _ in tqdm(range(num_runs)):
        start = time.perf_counter() 

        encode.set_tensor(encode_input_details[0]['index'], X)
        encode.invoke()
        encoded = encode.get_tensor(encode_output_details[0]['index'])
        detect.set_tensor(detect_input_details[0]['index'], encoded)

        total_time += time.perf_counter() - start

    return total_time/num_runs

def load_wavenet(model_dir):
    return load_model(args.tf_model_dir)

def load_crnn(model_dir):
    return load_model(args.tf_model_dir)

def load_tensorflow_model(model_dir, model_type):
    # load the appropriate model type
    if args.model_type == 'Wavenet':
        tf_model = load_wavenet(args.tf_model_dir)
    elif args.model_type == 'CRNN':
        tf_model = load_crnn(args.tf_model_dir) 

    return tf_model

def time_models(args):

    # Load the Tensorflow model
    tf_model = load_tensorflow_model(args.tf_model_dir, args.model_type)

    # run timings on Tensorflow model
    print(f'Running timings on Tensorflow {args.model_type} model')
    avg_time_tf = time_tf_model(tf_model, args.timesteps, args.num_features, args.model_type, args.num_runs)

    print(f'Tensorflow average time: {avg_time_tf} secs')

    # Load Tensorflow Lite models
    tf_lite_encode = tflite.Interpreter(model_path=os.path.join(args.tf_lite_model_dir, 'encode.tflite'))
    tf_lite_detect = tflite.Interpreter(model_path=os.path.join(args.tf_lite_model_dir, 'detect.tflite'))

    # run timings on Tensorflow Lite models
    print(f'Running timings on Tensorflow-Lite {args.model_type} models')
    avg_time_tf_lite = time_tf_lite_models(tf_lite_encode, tf_lite_detect, args.num_runs)

    print(f'TF-Lite average time: {avg_time_tf_lite} secs')

def parse_args():
    parser = argparse.ArgumentParser(description='Gets inference timings for Tensorflow and TF-Lite versions of Wavenet models.')
    parser.add_argument('--model_type', type=str, default='Wavenet', choices=['CRNN', 'Wavenet'], help='Model type being evaluated.')
    parser.add_argument('--tf_model_dir', type=str, default='tf_models', help='Directory with Tensorflow models')
    parser.add_argument('--tf_lite_model_dir', type=str, default='tf_lite_models', help='Directory with Tensorflow Lite models')
    parser.add_argument('--num_features', type=float, default=40, help='Number of features per-timestep')
    parser.add_argument('--timesteps', type=int, default=182, help='Number of timesteps per example, None for variable length')
    parser.add_argument('--num_runs', type=int, default=10, help='Number of runs to get average infernce time')
    return parser.parse_args()

def main(args):
    start = time.time()

    time_models(args) 

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
