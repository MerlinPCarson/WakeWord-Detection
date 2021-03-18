import os
import sys
import time
import argparse

import tensorflow as tf
from tensorflow.keras.models import load_model

def save_to_tflite(model, outdir):
    encode_converter = tf.lite.TFLiteConverter.from_keras_model(model.encoder)
    detect_converter = tf.lite.TFLiteConverter.from_keras_model(model.detect)

    tflite_encode_model = encode_converter.convert()
    tflite_detect_model = detect_converter.convert() 
    
    # Save the model.
    with open(os.path.join(outdir, 'encode.tflite'), 'wb') as f:
        f.write(tflite_encode_model)

    with open(os.path.join(outdir, 'detect.tflite'), 'wb') as f:
        f.write(tflite_detect_model)

    # enable weight quantization
    encode_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    encode_converter.target_spec.supported_types = [tf.float16]
    detect_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    detect_converter.target_spec.supported_types = [tf.float16]

    tflite_encode_model = encode_converter.convert()
    tflite_detect_model = detect_converter.convert()

    # Save the model.
    with open(os.path.join(outdir, 'encode-quant.tflite'), 'wb') as f:
        f.write(tflite_encode_model)

    with open(os.path.join(outdir, 'detect-quant.tflite'), 'wb') as f:
        f.write(tflite_detect_model)

def parse_args():
    
    parser = argparse.ArgumentParser(description='Converts Tensorflow model to TF-Lite models.')
    parser.add_argument('--tf_model_dir', type=str, default='CRNN_tf_model', help='Directory with Tensorflow models')
    return parser.parse_args()

def main(args):
    start = time.time()

    model = load_model(args.tf_model_dir)
    save_to_tflite(model, args.tf_model_dir)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
