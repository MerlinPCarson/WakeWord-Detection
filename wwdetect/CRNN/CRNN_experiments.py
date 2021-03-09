import os
import sys
import time
import pickle
import random
import argparse
import numpy as np

# for more determinisic results on GPU
os.environ['TF_CUDNN_DETERMINISTIC']='1'
import tensorflow as tf

from train import parse_args, train_basic, data_prep
from train import INPUT_SHAPE_FEATURES, INPUT_SHAPE_FRAMES
from evaluate import prep_test_data, eval_basics, load_model


def wakeword_reduce(args, experiment_num=1):
    '''
    Adapted from `wakeword_exps.py`. Performs a single round of
    experiments training the CRNN on a reduced set of wakewords.

    Outputs a pickle file containing relevant experimental details and
    results.
    '''
    print("Running experiment round", experiment_num, "!")
   
    # for reproducability
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # create dataloaders for training, test and validation sets
    train_generator, val_generator = data_prep(args.data_dir, args.ctc, use_enhanced=args.use_augmented_train)
    train_wakewords, train_negatives = train_generator.num_samples()
    val_wakewords, val_negatives = val_generator.num_samples()
    test_generator = prep_test_data(args.data_dir, INPUT_SHAPE_FRAMES, INPUT_SHAPE_FEATURES, args.ctc)
    print("Original number of training wakewords:", train_wakewords, "and negative samples:", train_negatives)
    print("Original number of validation wakewords:", val_wakewords, "and negative samples:", val_negatives)

    # keep ratio variables
    min_wakeword_ratio = 0.1
    max_wakeword_ratio = 1.0
    wakeword_ratio_step = 0.1

    # ratio of wakewords in dataset to use for training, iterating from max to min by step
    keep_ratios = np.arange(min_wakeword_ratio, max_wakeword_ratio+wakeword_ratio_step/10, wakeword_ratio_step)

    exps = {}

    print(f'Conducting training for models with wakeword keep ratios of: {keep_ratios}')
    for keep_ratio in reversed(keep_ratios):

        # prune 1-keep_ratio of wakewords from training set
        print(f'Pruning wakewords with a keep ratio of {keep_ratio}')
        num_kept, num_removed = train_generator.prune_wakewords(keep_ratio)
        train_wakewords, train_negatives = train_generator.num_samples()
        assert num_kept == train_wakewords
        print("New number of training wakewords:", train_wakewords, "and negative samples:", train_negatives)

        exps[keep_ratio] = {"num_train_wakewords_kept": train_wakewords,
                            "num_train_wakewords_removed": num_removed,
                            "num_train_negatives": train_negatives,
                            "keep_ratio": keep_ratio}

        # train the model with the keep ratio
        print("Training model...")
        model = train_basic(train_generator, val_generator, 
                            early_stopping=args.early_stopping,
                            ctc=args.ctc, positive_percentage=keep_ratio)
        model_path = model.pathname
        print("Model trained!")

        # clear out memory
        del model

        print("Loading tflite models...")
        # load .tflite models and evaluate
        encode_model, decode_model = load_model(os.path.join("models", model_path + "_encode.tflite"),
                                                os.path.join("models", model_path + "_detect.tflite"))
        stats = eval_basics(encode_model, decode_model, test_generator)
        exps[keep_ratio]["stats"] = stats

        print("Evaluation complete!")
    
    pickle.dump(exps, open(f'CRNN-{experiment_num}-wakeword_reduce-exps.npy', 'wb'))

    return 0

if __name__ == '__main__':
    args = parse_args()
    for i in range(1,9):
        wakeword_reduce(args, experiment_num=i)
    sys.exit()
