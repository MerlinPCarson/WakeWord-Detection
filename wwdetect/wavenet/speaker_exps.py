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

from train_wavenet import parse_args
from train_wavenet import train, load_datasets


def main(args):
    start = time.time()
   
    # for reproducability
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # make model directory for model experiments metadata file
    os.makedirs(os.path.dirname(args.model))

    # create dataloaders for training and validation sets
    trainset, valset = load_datasets(args)
    print(f'{trainset.number_of_examples()} training examples, {valset.number_of_examples()} validation examples')

    # keep ratio variables
    min_wakeword_ratio = args.wakeword_min_keep_ratio 
    max_wakeword_ratio = args.wakeword_max_keep_ratio 
    wakeword_ratio_step = args.wakeword_keep_ratio_step

    # save base model name
    model_name = args.model

    # ratio of wakewords in dataset to use for training, iterating from max to min by step
    keep_ratios = np.arange(min_wakeword_ratio, max_wakeword_ratio+wakeword_ratio_step/10, wakeword_ratio_step)

    # experiment metadata
    exps = {'model_name': os.path.basename(args.model), 'keep_ratios': [],
            'num_speakers': [], 'num_wakewords': []}

    print(f'Conducting training for models with speaker keep ratios of: {keep_ratios}')
    for keep_ratio in reversed(keep_ratios):

        # set experiment specific arguments
        args.wakeword_keep_ratio = f'{keep_ratio:0.2f}'
        args.model = f'{model_name}_{args.wakeword_keep_ratio}'

        print(f'Pruning wakewords with a keep ratio of {keep_ratio}')
        trainset.prune_speakers(keep_ratio)
        print(f'Size of training set {trainset.number_of_examples()} examples')

        # get new stats 
        num_speakers = trainset.number_of_speakers()
        num_wakewords = trainset.number_of_wakewords()

        print(f'Number of speakers in training set: {trainset.number_of_speakers()}')
        print(f'Number of wakewords examples in training set {trainset.number_of_wakewords()}')

        # train the model with the keep ratio
        #train(trainset, valset, args)

        # save stats to experiment metadata
        exps['keep_ratios'].append(keep_ratio)
        exps['num_speakers'].append(num_speakers)
        exps['num_wakewords'].append(num_wakewords)

        # saving experiment info 
        pickle.dump(exps, open(f'{model_name}-speaker_exps.npy', 'wb'))
    
    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
