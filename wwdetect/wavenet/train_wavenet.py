import os
import sys
import time
import argparse
import numpy as np
import tensorflow as tf
from wavenet import build_wavenet_model
from wavenet_loader import HeySnipsDataset 


def parse_args():
    parser = argparse.ArgumentParser(description='Wavenet trainer for wakeword detection.')
    parser.add_argument('--dataset_dir', type=str, default='data', help='directory with datasets as H5 files')
    parser.add_argument('--trainset', type=str, default='train.h5', help='H5 file containing training vectors')
    parser.add_argument('--valset', type=str, default='dev.h5', help='H5 file containing validation vectors')
    parser.add_argument('--testset', type=str, default='test.h5', help='H5 file containing test vectors')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save trained models to')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples per batch')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train for')
    parser.add_argument('--patience', type=int, default=10, help='Number of epochs of no improvement before early stopping')
    parser.add_argument('--lr', type=float, default=1e-4, help='Starting learning rate')
    parser.add_argument('--l2', type=float, default=1e-5, help='L2 regularization coefficient')
    parser.add_argument('--num_features', type=float, default=40, help='Number of features per-timestep')
    parser.add_argument('--timesteps', type=int, default=None, help='Number of timesteps per example, None for variable length')
    parser.add_argument('--seed', type=int, default=9999, help='Random seed for training')
    parser.add_argument('--eval_model', type=str, default='models/wavenet_model', help='Location of model to evaluate')

    return parser.parse_args()

def main(args):
    start = time.time()

    # for reproducability
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # create dataloaders
    trainset = HeySnipsDataset(os.path.join(args.dataset_dir, args.trainset), shuffle=True, 
                               batch_size=args.batch_size, num_features=args.num_features, workers=2)
    valset = HeySnipsDataset(os.path.join(args.dataset_dir, args.valset), shuffle=False, 
                             batch_size=args.batch_size, num_features=args.num_features, workers=2)

    print(f'{trainset.number_of_examples()} training examples, {valset.number_of_examples()} validation examples')

    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    # Open a strategy scope.
    with strategy.scope():
        # Everything that creates variables should be under the strategy scope.
        # In general this is only model construction & `compile()`.
        model = build_wavenet_model(args)

    # Keras callbacks
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(patience=args.patience//2, verbose=1)

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=args.patience, verbose=1)

    model_chkpt = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(args.model_dir, 'wavenet_model'),
                                                     save_best_only=True, verbose=1)

    # train model
    model.fit(trainset, epochs=args.epochs, validation_data=valset, 
              callbacks=[reduce_lr, early_stopping, model_chkpt])

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))

