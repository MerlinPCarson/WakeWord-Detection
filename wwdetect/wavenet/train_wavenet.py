import os
import sys
import time
import argparse
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
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-4, help='Starting learning rate')
    parser.add_argument('--num_features', type=float, default=40, help='Number of features per-timestep')
    parser.add_argument('--timesteps', type=int, default=None, help='Number of timesteps per example, None for variable length')

    return parser.parse_args()

def main(args):
    start = time.time()

    # create dataloaders
    trainset = HeySnipsDataset(os.path.join(args.dataset_dir, args.trainset), batch_size=args.batch_size, num_features=args.num_features, workers=2)
    valset = HeySnipsDataset(os.path.join(args.dataset_dir, args.valset), batch_size=args.batch_size, num_features=args.num_features, workers=2)
    #testset = HeySnipsDataset(os.path.join(args.dataset_dir, args.testset), batch_size=args.batch_size, num_features=args.num_features)

    print(f'{len(trainset)*args.batch_size} training examples, {len(valset)*args.batch_size} validation examples')

    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    # Open a strategy scope.
    with strategy.scope():
        # Everything that creates variables should be under the strategy scope.
        # In general this is only model construction & `compile()`.
        model = build_wavenet_model(args.timesteps, args.num_features)

    # train model
    model.fit(trainset, epochs=args.epochs, validation_data=valset)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))

