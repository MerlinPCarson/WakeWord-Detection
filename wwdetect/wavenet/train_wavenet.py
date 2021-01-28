import sys
import time
import argparse
from wavenet import build_wavenet_model
from wavenet_loader import HeySnipsDataset 



def parse_args():
    parser = argparse.ArgumentParser(description='Wavenet trainer for wakeword detection.')
    parser.add_argument('--trainset', type=str, default='data/train.h5', help='H5 file containing training vectors')
    parser.add_argument('--valset', type=str, default='data/dev.h5', help='H5 file containing validation vectors')
    parser.add_argument('--testset', type=str, default='data/test.h5', help='H5 file containing test vectors')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save trained models to')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of examples per batch')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-4, help='Starting learning rate')
    parser.add_argument('--num_features', type=float, default=40, help='Number of features per-timestep')
    parser.add_argument('--timesteps', type=int, default=None, help='Number of timesteps per example, None for variable length')

    return parser.parse_args()

def main(args):
    start = time.time()

    # create dataloaders
    trainset = HeySnipsDataset(args.trainset, batch_size=args.batch_size, num_features=args.num_features)
    valset = HeySnipsDataset(args.valset, batch_size=args.batch_size, num_features=args.num_features)
    #testset = HeySnipsDataset(args.testset, batch_size=args.batch_size, num_features=args.num_features)

    #print(f'{len(trainset)} training examples, {len(valset)} validation examples, {len(testset)} test examples')
    print(f'{len(trainset)*args.batch_size} training examples, {len(valset)*args.batch_size} validation examples')

    # instantiate model
    model = build_wavenet_model(args.timesteps, args.num_features)

    # train model
    model.fit_generator(trainset, epochs=args.epochs, validation_data=valset)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))