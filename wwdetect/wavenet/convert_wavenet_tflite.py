import os
import sys
import time
import numpy as np 
from tqdm import tqdm

from wavenet_model import build_wavenet_model
from wavenet_loader import HeySnipsDataset 
from train_wavenet import parse_args

def save_to_tflite(args):

    print('Converting models to TF-Lite')
    args.timesteps = 182 # receptive field, as described in paper Coucke et al.
    model = build_wavenet_model(args)
    model.load_weights(args.eval_model)
    model.save_to_tflite(os.path.dirname(args.eval_model))


def main(args):

    start = time.time()

    save_to_tflite(args)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))

