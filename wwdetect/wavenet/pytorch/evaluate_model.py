import os
import sys
import time
import torch
import pickle
import numpy as np
from torch import nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange

#from loader import Loader_HeySnips
from loaderH5 import Loader_HeySnips
from model import KeyWordSpotter

from sklearn.metrics import confusion_matrix, balanced_accuracy_score


def setup_gpus():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device_ids = [i for i in range(torch.cuda.device_count())]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, device_ids))
    return device_ids

def evaluate_model(model, model_dir, test_loader):

    # detect gpus and setup environment variables
    if torch.cuda.is_available() is True:
        device = 'cuda:0'
        device_ids = setup_gpus()
        print(f'Cuda devices found: {[torch.cuda.get_device_name(i) for i in device_ids]}')
    else:
        device = 'cpu'

    # if there are GPUs, prepare model for data parallelism (use multiple GPUs)
    if device == 'cuda:0':
        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()

    test_batches_per_epoch = test_loader.num_batches
    print(f'Number of test examples {len(test_loader.dataset)}')

    # test model
    model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pt')))

    model.eval()
    test_pos_acc = 0.0
    test_neg_acc = 0.0
    targets = []
    preds = []
    with torch.no_grad():
        avg_acc = 0.0
        bar = trange(test_batches_per_epoch)
        for batch in bar:
            x_np, y_np = test_loader.get_batch()
            x = torch.from_numpy(x_np).float().to(device)
            y = torch.from_numpy(y_np).float().to(device)

            out = model(x)

            targets.extend(y_np)
            preds.extend(out.cpu().numpy())

            #acc = float((out[y == 0] < .5).sum() + (out[y == 1] > .5).sum()) / float(y.shape[0])
            #avg_acc += acc / float(test_batches_per_epoch)

            #test_pos_acc += (out[y == 1] > .5).sum() / (y==1).sum()
            #test_neg_acc += (out[y == 0] < .5).sum() / (y==0).sum()

    preds = np.array([1 if i >= 0.5 else 0 for i in preds])
    targets = np.array(targets)
   
    print(f'num negative examples: {targets[targets==0].sum()}')
    print(f'num positive examples: {targets[targets==1].sum()}')
    print(targets)

    cf = confusion_matrix(targets, preds)
    print(cf)
    tn, fp, fn, tp = cf.ravel()
    print(f'True Positive: {tp}')
    print(f'True Negative: {tn}')
    print(f'False Positive: {fp}')
    print(f'False Negative: {fn}')
    print(f'Precision: {tp/(tp+fp)}')
    print(f'Recall: {tp/(tp+fn)}')

    print(f'Balanced Accuracy: {balanced_accuracy_score(targets, preds)}')


def main():
    start = time.time()

    model_dir = sys.argv[1]

    path = "/stash/tlab/mcarson/WakeWordData/"
    test_loader = Loader_HeySnips(os.path.join(path, "test.h5"),
                                   batch_size=32)
    model = KeyWordSpotter(40)

    print(model)
    print(f'total number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    evaluate_model(model, model_dir, test_loader)

    print(f'Script completed in {time.time()-start:.2f} secs')

if __name__ == '__main__':
    main()
