import os
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


def setup_gpus():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device_ids = [i for i in range(torch.cuda.device_count())]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, device_ids))
    return device_ids

# function to remove weight decay from output layer, or from PReLU
def weight_decay(model, final_layer='classifier.4'):
    print('Disabling weight decay for PReLU activation layers, batch normalization layers, and final output layer')
    params = []
    for name, param in model.named_parameters():
        if final_layer in name or 'classifier.0' in name or 'classifier.2' in name or 'norm' in name:
#            print(f'setting weight decay to 0 for layer {name}')
            params.append({'params': param, 'weight_decay': 0.0})
        else:
            params.append({'params': param})

    return params

def train(model, train_loader, val_loader, test_loader, num_epochs=1):

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

    # remove weight decay from final layer, batch norm layers, and PReLU activations
    params = model.parameters()
    #params = weight_decay(model)

    optimizer = AdamW(params, lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=1e-5, amsgrad=False)
    optimizer.zero_grad()
    L1_criterion = nn.L1Loss(reduction='none').to(device)
    #L1_criterion = nn.BCELoss().to(device)

    # schedulers
    patience = 10
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=patience//2)

    train_batches_per_epoch = train_loader.num_batches
    val_batches_per_epoch = val_loader.num_batches
    test_batches_per_epoch = test_loader.num_batches

    # intializiang best values for regularization via early stopping
    best_val_loss = 99999.0
    best_val_acc = 0.0
    epochs_since_improvement = 0
    history = {'loss': [], 'val_loss': [], 'pos_val_acc': [], 'neg_val_acc': []}
    model_dir = f'weights_{os.uname()[1].split(".")[0]}'
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        print("epoch: ", epoch)
        bar = trange(train_batches_per_epoch)
        avg_epoch_loss = 0.0
        avg_acc = 0.0
        for batch in bar:
            x_np, y_np = train_loader.get_batch()
            x = torch.from_numpy(x_np).float().to(device)
            y = torch.from_numpy(y_np).float().to(device)

            out = model(x)

            loss = weighted_L1_loss(L1_criterion, out, train_batches_per_epoch, y)
            loss.backward()
            avg_epoch_loss += loss.item() / train_batches_per_epoch
            acc = float((out[y == 0] < .5).sum() + (out[y == 1] > .5).sum()) / float(y.shape[0])
            avg_acc += acc / float(train_batches_per_epoch)
            curr_loss = loss.item()
            optimizer.step()
            optimizer.zero_grad()
            bar.set_description("epoch: %d, loss: %f, acc: %f " % (epoch, curr_loss, acc))
        bar.set_description("epoch: %d, loss: %f, acc: %f " % (epoch, avg_epoch_loss, avg_acc))

        model.eval()
        #sum_true_preds = 0
        avg_pos_acc = 0.0
        avg_neg_acc = 0.0
        with torch.no_grad():
            avg_epoch_val_loss = 0.0
            avg_acc = 0.0
            bar = trange(val_batches_per_epoch)
            for batch in bar:
                x_np, y_np = val_loader.get_batch()
                x = torch.from_numpy(x_np).float().to(device)
                y = torch.from_numpy(y_np).float().to(device)

                out = model(x)

                loss = weighted_L1_loss(L1_criterion, out, val_batches_per_epoch, y)
                pos_acc = (out[y == 1] > .5).sum() / (y==1).sum()
                neg_acc = (out[y == 0] < .5).sum() / (y==0).sum()
                avg_pos_acc += pos_acc
                avg_neg_acc += neg_acc
                curr_loss = loss.item()
                avg_epoch_val_loss += loss.item()
                bar.set_description("epoch: %d, val_loss: %f, pos_acc: %f, neg_acc %f " % (epoch, curr_loss, pos_acc, neg_acc))

            avg_epoch_val_loss /= val_batches_per_epoch
            avg_pos_acc /= val_batches_per_epoch
            avg_neg_acc /= val_batches_per_epoch
            bar.set_description("epoch: %d, val_loss: %f, pos_val_acc: %f , neg_val_acc: %f" % (epoch, avg_epoch_val_loss, avg_pos_acc, avg_neg_acc))

        print(f"epoch: {epoch}, loss: {avg_epoch_loss}, val_loss:{avg_epoch_val_loss}")
        print(f"True Positive: {avg_pos_acc}, True Negative: {avg_neg_acc}")

        # saving training stats
        history['loss'].append(avg_epoch_loss)
        history['val_loss'].append(avg_epoch_val_loss)
        history['pos_val_acc'].append(avg_pos_acc)
        history['neg_val_acc'].append(avg_neg_acc)
        pickle.dump(history, open(os.path.join(model_dir, 'model.npy'), 'wb'))
        
        # save if best model, reset patience counter
        if avg_epoch_val_loss < best_val_loss:
            print('Saving best model')
            best_val_loss = avg_epoch_val_loss
            best_epoch = epoch
            epochs_since_improvement = 0
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pt'))
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement > patience:
            print('Initiating early stopping')
            break

        # reduce learning rate if validation has leveled off
        scheduler.step(avg_epoch_val_loss)

        #torch.save(model.state_dict(), os.path.join(model_dir, f'epoch_{epoch}_{avg_epoch_val_loss}_{avg_acc}'))

    # test model
    print(f'Loading best model from epoch {best_epoch} [val_loss: {best_val_loss}]') 
    model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pt')))

    model.eval()
    test_pos_acc = 0.0
    test_neg_acc = 0.0
    with torch.no_grad():
        avg_acc = 0.0
        bar = trange(test_batches_per_epoch)
        for batch in bar:
            x_np, y_np = test_loader.get_batch()
            x = torch.from_numpy(x_np).float().to(device)
            y = torch.from_numpy(y_np).float().to(device)

            out = model(x)

            #acc = float((out[y == 0] < .5).sum() + (out[y == 1] > .5).sum()) / float(y.shape[0])
            #avg_acc += acc / float(test_batches_per_epoch)

            test_pos_acc += (out[y == 1] > .5).sum() / (y==1).sum()
            test_neg_acc += (out[y == 0] < .5).sum() / (y==0).sum()


    test_pos_acc /= test_batches_per_epoch
    test_neg_acc /= test_batches_per_epoch
    print(f"test_pos_acc: {test_pos_acc}, test_neg_acc: {test_neg_acc}")

    # save test stats to history file
    history['test_pos_acc'] = test_pos_acc
    history['test_neg_acc'] = test_neg_acc
    pickle.dump(history, open(os.path.join(model_dir, 'model.npy'), 'wb'))

def weighted_L1_loss(L1_criterion, out, batches_per_epoch, y):
    eps = (.001 / batches_per_epoch)
    return (L1_criterion(out, y) * (y + y.mean() + eps)).mean()


def main():
    start = time.time()

    path = "/stash/tlab/mcarson/WakeWordData/"
    train_loader = Loader_HeySnips(os.path.join(path, "train.h5"),
                                   batch_size=128)
    val_loader = Loader_HeySnips(os.path.join(path, "dev.h5"),
                                   batch_size=128)
    test_loader = Loader_HeySnips(os.path.join(path, "test.h5"),
                                   batch_size=128)
    model = KeyWordSpotter(40)

    print(model)
    print(f'total number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    train(model, train_loader, val_loader, test_loader, num_epochs=500)

    print(f'Script completed in {time.time()-start:.2f} secs')

if __name__ == '__main__':
    main()

