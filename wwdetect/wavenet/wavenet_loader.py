import h5py
import numpy as np
from tqdm import tqdm
import tensorflow as tf

class HeySnipsDataset(tf.keras.utils.Sequence):
    def __init__(self, data_file, batch_size = 32, num_features = 40, *args, **kwargs):
        # dataset pre-fetch
        self.dataset = self.preload_data(data_file)

        # state variables
        self.num_features = num_features
        self.batch_size = batch_size
        self.batch_idx = 0
        self.num_batches = len(self.dataset) // batch_size

    def __len__(self):
        # returns the number of batches
        return len(self.dataset) // self.batch_size

    def __getitem__(self, index):
        # returns one batch

        # find which segments make up the batch
        batch_start = self.batch_idx * self.batch_size
        batch = np.arange(batch_start, batch_start + self.batch_size)

        # load batch features and labels 
        X = [self.dataset[i]['features'] for i in batch]
        y = [self.dataset[i]['label'] for i in batch]

        # make each batch uniform length
        X = self.pad(X)

        # increment batch number
        self.batch_idx += 1

        return X, y

    def on_epoch_end(self):
        # option method to run some logic at the end of each epoch: e.g. reshuffling
        np.random.shuffle(self.dataset)
        self.batch_idx = 0

    def pad(self, X):
        # create new datastruct of max length timesteps 
        max_len = np.max([x.shape[0] for x in X])
        padded_X = np.zeros((len(X), max_len, self.num_features), dtype=np.float32)

        # pad features
        for i, x in enumerate(X):
            x = np.expand_dims(x, axis=0) 
            padded_X[i, : x.shape[1], : ] 

        return padded_X

    def preload_data(self, data_file):

        self.dataset = []
        print(f'pre-loading dataset from file {data_file}')
        with h5py.File(data_file, 'r') as h5:
            for key in tqdm(h5.keys()):
                self.dataset.append({'file_name': key, 
                                     'label': np.uint8(h5[key].attrs['is_hotword']),
                                     'start_speech': np.int16(h5[key].attrs['speech_start_ts']),
                                     'end_speech': np.int16(h5[key].attrs['speech_end_ts']),
                                     'features': np.array(h5[key][()], dtype=np.float32)})

        return self.dataset

if __name__ == '__main__':
    data_file = 'data/test.h5'
    dataloader = HeySnipsDataset(data_file, 32)