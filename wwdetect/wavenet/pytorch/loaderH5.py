import json
import os
import h5py
import random

import numpy as np
from tqdm import tqdm


class Loader_HeySnips:
    def __init__(self, h5_path, batch_size=32):

        self.dataset = self.preload_data(h5_path)

        self.batch_size = batch_size
        self.batch_idx = 0
        self.num_batches = len(self.dataset) // batch_size

    def get_batch(self):
        if self.batch_idx % self.num_batches:
            self.shuffle_samples()

        batch_start = self.batch_size * self.batch_idx
        batch_idxs = np.arange(batch_start, batch_start + self.batch_size)
        X = [self.dataset[i]['features'] for i in batch_idxs]
        X = self.pad_features(X)
        y = np.array([self.dataset[i]['label'] for i in batch_idxs])
        self.batch_idx += 1
        return X, y 

    def pad_features(self, X):
        max_length = max([x.shape[0] for x in X])
        X_padded = np.zeros((len(X), max_length, X[0].shape[-1]), dtype=np.float32)
        for idx, features in enumerate(X):
            features = np.expand_dims(features, 0)
            X_padded[idx, :features.shape[1], :features.shape[2]] = features 
        return X_padded 

    def shuffle_samples(self):
        random.shuffle(self.dataset)
        self.batch_idx = 0

    def preload_data(self, data_file):

        self.dataset = []
        print(f'pre-loading dataset from file {data_file}')
        with h5py.File(data_file, 'r') as h5:
            keys = list(h5.keys())
            for key in tqdm(keys):
                self.dataset.append({'file_name': key, 
                                     'label': np.uint8(h5[key].attrs['is_hotword']),
                                     'start_speech': np.int16(h5[key].attrs['speech_start_ts']),
                                     'end_speech': np.int16(h5[key].attrs['speech_end_ts']),
                                     'features': np.array(h5[key][()], dtype=np.float32)})

        return self.dataset
