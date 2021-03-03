import h5py
import numpy as np
from tqdm import tqdm
import tensorflow as tf


class HeySnipsDataset(tf.keras.utils.Sequence):
    def __init__(self, data_file, batch_size=32, num_features=40, 
                 timesteps=182, shuffle=False, *args, **kwargs):

        # state variables
        self.num_features = num_features
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.wakeword_window_step = 2 
        self.not_wakeword_window_step = 30 
        self.epoch_data_idx = 0 
        self.len_example = 0
        self.window_step = 0
        self.window_start = 0
        self.window_num = 0
        self.label_delta = 15
        self.shuffle = shuffle

        # dataset pre-fetch
        self.dataset, self.num_wakewords = self.preload_data(data_file)

        # calculate the total number of training examples
        self.num_examples = self.calc_sliding_windows()

        # calculate number of batches
        #self.num_batches = len(self.dataset) // batch_size
        self.num_batches = (self.num_examples // batch_size) - 1 

    def __len__(self):
        # returns the number of batches
        return self.num_batches 

    def __getitem__(self, index):
        # returns one batch

        # find which segments make up the batch
        #batch_start = self.batch_size * index 
        #batch_idxs = np.arange(batch_start, batch_start + self.batch_size)

        X = []
        y = []

        data_idx = self.epoch_data_idx
        batch_len = 0
        while batch_len < self.batch_size:

            if self.window_start == 0:
                # determine number of timesteps between subsequent examples in batch 
                # don't slide over negative examples
                if self.dataset[data_idx]['label'] == 0:
                    window_start = int(self.dataset[data_idx]['start_speech'])
                    #X.append(self.dataset[data_idx]['features'][window_start:window_start+self.timesteps])
                    X.append(self.dataset[data_idx]['features'][0:182])
                    y.append(0)
                    data_idx += 1
                    batch_len += 1
                    continue
                else:
                    # determine number of windows in example
                    self.window_step = self.wakeword_window_step
                    self.len_example = len(self.dataset[data_idx]['features'])
                    self.num_windows = ((self.len_example - self.timesteps) // self.window_step) + 1


            # add windows to batch until end of example or batch is full
            while batch_len < self.batch_size and self.window_num < self.num_windows:
                window_end = self.window_start + self.timesteps
                X.append(self.dataset[data_idx]['features'][self.window_start:window_end])

                # if it's a wakeword and
                # end of window is +/- 15 frames from the end of the keyword, label it positive
                label = 0
                if self.dataset[data_idx]['label'] == 1:
                    if window_end - self.label_delta <= self.dataset[data_idx]['end_speech'] <= window_end + self.label_delta:
                        label = 1
    
                y.append(label)
    
                # increment number of examples and start of next window
                batch_len += 1
                self.window_num += 1
                self.window_start += self.window_step 
    
            if self.window_num >= self.num_windows:
                # move to next example in dataset
                data_idx += 1
                self.window_num = 0
                self.window_start = 0
    

#        # load batch features and labels 
#        X = [self.dataset[i]['features'] for i in batch_idxs]
#        y = np.array([self.dataset[i]['label'] for i in batch_idxs])
#
#        # make each batch uniform length
#        X = self.pad_features(X)
        self.epoch_data_idx = data_idx
        return np.array(X), np.array(y)

    def calc_sliding_windows(self):

        num_examples = 0
        for audio in self.dataset:
            if audio['label'] == 1:
                window_step = self.wakeword_window_step
                # determine number of windows in example
                len_example = len(audio['features'])
                num_windows = ((len_example - self.timesteps) // window_step) + 1
                num_examples += num_windows
    
            else:
                num_examples += 1

        return num_examples

    def prune_wakewords(self, keep_ratio):

        # find indexes of all wakewords in dataset
        wakeword_idxs = []
        for i in range(len(self.dataset)):
            if self.dataset[i]['label'] == 1:
                wakeword_idxs.append(i)

        # move wakewords to new datastruct
        # must be done in reversed order or idxs won't match up as they are removed
        wakewords = []
        for i in reversed(wakeword_idxs):
            # copy example to new list
            wakewords.append(self.dataset[i])

            # delete example from original dataset to not keep duplicates
            del self.dataset[i]

        # shuffle so removal is not order dependent
        np.random.shuffle(wakewords)

        # discard 1-keep_ratio of wakeword examples from datset
        self.dataset = self.dataset + wakewords[:int(keep_ratio*self.num_wakewords)]

        # reshuffle dataset so wakewords are not at the end
        np.random.shuffle(self.dataset)

        # set new number of batchs in dataset
        self.num_batches = len(self.dataset) // self.batch_size

    def get_labels(self):
        assert not self.shuffle, "Order may not be correct due to shuffling being enabled"
        return [self.dataset[i]['label'] for i in range(self.number_of_examples())]

    def get_filenames(self):
        assert not self.shuffle, "Order may not be correct due to shuffling being enabled"
        return [self.dataset[i]['file_name'] for i in range(self.number_of_examples())]

    def number_of_examples(self):
        # returns number of examples in dataset
        return len(self.dataset)
    
    def number_of_wakewords(self):
        # returns number of wakewords in test set (does not account for pruning)
        return self.num_wakewords

    def on_epoch_end(self):
        # reset start position for dataset
        self.epoch_data_idx = 0 
        self.window_num = 0
        self.window_start = 0

        # option method to run some logic at the end of each epoch: e.g. reshuffling
        if self.shuffle:
            np.random.shuffle(self.dataset)

    def pad_features(self, X):

        # create new datastruct of max length timesteps 
        max_length = max([x.shape[0] for x in X])
        X_padded = np.zeros((len(X), max_length, X[0].shape[-1]), dtype=np.float32)

        # pad features
        for idx, features in enumerate(X):
            features = np.expand_dims(features, 0)
            X_padded[idx, :features.shape[1], :features.shape[2]] = features 
        return X_padded 

    def preload_data(self, data_file):

        self.dataset = []
        self.num_wakewords = 0
        print(f'pre-loading dataset from file {data_file}')
        with h5py.File(data_file, 'r') as h5:
            keys = list(h5.keys())
            for key in tqdm(keys):
                # don't load data with no detected speech
                if h5[key].attrs['speech_end_ts'] > 0:
                    self.dataset.append({'file_name': key, 
                                        'label': np.uint8(h5[key].attrs['is_hotword']),
                                        'start_speech': np.int16(h5[key].attrs['speech_start_ts']),
                                        'end_speech': np.int16(h5[key].attrs['speech_end_ts']),
                                        'features': np.array(h5[key][()], dtype=np.float32)})

                    # pad examples that are shorter than the input size
                    if len(self.dataset[-1]['features']) < self.timesteps:
                        #pad_len = self.timesteps - len(self.dataset[-1]['features'])
                        #self.dataset[-1]['features'] = np.pad(self.dataset[-1]['features'], (0,pad_len))
                        padded = np.zeros((self.timesteps, self.num_features), dtype=np.float32)
                        padded[:len(self.dataset[-1]['features']),:] = self.dataset[-1]['features']
                        self.dataset[-1]['features'] = padded

                    # count number of wakewords in dataset
                    if h5[key].attrs['is_hotword'] == 1:
                        self.num_wakewords += 1

        return self.dataset, self.num_wakewords

if __name__ == '__main__':
    # Test Loader
    data_file = 'data/dev.h5'
    dataloader = HeySnipsDataset(data_file)
    #X,y = dataloader[86]
    X,y = dataloader[0]
    X,y = dataloader[1]
    print(f'Data: \n {X}')
    print(f'Labels: \n {y}')
