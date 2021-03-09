from copy import deepcopy
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras import layers
from tqdm import tqdm
import h5py

class HeySnipsPreprocessed(Sequence):
    def __init__(self, h5_paths, batch_size,
               frame_num, feature_num, n_classes=2,
               shuffle=True, n_channels=1, ctc=False):
        self.frame_num = frame_num
        self.feature_num = feature_num
        self.h5_paths = h5_paths
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.n_channels = n_channels
        self.ctc = ctc
        self.wakeword_ids_all = None

        # Load features from h5 files.
        self.dataset, self.ids = self.load_data()
        self.original_dataset = deepcopy(self.dataset)
        self.original_ids = deepcopy(self.ids)

        # find ID's of all wakewords in original, non-pruned dataset
        # shuffle once
        self.wakeword_ids_all = [id for id in self.original_ids
                                if self.original_dataset[id]['label'] == 1]
        np.random.shuffle(self.wakeword_ids_all)

        self.on_epoch_end()

        if batch_size == 0:
            batch_size = len(self.ids)
        self.batch_size = batch_size

        if ctc:
            self.char2num = layers.experimental.preprocessing.StringLookup(
                vocabulary=list("nw"), num_oov_indices=0, mask_token=None)

            self.num2char = layers.experimental.preprocessing.StringLookup(
                vocabulary=self.char2num.get_vocabulary(), mask_token=None, invert=True)

    # Shuffles data after every epoch.
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    # Generates data w/ batch_size samples.
    def __data_generation(self, list_IDs_temp):
        # Initialization
        X = np.zeros((self.batch_size, self.feature_num, self.frame_num, self.n_channels))
        if self.ctc:
            y = np.empty((self.batch_size,3), dtype=int)
        else:
            y = np.zeros((self.batch_size,2), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample, cut to only include number of frames
            # to fit expected time dimension.
            X_i = self.dataset[ID]['features'].T[:,:self.frame_num]
            X_i = np.pad(X_i, pad_width=((0,0), (0, self.frame_num-X_i.shape[1])), mode="constant", constant_values=(0,0))
            X[i,] = np.expand_dims(X_i, -1)

            # Store class.
            if self.ctc:
                # If this is a wakeword.
                if label == 1:
                    y[i] = self.char2num(tf.strings.unicode_split("sws", input_encoding="UTF-8"))
                # If it is not.
                else:
                    y[i] = self.char2num(tf.strings.unicode_split("sns", input_encoding="UTF-8"))

            else:
                y[i][self.dataset[ID]['label']] = 1

        return X, y

    # Supplies number of batches per epoch.
    def __len__(self):
        return int(np.floor(len(self.ids) / self.batch_size))

    # Generate a batch of data.
    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.ids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def load_data(self):
        dataset = {}
        ids = []
        for path in self.h5_paths:
            with h5py.File(path, 'r') as h5:
                keys = list(h5.keys())
                for key in tqdm(keys):
                    assert dataset.get(key) is None

                    dataset[key] = {'label': np.uint8(h5[key].attrs['is_hotword']),
                                    'start_speech': np.int16(h5[key].attrs['speech_start_ts']),
                                    'end_speech': np.int16(h5[key].attrs['speech_end_ts']),
                                    'features': np.array(h5[key][()], dtype=np.float32)}
                    ids.append(key)
        return dataset, ids

    def num_samples(self):
        number_ww = [id for id in self.ids if self.dataset[id]['label'] == 1]
        number_other = len(self.ids) - len(number_ww)
        return len(number_ww), number_other

    # Adapted from WaveNet dataloader.
    def prune_wakewords(self, keep_ratio):
        num_kept = len(self.wakeword_ids_all)
        num_removed = 0
        if keep_ratio != 1.0:

            # Chop off portion of wakewords. If chopped in a previous
            # pruning, and this round is the same keep ratio or less,
            # same words will be pruned. This seems like the most
            # fair comparison.
            num_kept = int(len(self.wakeword_ids_all)*keep_ratio)
            wakewords_to_remove = self.wakeword_ids_all[num_kept:]

            # Copy original dataset into current dataset, and
            # remove the wakewords being removed. Remove from
            # ID list as well.
            self.dataset = deepcopy(self.original_dataset)
            self.ids = deepcopy(self.original_ids)
            for id in wakewords_to_remove:
                del self.dataset[id]
                self.ids.remove(id)
                num_removed += 1

        return num_kept, num_removed

if __name__ == "__main__":

    INPUT_SHAPE_FRAMES = 151
    INPUT_SHAPE_FEATURES = 40
    BATCH_SIZE = 64
    data_path = "/Users/amie/Desktop/OHSU/CS606 - Deep Learning II/FinalProject/spokestack-python/data_speech_isolated/silero/"

    dev_generator = HeySnipsPreprocessed([data_path + "dev.h5"], batch_size=BATCH_SIZE,
                                              frame_num=INPUT_SHAPE_FRAMES, feature_num=INPUT_SHAPE_FEATURES)
    print(dev_generator.prune_wakewords(0.9))

    test_generator = HeySnipsPreprocessed([data_path + "test.h5"], batch_size=0,
                                              frame_num=INPUT_SHAPE_FRAMES, feature_num=INPUT_SHAPE_FEATURES)
    training_generator = HeySnipsPreprocessed([data_path + "train.h5", data_path + "train_enhanced.h5"],
                                              batch_size=BATCH_SIZE, frame_num=INPUT_SHAPE_FRAMES,
                                                                     feature_num=INPUT_SHAPE_FEATURES)