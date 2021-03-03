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

        # Load features from h5 files.
        self.dataset, self.ids = self.load_data()
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
            y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample, cut to only include number of frames
            # to fit expected time dimension.
            X_i = self.dataset[ID]['features'].T[:,:self.frame_num]
            X_i = np.pad(X_i, pad_width=((0,0), (0, self.frame_num-X_i.shape[1])), mode="constant", constant_values=(0,0))
            X[i,] = np.expand_dims(X_i, -1)

            # Store class
            label = self.dataset[ID]['label']
            if self.ctc:
                # If this is a wakeword.
                if label == 1:
                    label = self.char2num(tf.strings.unicode_split("nwn", input_encoding="UTF-8"))
                # If it is not.
                else:
                    label = self.char2num(tf.strings.unicode_split("nnn", input_encoding="UTF-8"))
            y[i] = label

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


if __name__ == "__main__":

    INPUT_SHAPE_FRAMES = 151
    INPUT_SHAPE_FEATURES = 40
    BATCH_SIZE = 64
    data_path = "/Users/amie/Desktop/OHSU/CS606 - Deep Learning II/FinalProject/spokestack-python/data_isolated_enhanced/"

    dev_generator = HeySnipsPreprocessed([data_path + "dev.h5"], batch_size=BATCH_SIZE,
                                              frame_num=INPUT_SHAPE_FRAMES, feature_num=INPUT_SHAPE_FEATURES)
    test_generator = HeySnipsPreprocessed([data_path + "test.h5"], batch_size=0,
                                              frame_num=INPUT_SHAPE_FRAMES, feature_num=INPUT_SHAPE_FEATURES)
    training_generator = HeySnipsPreprocessed([data_path + "train.h5", data_path + "train_enhanced.h5"],
                                              batch_size=BATCH_SIZE, frame_num=INPUT_SHAPE_FRAMES,
                                                                     feature_num=INPUT_SHAPE_FEATURES)