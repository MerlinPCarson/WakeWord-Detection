import numpy as np
from tensorflow.keras.utils import Sequence
import librosa

class HeySnipsPreprocessed(Sequence):
  def __init__(self, list_IDs, labels, batch_size,
               dim, n_classes=2, shuffle=True, n_channels=1):
    self.dim = dim
    self.batch_size = batch_size
    self.labels = labels
    self.list_IDs = list_IDs
    self.n_classes = n_classes
    self.shuffle = shuffle
    self.n_channels = n_channels
    self.on_epoch_end()

  # Shuffles data after every epoch.
  def on_epoch_end(self):
    self.indexes = np.arange(len(self.list_IDs))
    if self.shuffle == True:
        np.random.shuffle(self.indexes)

  # Generates data w/ batch_size samples.
  def __data_generation(self, list_IDs_temp):
    # Initialization
    X = np.zeros((self.batch_size, *self.dim, self.n_channels))
    y = np.empty((self.batch_size), dtype=int)

    # Generate data
    for i, ID in enumerate(list_IDs_temp):
        # Store sample, cut to only include number of frames
        # to fit expected time dimension.
        try:
          X_i = np.load(ID).T[:,:self.dim[1]]
          X_i = np.pad(X_i, pad_width=((0,0), (0, self.dim[1]-X_i.shape[1])), mode="constant", constant_values=(0,0))
          X[i,] = np.expand_dims(X_i, -1)
        except IndexError as e:
          print(e)
          print(ID)
        # Store class
        y[i] = self.labels[ID]

    return X, y

  # Supplies number of batches per epoch.
  def __len__(self):
    return int(np.floor(len(self.list_IDs) / self.batch_size))

  # Generate a batch of data.
  def __getitem__(self, index):
    # Generate indexes of the batch
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

    # Find list of IDs
    list_IDs_temp = [self.list_IDs[k] for k in indexes]

    # Generate data
    X, y = self.__data_generation(list_IDs_temp)
    return X, y


# Dataloader for HeySnips which generates features
# on the fly. *Not yet thoroughly tested!*
class HeySnips(Sequence):
  def __init__(self, list_IDs, labels, batch_size,
               dim, n_classes=2, shuffle=True, n_channels=1):
    self.dim = dim
    self.batch_size = batch_size
    self.labels = labels
    self.list_IDs = list_IDs
    self.n_classes = n_classes
    self.shuffle = shuffle
    self.n_channels = n_channels
    self.on_epoch_end()

  # Shuffles data after every epoch.
  def on_epoch_end(self):
    self.indexes = np.arange(len(self.list_IDs))
    if self.shuffle == True:
        np.random.shuffle(self.indexes)

  # Generates data w/ batch_size samples.
  def __data_generation(self, list_IDs_temp):
    # Initialization
    X = np.empty((self.batch_size, *self.dim, self.n_channels))
    y = np.empty((self.batch_size), dtype=int)

    # Generate data
    for i, ID in enumerate(list_IDs_temp):
        # Store sample
        X[i,] = np.load(ID).T
        testwav, fs = librosa.load("ID",sr=None,duration=1.5)
        frame_length = 10
        sample_window = int(frame_length / 1000 * fs)
        X[i,] = librosa.feature.melspectrogram(testwav, fs,
                                        hop_length = sample_window,
                                        win_length = sample_window,
                                        center = True,
                                        pad_mode = "constant",
                                        n_mels=40)
        # Store class
        y[i] = self.labels[ID]
    return X, y

  # Supplies number of batches per epoch.
  def __len__(self):
    return int(np.floor(len(self.list_IDs) / self.batch_size))

  # Generate a batch of data.
  def __getitem__(self, index):
    # Generate indexes of the batch
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

    # Find list of IDs
    list_IDs_temp = [self.list_IDs[k] for k in indexes]

    # Generate data
    X, y = self.__data_generation(list_IDs_temp)
    return X, y

if __name__ == "__main__":
    import glob
    from itertools import chain

    INPUT_SHAPE_FRAMES = 151
    INPUT_SHAPE_FEATURES = 40
    BATCH_SIZE = 64
    data_path = "/content/machine-learning-data/wakeword_data_preprocessed"

    partition={}
    partition['train'] = glob.glob(data_path + "/train/*/*.npy")
    partition['test'] = glob.glob(data_path + "/test/*/*.npy")
    partition['dev'] = glob.glob(data_path + "/dev/*/*.npy")

    all_ids = list(chain(*[ids for ids in partition.values()]))
    labels = {id: (0 if "not-hey-snips" in id else 1) for id in all_ids}

    training_generator = HeySnipsPreprocessed(partition['train'], labels, batch_size=BATCH_SIZE,
                                              dim=(INPUT_SHAPE_FRAMES, INPUT_SHAPE_FEATURES))
    dev_generator = HeySnipsPreprocessed(partition['dev'], labels, batch_size=BATCH_SIZE,
                                         dim=(INPUT_SHAPE_FRAMES, INPUT_SHAPE_FEATURES))
    test_generator = HeySnipsPreprocessed(partition['test'], labels, batch_size=BATCH_SIZE,
                                          dim=(INPUT_SHAPE_FRAMES, INPUT_SHAPE_FEATURES))