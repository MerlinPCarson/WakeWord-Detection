import os
import sys
import time
import json
import argparse
import librosa
import numpy as np

from tqdm import tqdm
from glob import glob
from typing import Any
from tf_lite.filter import Filter
from tf_lite.tf_lite import TFLiteModel


class Dataset_Filter:
    
    def __init__(self,
                 dataset: str,
                 filter: TFLiteModel,
                 **kwargs: Any) -> None:

        # dataset variables
        self.dataset = dataset
        self.audio_metadata = json.load(open(dataset, 'r'))
        self.wake_word = kwargs['wake_word']

        # filter class variables
        self.filter = Filter(model_dir=args.models_dir)
        self.num_filter_outputs = self.filter.num_outputs()

        # audio parameters
        self.sr = kwargs['sample_rate']
        self.fw = kwargs['frame_width']
        self.frame_len = self.sr // 1000 * self.fw

        # data locations
        self.out_dir = kwargs['out_dir']
        self.data_dir = kwargs['data_dir']

        # make directory structure for dataset
        self.dataset_name = os.path.basename(dataset).replace('.json', '')
        self.dataset_dir = os.path.join(self.out_dir, self.dataset_name)
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(os.path.join(self.dataset_dir, self.wake_word), exist_ok=True)
        os.makedirs(os.path.join(self.dataset_dir, f'not-{self.wake_word}'), exist_ok=True)

    def filter_audio_file(self, audio_file: str) -> None:

        features = []

        # load audio from file 
        samples, _ = librosa.load(os.path.join(self.data_dir, audio_file), sr=self.sr)

        # frame audio and process it through filter
        for start_idx in np.arange(0, len(samples), self.frame_len):
            frame = samples[start_idx:start_idx+self.frame_len]
            if len(frame) < self.frame_len:
                pad_len = self.frame_len - len(frame)
                frame = np.pad(frame, (0,pad_len), mode='constant')
            
            frame = self.filter.filter_frame(frame)

            # if frame buffer is not full, filter cannot do overlapping windows, so nothing is returned
            if frame is not None:
                features.append(frame.squeeze())

        return np.array(features)

    def filter_dataset_audio(self) -> None:

        for audio in tqdm(self.audio_metadata):
            # pass audio file through filter model
            features = self.filter_audio_file(audio['audio_file_path']) 

            # save output features to file based on if it is a wake word or not
            out_file = os.path.basename(audio['audio_file_path']).replace('.wav', '.flt')
            if audio['is_hotword']:
                out_file = os.path.join(self.dataset_dir, self.wake_word, out_file)
            else:
                out_file = os.path.join(self.dataset_dir, f'not-{self.wake_word}', out_file)

            np.save(out_file, features)


def parse_args():
    parser = argparse.ArgumentParser(description='Builds and saves dataset arrays from Hey Snips audio data')
    parser.add_argument('--models_dir', type=str, default='tf_lite', help='directory with TF-Lite filter model')
    parser.add_argument('--data_dir', type=str, default='data/hey_snips_research_6k_en_train_eval_clean_ter', help='Directory with Hey Snips raw dataset')
    parser.add_argument('--out_dir', type=str, default='data', help='Directory to save datasets to')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate for audio (Hz)')
    parser.add_argument('--frame_width', type=int, default=20, help='Frame width for audio in (ms)')
    parser.add_argument('-wake_word', type=str, default='hey-snips', help='Wake work in dataset')
    args = parser.parse_args()

    return args

def main(args) -> int:
    start = time.time()

    filter = Filter(model_dir=args.models_dir)


    # load, filter and save each dataset to .h5 file
    for dataset in glob(os.path.join(args.data_dir, '*.json')):
        print(dataset)
        dataset_filter = Dataset_Filter(dataset, filter, **vars(args))
        dataset_filter.filter_dataset_audio()

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
