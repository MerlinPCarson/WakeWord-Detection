import os
import sys
import time
import json
import h5py
import argparse
import librosa
import numpy as np

from tqdm import tqdm
from glob import glob
from typing import Any
from tf_lite.filter import Filter
from tf_lite.tf_lite import TFLiteModel

import webrtcvad


class Dataset_Filter:
    
    def __init__(self,
                 dataset: str,
                 filter: TFLiteModel,
                 **kwargs: Any) -> None:

        # dataset variables
        self.dataset = dataset
        self.audio_metadata = json.load(open(dataset, 'r'))
        self.wake_word = kwargs['wake_word']

        # audio parameters
        self.sr = kwargs['sample_rate']
        self.fw = kwargs['frame_width']
        self.hw = kwargs['hop_width']
        self.frame_len = self.sr // 1000 * self.fw
        self.hop_len = self.sr // 1000 * self.hw

        # filter class variables
        self.filter = Filter(fft_hop_length=self.hw, model_dir=args.models_dir)
        self.num_filter_outputs = self.filter.num_outputs()

        # data locations
        self.out_dir = kwargs['out_dir']
        self.data_dir = kwargs['data_dir']

        # make directory structure for dataset
        os.makedirs(self.out_dir, exist_ok=True)
        self.dataset_file = os.path.join(self.out_dir, os.path.basename(dataset).replace('.json', '.h5'))

        # voice activity detector (0=lowest aggresiveness, 3=most agressive)
        self.vad = webrtcvad.Vad(3)

    def filter_audio_file(self, audio_file: str, label: int) -> None:

        features = []

        # load audio from file 
        samples, _ = librosa.load(os.path.join(self.data_dir, audio_file), sr=self.sr)

        # if wav file is empty, return None
        if len(samples) > 0:

            # start and end timesteps for voice in audio clip
            speech_start_ts = -1 
            speech_end_ts = -1 

            # frame audio and process it through filter
            for start_idx in np.arange(0, len(samples), self.frame_len):
                frame = samples[start_idx:start_idx+self.frame_len]
                if len(frame) < self.frame_len:
                    pad_len = self.frame_len - len(frame)
                    frame = np.pad(frame, (0,pad_len), mode='constant')

                # convert frame to bytes for WEBRTCVAD
                frame_bytes = np.int16(frame * 32768).tobytes()
                is_speech = self.vad.is_speech(frame_bytes, self.sr)

                # find timestep where speech starts
                if speech_start_ts == -1 and is_speech:
                    speech_start_ts = start_idx // self.hop_len

                ## find timestep where speech ends 
                if speech_start_ts > -1 and is_speech:
                    speech_end_ts = (start_idx + self.frame_len) // self.hop_len

                # filter audio through filter model                
                frame = self.filter.filter_frame(frame)

                # if frame buffer is not full, filter cannot do overlapping windows, so nothing is returned
                if len(frame) > 0:
                    features.extend(frame)

            #if (speech_start_ts == -1 or speech_end_ts == -1) and label==1:
            #    print(f'Error finding begining and ending of speech in: {audio_file}')

            return {'file_name': os.path.basename(audio_file).replace('.wav',''),
                    'is_hotword': label,
                    'features': np.array(features), 
                    'speech_start_ts': speech_start_ts, 
                    'speech_end_ts': speech_end_ts
                    }

        return None

    def filter_dataset_audio(self) -> None:

        audio_clips = []
        # process all audio files in dataset's json file
        for audio in tqdm(self.audio_metadata):
            # pass audio file through filter model
            audio_clip = self.filter_audio_file(audio['audio_file_path'], audio['is_hotword']) 

            # dont save empty feature maps (i.e. the audio file had too few samples)
            if audio_clip is None or len(audio_clip['features']) == 0:
                continue

            audio_clips.append(audio_clip)

        self.write_h5(audio_clips)

    def write_h5(self, audio_clips):

        print(f"Writing preprocessed dataset to {self.dataset_file}")
        with h5py.File(self.dataset_file, 'w') as h5f:
            for audio_clip in audio_clips:
                dset = h5f.create_dataset(audio_clip['file_name'], data=audio_clip['features'])
                dset.attrs['is_hotword'] = audio_clip['is_hotword']
                dset.attrs['speech_start_ts'] = audio_clip['speech_start_ts']
                dset.attrs['speech_end_ts'] = audio_clip['speech_end_ts']


def parse_args():
    parser = argparse.ArgumentParser(description='Builds and saves dataset arrays from Hey Snips audio data')
    parser.add_argument('--models_dir', type=str, default='tf_lite', help='directory with TF-Lite filter model')
    parser.add_argument('--data_dir', type=str, default='data/hey_snips_research_6k_en_train_eval_clean_ter', help='Directory with Hey Snips raw dataset')
    parser.add_argument('--out_dir', type=str, default='data', help='Directory to save datasets to')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate for audio (Hz)')
    parser.add_argument('--frame_width', type=int, default=20, help='Frame width for audio in (ms)')
    parser.add_argument('--hop_width', type=int, default=10, help='Hop width for audio in (ms)')
    parser.add_argument('-wake_word', type=str, default='hey-snips', help='Wake work in dataset')
    args = parser.parse_args()

    return args

def main(args) -> int:
    start = time.time()

    filter = Filter(model_dir=args.models_dir)


    # load, filter and save features of each audio file in dataset
    for dataset in glob(os.path.join(args.data_dir, '*.json')):
        print(f"Loading and preprocessing {os.path.basename(dataset).replace('.json', '')} dataset using metadata from {dataset}")
        dataset_filter = Dataset_Filter(dataset, filter, **vars(args))
        dataset_filter.filter_dataset_audio()

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
