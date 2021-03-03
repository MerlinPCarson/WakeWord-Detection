''' Module to enhance hey-snips dataset
    by removing silence at onset and offset,
    and add additional negative samples by
    grabbing segments of positive samples
    and replacing with other speech or
    silence, to limit FA's based on partial
    keyword matches.
'''

import os
import sys
import time
import json
import argparse
import librosa
import shutil
import numpy as np
import soundfile as sf

from tqdm import tqdm
from pathlib import Path
from typing import Any
from itertools import groupby
from matplotlib import pyplot as plt

import webrtcvad

# Class to perform preprocessing on hey-snips
# audio files.
class Dataset_Preprocessor:
    def __init__(self,
                 **kwargs: Any) -> None:

        self.sr = kwargs['sample_rate']
        self.out_dir = kwargs['out_dir']
        self.data_dir = kwargs['data_dir']
        self.train_meta = json.load(open(os.path.join(self.data_dir, "train.json"), 'r'))
        self.dev_meta = json.load(open(os.path.join(self.data_dir, "dev.json"), 'r'))
        self.test_meta = json.load(open(os.path.join(self.data_dir, "test.json"), 'r'))
        self.debug = kwargs['examine_audio']

        if not Path(os.path.join(self.out_dir, "audio_files")).exists():
            os.makedirs(os.path.join(self.out_dir, "audio_files"))
        self.vad = webrtcvad.Vad(2)

    # ----------------------------------------------------------- *
    # Use VAD to isolate speech in dataset, save out portion
    # of audio files which contain speech to self.out_dir. Updated
    # metadata .json files saved out as well, to same directory.
    def isolate_speech(self):
        vad_frame_ms = 10
        vad_frame = self.sr * vad_frame_ms // 1000
        splits = [("train", self.train_meta),
                  ("dev", self.dev_meta),
                  ("test", self.test_meta)]

        # Run through all audio files.
        for split, meta in splits:
            discarded = 0
            check_index = 0
            new_meta = []
            for i, file in enumerate(tqdm(meta, leave=True)):
                check_index += 1
                new_path = os.path.join(self.out_dir, file['audio_file_path'])

                # Load audio.
                samples, _ = librosa.load(os.path.join(self.data_dir, file['audio_file_path']), sr=self.sr)

                # Make sure file contains audio.
                if len(samples) > 0:
                    # Chop in to short frames.
                    frame_timepoints = np.arange(0, len(samples), vad_frame)
                    frames = [samples[index:index + vad_frame] for index in frame_timepoints]
                    # Pad last frame with zeros.
                    frames[-1] = np.pad(frames[-1], (0, vad_frame - len(frames[-1])))
                    frames_bytes = [np.int16(frame * 32768).tobytes() for frame in frames]
                    frames_speech = [self.vad.is_speech(frame_bytes, self.sr) for frame_bytes in frames_bytes]

                    # Remove implausably short speech chunks
                    # at onset or end of audio. (Adhoc chose
                    # less than 36 frames, which would be
                    # chunks <= 350 ms.
                    new_frames_speech = []
                    for chunk in groupby(frames_speech):
                        chunk = list(chunk[1])
                        if chunk[0] is True and len(chunk) <= 35:
                            new_frames_speech.extend([False for i in range(len(chunk))])
                        else:
                            new_frames_speech.extend(chunk)
                    assert len(new_frames_speech) == len(frames_speech)

                    if self.debug:
                        # Periodically inspect output.
                        if check_index % 2000 == 0:
                            self.examine_audio(file['audio_file_path'], samples, new_frames_speech, frame_timepoints,
                                        play_sound=False, plot=True)

                # Use results of VAD to chop out initial and final silence, leaving one
                # frame buffer on either side.
                try:
                    speech_start = frame_timepoints[new_frames_speech.index(True)-1]
                    speech_end = frame_timepoints[-(list(reversed(new_frames_speech)).index(True)+1)]
                    speech_samples = samples[speech_start:speech_end]
                    sf.write(new_path, speech_samples, self.sr)

                    # Update duration in metadata.
                    file['duration'] = len(speech_samples) / self.sr

                    new_meta.append(file)
                # Discard files for which no speech was found.
                except Exception as e:
                    if self.debug:
                        self.examine_audio(file['audio_file_path'], samples, new_frames_speech, frame_timepoints,
                                        play_sound=False, plot=True)
                    discarded += 1
                    print("discarding ", file['audio_file_path'])
                    #meta.remove(file)
            print("discarded", discarded)

            # Write out updated metadata.
            with open(os.path.join(self.out_dir, split + ".json"), 'w') as outfile:
                json.dump(new_meta, outfile, indent=4)

    # ----------------------------------------------------------- *
    # Take a look at which frames of an audio file have been
    # detected as to contain speech.
    def examine_audio(self, path, samples, is_speech, frames,
                      play_sound=True, plot=True):
        if play_sound:
            import pydub
            import simpleaudio
            audio = pydub.AudioSegment.from_wav(os.path.join(self.data_dir, path))
            playback = simpleaudio.play_buffer(
                audio.raw_data,
                num_channels=audio.channels,
                bytes_per_sample=audio.sample_width,
                sample_rate=audio.frame_rate
            )
            playback.wait_done()

        if plot:
            fig = plt.figure(figsize=(8, 4))
            plt.title(path)
            plt.plot(range(len(samples)), samples)
            if is_speech is not None and plot:
                for index, label in enumerate(is_speech):
                    if label is True and index+1 < len(frames):
                        plt.axvspan(frames[index],
                                    frames[index+1],
                                    alpha=0.1)
            plt.ylabel("Amplitude", size=12)
            plt.xlabel("Samples", size=12)
            plt.show()
            plt.close()

    # ----------------------------------------------------------- *
    # Enhance training set by replacing chunks of onset/offset
    # of positive samples with speech from negative samples,
    # or silence/noise. Pass path to dataset samples to be enhanced
    # if different from original path designated when creating class.
    def enhance_train_set(self, original_path=None):
        np.random.seed(42)
        enhanced_out_path = os.path.join(self.out_dir, "enhanced_train_negative")
        enhanced_meta = []
        if not Path(enhanced_out_path).exists():
            os.makedirs(enhanced_out_path)

        if original_path is None:
            original_path=self.data_dir

        negative_files = [file['audio_file_path'] for file in self.train_meta if not file['is_hotword'] and
                                                            Path(os.path.join(original_path, file['audio_file_path'])).exists()]
        for file in tqdm(self.train_meta, leave=True):
            if file['is_hotword'] and Path(os.path.join(original_path, file['audio_file_path'])).exists():
                #offset = np.random.randint(0,2) # 0: onset, 1: offset
                offset = 1 # For now, just replace offsets.
                percentage = np.random.uniform(0.45,0.6) # How much to remove/replace.
                speech = np.random.randint(0,2) # 0: remove, 1: replace with speech
                pos_samples, _ = librosa.load(os.path.join(original_path, file['audio_file_path']), sr=self.sr)
                if pos_samples.size == 0:
                    continue
                num_samples_to_remove = int(len(pos_samples) * percentage)

                if speech:
                    neg_samples = np.array([])
                    # Some samples are empty.
                    while neg_samples.size == 0:
                        replacement_index = np.random.randint(len(negative_files))  # choose random negative sample
                        replacement_file = negative_files[replacement_index]
                        neg_samples, _ = librosa.load(os.path.join(original_path, replacement_file), sr=self.sr)

                    # Construct informative filename.
                    if offset:
                        save_path = "pos_" + file['id'] + "_neg_" + Path(replacement_file).stem
                    else:
                        save_path = "neg_" + Path(replacement_file).stem + "_pos_" + file['id']
                else:
                    neg_samples = np.zeros(num_samples_to_remove)

                    # Construct informative filename.
                    if offset:
                        save_path = "pos_" + file['id'] + "_neg_silence"
                    else:
                        save_path = "neg_silence_pos_" + file['id']

                save_path += "__" + str(int(percentage*100)) + ".wav"

                if offset:
                    pos_samples_reduced = pos_samples[:-num_samples_to_remove]
                    new_neg_samples = np.append(pos_samples_reduced,neg_samples[-num_samples_to_remove:])
                else:
                    pos_samples_reduced = pos_samples[num_samples_to_remove:]
                    new_neg_samples = np.append(neg_samples[:num_samples_to_remove],pos_samples_reduced)
                sf.write(os.path.join(self.out_dir, "enhanced_train_negative", save_path), new_neg_samples, self.sr)

                # Add metadata to be written out as json file.
                enhanced_meta.append({'duration': len(new_neg_samples) / self.sr,
                                      'worker_id': 'n_a',
                                      'audio_file_path': os.path.join("enhanced_train_negative", save_path),
                                      'id': Path(save_path).stem,
                                      'is_hotword': 0})
        # Write out metadata.
        with open(os.path.join(self.out_dir, "train_enhanced.json"), 'w') as outfile:
            json.dump(enhanced_meta, outfile, indent=4)


    def reload_metadata(self, dir):
        self.train_meta = json.load(open(os.path.join(dir, "train.json"), 'r'))
        self.dev_meta = json.load(open(os.path.join(dir, "dev.json"), 'r'))
        self.test_meta = json.load(open(os.path.join(dir, "test.json"), 'r'))

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocesses audio data by isolating speech and '
                                                 'enhancing train samples')
    parser.add_argument('--data_dir', type=str, default='data/hey_snips_research_6k_en_train_eval_clean_ter',
                        help='Directory with Hey Snips raw dataset')
    parser.add_argument('--out_dir', type=str, default='data/data_speech_isolated/', help='Directory to save enhanced datasets to')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate for audio (Hz)')
    parser.add_argument('--examine_audio', default=False, action='store_true', help='Flag to examine problematic audio clips')
    args = parser.parse_args()

    return args

def main(args) -> int:
    start = time.time()

    dataset_preprocessor = Dataset_Preprocessor(**vars(args))
    dataset_preprocessor.isolate_speech()
    dataset_preprocessor.reload_metadata(args.out_dir)
    dataset_preprocessor.enhance_train_set(original_path=args.out_dir)

    print(f'Script completed in {time.time() - start:.2f} secs')

    return 0


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
