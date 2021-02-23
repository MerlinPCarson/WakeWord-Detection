'''
A script to evaluate and optionally compare performance
of model(s) designed to perform wakeword detection.
'''
import argparse
import sys
import json
import os
import pickle
import logging

from pydub import AudioSegment
from pathlib import Path
from tqdm import tqdm
import numpy as np
import librosa
from matplotlib import pyplot as plt

from spokestack.io.pyaudio import PyAudioInput
from spokestack.pipeline import SpeechPipeline
from spokestack.vad.webrtc import VoiceActivityDetector
from spokestack.wakeword.tflite import WakewordTrigger
from spokestack.activation_timeout import ActivationTimeout
from spokestack.models.tensorflow import TFLiteModel
from tf_lite.filter import Filter
from tf_lite.ring_buffer import RingBuffer

logging.basicConfig(level=logging.INFO)

# Currently not functional.
def evaluate_FRR_full_pipeline(model_type="CRNN", test_file="data/evaluation/hey_snips_2min_37_samples.wav"):
    audio = AudioSegment.from_wav(str(test_file))
    file = PyAudioInput(sample_rate=16000, frame_width=20, from_file=True, wav=audio)
    vad = VoiceActivityDetector(frame_width=20)
    wake = WakewordTrigger(model_dir=model_type , model_type=model_type, posterior_threshold=0.1)
    timeout = ActivationTimeout(frame_width=20, min_active=500, max_active=500)
    pipeline = SpeechPipeline(file, [vad,wake,timeout])
    pipeline.start()
    pipeline.run()


def get_posterior(models_dir, model_type, test_file, frame_width,
                  sample_rate):

    filter = Filter(model_dir=models_dir)
    encode_model: TFLiteModel = TFLiteModel(
        model_path=os.path.join(models_dir, "encode.tflite")
    )
    detect_model: TFLiteModel = TFLiteModel(
        model_path=os.path.join(models_dir, "detect.tflite")
    )

    encoder_len = encode_model.input_details[0]["shape"][1 if model_type == 'Wavenet' else 2]

    samples, _ = librosa.load(test_file, sr=sample_rate)
    window = []
    all_posterior = []
    frame_length = sample_rate // 1000 * frame_width

    # Pulled from filter_dataset.py to closely match training
    # preprocessing.
    print('Evaluating model')
    for start_idx in tqdm(np.arange(0, len(samples), frame_length)):
        frame = samples[start_idx:start_idx + frame_length]
        if len(frame) < frame_length:
            pad_len = frame_length - len(frame)
            frame = np.pad(frame, (0, pad_len), mode='constant')

        # filter audio through filter model
        frame = filter.filter_frame(frame)

        if len(frame) > 0:
            window.extend(frame)

        # Perform inference on current window.
        if len(window) >= encoder_len:
            # extract frames from window buffer
            frames = window[:encoder_len]
            # save remaining frames in window buffer
            window = window[encoder_len:]

            # inference for CRNN model
            if model_type == 'CRNN':
                x = np.expand_dims(np.array(frames).T, 0)
                x = np.expand_dims(x, -1)
                x = np.array(encode_model(x)).squeeze(0)
                x = detect_model(x)[0][0][0]

            # inference for Wavenet model
            elif model_type == 'Wavenet':
                x = np.expand_dims(np.array(frames), 0)
                x = np.array(encode_model(x)).squeeze(0)
                x = detect_model(x)[0][0][1]

            all_posterior.append(x)

    return all_posterior


def testset_files(base_path):

    # find all wav file paths from testset
    json_path = base_path + "test.json"
    test_data = json.load(open(json_path, 'r'))
    wakeword_files = [(base_path + path["audio_file_path"], path["is_hotword"]) \
                        for path in test_data if path["is_hotword"]]
    not_wakeword_files = [(base_path + path["audio_file_path"], path["is_hotword"]) \
                        for path in test_data if not path["is_hotword"]]

    return wakeword_files, not_wakeword_files 

def load_test(wakeword_files, not_wakeword_files):
    
    print('Loading all positive class wav files for test set')
    wakeword_wavs = [AudioSegment.from_wav(path[0]) for path in tqdm(wakeword_files)]
    print('Loading all negative class wav files for test set')
    not_wakeword_wavs = [AudioSegment.from_wav(path[0]) for path in tqdm(not_wakeword_files)]

    return wakeword_wavs, not_wakeword_wavs 


def concatenate_test(wakeword_wavs, not_wakeword_wavs, FFR_path, FAR_path):
    # Create long audio file with only wakeword 
    # to calculate FRR, and only not-wakeword for FAR.

    wakeword_wavs_concat = wakeword_wavs[0]
    not_wakeword_wavs_concat = not_wakeword_wavs[0]

    print('Concatenating positive samples')
    for wav in tqdm(wakeword_wavs[1:]):
        wakeword_wavs_concat += AudioSegment.silent(duration=3000) + wav

    wakeword_wavs_concat.export(FFR_path, format="wav")

    # Just take half of the samples...could even be less.
    print('Concatenating negative samples')
    for wav in tqdm(not_wakeword_wavs[1:len(wakeword_wavs)]):
        not_wakeword_wavs_concat += AudioSegment.silent(duration=3000) + wav

    not_wakeword_wavs_concat.export(FAR_path, format="wav")

def load_posteriors(models_dir, model_type, frame_width, sample_rate, input_path, out_path):
    if out_path.exists():
        with open(out_path, 'rb') as f:
            posteriors = pickle.load(f)
    else:
        posteriors = get_posterior(models_dir, model_type, input_path,
                                   frame_width, sample_rate)

        # make directory structure if it does not exist
        #out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'wb') as f:
            pickle.dump(posteriors, f)
    return posteriors

def duration_test(FRR_path, FAR_path, sample_rate):

    total_duration = 0

    # get duration in seconds for positive evaluation set
    s, _ = librosa.load(FRR_path, sr=sample_rate)
    total_duration += len(s)/sample_rate

    # get duration in seconds for negative evaluation set
    s, _ = librosa.load(FAR_path, sr=sample_rate)
    total_duration += len(s)/sample_rate

    return total_duration

def plot_FRR_FAR(keyword_posteriors, no_keyword_posteriors, num_wakewords, total_duration_hrs, model_type):

    thresholds = np.arange(0.98,0.99999,0.0005)

    FRR = []
    FAR = []
    for threshold in thresholds:

        prev_wake = False
        accepts = 0

        # measure true positives
        for posterior in keyword_posteriors:
            if posterior > threshold and not prev_wake:
                 accepts += 1
            prev_wake = False
            if posterior > threshold:
                 prev_wake = True
        rejects = num_wakewords - accepts
        reject_percentage = rejects / num_wakewords
        FRR.append(reject_percentage)

        prev_wake = False
        accepts = 0

        # measure false positives
        for posterior in no_keyword_posteriors:
            if posterior > threshold and not prev_wake:
                 accepts += 1
            prev_wake = False
            if posterior > threshold:
                 prev_wake = True
        accepts_rate = accepts / total_duration_hrs 
        FAR.append(accepts_rate)

    # plot data
    fig, ax = plt.subplots(1,1)
    ax.set_facecolor('lightgray')
    plt.plot(thresholds, FRR, label=model_type)
    plt.ylabel("False Rejection Rate")
    plt.xlabel("Posterior Threshold")
    plt.grid(color='white')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

    fig, ax = plt.subplots(1,1)
    ax.set_facecolor('lightgray')
    plt.plot(thresholds, FAR, label=model_type)
    plt.ylabel("False Accepts per Hour")
    plt.xlabel("Posterior Threshold")
    plt.grid(color='white')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

    fig, ax = plt.subplots(1,1)
    ax.set_facecolor('lightgray')
    plt.plot(FAR, FRR, label=model_type)
    plt.xlabel("False Alarms per Hour")
    plt.ylabel("False Rejection Rate")
    plt.grid(color='white')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluates wakeword model(s), reports useful metrics.')
    parser.add_argument('--model_type', type=str, default='Wavenet', choices=['CRNN', 'Wavenet'], help='Model type being evaluated.')
    parser.add_argument('--models_dir', type=str, default='wwdetect/wavenet/tf_models', help='Directory where trained models are stored.')
    parser.add_argument('--data_dir', type=str, default='data/hey_snips_research_6k_en_train_eval_clean_ter/',
                        help='Directory with Hey Snips raw dataset')
    parser.add_argument('--eval_dir', type=str, default='data/evaluation/',
                        help='Directory to save and load concatenated wav files from')
    parser.add_argument('--pos_samples', type=str, default='hey_snips_long.wav', help='File for concatenated positive class samples')
    parser.add_argument('--neg_samples', type=str, default='not_hey_snips_long.wav', help='File for concatenated negative class samples')
    parser.add_argument('--results_dir', type=str, default='utils/results', help="Directory to store results")
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate for audio (Hz)')
    parser.add_argument('--frame_width', type=int, default=20, help='Frame width for audio in (ms)')
    args = parser.parse_args()
    return args

def main(args) -> int:
    if not Path(args.results_dir).exists():
        try:
            os.mkdir(args.results_dir)
        except FileExistsError:
            print("results directory exists")

    # paths for concatenation of positive and negative samples wavs
    FRR_path = Path(os.path.join(args.eval_dir, args.pos_samples))
    FAR_path = Path(os.path.join(args.eval_dir, args.neg_samples))

    # load audio data
    wakeword_files, not_wakeword_files = testset_files(args.data_dir)

    # get number of wakeword examples 
    num_wakewords = len(wakeword_files)

    # if both negative and positive example evaluation files don't exists, create them
    if not FAR_path.exists() or not FRR_path.exists():
        try:
            os.mkdir(args.eval_dir)
        except FileExistsError:
            print("directory for eval files exists")
        wakeword_wavs, not_wakeword_wavs = load_test(wakeword_files, not_wakeword_files)
        concatenate_test(wakeword_wavs, not_wakeword_wavs, FRR_path, FAR_path)

        # clean up memory
        del wakeword_wavs
        del not_wakeword_wavs

    # get total duration
    print('Calculating total duration of test set')
    total_duration_hrs = duration_test(FRR_path, FAR_path, args.sample_rate)/3600
    print(f'Total duration of evaluation set is {total_duration_hrs:.2f} hrs')

    # get predictions from model on positive samples
    all_posterior_wakeword = load_posteriors(args.models_dir, args.model_type, args.frame_width, 
                                             args.sample_rate, FRR_path,
                                             Path(os.path.join(args.results_dir, args.model_type + "_all_wakeword.pkl")))

    # get predictions from model on negative samples
    all_posterior_not_wakeword = load_posteriors(args.models_dir, args.model_type, args.frame_width, 
                                                 args.sample_rate, FAR_path,
                                                 Path(os.path.join(args.results_dir, args.model_type + "_no_wakeword.pkl")))

    # plot results
    plot_FRR_FAR(all_posterior_wakeword, all_posterior_not_wakeword, num_wakewords, total_duration_hrs, args.model_type)

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
