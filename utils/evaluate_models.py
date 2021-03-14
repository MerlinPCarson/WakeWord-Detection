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


def get_posterior(models_dir, model_type, eval_type,
                  test_files, frame_width, sample_rate,
                  examine_audio=False):

    filter = Filter(model_dir=models_dir)
    encode_model: TFLiteModel = TFLiteModel(
        model_path=os.path.join(models_dir, "encode.tflite")
    )
    detect_model: TFLiteModel = TFLiteModel(
        model_path=os.path.join(models_dir, "detect.tflite")
    )

    encoder_len = encode_model.input_details[0]["shape"][1 if model_type == 'Wavenet' else 2]
    all_posterior = []

    frame_length = sample_rate // 1000 * frame_width
    inference_hop_length = 2

    print("Evaluating...")
    for file in tqdm(test_files):
        samples, _ = librosa.load(file, sr=sample_rate)

        all_posterior_file = []
        window_buffer = []

        # Pad with zeros to better examine trajectory.
        samples = np.pad(samples, (sample_rate//2, sample_rate//2),
                         mode='constant')

        # Pulled from filter_dataset.py to closely match training
        # preprocessing.
        for start_idx in tqdm(np.arange(0, len(samples), frame_length), leave=False):
            frame = samples[start_idx:start_idx + frame_length]
            if len(frame) < frame_length:
                pad_len = frame_length - len(frame)
                frame = np.pad(frame, (0, pad_len), mode='constant')

            # filter audio through filter model
            frame = filter.filter_frame(frame)

            if len(frame) > 0:
                window_buffer.extend(frame)

            # Perform inference on current window.
            if len(window_buffer) >= encoder_len:

                frames = window_buffer[:encoder_len]
                window_buffer = window_buffer[inference_hop_length:]

                # inference for CRNN model
                if model_type == 'CRNN':
                    x = np.expand_dims(np.array(frames).T, 0)
                    x = np.expand_dims(x, -1)
                    x = np.array(encode_model(x)).squeeze(0)
                    x = detect_model(x)[0][0][1]

                # inference for Wavenet model
                elif model_type == 'Wavenet':
                    x = np.expand_dims(np.array(frames), 0)
                    x = np.array(encode_model(x)).squeeze(0)
                    x = detect_model(x)[0][0][1]

                all_posterior_file.append(x)


        # TODO: Currently grabbing maximum
        # posterior output over all windows
        # for file. Eventually, if we can
        # model duration or otherwise, we
        # may refine this to be a bit more
        # rigorous about what constitutes
        # an accept.
        if eval_type == "false_negatives":
            all_posterior.append(np.max(all_posterior_file))
            if examine_audio and all_posterior[-1] < 0.75:
                    plot_wav_and_posterior(samples, all_posterior_file,
                                           sample_rate, Path(file).stem,
                                           frame_width, encoder_len,
                                           alignment="mid")
        else:
            all_posterior.extend(all_posterior_file)

    return all_posterior


def plot_wav_and_posterior(wav, posteriors, sample_rate,
                           filename, frame_width, encoder_len, alignment):
    first_frame_endpoint_sec = ((encoder_len + 1) * 10) / 1000
    sliding_window_sec = frame_width / 1000
    if alignment == 'mid':
        start_point = first_frame_endpoint_sec / 2
        posterior_x = [start_point+(frame_num*sliding_window_sec) for frame_num in range(len(posteriors))]
    elif alignment == 'end':
        posterior_x = [first_frame_endpoint_sec+(frame_num*sliding_window_sec) for frame_num in range(len(posteriors))]
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax[0].set_title("Posterior Trajectory for\n " + str(filename), size=18,
                    fontweight="bold")
    ax[0].plot(posterior_x, posteriors)
    ax[0].set_facecolor(color='lightgrey')
    ax[0].grid(color='white')
    ax[0].set_ylabel("Posterior\nProbability", size=12)
    x = [samp_num / sample_rate for samp_num in range(len(wav))]
    ax[1].plot(x, wav)
    ax[1].set_facecolor(color='lightgrey')
    ax[1].grid(color='white')
    ax[1].set_ylabel("Amplitude", size=12)
    ax[1].set_xlabel("Time (sec)", size=12)
    plt.show()
    plt.close()


def testset_files(base_path):

    # find all wav file paths from testset
    json_path = base_path + "test.json"
    test_data = json.load(open(json_path, 'r'))
    wakeword_files = [base_path + path["audio_file_path"] for path in test_data if path["is_hotword"]]
    not_wakeword_files = [base_path + path["audio_file_path"] for path in test_data if not path["is_hotword"]]

    return wakeword_files, not_wakeword_files


def concatenate_FA(wavs, num_files, FAR_path):
    # Create long audio file with only wakeword
    # to calculate FRR, and only not-wakeword for FAR.
    wavs_concat = wavs[0]

    # Just take a subset of files.
    print('Concatenating negative samples')
    for wav in tqdm(wavs[1:num_files]):
        wavs_concat += AudioSegment.silent(duration=100) + wav


    wavs_concat.export(FAR_path, format="wav")


def load_posteriors(models_dir, model_type, frame_width,
                    sample_rate, eval_type, input_path,
                    out_path, examine_audio=False):
    if out_path.exists():
        with open(out_path, 'rb') as f:
            posteriors = pickle.load(f)
    else:
        posteriors = get_posterior(models_dir, model_type, eval_type,
                                   input_path, frame_width, sample_rate,
                                   examine_audio)

        with open(out_path, 'wb') as f:
            pickle.dump(posteriors, f)
    return np.squeeze(np.array(posteriors))


def duration_test(FAR_path, sample_rate):
    # get duration in seconds for negative evaluation set
    s, _ = librosa.load(FAR_path, sr=sample_rate)
    return len(s)/sample_rate


def plot_FRR_FAR(keyword_posteriors, no_keyword_posteriors, num_wakewords, total_duration_hrs, model_type):

    thresholds = np.arange(0.5,0.99999,0.005)

    # smooth not keyword posteriors using average filter
    windowsize = 30
    no_keyword_posteriors = np.convolve(no_keyword_posteriors, np.ones((windowsize,))/windowsize, mode='same')

    FRR = []
    FAR = []
    print(f'Sweeping thresholds over posteriors')
    for threshold in tqdm(thresholds):

        prev_wake = False
        accepts = 0

        # measure true positives
        # each postierior is over a whole wakeword utterance
        accepts = (keyword_posteriors>threshold).sum()
        rejects = num_wakewords - accepts
        reject_percentage = rejects / num_wakewords
        FRR.append(reject_percentage)

        prev_wake = False
        accepts = 0

        # measure false positives
        # only count 1 false positive for consecutive false positives
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
    parser.add_argument('--model_type', type=str, default='CRNN', choices=['CRNN', 'Wavenet'],
                        help='Model type being evaluated.')
    parser.add_argument('--models_dir', type=str,
                        default='/Users/amie/Desktop/OHSU/CS606 - Deep Learning II/FinalProject/WakeWord-Detection/wwdetect/CRNN/models/Arik_CRNN_data_original/',
                        help='Directory where trained models are stored.')
    parser.add_argument('--data_dir', type=str, default='data/hey_snips_research_6k_en_train_eval_clean_ter/',
                        help='Directory with Hey Snips raw dataset')
    parser.add_argument('--eval_dir', type=str, default='data/evaluation/',
                        help='Directory to save and load concatenated wav files from')
    parser.add_argument('--pos_samples', type=str, default='hey_snips_long.wav',
                        help='File for concatenated positive class samples')
    parser.add_argument('--neg_samples', type=str, default='not_hey_snips_long.wav',
                        help='File for concatenated negative class samples')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate for audio (Hz)')
    parser.add_argument('--frame_width', type=int, default=20, help='Frame width for audio in (ms)')
    parser.add_argument('--examine_audio', default=False, action='store_true', help='Flag to examine problematic audio clips')
    args = parser.parse_args()

    assert Path(args.models_dir).exists(), "Directory for TF-Lite models and results is not found!"

    return args

def main(args) -> int:

    # paths for concatenation of negative samples wavs
    FAR_path = Path(os.path.join(args.eval_dir, args.neg_samples))

    # load audio data
    wakeword_paths, not_wakeword_paths = testset_files(args.data_dir)

    # get number of wakeword examples
    num_wakewords = len(wakeword_paths)

    # if negative evaluation file doesn't exist, create them
    if not FAR_path.exists():
        try:
            os.mkdir(args.eval_dir)
        except FileExistsError:
            print("directory for eval files exists")
        print(f'Loading all files for FA test set')
        not_wakeword_wavs = [AudioSegment.from_wav(path) for path in tqdm(not_wakeword_paths)]
        concatenate_FA(not_wakeword_wavs, num_wakewords, FAR_path)

        # clean up memory
        del not_wakeword_wavs

    # get total duration
    print('Calculating total duration of FA test set')
    total_duration_hrs = duration_test(FAR_path, args.sample_rate) / 3600
    print(f'Total duration of FA set is {total_duration_hrs:.2f} hrs')

    # # get predictions from model on positive samples
    all_posterior_wakeword = load_posteriors(args.models_dir, args.model_type, args.frame_width,
                                             args.sample_rate, "false_negatives", wakeword_paths,
                                             Path(os.path.join(args.models_dir,
                                                               args.model_type + "_all_wakeword.pkl")),
                                             args.examine_audio)

    # get predictions from model on negative samples
    all_posterior_not_wakeword = load_posteriors(args.models_dir, args.model_type, args.frame_width,
                                                 args.sample_rate, "false_accepts", [str(FAR_path)],
                                                 Path(os.path.join(args.models_dir,
                                                                   args.model_type + "_no_wakeword.pkl")),
                                                 args.examine_audio)

    # plot results
    plot_FRR_FAR(all_posterior_wakeword, all_posterior_not_wakeword, num_wakewords, total_duration_hrs,
                 args.model_type)

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))

