'''
A small module to inspect a particular waveform
and its resulting posterior trajectory; in order
to identify possible tweaks to system.

Script accepts a directory with audio files, and
will generate a posterior trajectory, used to create
a figure with the aligned trajectory, waveform,
and spectrogram.
'''
import os
import sys
import argparse
from pathlib import Path

import librosa
import librosa.display
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import grep

from tf_lite.tensorflow import TFLiteModel
from tf_lite.filter import Filter

def generate_plots(models_dir, model_type, audio_files,
                  frame_width, sample_rate):

    filter = Filter(model_dir=models_dir)
    encode_model: TFLiteModel = TFLiteModel(
        model_path=os.path.join(models_dir, "encode.tflite")
    )
    detect_model: TFLiteModel = TFLiteModel(
        model_path=os.path.join(models_dir, "detect.tflite")
    )

    encoder_len = encode_model.input_details[0]["shape"][1 if model_type == 'Wavenet' else 2]

    frame_length = sample_rate // 1000 * frame_width
    inference_hop_length = 2

    print("Generating plots...")
    for file in tqdm(os.listdir(audio_files)):
        if Path(file).suffix == '.wav':
            file = os.path.join(audio_files, file)
            image_path = Path(file).with_suffix(".png")
            samples, _ = librosa.load(file, sr=sample_rate)

            all_posterior_file = []
            window_buffer = []

            # Pad with zeros to better examine trajectory.
            samples = np.pad(samples, (sample_rate//2, sample_rate//2),
                             mode='constant')

            spectrogram = librosa.stft(samples, n_fft=512)
            spectrogram_dB = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)

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
                        x = detect_model(x)[0][0][0]

                    # inference for Wavenet model
                    elif model_type == 'Wavenet':
                        x = np.expand_dims(np.array(frames), 0)
                        x = np.array(encode_model(x)).squeeze(0)
                        x = detect_model(x)[0][0][1]

                    all_posterior_file.append(x)

            # Plot waveform, trajectory, and spectrogram.
            plot_wav_and_posterior(samples, all_posterior_file,
                                   sample_rate, frame_width,
                                   spectrogram_dB, Path(file).stem,
                                   encoder_len, 'mid', image_path)


    return None

def plot_wav_and_posterior(wav, posteriors, sample_rate, frame_width,
                           spectrogram, filename, encoder_len, alignment,
                           image_path):
    first_frame_endpoint_sec = ((encoder_len + 1) * 10) / 1000
    sliding_window_sec = frame_width / 1000
    if alignment == 'mid':
        start_point = first_frame_endpoint_sec / 2
        posterior_x = [start_point+(frame_num*sliding_window_sec) for frame_num in range(len(posteriors))]
    elif alignment == 'end':
        posterior_x = [first_frame_endpoint_sec+(frame_num*sliding_window_sec) for frame_num in range(len(posteriors))]
    fig, ax = plt.subplots(3, 1, figsize=(10, 5))
    x = [samp_num / sample_rate for samp_num in range(len(wav))]
    ax[0].set_title("Posterior Trajectory for\n " + str(filename), size=18,
                    fontweight="bold")
    ax[0].plot(posterior_x, posteriors)
    ax[0].set_xlim([0,x[-1]])
    ax[0].set_facecolor(color='lightgrey')
    ax[0].grid(color='white')
    ax[0].set_ylabel("Posterior\nProbability", size=12)
    y_spec = librosa.fft_frequencies(sample_rate, n_fft=512)
    ax[1].imshow(spectrogram, aspect='auto', cmap="Greys", origin='lower')
    ax[1].set_xticklabels([])
    ax[1].set_yticks([0,spectrogram.shape[0]//2, spectrogram.shape[0]])
    ax[1].set_yticklabels([0, y_spec[len(y_spec)//2],y_spec[-1]])
    ax[1].set_ylabel("Frequency", size=12)
    ax[1].set
    ax[2].plot(x, wav)
    ax[2].set_facecolor(color='lightgrey')
    ax[2].grid(color='white')
    ax[2].set_ylabel("Amplitude", size=12)
    ax[2].set_xlabel("Time (sec)", size=12)
    ax[2].set_xlim([0, x[-1]])
    plt.savefig(image_path)
    plt.show()
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluates wakeword model(s), reports useful metrics.')
    parser.add_argument('--model_type', type=str, default='Wavenet', choices=['CRNN', 'Wavenet'], help='Model type being evaluated.')
    parser.add_argument('--models_dir', type=str, default='Wavenet_files', help='Directory where trained models are stored.')
    parser.add_argument('--audio_dir', type=str, default='false_positives_wavenet', help='Directory where files to be examined are stored.')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate for audio (Hz)')
    parser.add_argument('--frame_width', type=int, default=20, help='Frame width for audio in (ms)')
    args = parser.parse_args()

    assert Path(args.models_dir).exists(), "Directory for TF-Lite models and results is not found!"

    return args

def main(args) -> int:
    generate_plots(args.models_dir, args.model_type, args.audio_dir, args.frame_width, args.sample_rate)

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
