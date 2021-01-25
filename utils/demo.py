import os
import sys
import time
import argparse
import logging

from spokestack.io.pyaudio import PyAudioInput
from spokestack.pipeline import SpeechPipeline
from spokestack.vad.webrtc import VoiceActivityDetector
from spokestack.wakeword.tflite import WakewordTrigger
from spokestack.activation_timeout import ActivationTimeout

logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description='Spokestack demo script for VAD and Wake Word detection')
    parser.add_argument('--models_dir', type=str, default='tf_models', help='directory with TF-Lite models filter, decode, detect')
    parser.add_argument('--sr', type=int, default=16000, help='Sample rate for audio (Hz)')
    parser.add_argument('--fw', type=int, default=20, help='Frame width for audio in (ms)')
    args = parser.parse_args()

    return args

def main(args):
    start = time.time()

    mic = PyAudioInput(sample_rate=args.sr, frame_width=args.fw)
    vad = VoiceActivityDetector()
    wake = WakewordTrigger(model_dir=args.models_dir)
    timeout = ActivationTimeout(frame_width=args.fw)

    pipeline = SpeechPipeline(mic, [vad, wake, timeout])
    pipeline.start()
    pipeline.run()

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
