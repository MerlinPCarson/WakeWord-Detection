import os
import numpy as np
from tf_lite.tf_lite import TFLiteModel
from tf_lite.ring_buffer import RingBuffer

class Filter:

    def __init__(
        self,
        pre_emphasis: float = 0.0,
        sample_rate: int = 16000,
        fft_window_type: str = "hann",
        fft_hop_length: int = 10,
        model_dir: str = "",
    ) -> None:

        self.pre_emphasis: float = pre_emphasis
        self.hop_length: int = int(fft_hop_length * sample_rate / 1000)

        if fft_window_type != "hann":
            raise ValueError("Invalid fft_window_type")

        self.filter_model: TFLiteModel = TFLiteModel(
            model_path=os.path.join(model_dir, "filter.tflite")
        )

        # window size calculated based on fft
        # the filter inputs are (fft_size - 1) / 2
        # which makes the window size (post_fft_size - 1) * 2
        self._window_size = (self.filter_model.input_details[0]["shape"][-1] - 1) * 2
        self._fft_window = np.hanning(self._window_size)

        # initialize sample buffer
        self.sample_window: RingBuffer = RingBuffer(shape=[self._window_size])
        self._prev_sample: float = 0.0

    def filter_frame(self, frame) -> None:
        # pull out a single value from the frame and apply pre-emphasis
        # with the previous sample then cache the previous sample
        # to be use in the next iteration
        prev_sample = frame[-1]
        frame -= self.pre_emphasis * np.append(self._prev_sample, frame[:-1])
        self._prev_sample = prev_sample

        # fill the sample window to analyze speech containing samples
        # after each window fill the buffer advances by the hop length
        # to produce an overlapping window
        for sample in frame:
            self.sample_window.write(sample)
            if self.sample_window.is_full:
                return self._analyze()
                self.sample_window.rewind().seek(self.hop_length)

        return None

    def _analyze(self) -> None:
        # read the full contents of the sample window to calculate a single frame
        # of the STFT by applying the DFT to a real-valued input and
        # taking the magnitude of the complex DFT
        frame = self.sample_window.read_all()
        frame = np.fft.rfft(frame * self._fft_window, n=self._window_size)
        frame = np.abs(frame).astype(np.float32)

        # compute mel spectrogram
        return self._filter(frame)

    def _filter(self, frame) -> None:
        # add the batch dimension and compute the mel spectrogram with filter model
        frame = np.expand_dims(frame, 0)
        frame = self.filter_model(frame)[0]

        return frame

    def num_outputs(self) -> None:
        # return number of output features from model
        return self.filter_model.output_details[0]["shape"][-1]
