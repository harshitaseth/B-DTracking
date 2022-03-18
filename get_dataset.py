

import torch
from torch.utils.data import Dataset

import os
import numpy as np


class ballroom(Dataset):
    

    def __init__(
            self,
            spectrogram_dir,
            label_dir,
            sr=22050,
            hop_size_in_seconds=0.01,
            trim_size=(81, 3000),
            downbeats=False):
        """
        Initialise the dataset object.

        Parameters:
            spectrogram_dir: directory holding spectrograms as NumPy dumps
            label_dir: directory containing labels as NumPy dumps

        Keyword Arguments:
            sr (=22050): Sample rate to use when converting annotations to
                         spectrogram frame indices
            hop_size_in_seconds (=0.01): Mel spectrogram hop size
            trim_size (=(81,3000)): Dimensions to trim spectrogram down to.
                                    Should match input dimensions of network.
        """
        self.spectrogram_dir = spectrogram_dir
        self.label_dir = label_dir
        self.names = self.get_data()

        self.sr = 22050
        self.hop_size = int(np.floor(hop_size_in_seconds * 22050))
        self.trim_size = trim_size

        self.downbeats = downbeats

    def __len__(self):
        """Overload len() calls on object."""
        return len(self.names)

    def __getitem__(self, i):
        """Overload square bracket indexing on object"""
        raw_spec, raw_beats, beat_times, downbeat_times = self.spectrogram_labels(i)
        x, y = self._trim_spec_and_labels(raw_spec, raw_beats)

        if self.downbeats:
            y = y.T
        # y[y==0.5] = 0
        # out_label = np.zeros(y[0].shape)
        # beats = np.argwhere(y[0] == 1).reshape(-1)
        # downbeats = np.argwhere(y[1] == 1).reshape(-1)
        # out_label[beats] = 1
        # out_label[downbeats] = 2
        # y = out_label
        
        return {
            'spectrogram': torch.from_numpy(np.expand_dims(x.T, axis=0)).float(),
            'target': torch.from_numpy(y[:3000].astype('float64')).float(),
            'beats': beat_times,
            'downbeats': downbeat_times,
            'path' : os.path.join(self.label_dir, self.names[i] + '.beats')
        }

    def get_name(self, i):
        """Fetches name of datapoint specified by index i"""
        return self.names[i]

    def get_ground_truth(self, i, quantised=True, downbeats=False):
        """
        Fetches ground truth annotations for datapoint specified by index i

        Parameters:
            i: Index signifying which datapoint to fetch truth for

        Keyword Arguments:
            quantised (=True): Whether to return a quantised grount truth
        """

        return self._get_quantised_ground_truth(i, downbeats)\
            if quantised else self._get_unquantised_ground_truth(i, downbeats)

    def _trim_spec_and_labels(self, spec, labels):
        """
        Trim spectrogram matrix and beat label vector to dimensions specified
        in self.trim_size. Returns tuple of trimmed NumPy arrays

        Parameters:
            spec: Spectrogram as NumPy array
            labels: Labels as NumPy array
        """

        x = np.zeros(self.trim_size)
        if not self.downbeats:
            y = np.zeros(self.trim_size[1])
        else:
            y = np.zeros((self.trim_size[1], 2))

        to_x = self.trim_size[0]
        to_y = min(self.trim_size[1], spec.shape[1])

        x[:to_x, :to_y] = spec[:, :to_y]
        y[:to_y] = labels[:to_y]

        return x, y

    def get_data(self):
        
        names = []
        for entry in os.scandir(self.label_dir):
            names.append(os.path.splitext(entry.name)[0])
        return names

    def _text_label_to_float(self, text):
        
        allowed = '1234567890. \t'
        filtered = ''.join([c for c in text if c in allowed])
        if '\t' in filtered:
            t = filtered.rstrip('\n').split('\t')
        else:
            t = filtered.rstrip('\n').split(' ')
        return float(t[0]), float(t[1])

    def _get_quantised_ground_truth(self, i, downbeats):
        

        with open(
                os.path.join(self.label_dir, self.names[i] + '.beats'),
                'r') as f:

            beat_times = []

            for line in f:
                time, index = self._text_label_to_float(line)
                if not downbeats:
                    beat_times.append(time * self.sr)
                else:
                    if index == 1:
                        beat_times.append(time * self.sr)
        quantised_times = []

        for time in beat_times:
            spec_frame = int(time / self.hop_size)
            quantised_time = spec_frame * self.hop_size / self.sr
            quantised_times.append(quantised_time)

        return np.array(quantised_times)

    def _get_unquantised_ground_truth(self, i, downbeats):
        """
        Fetches the ground truth (time labels) from the appropriate
        label file.
        """

        with open(
                os.path.join(self.label_dir, self.names[i] + '.beats'),
                'r') as f:
            
            beat_times = []

            for line in f:
                time, index = self._text_label_to_float(line)
                if not downbeats:
                    beat_times.append(time)
                else:
                    if index == 1:
                        beat_times.append(time)

        return np.array(beat_times)

    def spectrogram_labels(self, i):
        data_name = self.names[i]

        with open(
                os.path.join(self.label_dir, data_name + '.beats'),
                'r') as f:

            beat_floats = []
            beat_indices = []
            for line in f:
                parsed = self._text_label_to_float(line)
                beat_floats.append(parsed[0])
                beat_indices.append(parsed[1])
            beat_times = np.array(beat_floats) * self.sr

            if self.downbeats:
                downbeat_times = self.sr * np.array(
                    [t for t, i in zip(beat_floats, beat_indices) if i == 1])


        spectrogram =\
            np.load(os.path.join(self.spectrogram_dir, data_name + '.npy'))
        if not self.downbeats:
            beat_vector = np.zeros(spectrogram.shape[-1])
        else:
            beat_vector = np.zeros((spectrogram.shape[-1], 2))

        for time in beat_times:
            spec_frame =\
                min(int(time / self.hop_size), beat_vector.shape[0] - 1)
            for n in range(-2, 3):
                if 0 <= spec_frame + n < beat_vector.shape[0]:
                    if not self.downbeats:
                        beat_vector[spec_frame + n] = 1.0 if n == 0 else 0.5
                    else:
                        beat_vector[spec_frame + n, 0] = 1.0 if n == 0 else 0.5
        
        if self.downbeats:
            for time in downbeat_times:
                spec_frame =\
                    min(int(time / self.hop_size), beat_vector.shape[0] - 1)
                for n in range(-2, 3):
                    if 0 <= spec_frame + n < beat_vector.shape[0]:
                        beat_vector[spec_frame + n, 1] = 1.0 if n == 0 else 0.5


        return spectrogram, beat_vector, beat_times, downbeat_times
