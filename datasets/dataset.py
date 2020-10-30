import pandas as pd
import numpy as np


class DataLoader(object):
    def __init__(self, data_path='./',
                 is_train=True, is_reshape=True, split_factor=0.0, batch_size=32):
        self.img_rows = 28
        self.img_cols = 28
        self.num_classes = 10
        self.split_factor = split_factor
        self.data_path = data_path
        self.is_train = is_train
        self.is_reshape = is_reshape
        self.inputs, self.targets = self.get_data()
        self.batch_size = batch_size
        self.num_samples = self.inputs.shape[0]
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

    def get_data(self):
        raw_data = pd.read_csv(self.data_path)
        raw_data = raw_data.to_numpy()
        label = raw_data[:, 0]
        array_data = raw_data[:, 1:]
        if self.is_reshape:
            image_data = array_data.reshape(raw_data.shape[0], self.img_rows, self.img_cols, 1)
        else:
            image_data = array_data

        split_point = int(raw_data.shape[0] * (1 - self.split_factor))
        if self.is_train:
            inputs, targets = image_data[:split_point], label[:split_point]
        else:
            inputs, targets = image_data[split_point:], label[split_point:]

        inputs = inputs.astype(np.float32) / 255.0
        return inputs, targets

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_count < self.num_batches:
            start = self.batch_count * self.batch_size
            end = (self.batch_count + 1) * self.batch_size
            self.batch_count += 1
            return self.inputs[start:end], self.targets[start:end]
        else:
            self.batch_count = 0
            raise StopIteration
