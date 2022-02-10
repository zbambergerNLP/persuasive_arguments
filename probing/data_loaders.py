import torch
import os
import pandas as pd
import torch
from torch.utils.data import Dataset

import constants


class CMVHiddenStatesDataset(Dataset):
    def __init__(self, task_dir, key_phrase, batch_size):
        self.task_dir = task_dir
        self.batch_size = batch_size

        self.hidden_state_batch_file_names = list(
            filter(
                lambda file_name: file_name.endswith(constants.PARQUET) and key_phrase in file_name,
                os.listdir(task_dir)))
        self.probing_file_paths = list(
            map(
                lambda file_name: os.path.join(task_dir, file_name),
                self.hidden_state_batch_file_names))

    def __len__(self):
        return self.batch_size

    def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load('data/' + ID + '.pt')
        y = self.labels[ID]

        return X, y
