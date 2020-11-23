import csv
import dgl
import os
import sys
import torch
import argparse
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from ictuner import read_netlist
from ictuner import get_logger


class NetlistDataset(Dataset):
    def __init__(self, Gs, dataset_file):
        self.Gs = Gs
        self.flows_df = pd.read_csv(dataset_file)

        # clipping and min-max scaling
        self.flows_df['drv_total'] = self.flows_df['drv_total'].clip(0, 1000)
        #_min = self.flows_df['drv_total'].min()
        #_max = self.flows_df['drv_total'].max()
        #self.flows_df['drv_total'] = (self.flows_df['drv_total'] - _min) / (_max - _min)

    def __len__(self):
        return len(self.flows_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        g_id = self.flows_df.iloc[idx, 0].split('/')[-1]
        g = self.Gs[g_id]
        params = self.flows_df.iloc[idx, 1:7]           # params
        params = np.array([params]).astype('float')

        metrics = self.flows_df.loc[idx, 'drv_total']    # drv_total
        metrics = np.array([metrics]).astype('float')

        sample = {
            'id': g_id,
            'design': g,
            'params': torch.tensor(params).float(),
            'metrics': torch.tensor(metrics).float()
        }

        return sample