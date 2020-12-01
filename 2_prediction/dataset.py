import torch
import pandas as pd

from torch.utils.data import Dataset

class DesignDataset(Dataset):
    def __init__(self, Gs, dataset_file):
        self.Gs = Gs
        self.flows_df = pd.read_csv(dataset_file)

    def __len__(self):
        return len(self.flows_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        g_id = self.flows_df.iloc[idx, 0]
        g = self.Gs[g_id]

        metrics = self.flows_df.iloc[idx, 1]
        metrics = np.array([metrics]).astype('float')

        sample = {
            'id': g_id,
            'design': g,
            'metrics': torch.tensor(metrics).float()
        }

        return sample

