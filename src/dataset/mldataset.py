from torch.utils.data import Dataset
import numpy as np

class MLDataset(Dataset):
    def __init__(self, X, y, name=''):
        self.X = X
        self.y = y
        self.name = name
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        batch_inputs = self.X[idx]
        batch_labels = self.y[idx]
        return batch_inputs, batch_labels
    
    def __repr__(self):
        return f'{self.name}_{type(self).__name__} at {id(self)}'