import numpy as np
import pandas as pd
import random
import torch

def set_seeds(seed=1234):
    '''Set seeds for reproducibility.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # multi-GPU
    
def set_device(cuda=True):
    return torch.device("cuda" if (torch.cuda.is_available() and cuda) else "cpu")