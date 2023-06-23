import torch
import torch.nn as nn
import numpy as np
from ..smoothfunctions import smooth_max, smooth_sigmoid

class HyperNN(nn.Module):
    def __init__(self, nboxes, dim, tau=1, alpha=1, random_state=42):
        super().__init__()
        torch.manual_seed(random_state)
        self.nboxes = nboxes
        self.dim = dim
        self.alpha = alpha
        self.block = Block(nboxes, dim, tau)
        self.random_state = random_state
        
    def forward(self, X):
        if type(X) is np.ndarray:
            X = torch.from_numpy(X).float()

        out_block = self.block(X)
        contained_in_at_least_one_box = smooth_max(out_block, self.alpha)
        return contained_in_at_least_one_box
    
    def get_params(self):
        return self.block.get_params()
        
    def load_params(self,params):
        self.block.load_params(params)
        self.nboxes = self.block.nboxes
        self.dim = self.block.dim

class Block(nn.Module):
    def __init__(self, nboxes, dim, tau=0.1):
        super().__init__()
        self.nboxes = nboxes
        self.dim = dim
        self.tau = tau
        self.mins = nn.Parameter(torch.randn(nboxes, dim).type(torch.float32), requires_grad=True)
        self.length = nn.Parameter(torch.rand(nboxes, dim).type(torch.float32), requires_grad=True)
        
        self.init_parameters()
        
    def init_parameters(self):
        self.mins.data.normal_(mean=0.0, std=1.0)
        self.length.data.normal_(mean=0.0, std=1.0)
        
    def forward(self, X): 
        X = X.unsqueeze(1)
    
        maxs = self.mins + torch.abs(self.length)
        
        lower_then_max_point = maxs - X
        higher_then_min_point = X - self.mins
        
        lower_in_all_dimensions = torch.min(lower_then_max_point, -1)[0] # TODO: Fix multiple dimensions
        higher_in_all_dimensions = torch.min(higher_then_min_point, -1)[0]
        
        lower_soft_condition = smooth_sigmoid(lower_in_all_dimensions, self.tau)
        higher_soft_condition = smooth_sigmoid(higher_in_all_dimensions, self.tau)
        
        out = lower_soft_condition * higher_soft_condition # Logical AND
        
        return out

    def get_params(self):
        return (self.mins.detach(), torch.abs(self.length).detach()) 
    
    def load_params(self,params):
        mins,length = params
        self.nboxes,self.dim = mins.shape
        self.mins = nn.Parameter(mins,requires_grad=True)
        self.length = nn.Parameter(length,requires_grad=True)
        
    def get_mins_maxs(self):
        maxs = self.mins + torch.abs(self.length)
        return (self.mins.detach(), maxs.detach())