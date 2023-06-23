import torch
import numpy as np
from .abstractclassifier import AbstractClassifier
from .hypernn import HyperNN
from ..utils import numpy_to_torch

class HyperNNClassifier(AbstractClassifier):
    def __init__(self, nboxes=2, dim=2, tau=1, alpha=1, 
                 device='cuda', loss_fn=None, l1_reg=1e-3, l2_reg=1e-3,
                 training_epochs=100, learning_rate=1e-1, patience_early_stopping=50, 
                 alpha_tau_decay_step=1, prediction_threshold=0.5, 
                 verbose=False, verbosity=10, random_state=42):
        
        if loss_fn is None:
            loss_fn = torch.nn.BCELoss()
        
        if patience_early_stopping is None:
            patience_early_stopping = torch.inf
        
        self.nboxes = nboxes
        self.dim = dim
        self.device = device
        self.loss_fn = loss_fn
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg
        self.learning_rate = learning_rate
        self.tau = tau
        self.alpha = alpha
        self.training_epochs = training_epochs
        self.patience_early_stopping = patience_early_stopping
        self.alpha_tau_decay_step = alpha_tau_decay_step
        self.prediction_threshold = prediction_threshold
        self.verbose = verbose
        self.verbosity = verbosity
        self.random_state = random_state
        self.scheduler = None
        
        self.setup_classifier()

    def setup_classifier(self):
        self.model = HyperNN(
            nboxes=self.nboxes,
            dim = self.dim,
            tau=self.tau, 
            alpha=self.alpha,
            random_state=self.random_state)

        if self.device != "cpu":
            self.model.block = self.model.block.to(self.device)
        
        self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.learning_rate)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.1, 
                patience=30, min_lr=1e-5)
        
    def fit(self, X, y):
        self.dim = X.shape[1]
        self.setup_classifier()
        
        X = numpy_to_torch(X, self.device)
        y = numpy_to_torch(y, self.device)
        
        best_model = None
        best_loss = torch.inf
        
        self.model.train()
        for epoch in range(self.training_epochs):
            # Reset gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            z = self.model(X)
            
            # Compute loss
            J = self.loss_fn(z, y)
            J += self.l1_reg * torch.norm(self.model.block.length, p=1)
            J += self.l2_reg * torch.sum(torch.pow(self.model.block.length, 2))
            
            loss = J.detach().item()
            
            # Backward pass
            J.backward()  
            
            # Update weights
            self.optimizer.step()
            
            # Decay alpha and tau
            if self.alpha_tau_decay_step > 0 and ((epoch+1) % self.alpha_tau_decay_step) == 0:
                self.model.block.tau = max(self.model.block.tau * 0.95, 0.02)
                self.model.alpha = max(self.model.alpha * 0.95, 0.02)
                
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step(loss)
                
            if loss < best_loss:
                best_loss = loss
                best_model = self.model
                _patience = self.patience_early_stopping # Reset patience
            else:
                _patience -= 1 # Reduce patience
            
            if not _patience or loss == 0.:
                break
                
            # Print progress
            if self.verbose:
                if (((epoch+1) % self.verbosity) == 0) or (epoch == self.training_epochs-1):
                    print("Epoch: ", epoch+1)
                    print("Loss: ", loss)

    def predict(self, X):
        y_pred = []
        self.model.eval()
        with torch.inference_mode():
            X = numpy_to_torch(X, self.device)
            probs = self.model(X).detach()
            y_pred = (probs > self.prediction_threshold).long()

        return y_pred.cpu()

    def set_hyperparameters(self, hyperparameters:dict):
        return HyperNNClassifier(**hyperparameters)
    
    def get_num_boxes(self):
        return torch.sum(torch.abs(self.model.block.length) > 1e-3).item()/(self.dim)