import numpy as np
import pandas as pd
import prim
from .abstractclassifier import AbstractClassifier
import warnings
# The prim library used here throws RuntimeWarning when checking box limits.
# Since the warning is related to an special metric for PRIM, we can safely ignore these warnings.
warnings.filterwarnings("ignore", category=RuntimeWarning)

class PrimClassifier(AbstractClassifier):
    def __init__(self, peel_alpha=0.05, threshold=0.5, threshold_type='>'):
        self.boxes = []
        self.peel_alpha = peel_alpha
        self.threshold = threshold
        self.threshold_type = threshold_type
    
    def fit(self, X, y):
        self.boxes = []
        
        # Rename features using their index positions
        if type(X) is np.ndarray:
            X_df = pd.DataFrame(X, columns=[f'{i}' for i in range(X.shape[1])])
            
        alg_prim = prim.Prim(X_df, y, 
                             threshold=self.threshold, 
                             threshold_type=self.threshold_type, 
                             peel_alpha=self.peel_alpha)
        
        self.boxes = alg_prim.find_all()
    
    def predict(self, X):
        y_pred = np.zeros(len(X), dtype="int")

        for box in self.boxes:
            mins = box.limits['min'].values
            maxs = box.limits['max'].values
            
            # Check for empty boxes
            if not len(mins) or not len(maxs):
                continue

            # Since some boxes have only one dimension, we iterate to check bounds
            lower_bounds = np.full_like(y_pred, True, dtype=bool)
            for dim in box.limits['min'].index:
                dimension = int(dim)
                condition = X[:, dimension] >= box.limits['min'][dim]
                lower_bounds = np.logical_and(lower_bounds, condition)
                
            
            upper_bounds = np.full_like(y_pred, True, dtype=bool)
            for dim in box.limits['max'].index:
                dimension = int(dim)
                condition = X[:, dimension] <= box.limits['max'][dim]
                upper_bounds = np.logical_and(upper_bounds, condition)
                
            idx = np.logical_and(lower_bounds, upper_bounds)
            #idx = np.all(np.logical_and(mins <= X, X <= maxs), axis=1)
            y_pred[idx] = 1
            
        return y_pred
    
    def set_hyperparameters(self, hyperparameters:dict):
        new_model = PrimClassifier(**hyperparameters)
        new_model.boxes = self.boxes
        return new_model
    
    def get_num_boxes(self):
        return len([box for box in self.boxes if len(box.limits['min'].values) and len(box.limits['max'].values)])