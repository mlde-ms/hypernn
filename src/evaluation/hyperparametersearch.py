import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterGrid

class HyperparameterGridSearch:
    def __init__(self, model, hyperparameter_space, num_cross_validation_folds, evaluation_metric, random_state, verbose=False):
        self.model = model
        self.best_validation_score = -np.inf
        self.best_model_hyperparameters = dict()
        self.hyperparameter_grid = ParameterGrid(hyperparameter_space)
        self.num_cross_validation_folds = num_cross_validation_folds
        self.evaluation_metric = evaluation_metric
        self.verbose = verbose
        self.random_state = random_state
        
    def fit(self, X, y):
        self.best_validation_score = -np.inf
        
        for hyperparameters in self.hyperparameter_grid:
            # Set hyperparameters
            self.model = self.model.set_hyperparameters(hyperparameters)
            
            # Create cross-validation approach
            skfold_crossvalid = StratifiedKFold(
                n_splits=self.num_cross_validation_folds, 
                shuffle=True, 
                random_state=self.random_state)
            
            # Stores all scores
            score_history = []
            
            # Iterate over each fold
            for train_idx, test_idx in skfold_crossvalid.split(X, y):
                # Get train data
                X_train_fold, y_train_fold = X[train_idx], y[train_idx]
                # Get validation data
                X_test_fold, y_test_fold = X[test_idx], y[test_idx]
                # Train model
                self.model.fit(X_train_fold, y_train_fold)
                # Predict using validation data
                y_pred = self.model.predict(X_test_fold)
                # Compute validation score
                valid_score = self.evaluation_metric(y_test_fold, y_pred)
                # Store validation score
                score_history.append(valid_score)
            
            # Average validation scores
            overall_score = np.mean(score_history)
            
            if overall_score > self.best_validation_score:
                self.best_validation_score = overall_score
                self.best_model_hyperparameters = hyperparameters
                
        if self.verbose:
            print('Best hyperparameters:', self.best_model_hyperparameters) 
            print(f'Best validation score: {self.best_validation_score:.3}' )
            
        return self.model.set_hyperparameters(self.best_model_hyperparameters)