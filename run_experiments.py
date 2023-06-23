#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
import sklearn.metrics as mtr
import src.torchsetup as tsetup
from src.dataset.datasetcollector import DatasetCollector
from src.models.hypernnclassifier import HyperNNClassifier
from src.models.primclassifier import PrimClassifier
from src.evaluation.modelevaluator import ModelEvaluator

def parse_experiments_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='hypernn', type=str, help='which model to use: prim, hypernn.')
    parser.add_argument('--results_directory', default='results', type=str, help='where to store results.')
    parser.add_argument('--max_train_epochs', default=10000, type=int, help='maximum number of epochs.')
    parser.add_argument('--max_number_of_boxes', default=100, type=int, help='maximum number of boxes.')
    parser.add_argument('--patience', default=200, type=int, help='early stopping count.')
    parser.add_argument('--tau', default=1.0, type=float, help='initial smooth sigmoid factor.')
    parser.add_argument('--alpha', default=1.0, type=float, help='initial smoothmax factor.')
    parser.add_argument('--use_cuda', default=True, type=bool, help='set device as cuda if available.')
    parser.add_argument('--prim_threshold', default=0.5, type=float, help='prim threshold for finding boxes.')
    parser.add_argument('--test_size', default=0.3, type=float, help='split ration for the test dataset.')
    parser.add_argument('--num_cv_folds', default=5, type=int, help='number of cv folds.')
    parser.add_argument('--num_seeds', default=3, type=int, help='number of random seeds.')
    parser.add_argument('--scaling', default=True, type=bool, help='apply feature standard scaling.')
    parser.add_argument('--use_small_data', default=True, type=bool, help='use only small datasets.')
    
    return parser.parse_args()

def get_prim_and_hyperparameters(experiment_options):
    model = PrimClassifier()

    hparam_grid = {'threshold': [0.01, 0.025, 0.05, 0.1]}

    return model, hparam_grid

def get_hypernn_and_hyperparameters(experiment_options):
    device = tsetup.set_device(cuda=experiment_options.use_cuda)
    
    model = HyperNNClassifier()
   
    hparam_grid = {
        'nboxes'                  : [2, 3, 5, 10, 20, 50], #experiment_options.max_number_of_boxes],
        'device'                  : [device],  
        'training_epochs'         : [experiment_options.max_train_epochs],
        'patience_early_stopping' : [experiment_options.patience], 
        'learning_rate'           : [0.05, 0.1],
        'l1_reg'                  : [1e-3],
        'l2_reg'                  : [1e-3],
        'alpha_tau_decay_step'    : [10],
        'alpha'                   : [experiment_options.alpha],
        'tau'                     : [experiment_options.tau],
    }
    
    return model, hparam_grid

def main(experiment_options):
    model_name = experiment_options.model_name
    
    if model_name == 'prim':
        model, hyperparameters = get_prim_and_hyperparameters(experiment_options)
    elif model_name == 'hypernn':
        model, hyperparameters = get_hypernn_and_hyperparameters(experiment_options)
    else:
        return NotImplementedError(f'{model_name} is not implemented.')
    
    print(model_name)
    print(hyperparameters)

    collector = DatasetCollector()
    
    if not experiment_options.use_small_data:
        datasets = collector.get_all_datasets()
    else:
        datasets = datasets = [
            collector.load_iris(), 
            collector.load_wine(), 
            collector.load_blood(), 
            collector.load_cars(), 
        ]
    
    evaluator = ModelEvaluator(
        model = model,
        model_hyperparameter_space = hyperparameters,
        test_size=experiment_options.test_size,
        cross_validation_folds=experiment_options.num_cv_folds,
        scale_data=experiment_options.scaling,
        num_seeds=experiment_options.num_seeds,
        evaluation_metric=mtr.f1_score,
        results_logging_directory=experiment_options.results_directory,
    ) 

    evaluator.evaluate(datasets)

    print(evaluator.show_summary())

if __name__ == '__main__':
    main(parse_experiments_parameters())
