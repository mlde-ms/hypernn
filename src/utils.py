import matplotlib.patches as patches
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch

def plot_boxes(boxnet, X, y):
    assert X.shape[1] == 2,"Plotting only for 2D possible!"

    fig = plt.figure()
    ax = fig.add_subplot(111)

    min_points_boxes, lengths_boxes = boxnet.get_params()

    for i in range(len(min_points_boxes)):
        bottom = min_points_boxes[i]
        length = lengths_boxes[i]
        
        # Check for empty boxes
        if not len(bottom) or not len(length):
            continue
            
        rect = patches.Rectangle(
            tuple(detach_tensor(bottom)), 
            *tuple(detach_tensor(length)), 
            linewidth=1, 
            edgecolor='r', 
            facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    X_pd = pd.DataFrame(X)
    y_pd = pd.Series(y)

    ax.scatter(X_pd[0], X_pd[1], c=y_pd)

def plot_boxes_prim(prim_boxes, X, y):
    assert X.shape[1] == 2,"Plotting only for 2D possible!"

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for box in prim_boxes:
        mins = box.limits['min'].values
        maxs = box.limits['max'].values

        # Don't plot empty boxes
        if not len(mins) or not len(maxs):
            continue
            
        # Don't plot 1-dimensional boxes
        if len(mins) != X.shape[1] or len(maxs) != X.shape[1]:
            continue
            
        rect = patches.Rectangle(
            tuple(mins), 
            *tuple(maxs-mins), 
            linewidth=1, 
            edgecolor='r', 
            facecolor='none')

        ax.add_patch(rect)

    ax.scatter(X[:, 0], X[:, 1], c=y)

def detach_tensor(tensor):
    return tensor.cpu().detach().numpy()
    
def numpy_to_torch(data, device):
    if type(data) is np.ndarray:
        data = torch.from_numpy(data).float()
    
    if device != "cpu":
        data = data.to(device)

    return data

def binarize_labels(y, label):
    assert type(y) is np.ndarray, 'y must be of type numpy.ndarray'
    
    y_new = y.copy()
    y_new[np.where(y != label)] = 0
    y_new[np.where(y == label)] = 1
    y_new = y_new.astype(int)
    return y_new