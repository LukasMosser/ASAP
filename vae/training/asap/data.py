import torch 
import numpy as np
from sklearn.model_selection import ShuffleSplit

def load_dataset(near_stack_fname, far_stack_fname, window_size):
    near_traces = np.load(near_stack_fname)
    far_traces =  np.load(far_stack_fname)

    well_i, well_x = 38, 138
    well_variance_near = np.mean(np.std(near_traces[well_i-2:well_i+1, well_x-2:well_x+1], 2))
    well_variance_far = np.mean(np.std(far_traces[well_i-2:well_i+1, well_x-2:well_x+1], 2))
    
    near_traces /= well_variance_near
    far_traces /= well_variance_far
    
    near_traces_emb = near_traces.reshape(-1, 64)
    far_traces_emb =far_traces.reshape(-1, 64)

    X = torch.from_numpy(np.stack([near_traces_emb, far_traces_emb], 1)).float()
    y = torch.from_numpy(np.zeros((X.shape[0], 1))).float()

    split = ShuffleSplit(n_splits=1, test_size=0.8)
    for train_index, test_index in split.split(X):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

    return X_train, y_train, X_test, y_test, X, y