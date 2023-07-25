import sys
import os
import glob
from glob import glob
import numpy as np
import datetime
import matplotlib.pyplot as plt
import json


def euclidean_distance(tensor1, tensor2):
    squared_distance = np.sum((tensor1 - tensor2) ** 2)
    return np.sqrt(squared_distance)

def find_k_nearest_neighbors(tensor, missing_idx, k, distance_metric):
    distances = []
    for i in range(len(tensor)):
        if i == missing_idx:
            continue
        distance = distance_metric(tensor[missing_idx], tensor[i])
        distances.append((i, distance))
    distances.sort(key=lambda x: x[1])
    return [idx for idx, _ in distances[:k]]

def knn_imputation(tensor, k, distance_metric):
    filled_tensor = np.copy(tensor)

    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            for z in range(tensor.shape[2]):
                if np.isnan(tensor[i, j, z]):
                    neighbors = find_k_nearest_neighbors(tensor, i, k, distance_metric)
                    neighbor_values = [tensor[n, j, z] for n in neighbors if not np.isnan(tensor[n, j, z])]
                    filled_tensor[i, j, z] = np.mean(neighbor_values)

    return filled_tensor

def check_data(file):
    arr = np.load(file)
    res = np.zeros(shape=(arr.shape[1], arr.shape[2], 2))
    qual = arr[2, ...]
    for i in range(qual.shape[0]):
        for j in range(qual.shape[1]):
            if (qual[i][j] == 2) or (qual[i][j] == 3):
                res[i][j][0] = np.nan
                res[i][j][1] = np.nan
            else:
                res[i][j][0] = arr[0][i][j]
                res[i][j][1] = arr[1][i][j]
    
    return res