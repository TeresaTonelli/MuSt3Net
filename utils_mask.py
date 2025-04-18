#In this script we build teh masks for the float biogeochemical data

import numpy as np
import torch
import random
import math
from hyperparameter import *  


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def compute_mean_value(index, train_dataset):
    """function that computes the mean value of a variable on the entire domain in a week"""
    channel_total_mean = np.zeros(shape=(number_channel,))
    for train_tensor in train_dataset:
        tensor_mean = np.array(train_tensor.mean(axis=(0, 2, 3, 4)))  
        channel_total_mean = channel_total_mean + tensor_mean
    channel_total_mean = channel_total_mean / len(train_dataset)
    return channel_total_mean[index]   


def compute_std_value(index, train_dataset):
    """function that computes the std value of a variable on the entire domain in a week"""
    channel_total_std = np.zeros(shape=(number_channel,))
    for train_tensor in train_dataset:
        tensor_std = np.array(train_tensor.std(axis=(0, 2, 3, 4)))  
        channel_total_std = channel_total_std + tensor_std
    channel_total_std = channel_total_std / len(train_dataset)
    return channel_total_std[index]


def generate_input_mask(shape, n_points):
    """function that computes a mask which identifies a set of selected profiles"""
    mask = torch.zeros(shape) 
    _, _, d, h, w = shape
    selected_depths = random.choices(range(d), k=n_points)
    selected_longitudes = random.choices(range(h), k=n_points)
    selected_latitudes = random.choices(range(w), k=n_points)
    for i in range(len(selected_depths)):
        mask[:, :, selected_depths[i], selected_longitudes[i], selected_latitudes[i]] = 1.0
    return mask


def generate_sea_land_mask(tensor, depth_index):
    """function to generate the mask land - sea for a specific depth"""
    land_sea_mask = torch.zeros(1, 1, 1, int((longitude_interval[1] - longitude_interval[0]) * constant_longitude / resolution[1] + 1), int((latitude_interval[1] - latitude_interval[0]) * constant_latitude / resolution[0] + 1))
    for i_h in range(land_sea_mask.shape[3]):
        for i_w in range(land_sea_mask.shape[4]):
            if tensor[0, 0, depth_index, i_h, i_w] != 0.0:
                land_sea_mask[0, 0, 0, i_h, i_w] = 1.0
    return land_sea_mask


def generate_float_mask(float_tensor_coordinates):
    """function that computes a mask that indicates the location of float measures for a specific tensor"""
    float_mask = torch.zeros(batch, 1, d, h, w)
    for coordinates in float_tensor_coordinates:
        float_mask[:, :, :, coordinates[0], coordinates[1]] = 1.0
    return float_mask


def apply_masks(tensor, list_masks):
    """this function takes a tensor and properly multiply all the masks to it"""
    depth_index = range(tensor.shape[2])
    masked_tensor = tensor.clone()
    for i_d in depth_index:
        masked_tensor[:, :, i_d, :, :] = tensor[:, :, i_d, :, :].to(device) * list_masks[i_d][:, :, 0, :, 1:-1].to(device)
    return masked_tensor


def compute_weights(d, total_depht, main_depth):
    """function that implements the weights for the computation of the loss"""
    weights = torch.ones([d])
    for i in range(int(main_depth / resolution[2] + 1), int(total_depht / resolution[2] + 1)):
        weights[i] = 1 + ((-0.5) * (i - int(main_depth / resolution[2] + 1)) / ( int(total_depht / resolution[2]) - int(main_depth / resolution[2] + 1)) ) 
    weights_final = torch.zeros([1, 1, d, h, w])
    for i_h in range(h):
        for i_w in range(w):
            weights_final[:, :, :, i_h, i_w] = weights
    return weights_final[:, :, :-2, :, 1:-1]


def compute_exponential_weights(d, total_depht, main_depth):
    """function that implements the weights for the computation of the loss"""
    weights = torch.ones([d])
    y_B = 0.1
    for i in range(int(main_depth / resolution[2] + 1), int(total_depht / resolution[2] + 1)):  
        weights[i] = math.exp((math.log(y_B) / (int(total_depht / resolution[2]) - int(main_depth / resolution[2] + 1)))*i - (int(main_depth / resolution[2] + 1) * (math.log(y_B) / (int(total_depht / resolution[2]) - int(main_depth / resolution[2] + 1)))))
    weights_final = torch.zeros([1, 1, d, h, w])
    for i_h in range(h):
        for i_w in range(w):
            weights_final[:, :, :, i_h, i_w] = weights
    return weights_final[:, :, :-2, :, 1:-1]
