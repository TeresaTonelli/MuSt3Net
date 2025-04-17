"""
Normalization function that must be applied before proceeding with the training
bc the values of the unknown we want to estimate are way higher than 1
"""

import os
import torch 
from hyperparameter import *
from mean_pixel_value import MV_pixel, std_pixel


def Normalization(list_tensor, phase, phase_directory, number_channel=number_channel):
    mean_value_pixel = MV_pixel(list_tensor, number_channel=number_channel)
    mean_tensor = torch.tensor(mean_value_pixel.reshape(1, number_channel, 1, 1, 1))
    std_value_pixel = std_pixel(list_tensor, number_channel=number_channel)
    std_tensor = torch.tensor(std_value_pixel.reshape(1, number_channel, 1, 1, 1))
    if phase == "1p":
        mean_std_directory = phase_directory + "/mean_and_std_tensors/"
        if not os.path.exists(mean_std_directory):                   
            os.makedirs(mean_std_directory)
        torch.save(mean_tensor, mean_std_directory + "/mean_tensor.pt")
        torch.save(std_tensor, mean_std_directory + "/std_tensor.pt")
    if phase == "2p":
        mean_std_directory = phase_directory + "/mean_and_std_tensors/"
        if not os.path.exists(mean_std_directory):                   
            os.makedirs(mean_std_directory)
        torch.save(mean_tensor, mean_std_directory + "/mean_tensor.pt")
        torch.save(std_tensor, mean_std_directory + "/std_tensor.pt")
    normalized_list = []
    for tensor in list_tensor:
        tensor = (tensor - mean_tensor) / std_tensor   
        tensor = tensor[:, :, :-1, :, 1:-1]          
        tensor = tensor.float()
        normalized_list.append(tensor)
    return normalized_list, mean_tensor, std_tensor



def tmp_Normalization(list_tensor, phase, mean_std_directory):
    mean_tensor = torch.load(mean_std_directory + "/mean_tensor.pt")
    std_tensor = torch.load(mean_std_directory + "/std_tensor.pt")
    normalized_list = []
    for tensor in list_tensor:
        tensor = (tensor - mean_tensor) / std_tensor  
        tensor = tensor[:, :, :-1, :, 1:-1]         
        tensor = tensor.float()
        normalized_list.append(tensor)
    return normalized_list



def tmp_Normalization_float(tensor, phase, mean_std_directory, my_coord):
    mean_tensor = torch.squeeze(torch.load(mean_std_directory + "/mean_tensor.pt")[:, -1, :, :, :])
    std_tensor = torch.squeeze(torch.load(mean_std_directory + "/std_tensor.pt")[:, -1, :, :, :])
    norm_tensor = (tensor - mean_tensor) / std_tensor  
    norm_tensor = norm_tensor.float()
    return norm_tensor



def Normalization_Float(list_tensor, mean_tensor, std_tensor):
    """normalization routine for FLOAT data"""
    normalized_list = []
    for tensor in list_tensor:
        tensor = (tensor - mean_tensor) / std_tensor
        tensor = tensor[:, :, :-1, :, 1:-1]
        tensor = tensor.float()
        normalized_list.append(tensor)
    return normalized_list

