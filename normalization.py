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
    print("mean pixel value", mean_value_pixel)
    mean_tensor = torch.tensor(mean_value_pixel.reshape(1, number_channel, 1, 1, 1))
    print("mean tensor", mean_tensor[:,6, :, :, :])
    std_value_pixel = std_pixel(list_tensor, number_channel=number_channel)
    print("std pixel value", std_value_pixel)
    std_tensor = torch.tensor(std_value_pixel.reshape(1, number_channel, 1, 1, 1))
    print("std tebsor", std_tensor[:, 6, :, :, :])
    if phase == "1p":
        #torch.save(mean_tensor, os.getcwd() + "/path_results/mean_and_std_tensors/mean_tensor.pt")
        mean_std_directory = phase_directory + "/mean_and_std_tensors/"
        if not os.path.exists(mean_std_directory):                   
            os.makedirs(mean_std_directory)
        torch.save(mean_tensor, mean_std_directory + "/mean_tensor.pt")
        torch.save(std_tensor, mean_std_directory + "/std_tensor.pt")
    if phase == "2p":
        #torch.save(mean_tensor, os.getcwd() + "/path_results/mean_and_std_tensors/std_tensor.pt")
        mean_std_directory = phase_directory + "/mean_and_std_tensors/"
        if not os.path.exists(mean_std_directory):                   
            os.makedirs(mean_std_directory)
        torch.save(mean_tensor, mean_std_directory + "/mean_tensor.pt")
        torch.save(std_tensor, mean_std_directory + "/std_tensor.pt")
    normalized_list = []
    for tensor in list_tensor:
        #tensor = torch.from_numpy(tensor)     #questa mi sa che era nuova, la avevo aggiunta io ieri --> quindi forse potrebbe essere lei a creare il problema
        tensor = (tensor - mean_tensor) / std_tensor   #qua forse alucne divisioni per 0 generano i nan values
        tensor = tensor[:, :, :-1, :, 1:-1]            #riducendo le dimensioni riduco anche i9l numero dei nan --> qua non ne sto generado altri di nan
        tensor = tensor.float()
        normalized_list.append(tensor)
    return normalized_list, mean_tensor, std_tensor


def Normalization_Float(list_tensor, mean_tensor, std_tensor):
    """
    normalization routine for FLOAT data
    """
    normalized_list = []
    for tensor in list_tensor:
        tensor = (tensor - mean_tensor) / std_tensor
        tensor = tensor[:, :, :-1, :, 1:-1]
        tensor = tensor.float()
        normalized_list.append(tensor)
    return normalized_list

