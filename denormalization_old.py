import numpy as np
import torch


def Denormalization(normalized_tensor, mean_tensor, std_tensor):
    """this function performs the inverse of normalization"""
    denorm_tensor = normalized_tensor * std_tensor + mean_tensor
    return denorm_tensor
