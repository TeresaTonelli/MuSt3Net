""""
Implementation of the losses for the different train of the model
"""
import torch
from torch.nn.functional import mse_loss
from utils.utils_mask import apply_masks, compute_weights
from hyperparameter import * 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def convolutional_network_exp_weighted_loss(input, output, list_masks, exp_weights):
    """loss function weighted wrt depth"""
    masked_input = apply_masks(input, list_masks)  
    masked_output = apply_masks(output, list_masks)
    masked_weighted_input = masked_input[:, 6, :, :, :] * exp_weights.to(device)
    masked_weighted_output = masked_output * exp_weights.to(device)
    n = torch.count_nonzero(masked_output).float()
    return mse_loss(masked_weighted_output, masked_weighted_input, reduction='sum') / n



def convolutional_network_float_exp_weighted_loss(input, output, list_masks, float_coordinates_mask, exp_weights):
    """float loss function weighted wrt depth"""
    masked_input = apply_masks(input, list_masks)           
    masked_output = apply_masks(output, list_masks)
    masked_weighted_input = masked_input[:, 6, :, :, :] * exp_weights.to(device)
    masked_weighted_output = masked_output * exp_weights.to(device)
    n = torch.count_nonzero(float_coordinates_mask[:, :, :-2, :, 1:-1])
    return mse_loss(masked_weighted_output * float_coordinates_mask[:, :, :-2, :, 1:-1], masked_weighted_input * float_coordinates_mask[:, :, :-2, :, 1:-1], reduction='sum') / n 