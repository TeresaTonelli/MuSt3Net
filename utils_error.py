#this script contains all the functions that are useful to measure the effectiveness of our results


import numpy as np
import torch


def compute_chlorophyll_single_output(coordinates, chlorophyll_tensor):
    """this function takes some profiles and return a list that reshapes their values, in order to put them in 1D"""
    chlorophyll_values = torch.tensor()
    for coordinate in coordinates:
        chl_profile = chlorophyll_tensor[:, -1, :, coordinate[0], coordinate[1]]
        chl_profile.unsqueeze()
        chlorophyll_values = torch.cat((chlorophyll_values, chl_profile))
    return chlorophyll_values



def compute_RMSE_models(list_float_profiles_coordinates, test_data, BFM_test_data, model_1p, model_2p):
    model_1p.eval()
    model_2p.eval()
    rmse_BFM_float = 0
    rmse_model_1p_float = 0
    rmse_model_2p_float = 0
    BFM_chl_values = []
    float_chl_values = []
    rmse_BFM_float = torch.sqrt(torch.mean((BFM_chl_values - float_chl_values) ** 2))

    return rmse_BFM_float