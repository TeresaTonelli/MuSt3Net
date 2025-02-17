#In this script we can find all the functions to compute the rmse starting from an already trained model and a list of tensors, divides wrt geographical areas (ga) and seasons

import numpy as np
import torch
import os

from hyperparameter import *
from losses import convolutional_network_float_exp_weighted_loss
from utils_function import transform_latitude_index, transform_longitude_index
from utils_mask import generate_float_mask


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device", device)


def select_season_tensors(list_tensors, season, years_week_duplicates_list):
    """this function returns a list of all the tensors that belongs to a specific season"""
    list_season_tensor = []
    list_season_index = []
    for i in range(len(years_week_duplicates_list)):
        week = years_week_duplicates_list[i][1]
        if dict_season[season][0] <= int(week) <= dict_season[season][1]:
            list_season_index.append(years_week_duplicates_list[i])
            list_season_tensor.append(list_tensors[i])
    return list_season_tensor, list_season_index


def create_ga_mask(my_ga, list_float_coordinates, tensor_shape):
    """this function creates a mask of the same dimension of the tensor that identify the coordinates of a specific geografic area"""
    #list_float_coordinates = [transform_latitudes_longitudes([list_float_coordinates_index[i][1], list_float_coordinates_index[i][0]]) for i in range(len(list_float_coordinates_index))]
    #print("list coord before", list_float_coordinates[1])
    #for list_coord in list_float_coordinates:
     #   list_coord[0], list_coord[1] = list_coord[1], list_coord[0] 
    #print("list coord after", list_float_coordinates[1])
    my_lat_interval, my_long_interval = dict_ga[my_ga][1], dict_ga[my_ga][0]
    my_lat_index_interval = [transform_latitude_index(my_lat_interval[0]), transform_latitude_index(my_lat_interval[1])]
    my_long_index_interval = [transform_longitude_index(my_long_interval[0]), transform_longitude_index(my_long_interval[1])]
    ga_mask = torch.ones(tensor_shape)
    print("len list float coordinates", len(list_float_coordinates))
    for float_coord in list_float_coordinates:
        print("float coord", float_coord)
        if my_long_index_interval[0] <= float_coord[0] <= my_long_index_interval[1] and my_lat_index_interval[0] <= float_coord[1] <= my_lat_index_interval[1]:
            ga_mask[:, :, :, float_coord[0], float_coord[1]] = 1.0
    return ga_mask


def compute_rmse_ga_season_2(list_tensors, list_float_tensors, list_float_coordinates, land_sea_masks, ga, season, years_week_duplicates_list, exp_weights, CNN_model):
    """this function computes the root mean square error of a list of profiles related to a specific ga and a season"""
    #select a specific season
    season_tensors, season_index = select_season_tensors(list_tensors, season, years_week_duplicates_list)
    season_float_tensors = select_season_tensors(list_float_tensors, season, years_week_duplicates_list)
    #select the profiles of a specific season
    ga_masks = [create_ga_mask(ga, list_float_coordinates[i], season_tensors[i].shape) for i in range(len(list_float_coordinates))]
    #model evaluation and loss evaluation
    CNN_model.eval()
    with torch.no_grad():
        season_output_tensors = [CNN_model(input_tensor) for input_tensor in season_tensors]
        #loss computation
        float_coord_masks = [generate_float_mask(list_float_coordinates[i]) * ga_masks[i].to(device) for i in range(len(list_float_coordinates))]
        season_losses = [convolutional_network_float_exp_weighted_loss(season_float_tensors[i].float(), season_tensors[i].float(), land_sea_masks, float_coord_masks[i], exp_weights.to(device)) for i in range(len(season_output_tensors))]
        season_loss = np.mean(season_losses)
    return season_loss