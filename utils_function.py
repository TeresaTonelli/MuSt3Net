#This script contains all the functions not irectly related to neural networks architecture, but necessary for the algorithm of neural networks


import numpy as np
import torch 
import random
import os
import itertools
from hyperparameter import * 



def tensor_sample(tensor, k):
    """this function select k elemnt from a tensor and recreate a copy of the original tensor which contains only those elements"""
    depths_index_range = tensor.shape[2]   
    longitudes_index_range = tensor.shape[3]
    latitudes_index_range = tensor.shape[4]
    sampled_coordinates_depths = random.sample(range(depths_index_range), k)
    sampled_coordinates_longitudes = random.sample(range(longitudes_index_range), k)
    sampled_coordinates_latitudes = random.sample(range(latitudes_index_range), k)
    sampled_coordinates = [[sampled_coordinates_depths[i], sampled_coordinates_longitudes[i], sampled_coordinates_latitudes[i]] for i in range(k)]
    reduced_tensor = np.zeros([batch, number_channel_biogeoch, tensor.shape[2], tensor.shape[3], tensor.shape[4]])
    for i in range(k):
        reduced_tensor[:, :, sampled_coordinates[i][0], sampled_coordinates[i][1], sampled_coordinates[i][2]] = tensor[:, :, sampled_coordinates[i][0], sampled_coordinates[i][1], sampled_coordinates[i][2]]
    return reduced_tensor


def tensor_mask_sample(tensor, mask, k):
    """this function select k elemnt from a tensor and recreate a copy of the original tensor which contains only those elements, sampling only sea data"""
    longitudes_index_range = tensor.shape[3]
    latitudes_index_range = tensor.shape[4]
    couple_coordinates = itertools.product(range(longitudes_index_range), range(latitudes_index_range))  
    sea_coordinates = [couple_coordinate for couple_coordinate in couple_coordinates if mask[:, :, :, couple_coordinate[0], couple_coordinate[1]] == 1.0]
    deep_sea_coordinates = [sea_coordinate for sea_coordinate in sea_coordinates if torch.count_nonzero(tensor[:, :, :, sea_coordinate[0], sea_coordinate[1]]) > deep_sea_good_count]
    sampled_coordinates = random.sample(deep_sea_coordinates, k)  
    reduced_tensor = torch.zeros(batch, number_channel_biogeoch, tensor.shape[2], tensor.shape[3], tensor.shape[4])
    for i in range(k):
        reduced_tensor[:, :, :, sampled_coordinates[i][0], sampled_coordinates[i][1]] = tensor[:, :, :, sampled_coordinates[i][0], sampled_coordinates[i][1]]
    return reduced_tensor, sampled_coordinates


def tensor_mask_ngh_sample(tensor, mask, r):
    """this function select one element and its neighbours from a tensor and recreate a copy of the original tensor which contains only those elements, sampling only sea data"""
    longitudes_index_range = tensor.shape[3]
    latitudes_index_range = tensor.shape[4]
    couple_coordinates = itertools.product(range(longitudes_index_range), range(latitudes_index_range))  
    sea_coordinates = [couple_coordinate for couple_coordinate in couple_coordinates if mask[:, :, :, couple_coordinate[0], couple_coordinate[1]] == 1.0]
    sampled_coordinate = random.sample(sea_coordinates, 1)
    sampled_coordinates = [[sampled_coordinate[0][0] - i, sampled_coordinate[0][1] - j] for i, j in itertools.product(range(-r, r), range(-r, r))]
    #remove cooridnates which lies outside the domain of the figure
    for coordinates in sampled_coordinates[::-1]:
        if coordinates[0] not in range(0, h) or coordinates[1] not in range(0, w):
            sampled_coordinates.remove(coordinates)
    reduced_tensor = torch.zeros(batch, number_channel_biogeoch, tensor.shape[2], tensor.shape[3], tensor.shape[4])
    for i in range(len(sampled_coordinates)):
        reduced_tensor[:, :, :, sampled_coordinates[i][0], sampled_coordinates[i][1]] = tensor[:, :, :, sampled_coordinates[i][0], sampled_coordinates[i][1]]
    return reduced_tensor


def generate_sample_tensors(tensor, mask, k, m):
    """repeat the generation of tensor samples m times"""
    tensors_samples = []
    coordinates_samples = []
    for i in range(m):
        new_sample_tensor, new_sampled_coordinates = tensor_mask_sample(tensor, mask, k)
        tensors_samples.append(new_sample_tensor)
        coordinates_samples.append(new_sampled_coordinates)
    return tensors_samples, coordinates_samples


def generate_list_sample_tensors(list_tensors, mask, k, m):
    list_sample_tensors = []
    list_sample_coordinates = []
    for tensor in list_tensors:
        single_tensors_samples, single_coordinates_samples = generate_sample_tensors(tensor, mask, k, m)
        list_sample_tensors.extend(single_tensors_samples)
        list_sample_coordinates.extend(single_coordinates_samples)
    return list_sample_tensors, list_sample_coordinates


def fill_tensor_with_standard(tensor, list_mask, standard_value):
    """fill missing value of the tensor with a standard mean value of the variable contained in the tensor"""
    for i_0 in range(tensor.shape[0]):
        for i_1 in range(tensor.shape[1]):
            for i_2 in range(tensor.shape[2] - 1):   
                for i_3 in range(tensor.shape[3]):
                    for i_4 in range(tensor.shape[4]):
                        if tensor[i_0, i_1, i_2, i_3, i_4] == 0.0 and list_mask[i_2][i_0, i_1, 0, i_3, i_4] == 1.0:
                            tensor[i_0, i_1, i_2, i_3, i_4] = standard_value
    return tensor


def fill_tensor_opt(tensor, list_mask, standard_value):
    """fill missing value of the tensor with a standard mean value of the variable contained in the tensor"""
    tensor_mask = torch.cat(tuple(list_mask), 2)
    tensor[torch.logical_and(tensor == 0, tensor_mask == 1)] = standard_value
    return tensor


def compute_profile_coordinates(tensor):
    """function that finds the coordinates in which the original measures of a float lies"""
    list_coordinates_tensor = []
    longitude_indexes = np.arange(0, tensor.shape[3])
    latitude_indexes = np.arange(0, tensor.shape[4])
    for latitude_index in latitude_indexes:
        for longitude_index in longitude_indexes:
            profile_tensor_float = tensor[0, 0, :, longitude_index, latitude_index] 
            if torch.sum(profile_tensor_float) != 0.0:
                list_coordinates_tensor.append((longitude_index, latitude_index))
    return list_coordinates_tensor



def generate_sampled_profiles_tensor(original_tensor, sampled_float_prof_coord):
    """given a list of coordinates, it creates a new tensor that contains values of original tensor only for these coordinates"""
    tensor = torch.zeros(original_tensor.shape[0], original_tensor.shape[1], original_tensor.shape[2], original_tensor.shape[3], original_tensor.shape[4])
    for coord in sampled_float_prof_coord:
        tensor[:, :, :, coord[0], coord[1]] = original_tensor[:, :, :, coord[0], coord[1]]
    return tensor


def remove_float(tensor, list_to_remove_float):
    """this function takes a tensor and modiies it deleting some float measures"""
    modified_tensor = torch.clone(tensor)
    for coordinates in list_to_remove_float:
        modified_tensor[:, :, :, coordinates[0], coordinates[1]] = 0.0
    return modified_tensor



def extend_list(my_list, k):
    """this function copy the elment of a list for k times, in order"""
    extended_list = []
    for element in my_list:
        for i_k in range(k):
            extended_list.append(element)
    return extended_list


def transpose_latitudes(list_coordinates):
    """function used to modify the latitudes given the transposition occured with the normalization"""
    transposed_list_coordinates = []
    for list_couple in list_coordinates:
        single_list_coordinates = []
        for couple in list_couple:
            new_couple = (couple[0], couple[1]-1)
            single_list_coordinates.append(new_couple)
        transposed_list_coordinates.append(single_list_coordinates)
    return transposed_list_coordinates



def create_list_duplicates(my_list, m):
    """create a list that duplicates the elements of the original list m times"""
    new_list = []
    for element in my_list:
        list_element = []
        for i in range(m):
            list_element.append(element)
        new_list.extend(list_element)
    return new_list



def compute_indexes_from_depths(couple_layers, depth_resolution):
    """this function returns the indexes associated to each layer"""
    min_layer = couple_layers[0]
    max_layer = couple_layers[1]
    ind_min = int(min_layer / depth_resolution)
    ind_max = int(max_layer / depth_resolution)
    return ind_min, ind_max
    



def compute_mean_layers(my_tensor, my_list_layers, n_dim, my_size):
    """this function compyute the mean of the tensor wrt of specific layers"""
    my_tensor_mean_layers = torch.zeros(size=my_size)
    for j in range(len(my_list_layers)-1):
        ind_min, ind_max = compute_indexes_from_depths([my_list_layers[j], my_list_layers[j+1]], depth_resolution=resolution[2])
        dim_list = list(my_size)
        dim_list.pop(n_dim)
        dim_tuple = tuple(dim_list)
        my_tensor_mean_layers[:, :, j, :, :] = torch.mean(my_tensor[:, :, ind_min:ind_max, :, :], n_dim).unsqueeze(n_dim)
    return my_tensor_mean_layers


def compute_profile_mean(my_tensor, index_dim, list_to_plot_coordinates):
    tensor_float_coord = my_tensor[0, 0, :, list_to_plot_coordinates[0][0], list_to_plot_coordinates[0][1]]
    for i in range(1, len(list_to_plot_coordinates)):
        tensor_float_coord.stack((tensor_float_coord, my_tensor[0, 0, :, list_to_plot_coordinates[i][0], list_to_plot_coordinates[i][1]]), -1)
    depth_levels = np.arange(0, my_tensor.shape[2])
    profile_mean = torch.zeros([1, 1, my_tensor.shape[2], 1, 1])
    for i_d in depth_levels:
        profile_mean[:, :, i_d, :, :] = torch.mean(tensor_float_coord[:, :, i_d, :, :, :])
    return tensor_float_coord


def find_index_week(list_tensor, iw):
    for week_data in list_tensor:
        week = "".join(c for c in week_data if c.isdecimal())
        if int(week) == iw:
            wd_index = list_tensor.index(week_data)
    return wd_index


def concatenate_physics_tensor(path_directory_physics_tensor):
    """this function takes 6 directories and concatenate tensors relative to the same week"""
    list_directory_physics_tensor = [os.listdir(path_directory_physics_tensor[i]) for i in range(len(path_directory_physics_tensor))]
    concatenate_tensors = []
    n_week = len(list_directory_physics_tensor[0])
    for iw in range(1, n_week + 1):
        index = find_index_week(list_directory_physics_tensor[0], iw)
        my_tensor = torch.load(path_directory_physics_tensor[0] + list_directory_physics_tensor[0][index])  
        for i in range(1, len(list_directory_physics_tensor)):
            new_index = find_index_week(list_directory_physics_tensor[i], iw)
            week_tensor = torch.load(path_directory_physics_tensor[i] + list_directory_physics_tensor[i][new_index])
            my_tensor = torch.cat((my_tensor, week_tensor), axis=1)    
        concatenate_tensors.append(my_tensor)
        torch.save(my_tensor, os.getcwd() + "/dataset/MODEL/2022/final_tensor/" + "datetime_" + str(iw) + ".pt")  
    return concatenate_tensors


def re_order_weeks(list_weeks, list_tensor):
    n = len(list_weeks)
    list_order_tensor = [torch.zeros([1, 1, d, h, w]) for j in range(n)]
    for i in range(1, n+1):
        index_week = list_weeks.index(str(i))
        list_order_tensor[i-1] = list_tensor[index_week]
    return list_order_tensor, [str(j) for j in range(1, n+1)]


def sort_depths_old(list_depths):
    """this function sorts the elements of the list wrt increasing depths --> it works only for land_sea_maks"""
    sorted_list_depths = []
    n_dep = len(list_depths)
    for i in range(n_dep):
        for depth_file in list_depths:
            if str(i) in depth_file:
                sorted_list_depths.append(depth_file)
    return sorted_list_depths


def compute_season(week):
    """this function identifies if a data refers to summer or winter season"""
    if week < 14:
        return "winter"
    else: 
        return "summer"


def transform_latitudes_longitudes(lat_long_indexes):
    """this function transforms the latitudeds and longitudes indexes in the real values of latitude and longitude"""
    lat_index = lat_long_indexes[0]
    long_index = lat_long_indexes[1]
    lat_value = (lat_index + 1 - 1) * resolution[0] / constant_latitude + latitude_interval[0]
    long_value = (long_index - 1) * resolution[1] / constant_longitude + longitude_interval[0]
    return [lat_value, long_value]


def transform_latitude_index(lat_coord):
    """this function takes the latitude coordinate and reconstruct the index associated wrt current resolution"""
    return int((lat_coord - latitude_interval[0]) * constant_latitude / resolution[0] + 1)


def transform_longitude_index(long_coord):
    """this function takes the longitude coordinate and reconstruct the index associated wrt current resolution"""
    return int((long_coord - longitude_interval[0]) * constant_longitude / resolution[1] + 1)