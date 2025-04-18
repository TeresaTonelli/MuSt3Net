#In this script we can find all the functions to compute the rmse starting from an already trained model and a list of tensors, divides wrt geographical areas (ga) and seasons

import numpy as np
import torch
import os
import itertools

from convolutional_network import CompletionN
from denormalization import Denormalization
from hyperparameter import *
from losses import convolutional_network_float_exp_weighted_loss
from normalization import Normalization, tmp_Normalization
from utils_function import transform_latitude_index, transform_longitude_index, compute_profile_coordinates
from utils_generation_train_1p import write_list, read_list
from utils_mask import generate_float_mask
from utils_training_1 import load_land_sea_masks,  re_load_float_input_data, re_load_old_float_tensors
from utils_training_2 import compute_3D_ensemble_mean_std

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def create_ga_mask(my_ga, tensor_shape):
    """this function creates a mask of the same dimension of the tensor that identify the coordinates of a specific geografic area"""
    my_lat_interval, my_long_interval = dict_ga[my_ga][1], dict_ga[my_ga][0]
    my_lat_index_interval = [transform_latitude_index(my_lat_interval[0]), transform_latitude_index(my_lat_interval[1])]
    my_long_index_interval = [transform_longitude_index(my_long_interval[0]), transform_longitude_index(my_long_interval[1])]
    ga_mask = torch.zeros(tensor_shape)   
    ga_indices = list(itertools.product([i_long for i_long in range(my_long_index_interval[0], my_long_index_interval[1] + 1)], [i_lat for i_lat in range(my_lat_index_interval[0], my_lat_index_interval[1] + 1)]))
    indices_tensor = torch.tensor(ga_indices)
    ga_mask[:, :, :, indices_tensor[:, 0]-1, indices_tensor[:, 1]-1] = 1.0    
    return ga_mask


def rmse_float_CNN_BFM(list_input_tensors, indexes_train, indexes_test, list_float_tensors, list_float_coordinates, list_BFM_tensors, CNN_model, years_week_list, path_mean_std, season=""):
    #split dataset in training and testing
    list_input_training = [list_input_tensors[i_tr] for i_tr in indexes_train]
    list_input_testing = [list_input_tensors[i_te] for i_te in indexes_test]
    if season in list(dict_season.keys()): 
        #select a specific season
        season_tensors_train, season_indexes_train = select_season_tensors(list_input_training, season, [years_week_list[i_tr] for i_tr in indexes_train])
        season_float_tensors_train, season_float_indexes_train = select_season_tensors([list_float_tensors[i_tr] for i_tr in indexes_train], season, [years_week_list[i_tr] for i_tr in indexes_train])
        season_BFM_tensors_train, season_BFM_indexes_train = select_season_tensors([list_BFM_tensors[i_tr] for i_tr in indexes_train], season, [years_week_list[i_tr] for i_tr in indexes_train])
        season_tensors_test, season_indexes_test = select_season_tensors(list_input_testing, season, [years_week_list[i_te] for i_te in indexes_test])
        season_float_tensors_test, season_float_indexes_test = select_season_tensors([list_float_tensors[i_te] for i_te in indexes_test], season, [years_week_list[i_te] for i_te in indexes_test])
        season_BFM_tensors_test, season_BFM_indexes_test = select_season_tensors([list_BFM_tensors[i_te] for i_te in indexes_test], season, [years_week_list[i_te] for i_te in indexes_test])
    else:
        season_tensors_train = list_input_training
        season_float_tensors_train = [list_float_tensors[i_tr] for i_tr in indexes_train]
        season_BFM_tensors_train = [list_BFM_tensors[i_tr] for i_tr in indexes_train]
        season_tensors_test = list_input_testing
        season_float_tensors_test = [list_float_tensors[i_te] for i_te in indexes_test]
        season_BFM_tensors_test = [list_BFM_tensors[i_te] for i_te in indexes_test]
        season_indexes_train = season_float_indexes_train = season_BFM_indexes_train = indexes_train
        season_indexes_test = season_float_indexes_test = season_BFM_indexes_test = indexes_test
    #prepare mean, std and land_sea_masks
    my_mean = torch.unsqueeze(torch.load(path_mean_std+ "/mean_tensor.pt")[:, 6, :, :, :], 1).to(device)
    my_std = torch.unsqueeze(torch.load(path_mean_std + "/std_tensor.pt")[:, 6, :, :, :], 1).to(device)
    land_sea_masks = load_land_sea_masks("dataset_training/land_sea_masks/")
    #model evaluation and loss evaluation, both for training and testing dataset
    CNN_model.eval()
    with torch.no_grad():
        season_output_tensors_train = [CNN_model(input_tensor) for input_tensor in season_tensors_train]
        season_output_tensors_train = [Denormalization(output_tensor, my_mean, my_std) for output_tensor in season_output_tensors_train]
        season_output_tensors_test = [CNN_model(input_tensor) for input_tensor in season_tensors_test]
        season_output_tensors_test = [Denormalization(output_tensor, my_mean, my_std) for output_tensor in season_output_tensors_test]
        #loss computation
        float_coord_masks = [generate_float_mask(list_float_coordinates[i]) for i in range(len(list_float_coordinates))]
        season_losses_CNN_train = [convolutional_network_float_exp_weighted_loss(season_float_tensors_train[i][:, :, :-1, :, 1:-1].float(), season_output_tensors_train[i].float(), land_sea_masks, float_coord_masks[years_week_list.index(season_indexes_train[i])], torch.ones([1, 1, d-2, h, w-2]).to(device)) for i in range(len(season_output_tensors_train))]
        mean_season_loss_CNN_train = np.nanmean(season_losses_CNN_train)
        season_losses_CNN_test = [convolutional_network_float_exp_weighted_loss(season_float_tensors_test[i][:, :, :-1, :, 1:-1].float(), season_output_tensors_test[i].float(), land_sea_masks, float_coord_masks[years_week_list.index(season_indexes_test[i])], torch.ones([1, 1, d-2, h, w-2]).to(device)) for i in range(len(season_output_tensors_test))]
        mean_season_loss_CNN_test = np.nanmean(season_losses_CNN_test)
        season_losses_BFM_train = [convolutional_network_float_exp_weighted_loss(season_float_tensors_train[i][:, :, :-1, :, 1:-1].float(), season_BFM_tensors_train[i][:,:, :-1, :, 1:-1].float(), land_sea_masks, float_coord_masks[years_week_list.index(season_BFM_indexes_train[i])], torch.ones([1, 1, d-2, h, w-2]).to(device)) for i in range(len(season_BFM_tensors_train))]
        mean_season_loss_BFM_train = np.nanmean(season_losses_BFM_train)
        season_losses_BFM_test = [convolutional_network_float_exp_weighted_loss(season_float_tensors_test[i][:, :, :-1, :, 1:-1].float(), season_BFM_tensors_test[i][:,:, :-1, :, 1:-1].float(), land_sea_masks, float_coord_masks[years_week_list.index(season_BFM_indexes_test[i])], torch.ones([1, 1, d-2, h, w-2]).to(device)) for i in range(len(season_BFM_tensors_test))]
        mean_season_loss_BFM_test = np.nanmean(season_losses_BFM_test)
    return mean_season_loss_CNN_train, mean_season_loss_CNN_test, mean_season_loss_BFM_train, mean_season_loss_BFM_test


def rmse_ga_season_2(list_input_tensors, indexes_train, indexes_test, list_float_tensors, list_float_coordinates, CNN_model, years_week_list, ga, season, path_mean_std):
    """this function computes the root mean square error of a list of profiles related to a specific ga and a season"""
    #split dataset in training and testing
    list_input_training = [list_input_tensors[i_tr] for i_tr in indexes_train]
    list_input_testing = [list_input_tensors[i_te] for i_te in indexes_test]
    #select a specific season
    season_tensors_train, season_indexes_train = select_season_tensors(list_input_training, season, [years_week_list[i_tr] for i_tr in indexes_train])
    season_float_tensors_train, season_float_indexes_train = select_season_tensors([list_float_tensors[i_tr] for i_tr in indexes_train], season, [years_week_list[i_tr] for i_tr in indexes_train])
    season_tensors_test, season_indexes_test = select_season_tensors(list_input_testing, season, [years_week_list[i_te] for i_te in indexes_test])
    season_float_tensors_test, season_float_indexes_test = select_season_tensors([list_float_tensors[i_te] for i_te in indexes_test], season, [years_week_list[i_te] for i_te in indexes_test])
    #select the profiles of a specific season applying GA mask
    ga_masks = [create_ga_mask(ga, (1,1,d,h,w)) for i in range(len(list_float_coordinates))]
    #prepare mean, std and land_sea_masks
    my_mean = torch.unsqueeze(torch.load(path_mean_std+ "/mean_tensor.pt")[:, 6, :, :, :], 1).to(device)
    my_std = torch.unsqueeze(torch.load(path_mean_std + "/std_tensor.pt")[:, 6, :, :, :], 1).to(device)
    land_sea_masks = load_land_sea_masks("dataset_training/land_sea_masks/")
    #model evaluation and loss evaluation, both for training and testing dataset
    CNN_model.eval()
    with torch.no_grad():
        season_output_tensors_train = [CNN_model(input_tensor) for input_tensor in season_tensors_train]
        season_output_tensors_train = [Denormalization(output_tensor, my_mean, my_std) for output_tensor in season_output_tensors_train]
        season_output_tensors_test = [CNN_model(input_tensor) for input_tensor in season_tensors_test]
        season_output_tensors_test = [Denormalization(output_tensor, my_mean, my_std) for output_tensor in season_output_tensors_test]
        #loss computation
        float_coord_masks = [generate_float_mask(list_float_coordinates[i]) * ga_masks[i].to(device) for i in range(len(list_float_coordinates))]
        season_losses_train = [convolutional_network_float_exp_weighted_loss(season_float_tensors_train[i][:, :, :-1, :, 1:-1].float(), season_output_tensors_train[i].float(), land_sea_masks, float_coord_masks[years_week_list.index(season_indexes_train[i])], torch.ones([1, 1, d-2, h, w-2]).to(device)) for i in range(len(season_output_tensors_train))]
        mean_season_loss_train = np.nanmean(season_losses_train)
        season_losses_test = [convolutional_network_float_exp_weighted_loss(season_float_tensors_test[i][:, :, :-1, :, 1:-1].float(), season_output_tensors_test[i].float(), land_sea_masks, float_coord_masks[years_week_list.index(season_indexes_test[i])], torch.ones([1, 1, d-2, h, w-2]).to(device)) for i in range(len(season_output_tensors_test))]
        mean_season_loss_test = np.nanmean(season_losses_test)
    return mean_season_loss_train, mean_season_loss_test


def rmse_ga_season_2_final(list_input_tensors, indexes_train, indexes_test, list_float_tensors, list_float_coordinates, CNN_model, years_week_list, ga, season, path_mean_std, threshold = 20):
    """this function computes the root mean square error of a list of profiles related to a specific ga and a season"""
    if threshold == 20:
        list_input_tensors = list_input_tensors[:157]
        indexes_train = [i for i in indexes_train if i < 157]
        indexes_test = [i for i in indexes_test if i < 157]
        list_float_tensors = list_float_tensors[:157]
        list_float_coordinates = list_float_coordinates[:157]
    #split dataset in training and testing
    list_input_training = [list_input_tensors[i_tr] for i_tr in indexes_train]
    list_input_testing = [list_input_tensors[i_te] for i_te in indexes_test]
    #prepare mean, std and land_sea_masks
    my_mean = torch.unsqueeze(torch.load(path_mean_std+ "/mean_tensor.pt")[:, 6, :, :, :], 1).to(device)
    my_std = torch.unsqueeze(torch.load(path_mean_std + "/std_tensor.pt")[:, 6, :, :, :], 1).to(device)
    land_sea_masks = load_land_sea_masks("dataset_training/land_sea_masks/")
    #select a specific season
    if season in list(dict_season.keys()):
        season_tensors_train, season_indexes_train = select_season_tensors(list_input_training, season, [years_week_list[i_tr] for i_tr in indexes_train])
        season_float_tensors_train, season_float_indexes_train = select_season_tensors([list_float_tensors[i_tr] for i_tr in indexes_train], season, [years_week_list[i_tr] for i_tr in indexes_train])
        season_tensors_test, season_indexes_test = select_season_tensors(list_input_testing, season, [years_week_list[i_te] for i_te in indexes_test])
        season_float_tensors_test, season_float_indexes_test = select_season_tensors([list_float_tensors[i_te] for i_te in indexes_test], season, [years_week_list[i_te] for i_te in indexes_test])
    elif season == "all_seasons":
        season_tensors_train = list_input_training
        season_indexes_train = [years_week_list[i_tr] for i_tr in indexes_train]
        season_float_tensors_train = [list_float_tensors[i_tr] for i_tr in indexes_train]
        season_tensors_test = list_input_testing
        season_indexes_test = [years_week_list[i_te] for i_te in indexes_test]
        season_float_tensors_test = [list_float_tensors[i_te] for i_te in indexes_test]
    #select the profiles of a specific season applying GA mask
    if ga in list(dict_ga.keys()):
        ga_masks = [create_ga_mask(ga, (1,1,d,h,w)) for i in range(len(list_float_coordinates))]
    elif ga == "all_ga":
        ga_masks = [torch.cat(land_sea_masks + [land_sea_masks[-1]], 2) for i in range(len(list_float_coordinates))]     
    #model evaluation and loss evaluation, both for training and testing dataset
    CNN_model.eval()
    with torch.no_grad():
        season_output_tensors_train = [CNN_model(input_tensor) for input_tensor in season_tensors_train]
        season_output_tensors_train = [Denormalization(output_tensor, my_mean, my_std) for output_tensor in season_output_tensors_train]
        season_output_tensors_test = [CNN_model(input_tensor) for input_tensor in season_tensors_test]
        season_output_tensors_test = [Denormalization(output_tensor, my_mean, my_std) for output_tensor in season_output_tensors_test]
        #loss computation
        float_coord_masks = [generate_float_mask(list_float_coordinates[i]) * ga_masks[i].to(device) for i in range(len(list_float_coordinates))]
        season_losses_train = [convolutional_network_float_exp_weighted_loss(season_float_tensors_train[i][:, :, :-1, :, 1:-1].float(), season_output_tensors_train[i].float(), land_sea_masks, float_coord_masks[years_week_list.index(season_indexes_train[i])], torch.ones([1, 1, d-2, h, w-2]).to(device)) for i in range(len(season_output_tensors_train))]
        season_losses_train = [np.sqrt(season_loss_train.float()) for season_loss_train in season_losses_train]
        mean_season_loss_train = np.nanmean(season_losses_train)
        season_losses_test = [convolutional_network_float_exp_weighted_loss(season_float_tensors_test[i][:, :, :-1, :, 1:-1].float(), season_output_tensors_test[i].float(), land_sea_masks, float_coord_masks[years_week_list.index(season_indexes_test[i])], torch.ones([1, 1, d-2, h, w-2]).to(device)) for i in range(len(season_output_tensors_test))]
        season_losses_test = [np.sqrt(season_loss_test) for season_loss_test in season_losses_test]
        mean_season_loss_test = np.nanmean(season_losses_test)
    return mean_season_loss_train, mean_season_loss_test


def RMSE_ga_season(path_job, years_week_indexes):
    path_mean_std = path_job + "/results_training_2/mean_and_std_tensors"
    index_training_2 = read_list(path_job + "/results_training_2/P_l/40/lrc_0.001" + "/index_training.txt")
    index_int_testing_2 = read_list(path_job + "/results_training_2/P_l/40/lrc_0.001" + "/index_internal_testing.txt")
    index_ext_testing_2 = read_list(path_job + "/results_training_2/P_l/40/lrc_0.001" + "/index_external_testing.txt")
    #float dataset
    list_float_tensors = re_load_old_float_tensors("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset", years_week_indexes)
    list_float_coordinates = [compute_profile_coordinates(torch.unsqueeze(float_tensor[:, -1, :, :, :], 1)) for float_tensor in list_float_tensors]
    list_float_tensors = [float_tensor[:, :, :-1, :, :] for float_tensor in list_float_tensors]
    #total input dataset phase 2
    input_dataset_2 = re_load_float_input_data("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset", years_week_indexes)
    input_dataset_2 = tmp_Normalization(input_dataset_2, "2p", path_mean_std)
    #call CNN and evaluate RMSE                   
    CNN_model = CompletionN()
    checkpoint_CNN = torch.load(path_job + "/results_training_2/model_checkpoint_2.pth", map_location=device)
    CNN_model.load_state_dict(checkpoint_CNN['model_state_dict'])
    loss_results = []
    for my_ga in list(dict_ga.keys()):
        for season in list(dict_season.keys()):
            list_loss_ga_season = []
            list_loss_ga_season.append(my_ga)
            list_loss_ga_season.append(season)
            loss_ga_season_train, loss_ga_season_test = rmse_ga_season_2(input_dataset_2, index_training_2, index_ext_testing_2 + index_int_testing_2, list_float_tensors, list_float_coordinates, CNN_model, years_week_indexes, my_ga, season, path_mean_std)
            list_loss_ga_season.append([loss_ga_season_train, loss_ga_season_test])
            loss_results.append(list_loss_ga_season)
    #Save RMSE results in a .txt file
    for list_loss in loss_results:
        with open(path_job + "/results_training_2/" + "losses_ga_season.txt", "a") as f:
            f.write("\n")
            f.write(",".join([list_loss[0], list_loss[1], str(list_loss[2])]))
    return None


def RMSE_ensemble_ga_season(path_job, years_week_indexes, n_ens):
    path_results_ensemble = path_job + "/results_training_2_ensemble"
    path_mean_std =path_results_ensemble + "/mean_and_std_tensors"
    path_lr = path_results_ensemble + "/P_l/20/lrc_0.001"
    list_index_training_2 = [read_list(path_lr + "/ensemble_model_" + str(i_ens) + "/index_training.txt") for i_ens in range(n_ens)]
    list_index_int_testing_2 = [read_list(path_lr + "/ensemble_model_" + str(i_ens) + "/index_internal_testing.txt") for i_ens in range(n_ens)]
    list_index_ext_testing_2 = [read_list(path_lr + "/ensemble_model_" + str(i_ens) + "/index_external_testing.txt") for i_ens in range(n_ens)]
    #float dataset
    list_float_tensors = re_load_old_float_tensors("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset", years_week_indexes)
    list_float_coordinates = [compute_profile_coordinates(torch.unsqueeze(float_tensor[:, -1, :, :, :], 1)) for float_tensor in list_float_tensors]
    list_float_tensors = [float_tensor[:, :, :-1, :, :] for float_tensor in list_float_tensors]
    #total input dataset phase 2
    input_dataset_2 = re_load_float_input_data("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset", years_week_indexes)
    input_dataset_2 = tmp_Normalization(input_dataset_2, "2p", path_mean_std)
    tensor_loss_ga_season_ens = torch.zeros([n_ens, 2, len(list(dict_ga.keys())), len(list(dict_season.keys()))])
    #call CNN models
    list_CNN_model = [CompletionN() for i in range(n_ens)]
    list_checkpoints_CNN = [torch.load(path_lr + "/ensemble_model_" + str(i_ens) + "/" + 'model_checkpoint_2_ens_' + str(i_ens) + '.pth', map_location=device) for i_ens in range(10)]
    for i_ens in range(len(list_CNN_model)):
        list_CNN_model[i_ens].load_state_dict(list_checkpoints_CNN[i_ens]['model_state_dict'])
    #start RMSE evaluation
    for i_ens in range(n_ens):
        for my_ga in list(dict_ga.keys()):
            for my_season in list(dict_season.keys()):
                loss_ga_season_train, loss_ga_season_test = rmse_ga_season_2(input_dataset_2, list_index_training_2[i_ens], list_index_ext_testing_2[i_ens] + list_index_int_testing_2[i_ens], list_float_tensors, list_float_coordinates, list_CNN_model[i_ens], years_week_indexes, my_ga, my_season, path_mean_std)
                tensor_loss_ga_season_ens[i_ens, 0, list(dict_ga.keys()).index(my_ga), list(dict_season.keys()).index(my_season)] = float(loss_ga_season_train)
                tensor_loss_ga_season_ens[i_ens, 1, list(dict_ga.keys()).index(my_ga), list(dict_season.keys()).index(my_season)] = float(loss_ga_season_test)
    mean_loss_ga_season_ensemble = torch.mean(tensor_loss_ga_season_ens, dim=0)
    #Save the results on a txt file
    for i_ga in range(len(list(dict_ga.keys()))):
        for i_s in range(len(list(dict_season.keys()))):
            with open(path_results_ensemble + "/losses_ga_season.txt", "a") as f:
                f.write("\n")
                f.write(",".join([list(dict_ga.keys())[i_ga], list(dict_season.keys())[i_s], "train", str(mean_loss_ga_season_ensemble[0, i_ga, i_s])]))
                f.write("\n")
                f.write(",".join([list(dict_ga.keys())[i_ga], list(dict_season.keys())[i_s], "test", str(mean_loss_ga_season_ensemble[1, i_ga, i_s])]))
    return None



def RMSE_ensemble_ga(path_job, years_week_indexes, n_ens):
    path_results_ensemble = path_job + "/results_training_2_ensemble"
    path_mean_std =path_results_ensemble + "/mean_and_std_tensors"
    path_lr = path_results_ensemble + "/P_l/20/lrc_0.001"
    list_index_training_2 = [read_list(path_lr + "/ensemble_model_" + str(i_ens) + "/index_training.txt") for i_ens in range(n_ens)]
    list_index_int_testing_2 = [read_list(path_lr + "/ensemble_model_" + str(i_ens) + "/index_internal_testing.txt") for i_ens in range(n_ens)]
    list_index_ext_testing_2 = [read_list(path_lr + "/ensemble_model_" + str(i_ens) + "/index_external_testing.txt") for i_ens in range(n_ens)]
    #float dataset
    list_float_tensors = re_load_old_float_tensors("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset", years_week_indexes[:105])
    list_float_coordinates = [compute_profile_coordinates(torch.unsqueeze(float_tensor[:, -1, :, :, :], 1)) for float_tensor in list_float_tensors]
    list_float_tensors = [float_tensor[:, :, :-1, :, :] for float_tensor in list_float_tensors]
    print("count number floats", [torch.count_nonzero(float_tensor[:, -1, :, :, :]) for float_tensor in list_float_tensors], flush=True)
    #total input dataset phase 2
    input_dataset_2 = re_load_float_input_data("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset", years_week_indexes)
    input_dataset_2 = tmp_Normalization(input_dataset_2, "2p", path_mean_std)
    tensor_loss_ga_season_ens = torch.zeros([n_ens, 2, len(list(dict_ga.keys()))])
    #call CNN models
    list_CNN_model = [CompletionN() for i in range(n_ens)]
    list_checkpoints_CNN = [torch.load(path_lr + "/ensemble_model_" + str(i_ens) + "/" + 'model_checkpoint_2_ens_' + str(i_ens) + '.pth', map_location=device) for i_ens in range(10)]
    for i_ens in range(len(list_CNN_model)):
        list_CNN_model[i_ens].load_state_dict(list_checkpoints_CNN[i_ens]['model_state_dict'])
    #start RMSE evaluation
    for i_ens in range(n_ens):
        for my_ga in list(dict_ga.keys()):
            loss_ga_season_train, loss_ga_season_test = rmse_ga_season_2_final(input_dataset_2, list_index_training_2[i_ens], list_index_ext_testing_2[i_ens] + list_index_int_testing_2[i_ens], list_float_tensors, list_float_coordinates, list_CNN_model[i_ens], years_week_indexes, my_ga, "all_seasons", path_mean_std)
            tensor_loss_ga_season_ens[i_ens, 0, list(dict_ga.keys()).index(my_ga)] = float(loss_ga_season_train)
            tensor_loss_ga_season_ens[i_ens, 1, list(dict_ga.keys()).index(my_ga)] = float(loss_ga_season_test)
    mean_loss_ga_season_ensemble = torch.mean(tensor_loss_ga_season_ens, dim=0)
    #Save the results on a txt file
    for i_ga in range(len(list(dict_ga.keys()))):
        with open(path_results_ensemble + "/losses_ga_final.txt", "a") as f:
            f.write("\n")
            f.write(",".join([list(dict_ga.keys())[i_ga], "train", str(mean_loss_ga_season_ensemble[0, i_ga])]))
            f.write("\n")
            f.write(",".join([list(dict_ga.keys())[i_ga], "test", str(mean_loss_ga_season_ensemble[1, i_ga])]))
    return None
    

def RMSE_ensemble_season(path_job, years_week_indexes, n_ens, threshold=20, behavior_season = "traditional"):
    path_results_ensemble = path_job + "/results_training_2_ensemble"
    path_mean_std =path_results_ensemble + "/mean_and_std_tensors"
    path_lr = path_results_ensemble + "/P_l/20/lrc_0.001"
    list_index_training_2 = [read_list(path_lr + "/ensemble_model_" + str(i_ens) + "/index_training.txt") for i_ens in range(n_ens)]
    list_index_int_testing_2 = [read_list(path_lr + "/ensemble_model_" + str(i_ens) + "/index_internal_testing.txt") for i_ens in range(n_ens)]
    list_index_ext_testing_2 = [read_list(path_lr + "/ensemble_model_" + str(i_ens) + "/index_external_testing.txt") for i_ens in range(n_ens)]
    #float dataset
    list_float_tensors = re_load_old_float_tensors("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset", years_week_indexes)
    list_float_coordinates = [compute_profile_coordinates(torch.unsqueeze(float_tensor[:, -1, :, :, :], 1)) for float_tensor in list_float_tensors]
    list_float_tensors = [float_tensor[:, :, :-1, :, :] for float_tensor in list_float_tensors]
    print("count number floats", [torch.count_nonzero(float_tensor[:, -1, :, :, :]) for float_tensor in list_float_tensors], flush=True)
    #total input dataset phase 2
    input_dataset_2 = re_load_float_input_data("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset", years_week_indexes)
    input_dataset_2 = tmp_Normalization(input_dataset_2, "2p", path_mean_std)
    tensor_loss_ga_season_ens = torch.zeros([n_ens, 2, len(list(dict_season.keys()))])
    #call CNN models
    list_CNN_model = [CompletionN() for i in range(n_ens)]
    list_checkpoints_CNN = [torch.load(path_lr + "/ensemble_model_" + str(i_ens) + "/" + 'model_checkpoint_2_ens_' + str(i_ens) + '.pth', map_location=device) for i_ens in range(10)]
    for i_ens in range(len(list_CNN_model)):
        list_CNN_model[i_ens].load_state_dict(list_checkpoints_CNN[i_ens]['model_state_dict'])
    #start RMSE evaluation
    for i_ens in range(n_ens):
        if behavior_season == "traditional":
            list_season = list_traditional_season
        elif behavior_season == "bloom_DCM":
            list_season = list_bloom_DCM_season
        for my_season in list_season:    #list(dict_season.keys()):
            loss_ga_season_train, loss_ga_season_test = rmse_ga_season_2_final(input_dataset_2, list_index_training_2[i_ens], list_index_ext_testing_2[i_ens] + list_index_int_testing_2[i_ens], list_float_tensors, list_float_coordinates, list_CNN_model[i_ens], years_week_indexes, "all_ga", my_season, path_mean_std)
            tensor_loss_ga_season_ens[i_ens, 0, list(dict_season.keys()).index(my_season)] = float(loss_ga_season_train)
            tensor_loss_ga_season_ens[i_ens, 1, list(dict_season.keys()).index(my_season)] = float(loss_ga_season_test)
    mean_loss_ga_season_ensemble = torch.mean(tensor_loss_ga_season_ens, dim=0)
    #Save the results on a txt file
    for i_s in range(len(list(dict_season.keys()))):
        with open(path_results_ensemble + "/losses_season_final.txt", "a") as f:
            f.write("\n")
            f.write(",".join([list(dict_season.keys())[i_s], "train", str(mean_loss_ga_season_ensemble[0, i_s])]))
            f.write("\n")
            f.write(",".join([list(dict_season.keys())[i_s], "test", str(mean_loss_ga_season_ensemble[1, i_s])]))
    return None


#OLD functions

def create_ga_mask_full(my_ga, list_float_coordinates, tensor_shape):
    """this function creates a mask of the same dimension of the tensor that identify the coordinates of a specific geografic area"""
    my_lat_interval, my_long_interval = dict_ga[my_ga][1], dict_ga[my_ga][0]
    my_lat_index_interval = [transform_latitude_index(my_lat_interval[0]), transform_latitude_index(my_lat_interval[1])]
    my_long_index_interval = [transform_longitude_index(my_long_interval[0]), transform_longitude_index(my_long_interval[1])]
    print("my lat index interval", my_lat_index_interval)
    print("my long index interval", my_long_index_interval)
    ga_mask = torch.zeros(tensor_shape)    #torch.ones(tensor_shape)
    for float_coord in list_float_coordinates:
        if my_long_index_interval[0] <= float_coord[0] <= my_long_index_interval[1] and my_lat_index_interval[0] <= float_coord[1] <= my_lat_index_interval[1]:
            print("float coord ga", float_coord)
            ga_mask[:, :, :, float_coord[0], float_coord[1]] = 1.0
    return ga_mask


def compute_rmse_ga_season_2(list_tensors, list_float_tensors, list_float_coordinates, ga, season, years_week_list, exp_weights, CNN_model, path_mean_std):
    """this function computes the root mean square error of a list of profiles related to a specific ga and a season"""
    #select a specific season
    season_tensors, season_index = select_season_tensors(list_tensors, season, years_week_list)
    season_float_tensors, season_float_index = select_season_tensors(list_float_tensors, season, years_week_list)
    #select the profiles of a specific season applying GA mask
    ga_masks = [create_ga_mask(ga, (1,1,d,h,w)) for i in range(len(list_float_coordinates))]
    #prepare mean, std and land_sea_masks
    my_mean = torch.unsqueeze(torch.load(path_mean_std+ "/mean_tensor.pt")[:, 6, :, :, :], 1).to(device)
    my_std = torch.unsqueeze(torch.load(path_mean_std + "/std_tensor.pt")[:, 6, :, :, :], 1).to(device)
    land_sea_masks = load_land_sea_masks("dataset_training/land_sea_masks/")
    #model evaluation and loss evaluation
    CNN_model.eval()
    with torch.no_grad():
        season_output_tensors = [CNN_model(input_tensor) for input_tensor in season_tensors]
        season_output_tensors = [Denormalization(output_tensor, my_mean, my_std) for output_tensor in season_output_tensors]
        #loss computation
        float_coord_masks = [generate_float_mask(list_float_coordinates[i]) * ga_masks[i].to(device) for i in range(len(list_float_coordinates))]
        #change float tensors shape
        season_losses = [convolutional_network_float_exp_weighted_loss(season_float_tensors[i][:, :, :-1, :, 1:-1].float(), season_output_tensors[i].float(), land_sea_masks, float_coord_masks[i], exp_weights.to(device)) for i in range(len(season_output_tensors))]
        season_loss = np.mean(season_losses)
        #season_losses_ga = [convolutional_network_float_exp_weighted_loss(season_float_tensors[i][:, :, :-1, :, 1:-1].float(), season_output_tensors[i].float(), land_sea_masks, ga_masks_full[i].to(device), exp_weights.to(device)) for i in range(len(season_output_tensors))]
        #season_loss_ga = np.mean(season_losses_ga)
        print("season loss", season_loss, flush=True)
        #print("season loss ga", season_loss_ga, flush=True)
    return season_loss



def compute_rmse_ga_season_2_ensemble(list_tensors, list_float_tensors, list_float_coordinates, ga, season, years_week_duplicates_list, exp_weights, list_CNN_model, path_mean_std):
    """this function computes the root mean square error of a list of profiles related to a specific ga and a season"""
    #select a specific season
    season_tensors, season_index = select_season_tensors(list_tensors, season, years_week_duplicates_list)
    season_float_tensors, season_float_index = select_season_tensors(list_float_tensors, season, years_week_duplicates_list)
    #select the profiles of a specific season
    ga_masks = [create_ga_mask(ga, (1, 1, d, h, w)) for i in range(len(list_float_coordinates))]
    #prepare mean, std and land_sea_masks
    my_mean = torch.unsqueeze(torch.load(path_mean_std+ "/mean_tensor.pt")[:, 6, :, :, :], 1).to(device)
    my_std = torch.unsqueeze(torch.load(path_mean_std + "/std_tensor.pt")[:, 6, :, :, :], 1).to(device)
    land_sea_masks = load_land_sea_masks("dataset_training/land_sea_masks/")
    #model evaluation and loss evaluation
    for CNN_model in list_CNN_model:
        CNN_model.eval()
    with torch.no_grad():
        season_output_tensors = [compute_3D_ensemble_mean_std(input_tensor, list_CNN_model, path_mean_std)[0] for input_tensor in season_tensors]
        season_output_tensors = [Denormalization(output_tensor, my_mean, my_std) for output_tensor in season_output_tensors]
        #loss computation
        float_coord_masks = [generate_float_mask(list_float_coordinates[i]) * ga_masks[i].to(device) for i in range(len(list_float_coordinates))]
        print("end float ga masks creation", flush=True)
        print("count non zero elements final masks", torch.count_nonzero(float_coord_masks[2]))
        #change float tensors shape
        season_losses = [convolutional_network_float_exp_weighted_loss(season_float_tensors[i][:, :, :-1, :, 1:-1].float(), season_output_tensors[i].float(), land_sea_masks, float_coord_masks[i], exp_weights.to(device)) for i in range(len(season_output_tensors))]
        season_loss = np.mean(season_losses)
        print("season loss", season_loss, flush=True)
    return season_loss