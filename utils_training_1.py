#In this script we write all the functions useful for preparing data from training 1


import numpy as np
import torch
import pandas as pd
import os
import random
from utils_generation_train_1p import read_list


def prepare_paths(name_datetime_folder, biogeoch_var_to_predict, epoch_c, epoch_pretrain, lr_c):
    """this function defines all the path in which we will save the results of training 1 procedure"""
    path_job = "results_job_" + name_datetime_folder
    if not os.path.exists(path_job):
        print("new path created")
        os.mkdir(path_job)
    path_results = path_job + "/results_training_1"           
    if not os.path.exists(path_results):                   
        os.mkdir(path_results)
    path_mean_std = path_results + "/mean_and_std_tensors"
    if not os.path.exists(path_mean_std):
        os.makedirs(path_mean_std)
    path_land_sea_masks = path_results + "/land_sea_masks"
    if not os.path.exists(path_land_sea_masks):
        os.makedirs(path_land_sea_masks)
    path_configuration = path_results + "/" + str(biogeoch_var_to_predict) + "/" + str(epoch_c + epoch_pretrain) 
    if not os.path.exists(path_configuration):
        os.makedirs(path_configuration)
    path_lr = path_configuration + "/lrc_" + str(lr_c) 
    if not os.path.exists(path_lr):
        os.makedirs(path_lr)
    path_losses = path_lr + "/losses"
    if not os.path.exists(path_losses):
        os.makedirs(path_losses)
    path_model = path_lr + "/partial_models/"
    if not os.path.exists(path_model):
        os.mkdir(path_model)
    path_plots = path_lr + "/plots"
    if not os.path.exists(path_plots):
        os.makedirs(path_plots)
    return path_job, path_results, path_mean_std, path_land_sea_masks, path_configuration, path_lr, path_losses, path_model, path_plots


def reload_paths_1p(path_job, biogeoch_var, n_epochs_1p, epoch_pretrain_1p, lr_1p):
    """this function return the paths from an old path_job"""
    path_results = path_job + "/results_training_1"           
    path_mean_std = path_results + "/mean_and_std_tensors"
    path_land_sea_masks = path_results + "/land_sea_masks"
    path_configuration = path_results + "/" + str(biogeoch_var) + "/" + str(n_epochs_1p + epoch_pretrain_1p) 
    path_lr = path_configuration + "/lrc_" + str(lr_1p) 
    path_losses = path_lr + "/losses"
    path_model = path_lr + "/partial_models/"
    path_plots = path_plots = path_lr + "/plots"
    return path_results, path_mean_std, path_land_sea_masks, path_configuration, path_lr, path_losses, path_model, path_plots


def prepare_paths_2(path_job, biogeoch_var, n_epochs_2, epoch_pretrain_2, lr_2):
    path_results_2 = path_job + "/results_training_2" 
    path_configuration_2 = path_results_2 + "/" + str(biogeoch_var) + "/" + str(n_epochs_2 + epoch_pretrain_2) 
    if not os.path.exists(path_configuration_2):
        os.makedirs(path_configuration_2)
    path_mean_std_2 = path_results_2 + "/mean_and_std_tensors"
    if not os.path.exists(path_mean_std_2):
        os.makedirs(path_mean_std_2)
    path_lr_2 = path_configuration_2 + "/lrc_" + str(lr_2) 
    if not os.path.exists(path_lr_2):
        os.makedirs(path_lr_2)
    path_losses_2 = path_lr_2 + "/losses"
    if not os.path.exists(path_losses_2):
        os.makedirs(path_losses_2)
    path_model_2 = path_lr_2 + "/partial_models/"
    if not os.path.exists(path_model_2):
        os.mkdir(path_model_2)
    path_plots_2 = path_lr_2 + "/plots"
    if not os.path.exists(path_plots_2):
        os.makedirs(path_plots_2)
    return path_results_2, path_configuration_2, path_mean_std_2, path_lr_2, path_losses_2, path_model_2, path_plots_2


def generate_random_week_indexes(desired_sum, week_per_year):
    """it random samples week_per_year random number whose sum is exactly the m, numer of tensor that I want for a specific year"""
    # Step 1: Generate 52 random numbers
    numbers = np.random.randint(1, int(desired_sum / week_per_year + 1), size=week_per_year)
    # Step 2: Scale the numbers so that their sum equals desired_sum
    current_sum = np.sum(numbers)
    # Adjust the numbers by scaling them
    scaling_factor = desired_sum / current_sum
    scaled_numbers = np.round(numbers * scaling_factor)
    # Step 3: Make sure the sum is exactly 200
    # If there is any rounding error, adjust the largest or smallest values
    final_sum = np.sum(scaled_numbers)
    difference = desired_sum - final_sum
    # Adjust the numbers to match the desired sum without going negative
    if difference != 0:
        if difference > 0:
            # If the difference is positive, just add 1 to the first 'difference' numbers
            for j in range(int(difference)):
                scaled_numbers[j] += 1
        elif difference < 0:
            # If the difference is negative, subtract 1, but only from numbers that are greater than 0
            for j in range(int(abs(difference))):
                if scaled_numbers[j] > 0:
                    scaled_numbers[j] -= 1
    return scaled_numbers


def generate_random_duplicates_indexes(week_indexes, n_dupl_per_week):
    """this function samples k duplicates for each week, where k is the element of week list refered to that week"""
    total_dupl_indexes = []
    for index_week in range(len(week_indexes)):
        k = week_indexes[index_week]
        dupl_indexes = random.sample(range(int(n_dupl_per_week)), k)
        total_dupl_indexes.append(dupl_indexes)
    return total_dupl_indexes


def add_year_indexes(year, week_dupl_indexes):
    """this function takes an year an add it to the week and duplicates indexes"""
    year_week_dupl_indexes = [[year] + week_dupl_indexes[i] for i in range(len(week_dupl_indexes))]
    return year_week_dupl_indexes


def sort_depths(list_file_depths):
    """this function sorts a list of objects depending on their depths, expressed as a string in their name"""
    list_file_depths.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    return list_file_depths


def load_land_sea_masks(directory_land_sea_masks):
    """this function load the land_sea_masks tensors"""
    land_sea_masks = []
    list_masks = os.listdir(directory_land_sea_masks)
    print("list masks", list_masks)
    list_masks = sort_depths(list_masks)
    print("list masks sorted", list_masks)
    for file_mask in list_masks:
        land_sea_mask = torch.load(directory_land_sea_masks + file_mask, map_location=torch.device('cpu'))
        land_sea_masks.append(land_sea_mask)
    print("type land sea mask", type(land_sea_masks[1]))
    return land_sea_masks


def load_old_total_tensor(directory_old_total_dataset, index, years_week_dupl_indexes):
    """this function loads the old total tensor which corresponds to a specific train / test data"""
    year, week = years_week_dupl_indexes[index][0], years_week_dupl_indexes[index][1]
    old_total_tensor = torch.load(directory_old_total_dataset + str(year) + "/P_l/datetime_" + str(week) + ".pt")
    return old_total_tensor


def load_tensors(tensors_directory_year, week_dupl_indexes):
    """this function takes a list of indexes and load the relative tensors"""
    total_dataset = []
    for i in range(len(week_dupl_indexes)):
        week_dupl_index = week_dupl_indexes[i]
        tensor = torch.load(tensors_directory_year + "week_" + str(week_dupl_index[0]) + "/duplicate_" + str(week_dupl_index[1]) + "/tensor.pt")
        total_dataset.append(tensor)
    return total_dataset


def re_load_tensors(tensors_directory, years_week_dupl_indexes):
    """this function takes a list of indexes and load the relative tensors"""
    total_dataset = []
    for i in range(len(years_week_dupl_indexes)):
        year_week_dupl_index = years_week_dupl_indexes[i]
        tensor = torch.load(tensors_directory + "/P_l/" + str(year_week_dupl_index[0]) + "/week_" + str(year_week_dupl_index[1]) + "/duplicate_" + str(year_week_dupl_index[2]) + "/tensor.pt")
        total_dataset.append(tensor)
    return total_dataset


def load_transp_lat_coordinates(tensors_directory_year, week_dupl_indexes):
    """this function takes a list of indexes and load the relative list of transposed latitude coordinates"""
    transp_lat_coordinates = []
    for i in range(len(week_dupl_indexes)):
        week_dupl_index = week_dupl_indexes[i]
        transp_lat_coord = read_list(tensors_directory_year + "week_" + str(week_dupl_index[0]) + "/duplicate_" + str(week_dupl_index[1]) + "/file_transpose_latitudes.txt")
        transp_lat_coordinates.append(transp_lat_coord)
    return transp_lat_coordinates


def re_load_transp_lat_coordinates(tensors_directory, years_week_dupl_indexes):
    """this function takes a list of indexes and load the list of transposed latitude coordinates"""
    transp_lat_coordinates = []
    for i in range(len(years_week_dupl_indexes)):
        year_week_dupl_index = years_week_dupl_indexes[i]
        transp_lat_coord = read_list(tensors_directory + "/P_l/" + str(year_week_dupl_index[0]) + "/week_" + str(year_week_dupl_index[1]) + "/duplicate_" + str(year_week_dupl_index[2]) + "/file_transpose_latitudes.txt")
        transp_lat_coordinates.append(transp_lat_coord)
    return transp_lat_coordinates


def generate_training_dataset_1(tensors_directory, biogeoch_var, years, n, n_dupl_per_week):
    """this function loads the tensor to generate the total dataset"""
    desired_n_tensor_per_year= int(n/len(years))
    total_dataset = []
    transposed_latitudes_coordinates = []
    years_week_dupl_indexes = []
    for year in years:
        #generate a list of k (k = n week of a specific year) random number whose sum is = n_year
        week_indexes = generate_random_week_indexes(desired_n_tensor_per_year, 52)
        week_indexes = [int(week_indexes[i]) for i in range(len(week_indexes))]
        #for each week, sample k duplicates, where k is the element is the previous list refered to that week
        duplicates_indexes = generate_random_duplicates_indexes(week_indexes, n_dupl_per_week)
        #build a unique list of week and duplicates indexes
        week_dupl_indexes = [[i+1, duplicates_indexes[i][j]] for i in range(len(week_indexes)) for j in range(len(duplicates_indexes[i]))]
        #now, load the tensor realtive to the sampled year, week and duplicate, and put inside total_dataset list
        total_year_dataset = load_tensors(tensors_directory + "/" + str(biogeoch_var) + "/" + str(year) + "/", week_dupl_indexes)
        #now, load the transposed latitudes coordinates
        transposed_year_latitudes_coordinates = load_transp_lat_coordinates(tensors_directory + "/" + str(biogeoch_var) + "/" + str(year) + "/", week_dupl_indexes)
        #now modifies the indexes relative to week and duplicates, adding the year
        year_week_dupl_indexes = add_year_indexes(year, week_dupl_indexes)
        #append the result of a specific year at the end of the for loop iteration
        #total_dataset.append(total_year_dataset)
        #transposed_latitudes_coordinates.append(transposed_year_latitudes_coordinates)
        #years_week_dupl_indexes.append(year_week_dupl_indexes)
        #extend the result of a specific year at the end of the for loop iteration
        total_dataset.extend(total_year_dataset)
        transposed_latitudes_coordinates.extend(transposed_year_latitudes_coordinates)
        years_week_dupl_indexes.extend(year_week_dupl_indexes)
    return total_dataset, transposed_latitudes_coordinates, years_week_dupl_indexes


def split_train_test_data(total_dataset, train_perc=0.8, int_test_perc=0.1, ext_test_perc=0.1):
    """"this function creates the train, internal_test and external_test dataset"""
    index_testing = random.sample(range(len(total_dataset)), int(len(total_dataset) * (ext_test_perc + int_test_perc)))  
    index_internal_testing = random.sample(index_testing, int(len(index_testing) / 2))
    index_external_testing = [i for i in index_testing if i not in index_internal_testing]
    index_training = [i for i in range(len(total_dataset)) if i not in index_testing]
    test_dataset = [total_dataset[i] for i in index_external_testing] 
    internal_test_dataset = [total_dataset[i] for i in index_internal_testing]                       
    train_dataset = [total_dataset[i] for i in index_training]
    del total_dataset
    return train_dataset, internal_test_dataset, test_dataset, index_training, index_internal_testing, index_external_testing



def recreate_train_test_datasets(total_dataset_norm, indexes_train, indexes_internal_test, indexes_external_test):
    """this function recreates the train dataset, the internal test dataset and the external test dataset"""
    train_dataset = [total_dataset_norm[i] for i in indexes_train]
    internal_test_dataset = [total_dataset_norm[i] for i in indexes_internal_test]
    test_dataset = [total_dataset_norm[i] for i in indexes_external_test]
    return train_dataset, internal_test_dataset, test_dataset
    #for i_train in indexes_train:
     #   y_w_d_train_index = years_week_dupl_indexes[i_train]
      #  train_tensor = torch.load(tensors_directory + "/P_l/" + str(y_w_d_train_index[0]) + "/week_" + str(y_w_d_train_index[1]) + "/duplicate_" + str(y_w_d_train_index[2]) + "/tensor.pt")
       # train_dataset.append(train_tensor)

