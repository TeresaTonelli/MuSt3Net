#This script contains the functions fro the generatiorn of the dataset that we need to train our netwprk
#USEFUL TO SAVE RAM 


import numpy as np
import torch
import os
import datetime
import pickle

from get_dataset import concatenate_tensors, get_list_float_tensor_weeks, get_list_model_tensor_weeks, get_list_float_tensor_year_weeks, get_list_model_tensor_year_weeks
from utils_function import *
from utils_mask import *
from normalization import Normalization



def generate_land_sea_masks(path_saving, biogeoch_var, year):
    """this function generates and saves the land_sea_masks"""
    biogeoch_train_dataset, list_biogeoch_weeks = get_list_model_tensor_year_weeks(biogeoch_var, year)
    biogeoch_train_dataset, list_biogeoch_weeks = re_order_weeks(list_biogeoch_weeks, biogeoch_train_dataset)
    #Generation of land - sea masks
    land_sea_masks = []
    for i_d in range(int((depth_interval[1] - depth_interval[0]) / resolution[2] + 1) - 1):
        land_sea_mask = generate_sea_land_mask(biogeoch_train_dataset[0], i_d)
        land_sea_masks.append(land_sea_mask)
    #saves land_sea_masks
    for i in range(len(land_sea_masks)):
        land_sea_path = path_saving + "/land_sea_masks/"
        torch.save(land_sea_masks[i], land_sea_path + "depth_" + str(i) + ".pt")
    return None


#generate_land_sea_masks("dataset_training", "P_l", 2019)


def generate_old_total_dataset(biogeoch_var, year, path_saving):
    """this function concatenate the physycs and biogeochemical variable in a single tensor, and save it"""
    physics_train_dataset, list_physics_weeks = get_list_model_tensor_year_weeks("physics_vars", year)
    biogeoch_train_dataset, list_biogeoch_weeks = get_list_model_tensor_year_weeks(biogeoch_var, year)
    #re_order the tensors and the weeks
    physics_train_dataset, list_physics_weeks = re_order_weeks(list_physics_weeks, physics_train_dataset)
    biogeoch_train_dataset, list_biogeoch_weeks = re_order_weeks(list_biogeoch_weeks, biogeoch_train_dataset)
    #concatenate together physical and biogeochemical variables
    old_total_dataset = [concatenate_tensors(physics_train_dataset[i], biogeoch_train_dataset[i], axis=1) for i in range(len(physics_train_dataset))]
    #save old total dataset
    old_dataset_path = path_saving + "/old_total_dataset/" + str(year) + "/" + str(biogeoch_var) + "/"
    if not os.path.exists(old_dataset_path):
        os.makedirs(old_dataset_path)
    for i in range(len(old_total_dataset)):
        torch.save(old_total_dataset[i], old_dataset_path + "datetime_" + str(i+1) +".pt")
    return None

#generate_old_total_dataset("P_l", 2022, "dataset_training")




def write_list(a_list, file_dir):
    # store list in binary file so 'wb' mode
    with open(file_dir, 'wb') as fp:
        pickle.dump(a_list, fp)
        print('Done writing list into a binary file')


def read_list(file_dir):
    # for reading also binary mode is important
    with open(file_dir, 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list
    