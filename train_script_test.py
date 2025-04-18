"""
Implementation of the training routine for the 3D CNN with GAN
- train_dataset : list/array of 5D (or 5D ?) tensor in form (bs, input_channels, D_in, H_in, W_in)
"""
import numpy as np
import torch.nn as nn
from torch.optim import Adadelta
import matplotlib.pyplot as plt
import random
import datetime

#from convolutional_network import CompletionN
#from losses import convolutional_network_weighted_loss, convolutional_network_float_weighted_loss, convolutional_network_exp_weighted_loss, convolutional_network_float_exp_weighted_loss
#from mean_pixel_value import MV_pixel
#from utils_mask import generate_input_mask, generate_sea_land_mask
#from normalization import Normalization
#from denormalization import Denormalization
#from get_dataset import *
#from plot_error import Plot_Error
#from plot_results import *
#from utils_function import *
#from utils_mask import generate_float_mask, compute_weights, compute_exponential_weights
#from generation_training_dataset import generate_dataset_phase_1_saving, generate_dataset_phase_2_saving
#from utils_training_1 import prepare_paths, reload_paths_1p, prepare_paths_2, generate_random_week_indexes, generate_random_duplicates_indexes, add_year_indexes, load_tensors, load_transp_lat_coordinates, generate_training_dataset_1, generate_training_dataset_1_winter, generate_training_dataset_1_summer,split_train_test_data, load_land_sea_masks, load_old_total_tensor, re_load_tensors, recreate_train_test_datasets, re_load_transp_lat_coordinates
#from utils_generation_train_1p import write_list, read_list
#from training_testing_functions import training_1p, testing_1p, training_2p, testing_2p


from convolutional_network import CompletionN
from normalization import Normalization
from get_dataset import *
from plot_results import *
from utils_function import *
from generation_training_dataset import generate_dataset_phase_2_saving
from utils_training_1 import prepare_paths, reload_paths_1p, prepare_paths_2, generate_training_dataset_1,split_train_test_data, load_land_sea_masks, load_old_total_tensor, re_load_tensors, recreate_train_test_datasets, re_load_transp_lat_coordinates
from utils_generation_train_1p import write_list, read_list
from training_testing_functions import training_1p, testing_1p, training_2p, testing_2p


#3 parameters to define the jobs pypeline
first_run_id = 0
end_train_1p = 0
end_1p = 0
path_job = ""


num_channel = number_channel  
name_datetime_folder = str(datetime.datetime.utcnow())


#1: create paths 1 phase
if first_run_id == 0:
    path_job, path_results, path_mean_std, path_land_sea_masks, path_configuration, path_lr, path_losses, path_model, path_plots = prepare_paths(name_datetime_folder, "P_l", 200, 0, 0.001)
    f_job_dev = open(path_job + "/file_job_dev.txt", 'a')
elif first_run_id == 1:
    path_results, path_mean_std, path_land_sea_masks, path_configuration, path_lr, path_losses, path_model, path_plots = reload_paths_1p(path_job, 'P_l', 200, 0, 0.001)
    f_job_dev = open(path_job + "/file_job_dev.txt", "a")


#2: create GPU device 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#3: Load dei tensori e delle varie liste per definire il total_dataset
if first_run_id == 0:
    f_job_dev.write(f"[first_run_id]:{first_run_id:.12f} \n")
    f_job_dev.close()
    n_dupl_per_week = len(os.listdir("dataset_training/total_dataset/P_l/2022/week_1/"))
    total_dataset, new_transposed_latitudes_coordinates, years_week_dupl_indexes = generate_training_dataset_1("dataset_training/total_dataset", "P_l", [2019, 2020, 2021, 2022], 400, n_dupl_per_week)
    write_list(years_week_dupl_indexes, path_lr + "/ywd_indexes.txt")
    #3a: normalize the dataset
    total_dataset_norm, _, _ = Normalization(total_dataset, "1p", path_results)
    #del total_dataset to save memory
    del total_dataset
    train_dataset, internal_test_dataset, test_dataset, index_training, index_internal_testing, index_external_testing  = split_train_test_data(total_dataset_norm)
    write_list(index_training, path_lr + "/index_training.txt")
    write_list(index_internal_testing, path_lr + "/index_internal_testing.txt")
    write_list(index_external_testing, path_lr + "/index_external_testing.txt")
    #3b: load the land_sea_masks
    land_sea_masks = load_land_sea_masks("dataset_training/land_sea_masks/")

elif first_run_id == 1:
    f_job_dev.write(f"[first_run_id]:{first_run_id:.12f} \n")
    f_job_dev.close()
    years_week_dupl_indexes = read_list(path_lr + "/ywd_indexes.txt")
    index_training = read_list(path_lr + "/index_training.txt")
    index_internal_testing = read_list(path_lr + "/index_internal_testing.txt")
    index_external_testing = read_list(path_lr + "/index_external_testing.txt")
    total_dataset = re_load_tensors("dataset_training/total_dataset", years_week_dupl_indexes)
    new_transposed_latitudes_coordinates = re_load_transp_lat_coordinates("dataset_training/total_dataset", years_week_dupl_indexes)
    total_dataset_norm, _, _ = Normalization(total_dataset, "1p", path_results)
    del total_dataset
    train_dataset, internal_test_dataset, new_test_dataset = recreate_train_test_datasets(total_dataset_norm, index_training, index_internal_testing, index_external_testing)
    del total_dataset_norm
    land_sea_masks = load_land_sea_masks("dataset_training/land_sea_masks/")

#Start the 1 phase of the training procedure: different if-else are related to the different steps of the training procedure
if end_1p == 0:
    if end_train_1p == 0:
        #train 1 phase
        n_epochs_1p = 4 #400
        snaperiod = 2
        l_r = 0.001
        f, f_test = open(path_losses + "/train_loss.txt", "w+"), open(path_losses + "/test_loss.txt", "w+")
        my_mean_tensor = torch.unsqueeze(torch.load(path_mean_std + "/mean_tensor.pt")[:, 6, :, :, :], 1).to(device)
        my_std_tensor = torch.unsqueeze(torch.load(path_mean_std + "/std_tensor.pt")[:, 6, :, :, :], 1).to(device)
        exp_weights = torch.ones([1, 1, d-2, h, w-2])
        losses_1p = []
        train_losses_1p = []
        test_losses_1p = []
        model_save_path = path_results
        training_1p(n_epochs_1p, snaperiod, l_r, years_week_dupl_indexes,  my_mean_tensor, my_std_tensor,train_dataset, internal_test_dataset, index_training, index_internal_testing, land_sea_masks, exp_weights, f, f_test, f_job_dev, losses_1p, train_losses_1p, test_losses_1p, model_save_path, path_model, path_losses, path_lr, new_transposed_latitudes_coordinates)
    elif end_train_1p == 1:
        #test 1 phase
        path_results, path_mean_std, path_land_sea_masks, path_configuration, path_lr, path_losses, path_model, path_plots = reload_paths_1p(path_job, "P_l", 200, 0, 0.001)
        years_week_dupl_indexes = read_list(path_lr + "/ywd_indexes.txt")
        index_external_testing = read_list(path_lr + "/index_external_testing.txt")
        my_mean_tensor = torch.unsqueeze(torch.load(path_mean_std + "/mean_tensor.pt")[:, 6, :, :, :], 1).to(device)
        my_std_tensor = torch.unsqueeze(torch.load(path_mean_std + "/std_tensor.pt")[:, 6, :, :, :], 1).to(device)
        model_1p = CompletionN()  
        checkpoint = torch.load(path_results + '/model_checkpoint.pth')
        model_1p.load_state_dict(checkpoint['model_state_dict'])  
        testing_1p("P_l", path_plots, years_week_dupl_indexes, model_1p, new_test_dataset, index_external_testing, land_sea_masks, new_transposed_latitudes_coordinates, my_mean_tensor, my_std_tensor)

#Start the 2 phase of the training procedure: different if-else are related to the different steps of the training procedure
elif end_1p == 1:
    #start 2 phase
    path_results, path_mean_std, path_land_sea_masks, path_configuration, path_lr, path_losses, path_model, path_plots = reload_paths_1p(path_job, 'P_l', 400, 0, 0.001)
    f_job_dev = open(path_job + "/file_job_dev.txt", "a")
    land_sea_masks = load_land_sea_masks("dataset_training/land_sea_masks/")
    path_results_2, path_configuration_2, path_mean_std_2, path_lr_2, path_losses_2, path_model_2, path_plots_2 = prepare_paths_2(path_job, "P_l", 40, 0, 0.001)
    list_year_week_indexes, old_float_total_dataset, list_float_profiles_coordinates, sampled_list_float_profile_coordinates, index_training_2, index_internal_testing_2, index_external_testing_2, train_dataset_2, internal_test_dataset_2, test_dataset_2 = generate_dataset_phase_2_saving("P_l", path_results_2, [2019, 2020, 2021], "dataset_training/float", land_sea_masks)
    #saves indexes of phase 2
    write_list(list_year_week_indexes, path_lr_2 + "/ywd_indexes.txt")
    write_list(index_training_2, path_lr_2 + "/index_training.txt")
    write_list(index_internal_testing_2, path_lr_2 + "/index_internal_testing.txt")
    write_list(index_external_testing_2, path_lr_2 + "/index_external_testing.txt")
    #preparation of training:
    #3b: load the land_sea_masks
    land_sea_masks = load_land_sea_masks("dataset_training/land_sea_masks/")
    #train 2 phase:
    n_epochs_2p = 20    
    snaperiod_2p = 5
    l_r_2p = 0.001
    f_2, f_2_test = open(path_losses_2 + "/train_loss.txt", "w+"), open(path_losses_2 + "/test_loss.txt", "w+")
    my_mean_tensor_2p = torch.unsqueeze(torch.load(path_mean_std_2 + "/mean_tensor.pt")[:, 6, :, :, :], 1).to(device)
    my_std_tensor_2p = torch.unsqueeze(torch.load(path_mean_std_2 + "/std_tensor.pt")[:, 6, :, :, :], 1).to(device)
    exp_weights = torch.ones([1, 1, d-2, h, w-2])
    losses_2p = []
    train_losses_2p = []
    test_losses_2p = []
    model_2p_save_path = path_results_2
    training_2p(n_epochs_2p, snaperiod_2p, l_r_2p, my_mean_tensor_2p, my_std_tensor_2p, train_dataset_2, internal_test_dataset_2, index_training_2, index_internal_testing_2, land_sea_masks, exp_weights, old_float_total_dataset, sampled_list_float_profile_coordinates, f_2, f_2_test, losses_2p, train_losses_2p, test_losses_2p, path_results, model_2p_save_path, path_model_2, path_losses_2)
    #test 2 phase:
    model_1p = CompletionN()  
    checkpoint = torch.load(path_results + '/model_checkpoint.pth')
    model_1p.load_state_dict(checkpoint['model_state_dict']) 
    model_2p = CompletionN()  
    checkpoint_2 = torch.load(path_results_2 + '/model_checkpoint_2.pth')
    model_2p.load_state_dict(checkpoint_2['model_state_dict']) 
    biogeoch_total_dataset = [torch.unsqueeze(load_old_total_tensor("dataset_training/old_total_dataset/", i_test_2, list_year_week_indexes)[:, -1, :, :, :], 1) for i_test_2 in index_external_testing_2]
    model_1p.eval()
    testing_2p("P_l", path_plots_2, list_year_week_indexes, biogeoch_total_dataset, old_float_total_dataset, model_1p, model_2p, test_dataset_2, index_external_testing_2, land_sea_masks, list_float_profiles_coordinates, my_mean_tensor_2p, my_std_tensor_2p)
