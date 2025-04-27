"""
Implementation of the 2-step training routine for the 3D CNN
"""
import numpy as np
import torch.nn as nn
from torch.optim import Adadelta
import matplotlib.pyplot as plt
import random
import datetime

from CNN_3DMedSea.convolutional_network import CompletionN
#from losses import  convolutional_network_exp_weighted_loss, convolutional_network_float_exp_weighted_loss
#from mean_pixel_value import MV_pixel
#from utils_mask import generate_input_mask, generate_sea_land_mask
from CNN_3DMedSea.normalization_functions import Normalization, Denormalization
from data_preprocessing.get_dataset import *
#from plot_error import Plot_Error
from CNN_plots.plot_results import *
from utils.utils_general import *
#from utils_mask import generate_float_mask, compute_exponential_weights
from data_preprocessing.generation_training_dataset import generate_dataset_phase_2_saving
from utils.utils_training import prepare_paths, reload_paths_1p, prepare_paths_2_ensemble, generate_training_dataset_1, split_train_test_data, load_land_sea_masks, load_old_total_tensor, re_load_tensors, recreate_train_test_datasets, re_load_transp_lat_coordinates, compute_ensemble_mean, compute_ensemble_std, compute_3D_ensemble_mean_std
from utils.utils_dataset_generation import write_list, read_list
from CNN_3DMedSea.training_testing_functions import training_1p, testing_1p, training_2p, testing_2p_ensemble


#3 parameters to define the jobs pypeline
first_run_id = 2
end_train_1p = 1
end_1p = 1
path_job = "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/results_job_2025-04-27 08:25:17.224415" 


num_channel = number_channel  
name_datetime_folder = str(datetime.datetime.utcnow())


#1: create paths 1 phase
if first_run_id == 0:
    path_job, path_results, path_mean_std, path_land_sea_masks, path_configuration, path_lr, path_losses, path_model, path_plots = prepare_paths(name_datetime_folder, "P_l", 200, 0, 0.001)
    f_job_dev = open(path_job + "/file_job_dev.txt", 'a')
elif first_run_id == 1:
    path_results, path_mean_std, path_land_sea_masks, path_configuration, path_lr, path_losses, path_model, path_plots = reload_paths_1p(path_job, 'P_l', 200, 0, 0.001)
    f_job_dev = open(path_job + "/file_job_dev.txt", "a")

#2: create the GPU device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#3: Load the tensor for the training dataset
if first_run_id == 0:
    f_job_dev.write(f"[first_run_id]:{first_run_id:.12f} \n")
    f_job_dev.close()
    n_dupl_per_week = len(os.listdir("dataset_training/total_dataset/P_l/2022/week_1/"))
    total_dataset, new_transposed_latitudes_coordinates, years_week_dupl_indexes = generate_training_dataset_1("dataset_training/total_dataset", "P_l", [2019], 400, n_dupl_per_week)
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

#Start of the 1 phase training procedure
if end_1p == 0:
    if end_train_1p == 0:
        #train 1 phase
        n_epochs_1p = 400 
        snaperiod = 50
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

#Start the 2 phase of the training
elif end_1p == 1:
    #start 2 phase --> ensemble phase
    n_ensemble = 2
    path_results, path_mean_std, path_land_sea_masks, path_configuration, path_lr, path_losses, path_model, path_plots = reload_paths_1p(path_job, 'P_l', 200, 0, 0.001)
    f_job_dev = open(path_job + "/file_job_dev.txt", "a")
    years_week_dupl_indexes_1 = read_list(path_lr + "/ywd_indexes.txt")
    index_1_ext_test = read_list(path_lr + "/index_external_testing.txt")
    transposed_latitudes_coordinates_1 = re_load_transp_lat_coordinates("dataset_training/total_dataset", years_week_dupl_indexes_1)
    land_sea_masks = load_land_sea_masks("dataset_training/land_sea_masks/")
    path_results_2, path_configuration_2, path_mean_std_2, path_lr_2, paths_ensemble_models = prepare_paths_2_ensemble(path_job, "P_l", 20, 0, 0.001, n_ensemble)
    land_sea_masks = load_land_sea_masks("dataset_training/land_sea_masks/")
    n_epochs_2p = 20  
    snaperiod_2p = 5
    l_r_2p = 0.001
    for i_ens in range(n_ensemble):
        list_year_week_indexes, old_float_total_dataset, list_float_profiles_coordinates, sampled_list_float_profile_coordinates, index_training_2, index_internal_testing_2, index_external_testing_2, train_dataset_2, internal_test_dataset_2, test_dataset_2 = generate_dataset_phase_2_saving("P_l", path_results_2, [2019], "dataset_training/float", land_sea_masks)
        #saves indexes of phase 2 for each ensemble
        write_list(list_year_week_indexes, path_lr_2 + "/ensemble_ywd_indexes.txt")
        write_list(index_training_2, path_lr_2 + "/index_training.txt")
        write_list(index_internal_testing_2, path_lr_2 + "/index_internal_testing.txt")
        write_list(index_external_testing_2, path_lr_2 + "/index_external_testing.txt")
        path_ensemble_model = paths_ensemble_models[i_ens]
        #saves indexes of phase 2 for each ensemble
        write_list(list_year_week_indexes, path_ensemble_model + "/ywd_indexes.txt")
        write_list(index_training_2, path_ensemble_model + "/index_training.txt")
        write_list(index_internal_testing_2, path_ensemble_model + "/index_internal_testing.txt")
        write_list(index_external_testing_2, path_ensemble_model + "/index_external_testing.txt")
        path_losses_2 = path_ensemble_model + "/losses"
        if not os.path.exists(path_losses_2):
            os.makedirs(path_losses_2)
        path_model_2 = path_ensemble_model + "/partial_models/"
        if not os.path.exists(path_model_2):
            os.makedirs(path_model_2)
        path_plots_2 = path_ensemble_model + "/plots"
        if not os.path.exists(path_plots_2):
            os.makedirs(path_plots_2)
        #train 2 phase:
        f_2, f_2_test = open(path_losses_2 + "/train_losses_2p.txt", "w+"), open(path_losses_2 + "/test_losses_2p.txt", "w+")
        my_mean_tensor_2p = torch.unsqueeze(torch.load(path_mean_std_2 + "/mean_tensor.pt")[:, 6, :, :, :], 1).to(device)
        my_std_tensor_2p = torch.unsqueeze(torch.load(path_mean_std_2 + "/std_tensor.pt")[:, 6, :, :, :], 1).to(device)
        exp_weights = torch.ones([1, 1, d-2, h, w-2])
        losses_2p = []
        train_losses_2p = []
        test_losses_2p = []
        model_2p_save_path = path_results_2
        training_2p(n_epochs_2p, snaperiod_2p, l_r_2p, my_mean_tensor_2p, my_std_tensor_2p, train_dataset_2, internal_test_dataset_2, index_training_2, index_internal_testing_2, land_sea_masks, exp_weights, old_float_total_dataset, sampled_list_float_profile_coordinates, f_2, f_2_test, losses_2p, train_losses_2p, test_losses_2p, path_results, model_2p_save_path, path_model_2, path_losses_2)
        #copy the model checkpoint_2 inside the directory of the current ensemble model
        source = path_results_2 + "/model_checkpoint_2.pth"
        destination = path_ensemble_model + "/model_checkpoint_2_ens_" + str(i_ens) + ".pth"
        with open(source, 'rb') as src, open(destination, 'wb') as dst:
            dst.write(src.read())
        os.remove(source)
        #test 2 phase:
        model_1p = CompletionN()  
        checkpoint = torch.load(path_results + '/model_checkpoint.pth')
        model_1p.load_state_dict(checkpoint['model_state_dict']) 
        model_2p = CompletionN()  
        checkpoint_2 = torch.load(path_ensemble_model + "/model_checkpoint_2_ens_" + str(i_ens) + ".pth")    
        model_2p.load_state_dict(checkpoint_2['model_state_dict']) 
        biogeoch_total_dataset = [torch.unsqueeze(load_old_total_tensor("dataset_training/old_total_dataset/", i_test_2, list_year_week_indexes)[:, -1, :, :, :], 1) for i_test_2 in index_external_testing_2]
        model_1p.eval()
        testing_2p_ensemble("P_l", path_plots_2, list_year_week_indexes, biogeoch_total_dataset, old_float_total_dataset, model_1p, model_2p, test_dataset_2, index_external_testing_2, land_sea_masks, list_float_profiles_coordinates, sampled_list_float_profile_coordinates,my_mean_tensor_2p, my_std_tensor_2p, exp_weights, path_losses_2)
    #phase of evaluation of mean and standard deviation of ensemble models
    checkpoint_list = [torch.load(paths_ensemble_models[i_ens] + '/model_checkpoint_2_ens_' + str(i_ens) + '.pth') for i_ens in range(n_ensemble)]
    models_list = [CompletionN() for i in range(n_ensemble)]
    for i_m in range(len(models_list)):
        model = models_list[i_m]
        model.load_state_dict(checkpoint_list[i_m]['model_state_dict'])
        #check of weights
        mismatch = False
        for (name1, param1), (name2, param2) in zip(checkpoint_list[i_m]['model_state_dict'].items(), models_list[i_m].state_dict().items()):
            if not torch.equal(param1.cpu(), param2.cpu()):
                print(f"Mismatch in parameter: {name1}")
                mismatch = True
        if not mismatch:
            print("All model parameters match exactly!")
        models_list[i_m] = model   
        #check of weights
        mismatch = False
        for (name1, param1), (name2, param2) in zip(checkpoint_list[i_m]['model_state_dict'].items(), models_list[i_m].state_dict().items()):
            if not torch.equal(param1.cpu(), param2.cpu()):
                print(f"Mismatch in parameter: {name1}")
                mismatch = True
        if not mismatch:
            print("All model parameters match exactly!")
    #1a approach: 
    test_ensemble_mean = compute_ensemble_mean([paths_ensemble_models[i_ens] + "/losses" + "/test_loss_final.txt" for i_ens in range(n_ensemble)])
    test_ensemble_std = compute_ensemble_std([paths_ensemble_models[i_ens] + "/losses" + "/test_loss_final.txt" for i_ens in range(n_ensemble)])
    #1b approach:
    test_ensemble_mean_winter = compute_ensemble_mean([paths_ensemble_models[i_ens] + "/losses" + "/test_loss_final_winter.txt" for i_ens in range(n_ensemble)])
    test_ensemble_std_winter = compute_ensemble_std([paths_ensemble_models[i_ens] + "/losses" + "/test_loss_final_winter.txt" for i_ens in range(n_ensemble)])
    test_ensemble_mean_summer = compute_ensemble_mean([paths_ensemble_models[i_ens] + "/losses" + "/test_loss_final_summer.txt" for i_ens in range(n_ensemble)])
    test_ensemble_std_summer = compute_ensemble_std([paths_ensemble_models[i_ens] + "/losses" + "/test_loss_final_summer.txt" for i_ens in range(n_ensemble)])
    #2 approach:
    test_data_winter = []
    test_data_summer = []
    path_ensemble_plots = path_lr_2 + "/plots_ensemble"
    if not os.path.exists(path_ensemble_plots):
        os.makedirs(path_ensemble_plots)
    for i_test in range(len(test_dataset_2)):
        path_ensemble_test_data = path_ensemble_plots + "/test_data_year_" + str(list_year_week_indexes[index_external_testing_2[i_test]][0]) + "_week_" + str(list_year_week_indexes[index_external_testing_2[i_test]][1])
        if not os.path.exists(path_ensemble_test_data):
            os.makedirs(path_ensemble_test_data)
        path_ensemble_plot_test_mean = path_ensemble_test_data + "/mean"
        if not os.path.exists(path_ensemble_plot_test_mean):
            os.makedirs(path_ensemble_plot_test_mean)
        path_ensemble_plot_test_std = path_ensemble_test_data + "/std"
        if not os.path.exists(path_ensemble_plot_test_std):
            os.makedirs(path_ensemble_plot_test_std)
        path_ensemble_profiles = path_ensemble_test_data + "/profiles"
        if not os.path.exists(path_ensemble_profiles):
            os.makedirs(path_ensemble_profiles)
        tensor_ensemble_mean, tensor_ensemble_std = compute_3D_ensemble_mean_std(test_dataset_2[i_test], models_list, path_mean_std_2)
        #plot the maps of chlorophyll mean over the ensembles and its std
        plot_NN_maps(tensor_ensemble_mean, land_sea_masks, "P_l", path_ensemble_plot_test_mean)
        plot_NN_maps_std_percentage(tensor_ensemble_std / tensor_ensemble_mean * 100, land_sea_masks, "P_l", path_ensemble_plot_test_std)
        #plot the mean profile of the whole ensemble
        tensor_output_float = torch.unsqueeze(old_float_total_dataset[index_external_testing_2[i_test]][:, 6, :, :, :], 1)
        tensor_output_NN_model = tensor_ensemble_mean
        tensor_output_num_model = biogeoch_total_dataset[i_test][:, :, :-1, :, 1:-1]
        model_1p.to(device)
        model_1p.eval()
        tensor_output_NN_1_model = model_1p(test_dataset_2[i_test].to(device))
        tensor_output_NN_1_model = Denormalization(tensor_output_NN_1_model, my_mean_tensor_2p, my_std_tensor_2p)
        comparison_profiles_1_2_phases(tensor_output_float, tensor_output_NN_model, tensor_output_num_model, tensor_output_NN_1_model, "P_l", path_ensemble_profiles)

    
