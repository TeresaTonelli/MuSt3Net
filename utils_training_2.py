#In this script we write all the functions useful for preparing data from training 2


import numpy as np
import torch
import pandas as pd
import os
import random
from denormalization import Denormalization
from utils_generation_train_1p import read_list


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def prepare_paths_2_ensemble(path_job, biogeoch_var, n_epochs_2, epoch_pretrain_2, lr_2, n_ensemble):
    path_results_2 = path_job + "/results_training_2_ensemble" 
    path_configuration_2 = path_results_2 + "/" + str(biogeoch_var) + "/" + str(n_epochs_2 + epoch_pretrain_2) 
    if not os.path.exists(path_configuration_2):
        os.makedirs(path_configuration_2)
    path_mean_std_2 = path_results_2 + "/mean_and_std_tensors"
    if not os.path.exists(path_mean_std_2):
        os.makedirs(path_mean_std_2)
    path_lr_2 = path_configuration_2 + "/lrc_" + str(lr_2) 
    if not os.path.exists(path_lr_2):
        os.makedirs(path_lr_2)
    paths_ensemble_models = []
    for i in range(n_ensemble):
        path_ensemble_model = path_lr_2 + "/ensemble_model_" + str(i)
        if not os.path.exists(path_ensemble_model):
            os.makedirs(path_ensemble_model)
        paths_ensemble_models.append(path_ensemble_model)
    return path_results_2, path_configuration_2, path_mean_std_2, path_lr_2, paths_ensemble_models


def compute_ensemble_mean(list_path_test_loss):
    n_ens = len(list_path_test_loss)
    n_ens_means = []
    for i_ens in range(n_ens):
        test_loss_list = read_list(list_path_test_loss[i_ens])
        test_loss_list = [test_loss_list[i].cpu() for i in range(len(test_loss_list))]
        test_mean = np.mean(np.array(test_loss_list))
        n_ens_means.append(test_mean)
    return np.mean(np.array(n_ens_means))


def compute_ensemble_std(list_path_test_loss):
    n_ens = len(list_path_test_loss)
    n_ens_means = []
    for i_ens in range(n_ens):
        test_loss_list = read_list(list_path_test_loss[i_ens])
        test_loss_list = [test_loss_list[i].cpu() for i in range(len(test_loss_list))]
        test_std = np.std(np.array(test_loss_list))
        n_ens_means.append(test_std)
    return np.mean(np.array(n_ens_means))    


def compute_3D_ensemble_mean_std(test_data, list_ensemble_models, path_mean_std):
    mean_tensor_2 = torch.unsqueeze(torch.load(path_mean_std + "/mean_tensor.pt")[:, 6, :, :, :], 1).to(device)
    std_tensor_2 = torch.unsqueeze(torch.load(path_mean_std + "/std_tensor.pt")[:, 6, :, :, :], 1).to(device)
    list_models_outputs = []
    for ens_model in list_ensemble_models:
        ens_model.to(device)
        ens_model.eval()
        with torch.no_grad():
            test_data = test_data.to(device)
            test_output = ens_model(test_data.float())
            denorm_test_output = Denormalization(test_output, mean_tensor_2, std_tensor_2)
            list_models_outputs.append(denorm_test_output)
    #compute the mean of the n_ensemble output tensors
    stacked_tensor_outputs = torch.stack([list_models_outputs[i_ens] for i_ens in range(len(list_ensemble_models))])
    tensor_mean = torch.mean(stacked_tensor_outputs, dim=0)
    tensor_std = torch.std(stacked_tensor_outputs, dim=0)    
    return tensor_mean, tensor_std