#In this script we write all the function to generate plots FOR POST-PROCESSING, so the ones that I can recall after the network training 


import numpy as np
import torch 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
from mpl_toolkits.basemap import Basemap
import os
import itertools

from convolutional_network import CompletionN
from denormalization import Denormalization
from hyperparameter import * 
from utils_mask import apply_masks
from utils_function import compute_mean_layers, compute_profile_mean


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sns.set_theme(context='paper', style='whitegrid', font='sans-serif', font_scale=1.0, color_codes=True, rc=None)


def moving_average(data, window_size):
    if window_size % 2 == 0:
        window_size += 1  
    pad_size = window_size // 2
    padded_data = np.pad(data, pad_size, mode='edge')
    cumsum_vec = np.cumsum(np.insert(padded_data, 0, 0))
    return (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size


def compute_channel(var):
   """this method returns the index of the channel in which this variable is saved inside the tensor"""
   if var in list_physics_vars:
      return list_physics_vars.index(var)
   elif var in list_biogeoch_vars:
      return 0
   

def compute_cmap(string_colormap):
    """this function defines a modified cmap, that leaves white all the lands and paint the correct scale of chl for the sea"""
    if string_colormap == 'jet':
        jet = plt.get_cmap('jet')
    elif string_colormap == 'viridis':
        jet = plt.get_cmap('viridis')
    newcolors = jet(np.linspace(0, 1, 256))
    newcolors[0, :] = np.array([1, 1, 1, 1])  
    newcmap = ListedColormap(newcolors)
    return newcmap


def plot_BFM_maps(BFM_tensor, list_masks, var, path_fig_channel, list_layers):
    """function that plot the tensor resulted from the BFM"""
    sns.set_theme(context='paper', style='whitegrid', font='sans-serif', font_scale=2.0, color_codes=True, rc=None)
    channel = compute_channel(var)
    masked_NN_tensor = apply_masks(BFM_tensor, list_masks)
    #compute the mean wrt layers and return the new tensor
    masked_NN_tensor_mean_layer = compute_mean_layers(masked_NN_tensor, list_layers, 2, (1, 1, len(list_layers), masked_NN_tensor.shape[3], masked_NN_tensor.shape[4]))   
    #plot the resulted tensor
    for layers_level in range(len(list_layers) - 1):
        plot_tensor = torch.clone(masked_NN_tensor_mean_layer)   
        plot_tensor = np.transpose(plot_tensor.cpu(), [0,1,2,4,3])
        plot_tensor = torch.from_numpy(np.flip(plot_tensor.numpy(), 3).copy())
        cmap = plt.get_cmap("jet")   
        newcmap = compute_cmap('jet')
        depth_level = list_layers[layers_level] // resolution[2]
        plt.imshow(plot_tensor[0, channel, layers_level, :, :], cmap=newcmap, vmin = parameters_plots[var][0][depth_level], vmax = parameters_plots[var][1][depth_level], interpolation='spline16')    
        plt.colorbar(shrink=0.6, pad=0.01)
        my_xticks= np.arange(0, h, 70)
        my_yticks = np.arange(0, w, 65)
        my_xticks_label = np.array([int((index - 1) * resolution[1] / constant_longitude + 1 + longitude_interval[0]) for index in np.arange(0, h, 70)])
        my_yticks_label = np.array([int((index - 1) * resolution[0] / constant_latitude + 1 + latitude_interval[0]) for index in np.arange(w-1, -1, -65)])   #prima era -1
        plt.xticks(my_xticks, my_xticks_label)
        plt.yticks(my_yticks, my_yticks_label)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tight_layout()
        plt.savefig(path_fig_channel + "/depth_" + str(list_layers[layers_level]) + ".png", dpi=800)
        plt.close()


def plot_NN_maps_final_1(input_tensor, CNN_model, path_job, list_masks, var, path_mean_std, path_fig_channel, mean_layer=False, list_layers = []):
    """function that plot the tensor resulted from the NN after the first phase"""
    sns.set_theme(context='paper', style='whitegrid', font='sans-serif', font_scale=2.0, color_codes=True, rc=None)
    input_tensor = input_tensor.to(device)
    CNN_checkpoint = torch.load(path_job + '/results_training_1/model_checkpoint.pth', map_location=device)
    CNN_model.load_state_dict(CNN_checkpoint['model_state_dict'])  
    CNN_model = CNN_model.to(device)
    CNN_model.eval()
    with torch.no_grad():
        chl_tensor_out = CNN_model(input_tensor.float())
        my_mean_tensor = torch.unsqueeze(torch.load(path_mean_std + "/mean_tensor.pt")[:, 6, :, :, :], 1).to(device)
        my_std_tensor = torch.unsqueeze(torch.load(path_mean_std + "/std_tensor.pt")[:, 6, :, :, :], 1).to(device)
        chl_tensor = Denormalization(chl_tensor_out, my_mean_tensor, my_std_tensor).to(device)
        if mean_layer == False:
            depth_levels = np.arange(0, chl_tensor.shape[2])
            masked_chl_tensor = apply_masks(chl_tensor, list_masks)
            channel = compute_channel(var)
        else: 
            depth_levels = np.arange(0, len(list_layers) - 1)
            masked_NN_tensor = apply_masks(chl_tensor, list_masks)
            masked_chl_tensor = compute_mean_layers(masked_NN_tensor, list_layers, 2, (1, 1, len(list_layers), masked_NN_tensor.shape[3], masked_NN_tensor.shape[4]))   
            channel = compute_channel(var)
        for depth_level in depth_levels:
            if mean_layer == False:
                color_level = depth_level
            elif mean_layer == True:
                color_level = list_layers[depth_level] // resolution[2]
            plot_tensor = torch.clone(masked_chl_tensor) 
            plot_tensor = np.transpose(plot_tensor.cpu(), [0,1,2,4,3])
            plot_tensor = torch.from_numpy(np.flip(plot_tensor.numpy(), 3).copy()) 
            newcmap = compute_cmap('jet')
            plt.imshow(plot_tensor[0, channel, depth_level, :, :], cmap=newcmap, vmin = parameters_plots[var][0][color_level + 1], vmax = parameters_plots[var][1][color_level + 1], interpolation='spline16')
            plt.colorbar(shrink=0.95, pad=0.01)
            my_xticks= np.arange(0, h, 70)
            my_yticks = np.arange(0, w, 65)
            my_xticks_label = np.array([int((index - 1) * resolution[1] / constant_longitude + 1 + longitude_interval[0]) for index in np.arange(0, h, 70)])
            my_yticks_label = np.array([int((index - 1) * resolution[0] / constant_latitude + 1 + latitude_interval[0]) for index in np.arange(w-1, -1, -65)]) 
            plt.xticks(my_xticks, my_xticks_label)
            plt.yticks(my_yticks, my_yticks_label)
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.tight_layout()
            plt.savefig(path_fig_channel + "/depth_" + str(list_layers[depth_level]) + "_" + str(list_layers[depth_level + 1]) +  ".png", dpi=800)
            plt.close()


def plot_NN_maps_final_2(input_tensor, path_job, list_masks, var, path_fig_channels, n_ensemble, mean_layer=False, list_layers = []):
    """function that plot the tensor resulted from the NN"""
    sns.set_theme(context='paper', style='whitegrid', font='sans-serif', font_scale=2.0, color_codes=True, rc=None)
    path_mean_std = path_job + "/results_training_2_ensemble/mean_and_std_tensors"
    path_lr = path_job + "/results_training_2_ensemble/" + var + "/20/lrc_0.001"
    my_mean_tensor = torch.unsqueeze(torch.load(path_mean_std + "/mean_tensor.pt")[:, 6, :, :, :], 1).to(device)
    my_std_tensor = torch.unsqueeze(torch.load(path_mean_std + "/std_tensor.pt")[:, 6, :, :, :], 1).to(device)
    single_chl_tensor_shape = (n_ensemble, input_tensor.shape[0], 1, input_tensor.shape[2], input_tensor.shape[3], input_tensor.shape[4])
    ensemble_chl_tensor = torch.ones(single_chl_tensor_shape)
    for i_ens in range(n_ensemble):
        CNN_checkpoint = torch.load(path_lr + "/ensemble_model_" + str(i_ens) + "/model_checkpoint_2_ens_" + str(i_ens) + ".pth", map_location=device)
        CNN_model = CompletionN()
        CNN_model.load_state_dict(CNN_checkpoint['model_state_dict'])  
        CNN_model = CNN_model.to(device)
        CNN_model.eval()
        with torch.no_grad():
            input_tensor = input_tensor.to(device)
            chl_tensor_out = CNN_model(input_tensor.float())
            chl_tensor = Denormalization(chl_tensor_out, my_mean_tensor, my_std_tensor).to(device)
            ensemble_chl_tensor[i_ens, :, :, :, :, :] = chl_tensor
    mean_chl_tensor = torch.mean(ensemble_chl_tensor, dim=0)
    std_chl_tensor = torch.std(ensemble_chl_tensor, dim=0)
    if mean_layer == False:
        depth_levels = np.arange(0, mean_chl_tensor.shape[2])
        masked_chl_tensor = apply_masks(mean_chl_tensor, list_masks)
        masked_chl_std_tensor = apply_masks(std_chl_tensor, list_masks)
        channel = compute_channel(var)
    else: 
        depth_levels = np.arange(0, len(list_layers) - 1)
        masked_NN_tensor = apply_masks(mean_chl_tensor, list_masks)
        masked_NN_std_tensor = apply_masks(std_chl_tensor, list_masks)
        masked_chl_tensor = compute_mean_layers(masked_NN_tensor, list_layers, 2, (1, 1, len(list_layers), masked_NN_tensor.shape[3], masked_NN_tensor.shape[4]))   
        masked_chl_std_tensor = compute_mean_layers(masked_NN_std_tensor, list_layers, 2, (1, 1, len(list_layers), masked_NN_std_tensor.shape[3], masked_NN_std_tensor.shape[4]))   
        channel = compute_channel(var)
    for depth_level in depth_levels:
        if mean_layer == False:
            color_level = depth_level
        elif mean_layer == True:
            color_level = list_layers[depth_level] // resolution[2]
        plot_tensor_mean = torch.clone(masked_chl_tensor)   
        plot_tensor_mean = np.transpose(plot_tensor_mean.cpu(), [0,1,2,4,3])
        plot_tensor_mean = torch.from_numpy(np.flip(plot_tensor_mean.numpy(), 3).copy())
        plot_tensor_std = torch.clone(masked_chl_std_tensor) 
        plot_tensor_std = np.transpose(plot_tensor_std.cpu(), [0,1,2,4,3])
        plot_tensor_std = torch.from_numpy(np.flip(plot_tensor_std.numpy(), 3).copy())
        cmap = plt.get_cmap("jet")   
        newcmap = compute_cmap('jet')
        #plot the mean
        fig_mean, ax_mean = plt.subplots()
        m = ax_mean.imshow(plot_tensor_mean[0, channel, depth_level, :, :], cmap=newcmap, vmin = parameters_plots[var][0][color_level + 2], vmax = parameters_plots[var][1][color_level + 2], interpolation='gaussian')
        plt.colorbar(m, shrink=0.95, pad=0.01)
        my_xticks = np.arange(0, h, 70)
        my_yticks = np.arange(0, w, 65)
        my_xticks_label = np.array([int((index - 1) * resolution[1] / constant_longitude + 1 + longitude_interval[0]) for index in np.arange(0, h, 70)])
        my_yticks_label = np.array([int((index - 1) * resolution[0] / constant_latitude + 1 + latitude_interval[0]) for index in np.arange(w-1, -1, -65)])  
        ax_mean.set_xticks(my_xticks, my_xticks_label)
        ax_mean.set_yticks(my_yticks, my_yticks_label)
        ax_mean.set_xlabel("Longitude")
        ax_mean.set_ylabel("Latitude")
        plt.tight_layout()
        plt.savefig(path_fig_channels[0] + "/depth_" + str(list_layers[depth_level]) + "_" + str(list_layers[depth_level + 1]) + ".png", dpi=800)
        plt.clf()
        plt.close(fig_mean)
        #plot the std
        fig_std, ax_std = plt.subplots()
        s = ax_std.imshow(plot_tensor_std[0, channel, depth_level, :, :], cmap=newcmap, vmin = parameters_plots[var][0][color_level], vmax = parameters_plots[var][1][color_level], interpolation='spline16')
        plt.colorbar(s, shrink=0.95, pad=0.01)
        my_xticks= np.arange(0, h, 70)
        my_yticks = np.arange(0, w, 65)
        my_xticks_label = np.array([int((index - 1) * resolution[1] / constant_longitude + 1 + longitude_interval[0]) for index in np.arange(0, h, 70)])
        my_yticks_label = np.array([int((index - 1) * resolution[0] / constant_latitude + 1 + latitude_interval[0]) for index in np.arange(w-1, -1, -65)])  
        ax_std.set_xticks(my_xticks, my_xticks_label)
        ax_std.set_yticks(my_yticks, my_yticks_label)
        ax_std.set_xlabel("Longitude")
        ax_std.set_ylabel("Latitude")
        plt.tight_layout()
        plt.savefig(path_fig_channels[1] + "/depth_" + str(list_layers[depth_level]) + "_" + str(list_layers[depth_level + 1]) + ".png", dpi=800)
        plt.clf()
        plt.close(fig_std)



def plot_models_profiles_1(tensor_input_NN, CNN_model, tensor_output_num_model, path_job, var, path_mean_std, path_fig_channel, list_to_plot_coordinates):
    """function that plot the profiles resulted from the NN and compares them with BFM's profiles"""
    sns.set_theme(context='paper', style='whitegrid', font='sans-serif', font_scale=1.25, color_codes=True, rc=None)
    tensor_input_NN = tensor_input_NN.to(device)
    CNN_checkpoint = torch.load(path_job + '/results_training_1/model_checkpoint.pth', map_location=device)
    CNN_model.load_state_dict(CNN_checkpoint['model_state_dict'])  
    CNN_model = CNN_model.to(device)
    CNN_model.eval()
    with torch.no_grad():
        tensor_output_NN_model = CNN_model(tensor_input_NN.float())
        my_mean_tensor = torch.unsqueeze(torch.load(path_mean_std + "/mean_tensor.pt")[:, 6, :, :, :], 1).to(device)
        my_std_tensor = torch.unsqueeze(torch.load(path_mean_std + "/std_tensor.pt")[:, 6, :, :, :], 1).to(device)
        tensor_input_NN_model = Denormalization(tensor_input_NN[:, -1, :, :, :], my_mean_tensor, my_std_tensor)
        tensor_output_NN_model = Denormalization(tensor_output_NN_model, my_mean_tensor, my_std_tensor).to(device)
        depth_levels = resolution [2] * np.arange(tensor_input_NN.shape[2]-1, -1, -1)  #20, non resolution[2] prima
        channel = compute_channel(var)
        for plot_coordinate in list_to_plot_coordinates:
            profile_tensor_num_model = tensor_output_num_model[0, channel, :, plot_coordinate[0], plot_coordinate[1]]
            profile_tensor_input_NN = tensor_input_NN_model[0, channel, :, plot_coordinate[0], plot_coordinate[1]]
            profile_tensor_NN_model = tensor_output_NN_model[0, channel, :, plot_coordinate[0], plot_coordinate[1]]
            #addition of moving_average
            profile_tensor_num_model = moving_average(profile_tensor_num_model.detach().cpu().numpy(), 3)
            profile_tensor_input_NN = moving_average(profile_tensor_input_NN.detach().cpu().numpy(), 3)
            profile_tensor_NN_model = moving_average(profile_tensor_NN_model.detach().cpu().numpy(), 3)
            profile_tensor_NN_model = np.maximum(profile_tensor_NN_model, 0)
            #plot profiles
            path_fig_channel_coordinates = path_fig_channel + "/lat_" + str(plot_coordinate[1]) + "_lon_" + str(plot_coordinate[0])
            plt.yticks(depth_levels, resolution[2] * np.arange(0, tensor_input_NN.shape[2]), fontsize=6)  
            plt.plot(profile_tensor_NN_model, depth_levels, color="#2CA02C", label="CNN-3DMedSea")       
            plt.plot(profile_tensor_num_model, depth_levels, color="#1F77B4", label="BFM")      
            plt.grid(axis = 'x')
            plt.xlabel(r"Chlorophyll [$mg \ m^{-3}$]")
            plt.ylabel(r"Depth [$m$]")
            plt.legend(loc="lower right", prop={'size': 8})
            plt.tight_layout()
            plt.savefig(path_fig_channel_coordinates + ".png", dpi=400)
            plt.close()



def plot_models_profiles_2(tensor_input_NN, tensor_output_num_model, tensor_float, path_job, var, path_mean_std, path_fig_channel, list_to_plot_coordinates, n_ensemble):
    """function that plot the profiles resulted from the NN and compares them with BFM's profiles and real BGC-Argo floats"""
    sns.set_theme(context='paper', style='whitegrid', font='sans-serif', font_scale=2.5, color_codes=True, rc=None)
    my_mean_tensor = torch.unsqueeze(torch.load(path_mean_std + "/mean_tensor.pt")[:, 6, :, :, :], 1).to(device)
    my_std_tensor = torch.unsqueeze(torch.load(path_mean_std + "/std_tensor.pt")[:, 6, :, :, :], 1).to(device)
    single_chl_tensor_shape = (n_ensemble, tensor_input_NN.shape[0], 1, tensor_input_NN.shape[2], tensor_input_NN.shape[3], tensor_input_NN.shape[4])
    ensemble_chl_tensor = torch.ones(single_chl_tensor_shape)
    for i_ens in range(n_ensemble):
        CNN_checkpoint = torch.load(path_job + "/results_training_2_ensemble/" + var + "/20/lrc_0.001/ensemble_model_" + str(i_ens) + "/model_checkpoint_2_ens_" + str(i_ens) + ".pth", map_location=device)
        CNN_model = CompletionN()
        CNN_model.load_state_dict(CNN_checkpoint['model_state_dict'])  
        CNN_model = CNN_model.to(device)
        CNN_model.eval()
        with torch.no_grad():
            tensor_output_NN_model = CNN_model(tensor_input_NN.float())
            tensor_input_NN_model = Denormalization(tensor_input_NN[:, -1, :, :, :], my_mean_tensor, my_std_tensor)
            tensor_output_NN_model = Denormalization(tensor_output_NN_model, my_mean_tensor, my_std_tensor).to(device)
            ensemble_chl_tensor[i_ens, :, :, :, :, :] = tensor_output_NN_model
    mean_chl_tensor = torch.mean(ensemble_chl_tensor, dim=0)
    CNN_checkpoint_1 = torch.load(path_job + "/results_training_1/model_checkpoint.pth", map_location=device)
    CNN_model_1 = CompletionN()
    CNN_model_1.load_state_dict(CNN_checkpoint_1['model_state_dict'])  
    CNN_model_1 = CNN_model_1.to(device)
    CNN_model_1.eval()
    with torch.no_grad():
        tensor_output_NN_1_model = CNN_model_1(tensor_input_NN.float())
        tensor_input_NN_1_model = Denormalization(tensor_input_NN[:, -1, :, :, :], my_mean_tensor, my_std_tensor)
        tensor_output_NN_1_model = Denormalization(tensor_output_NN_1_model, my_mean_tensor, my_std_tensor).to(device)
    depth_levels = resolution [2] * np.arange(tensor_input_NN.shape[2]-1, -1, -1) 
    channel = compute_channel(var)
    for plot_coordinate in list_to_plot_coordinates:
        profile_tensor_num_model = tensor_output_num_model[0, channel, :, plot_coordinate[0], plot_coordinate[1]]
        profile_tensor_float = tensor_float[0, channel, :, plot_coordinate[0], plot_coordinate[1]]
        profile_tensor_NN_model = mean_chl_tensor[0, channel, :, plot_coordinate[0], plot_coordinate[1]]
        profile_tensor_NN_1_model = tensor_output_NN_1_model[0, channel, :, plot_coordinate[0], plot_coordinate[1]]
        #addition of moving_average
        profile_tensor_num_model = moving_average(profile_tensor_num_model.detach().cpu().numpy(), 3)
        profile_tensor_float = moving_average(profile_tensor_float.detach().cpu().numpy(), 3)
        profile_tensor_NN_model = moving_average(profile_tensor_NN_model.detach().cpu().numpy(), 3)
        profile_tensor_NN_1_model = moving_average(profile_tensor_NN_1_model.detach().cpu().numpy(), 3)
        profile_tensor_NN_model = np.maximum(profile_tensor_NN_model, 0)
        #plot profiles
        path_fig_channel_coordinates = path_fig_channel + "/lat_" + str(plot_coordinate[1]) + "_lon_" + str(plot_coordinate[0])
        plt.yticks(np.arange(280, -40, -40), np.arange(0, 320, 40))  
        plt.plot(profile_tensor_float, depth_levels, color="#d32a16", label=r"ARGO-float", linewidth=2.0)  
        plt.plot(profile_tensor_NN_model, depth_levels, color="#2CA02C", label=r"CNN-3DMedSea", linewidth=2.0)       
        plt.plot(profile_tensor_num_model, depth_levels, color="#1F77B4", label=r"MedBFM", linewidth=2.0)    
        plt.plot(profile_tensor_NN_1_model, depth_levels, color="#2CA02C", linestyle="dashed", label=r"CNN-3DMedSea ($1^{st}$ phase)", linewidth=2.0 )
        plt.grid(axis = 'x')
        plt.xlabel(r"Chlorophyll [$mg \ m^{-3}$]")
        plt.ylabel(r"Depth [$m$]")
        plt.tight_layout()
        plt.savefig(path_fig_channel_coordinates + ".png", dpi=1200)
        plt.close()



def plot_Hovmoller(list_week_tensors, list_week_tensors_BFM, path_job, list_masks, var, path_plots, n_ensemble, ga, dict_coord_ga, mean_layer=False, list_layers = [], mean_ga=True):
    """function to generate the Hovmoller plot"""
    #part 1 --> network evaluation
    sns.set_theme(context='paper', style='whitegrid', font='sans-serif', font_scale=2.1, color_codes=True, rc=None)
    path_mean_std = path_job + "/results_training_2_ensemble/mean_and_std_tensors"
    path_lr = path_job + "/results_training_2_ensemble/" + var + "/20/lrc_0.001"
    my_mean_tensor = torch.unsqueeze(torch.load(path_mean_std + "/mean_tensor.pt")[:, 6, :, :, :], 1).to(device)
    my_std_tensor = torch.unsqueeze(torch.load(path_mean_std + "/std_tensor.pt")[:, 6, :, :, :], 1).to(device)
    Hovmoller_tensor = torch.zeros(list_week_tensors[0].shape[2], len(list_week_tensors))
    Hovmoller_tensor_BFM = torch.zeros(list_week_tensors_BFM[0].shape[2], len(list_week_tensors_BFM))
    for i in range(len(list_week_tensors)):
        input_tensor = list_week_tensors[i]
        single_chl_tensor_shape = (n_ensemble, input_tensor.shape[0], 1, input_tensor.shape[2], input_tensor.shape[3], input_tensor.shape[4])
        ensemble_chl_tensor = torch.ones(single_chl_tensor_shape)
        for i_ens in range(n_ensemble):
            CNN_checkpoint = torch.load(path_lr + "/ensemble_model_" + str(i_ens) + "/model_checkpoint_2_ens_" + str(i_ens) + ".pth", map_location=device)
            CNN_model = CompletionN()
            CNN_model.load_state_dict(CNN_checkpoint['model_state_dict'])  
            CNN_model = CNN_model.to(device)
            CNN_model.eval()
            with torch.no_grad():
                input_tensor = input_tensor.to(device)
                chl_tensor_out = CNN_model(input_tensor.float())
                chl_tensor = Denormalization(chl_tensor_out, my_mean_tensor, my_std_tensor).to(device)
                ensemble_chl_tensor[i_ens, :, :, :, :, :] = chl_tensor
        mean_chl_tensor = torch.mean(ensemble_chl_tensor, dim=0)
        std_chl_tensor = torch.std(ensemble_chl_tensor, dim=0)
        #part 2 --> creation of the tensor for the Hovmoller
        if mean_ga == "ga":
            mean_chl_tensor = mean_chl_tensor * torch.cat(tuple([list_masks[i][:, :, :, :, 1:-1] for i in range(len(list_masks)-1)]), 2)
            ga_limits = dict_indexes_ga[ga]
            ga_chl_tensor = mean_chl_tensor[:, :, : , ga_limits[0][0]:ga_limits[0][1], ga_limits[1][0]:ga_limits[1][1]]
            non_zero_mask = torch.where(ga_chl_tensor == 0, torch.nan, ga_chl_tensor)
            ga_mean_profile = torch.nanmean(non_zero_mask, dim=(0,1,3,4))
        elif mean_ga == "single_point":
            coord_prof = dict_coord_ga[ga]   
            ga_mean_profile = torch.squeeze(mean_chl_tensor[:, :, : ,coord_prof[0], coord_prof[1]])
        elif mean_ga == "mean_ngh":
            coord_prof = dict_coord_ga[ga]     
            ngh_amphitude = 30
            mean_chl_tensor = mean_chl_tensor * torch.cat(tuple([list_masks[i][:, :, :, :, 1:-1] for i in range(len(list_masks)-1)]), 2)
            ga_limits = dict_indexes_ga[ga]
            ga_chl_ngh_tensor = mean_chl_tensor[:, :, : , coord_prof[0]-ngh_amphitude:coord_prof[0]+ngh_amphitude, coord_prof[1]-ngh_amphitude:coord_prof[1]+ngh_amphitude]
            non_zero_ngh_mask = torch.where(ga_chl_ngh_tensor == 0, torch.nan, ga_chl_ngh_tensor)
            ga_mean_profile = torch.nanmean(non_zero_ngh_mask, dim=(0,1,3,4))
        ga_mean_profile = torch.clamp_min(ga_mean_profile, min=0)
        ga_mean_profile = torch.from_numpy(moving_average(ga_mean_profile.detach().cpu().numpy(), 3))
        Hovmoller_tensor[:, i] = ga_mean_profile
    #part 3 --> plot the Hovmoller
    fig, axs = plt.subplots(1, figsize=(8, 6)) 
    cmap = plt.get_cmap('viridis')
    cmap.set_under('white')
    plt.grid(False)
    im = axs.imshow(Hovmoller_tensor[:20, :], vmin=0.0, vmax=0.4,cmap=cmap,aspect='auto')  
    plt.colorbar(im, shrink=0.9, pad=0.05)
    axs.set_title("CNN-3DMedSea CHLA timeline")
    axs.set_xticks(np.arange(0, len(list_week_tensors), 1), np.array([i for i in range(len(list_week_tensors))]), rotation=45)
    x_ticks_labels = np.array(["01/2019", "03/2019", "06/2019", "09/2019", "12/2019", "03/2020", "06/2020", "09/2020", "12/2020", "03/2021", "06/2021", "09/2021", "12/2021"])
    #x_ticks_labels = np.array(["09/2022", "12/2022", "03/2023", "06/2023"])
    #axs.set_xticks(np.arange(0, len(list_week_tensors), 13), x_ticks_labels, rotation=45)
    axs.set_yticks(np.arange(0, 20, 5), np.arange(0, 200, 50))
    axs.set_ylabel(r"depth [$m$]")
    plt.tight_layout()
    path_plots_hov = path_plots + "/hovmoller"
    if not os.path.exists(path_plots_hov):
        os.makedirs(path_plots_hov)
    plt.savefig(path_plots_hov + "/hovmoller_external_sept_aug_" + str(ga) +".png", dpi=600)
    plt.close()



def plot_Hovmoller_real_float(float_device_tensor, path_plots, ga, mean_layer=False, list_layers = [], mean_ga=True, tensor_order="standard", name_fig="daily_total", apply_prof_smooting=False):
    """this function creates the Hovmoller relative to a selected, or more selected floats devices"""
    sns.set_theme(context='paper', style='whitegrid', font='sans-serif', font_scale=2.1, color_codes=True, rc=None)
    #part 1: creation of the tensor
    Hovmoller_tensor = float_device_tensor
    if tensor_order == "standard":
        Hovmoller_tensor = float_device_tensor
    elif tensor_order == "LEV_order":
        Hovmoller_tensor = torch.cat((float_device_tensor[:, 46:52], float_device_tensor[:, :46]), axis = 1)
    elif tensor_order == "NWM_order":
        #Hovmoller_tensor = torch.cat((float_device_tensor[:, 475:477], float_device_tensor[:, 292:302], float_device_tensor[:, 290:292],float_device_tensor[:, 265:275], float_device_tensor[:, 479:483], float_device_tensor[:, 275:290], float_device_tensor[:, 483:486]), axis=1)
        #Hovmoller_tensor = torch.cat((float_device_tensor[:, 475:477], float_device_tensor[:, 292:302], + float_device_tensor[:, 290:292], float_device_tensor[:, 265:273], float_device_tensor[:, 275:276], float_device_tensor[:, 288:290], float_device_tensor[:, 279:287], float_device_tensor[:, 274:275], float_device_tensor[:, 479:483], float_device_tensor[:, 276:277]), axis=1)
        #Hovmoller_tensor = torch.cat((float_device_tensor[:, 288:290], float_device_tensor[:, 276:277], float_device_tensor[:, 292:302], float_device_tensor[:, 290:292], float_device_tensor[:, 265:273], float_device_tensor[:, 279:287], float_device_tensor[:, 481:483], float_device_tensor[:, 88:90], float_device_tensor[:, 100:103], float_device_tensor[:, 475:477], float_device_tensor[:, 90:96], float_device_tensor[:, 97:100]), axis=1)
        Hovmoller_tensor = torch.cat((float_device_tensor[:, 108:111], float_device_tensor[:, 288:290], float_device_tensor[:, 276:277], float_device_tensor[:, 292:302], float_device_tensor[:, 290:292], float_device_tensor[:, 265:273], float_device_tensor[:, 279:287], float_device_tensor[:, 481:483], float_device_tensor[:, 88:90], float_device_tensor[:, 90:96], float_device_tensor[:, 97:99], float_device_tensor[:, 102:108]), axis=1)
    if apply_prof_smooting == True:
        for i in range(Hovmoller_tensor.shape[1]):
            Hovmoller_tensor[:, i] = torch.from_numpy(moving_average(Hovmoller_tensor[:,i].detach().cpu().numpy(), 5))
    #part 2: plot the tensor
    fig, axs = plt.subplots(1, figsize=(8,6)) 
    cmap = plt.get_cmap('viridis')
    cmap.set_under('white')
    plt.grid(False)
    im = axs.imshow(Hovmoller_tensor[:20, :], vmin=0.0, vmax = 0.40, cmap=cmap,aspect='auto') 
    plt.colorbar(im, shrink=0.9, pad=0.05)
    x_ticks_labels = np.array(["10/2022", "01/2023", "04/2023", "07/2023"])
    axs.set_xticks(np.arange(0, Hovmoller_tensor.shape[1], 13), x_ticks_labels, rotation=45)
    axs.set_yticks(np.arange(0, 20, 5), np.arange(0, 200, 50))
    axs.set_ylabel(r"depth [$m$]")
    plt.tight_layout()
    path_plots_hov = path_plots + "/hovmoller"
    if not os.path.exists(path_plots_hov):
        os.makedirs(path_plots_hov)
    plt.savefig(path_plots_hov + "/hovmoller_float_" + str(name_fig) + "_" + str(ga) +".png", dpi=600)
    return None