#In this script there are the functions to plot the maps of NN model and profiles of NN models, compared with real distribution of data


import numpy as np
import torch 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import os
import itertools
import seaborn as sns
from mpl_toolkits.basemap import Basemap

from hyperparameter import * 
from utils.utils_mask import apply_masks
from utils.utils_general import compute_mean_layers, compute_profile_mean

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def moving_average(data, window_size):
    if window_size % 2 == 0:
        window_size += 1  # Ensure window size is odd for symmetry
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


def plot_NN_maps(NN_tensor, list_masks, var, path_fig_channel):
    """function that plot the tensor resulted from the NN"""
    sns.set_theme(context='paper', style='whitegrid', font='sans-serif', font_scale=1.0,color_codes=True, rc=None)
    channel = compute_channel(var)
    depth_levels = np.arange(0, NN_tensor.shape[2])
    masked_NN_tensor = apply_masks(NN_tensor, list_masks)
    for d in range(len(depth_levels)):
        masked_NN_tensor[:, :, d, :, :] = masked_NN_tensor[:, :, d, :, :].masked_fill(list_masks[d].to(device)[:, :, :, :, 1:-1] == 0, -1.0)
    for depth_level in depth_levels:
        plot_tensor = torch.clone(masked_NN_tensor)   
        plot_tensor = np.transpose(plot_tensor.cpu(), [0,1,2,4,3])
        plot_tensor = torch.from_numpy(np.flip(plot_tensor.numpy(), 3).copy())
        plot_tensor = np.ma.masked_where(plot_tensor == -1.0, plot_tensor)
        cmap = plt.get_cmap("jet")   
        cmap.set_bad(color='white')
        plt.imshow(plot_tensor[0, channel, depth_level, :, :], cmap=cmap, vmin = parameters_plots[var][0][depth_level], vmax = parameters_plots[var][1][depth_level], interpolation='spline16')
        plt.colorbar(shrink=0.6, pad=0.01)
        my_xticks= np.arange(0, h, 30)
        my_yticks = np.arange(0, w, 30)
        my_xticks_label = np.array([int((index - 1) * resolution[1] / constant_longitude + 1 + longitude_interval[0]) for index in np.arange(0, h, 30)])
        my_yticks_label = np.array([int((index - 1) * resolution[0] / constant_latitude + 1 + latitude_interval[0]) for index in np.arange(w-1, -1, -30)])   #prima era -1
        plt.xticks(my_xticks, my_xticks_label, fontsize=6)
        plt.yticks(my_yticks, my_yticks_label, fontsize=6)
        plt.xlabel("longitude")
        plt.ylabel("latitude")
        plt.savefig(path_fig_channel + "/depth_" + str(depth_level) + ".png", dpi=600)
        plt.close()


def plot_NN_maps_std_percentage(NN_tensor, list_masks, var, path_fig_channel):
    """function that plot the tensor resulted from the NN"""
    channel = compute_channel(var)
    depth_levels = np.arange(0, NN_tensor.shape[2])
    masked_NN_tensor = apply_masks(NN_tensor, list_masks)
    for depth_level in depth_levels:
        plot_tensor = torch.clone(masked_NN_tensor)   
        plot_tensor = np.transpose(plot_tensor.cpu(), [0,1,2,4,3])
        plot_tensor = torch.from_numpy(np.flip(plot_tensor.numpy(), 3).copy())
        cmap = plt.get_cmap("jet")  
        newcmap = compute_cmap('jet')
        plt.imshow(plot_tensor[0, channel, depth_level, :, :], cmap=newcmap, vmin=0, vmax=100, interpolation='none')  
        plt.colorbar(shrink=0.6, pad=0.01)
        my_xticks= np.arange(0, h, 30)
        my_yticks = np.arange(0, w, 30)
        my_xticks_label = np.array([int((index - 1) * resolution[1] / constant_longitude + 1 + longitude_interval[0]) for index in np.arange(0, h, 30)])
        my_yticks_label = np.array([int((index - 1) * resolution[0] / constant_latitude + 1 + latitude_interval[0]) for index in np.arange(w-1, -1, -30)])  
        plt.xticks(my_xticks, my_xticks_label, fontsize=6)
        plt.yticks(my_yticks, my_yticks_label, fontsize=6)
        plt.xlabel("longitude")
        plt.ylabel("latitude")
        plt.savefig(path_fig_channel + "/depth_" + str(depth_level) + ".png")
        plt.close()


def plot_models_profiles_1p(tensor_input_NN, tensor_output_NN_model, tensor_output_num_model, var, path_fig_channel, list_to_plot_coordinates):
    sns.set_theme(context='paper', style='whitegrid', font='sans-serif', font_scale=1.0, color_codes=True, rc=None)
    channel = compute_channel(var)
    depth_levels = resolution [2] * np.arange(tensor_input_NN.shape[2]-1, -1, -1)  
    for plot_coordinate in list_to_plot_coordinates:
        profile_tensor_num_model = tensor_output_num_model[0, channel, :, plot_coordinate[0], plot_coordinate[1]]
        profile_tensor_input_NN = tensor_input_NN[0, channel, :, plot_coordinate[0], plot_coordinate[1]]
        profile_tensor_NN_model = tensor_output_NN_model[0, channel, :, plot_coordinate[0], plot_coordinate[1]]
        #addition of moving_average
        profile_tensor_num_model = moving_average(profile_tensor_num_model.detach().cpu().numpy(), 3)
        profile_tensor_input_NN = moving_average(profile_tensor_input_NN.detach().cpu().numpy(), 3)
        profile_tensor_NN_model = moving_average(profile_tensor_NN_model.detach().cpu().numpy(), 3)
        #plot of profiles
        path_fig_channel_coordinates = path_fig_channel + "/lat_" + str(plot_coordinate[1]) + "_lon_" + str(plot_coordinate[0])
        plt.yticks(depth_levels, resolution[2] * np.arange(0, tensor_input_NN.shape[2]), fontsize=6)  
        plt.plot(profile_tensor_input_NN, depth_levels, color="red", label="input CNN profile")   
        plt.plot(profile_tensor_NN_model, depth_levels, color="green", label="CNN profile")       
        plt.plot(profile_tensor_num_model, depth_levels, color="blue", label="BFM profile")       
        plt.grid(axis = 'x')    
        plt.xlabel(var + " values")
        plt.ylabel("depths values")
        plt.legend(loc="lower right", prop={'size': 6})
        plt.savefig(path_fig_channel_coordinates + ".png")
        plt.close()


def plot_NN_maps_layer_mean(NN_tensor, list_masks, var, path_fig_channel, list_layers):
    """function that plot the mean computed on different layer of the NN output"""
    channel = compute_channel(var)
    masked_NN_tensor = apply_masks(NN_tensor, list_masks)
    #compute the mean wrt layers and return the new tensor
    masked_NN_tensor_mean_layer = compute_mean_layers(masked_NN_tensor, list_layers, 2, (1, 1, len(list_layers), masked_NN_tensor.shape[3], masked_NN_tensor.shape[4]))   #prima erano 181 e 73 le ultime due shape
    #plot the resulted tensor
    for layers_level in range(len(list_layers) - 1):
        plot_tensor = torch.clone(masked_NN_tensor_mean_layer)   
        plot_tensor = np.transpose(plot_tensor.cpu(), [0,1,2,4,3])
        plot_tensor = torch.from_numpy(np.flip(plot_tensor.numpy(), 3).copy())
        cmap = plt.get_cmap("jet")   #trovata su internet, dovrebbe dìandare dai rossi ai blu
        newcmap = compute_cmap('jet')
        depth_level = list_layers[layers_level] // resolution[2]
        print("depth level", depth_level)
        plt.imshow(plot_tensor[0, channel, layers_level, :, :], cmap=newcmap, vmin = parameters_plots[var][0][depth_level], vmax = parameters_plots[var][1][depth_level], interpolation='spline16')    
        plt.colorbar(shrink=0.6, pad=0.01)
        my_xticks= np.arange(0, h, 30)
        my_yticks = np.arange(0, w, 30)
        my_xticks_label = np.array([int((index - 1) * resolution[1] / constant_longitude + 1 + longitude_interval[0]) for index in np.arange(0, h, 30)])
        my_yticks_label = np.array([int((index - 1) * resolution[0] / constant_latitude + 1 + latitude_interval[0]) for index in np.arange(w-1, -1, -30)])  
        plt.xticks(my_xticks, my_xticks_label, fontsize=6)
        plt.yticks(my_yticks, my_yticks_label, fontsize=6)
        plt.xlabel("longitude")
        plt.ylabel("latitude")
        plt.savefig(path_fig_channel + "/depth_" + str(list_layers[layers_level]) + ".png")
        plt.close()


def comparison_profiles_1_2_phases(tensor_output_float, tensor_output_NN_model, tensor_output_num_model, tensor_output_NN_1_model, var, path_fig_channel):
    """function that compares the profiles of float, BFM, 1 and 2 CNN output"""
    channel = compute_channel(var)
    depth_levels = resolution [2] * np.arange(tensor_output_float.shape[2]-1, -1, -1)  
    longitude_indexes = np.arange(0, tensor_output_float.shape[3])
    latitude_indexes = np.arange(0, tensor_output_float.shape[4])
    counter_float = 0
    for latitude_index in latitude_indexes:
        for longitude_index in longitude_indexes:
            profile_tensor_float = tensor_output_float[0, channel, :, longitude_index, latitude_index] 
            if torch.sum(profile_tensor_float) != 0.0 and not torch.equal(profile_tensor_float, profile_tensor_float[0] * torch.ones(tensor_output_float.shape[2])):   
                counter_float += 1
                profile_tensor_NN_model = tensor_output_NN_model[0, channel, :, longitude_index, latitude_index] 
                profile_tensor_num_model = tensor_output_num_model[0, channel, :, longitude_index, latitude_index] 
                profile_tensor_NN_1_model = tensor_output_NN_1_model[0, channel, :, longitude_index, latitude_index]
                #add moving average
                profile_tensor_float = moving_average(profile_tensor_float.detach().cpu().numpy(), 3)
                profile_tensor_NN_model = moving_average(profile_tensor_NN_model.detach().cpu().numpy(), 3)
                profile_tensor_NN_1_model = moving_average(profile_tensor_NN_1_model.detach().cpu().numpy(), 3)
                profile_tensor_num_model = moving_average(profile_tensor_num_model.detach().cpu().numpy(), 3)
                #plots profiles
                path_fig_channel_coordinates = path_fig_channel + "/lat_" + str(latitude_index) + "_lon_" + str(longitude_index)
                plt.yticks(depth_levels, resolution[2] * np.arange(0, tensor_output_float.shape[2]), fontsize=6)   
                plt.plot(profile_tensor_float, depth_levels, color="red", label="float profile")           
                plt.plot(profile_tensor_NN_model, depth_levels, color="green", label="CNN profile")         
                plt.plot(profile_tensor_NN_1_model, depth_levels, color="green", linestyle="dashed", label = "CNN profile first phase")   
                plt.plot(profile_tensor_num_model, depth_levels, color="blue", label="BFM profile")         
                plt.grid(axis = 'x')   
                plt.xlabel(var + " values")
                plt.ylabel("depths values")
                plt.legend(loc="lower right", prop={'size': 6})
                plt.savefig(path_fig_channel_coordinates + ".png")
                plt.close()