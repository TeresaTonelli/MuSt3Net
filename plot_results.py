#In this script there are the functions to plot the maps of NN model and profiles of NN models, compared with real distribution of data


import numpy as np
import torch 
import matplotlib.pyplot as plt
import os
import itertools
from hyperparameter import * 
from utils_mask import apply_masks
from utils_function import compute_mean_layers, compute_profile_mean

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device", device)


def compute_channel(var):
   """this method returns the index of the channel in which this variable is saved inside the tensor"""
   if var in list_physics_vars:
      return list_physics_vars.index(var)
   elif var in list_biogeoch_vars:
      return 0


def plot_NN_maps(NN_tensor, list_masks, var, path_fig_channel):
    """function that plot the tensor resulted from the NN"""
    channel = compute_channel(var)
    depth_levels = np.arange(0, NN_tensor.shape[2])
    masked_NN_tensor = apply_masks(NN_tensor, list_masks)
    for depth_level in depth_levels:
        plot_tensor = torch.clone(masked_NN_tensor)   
        plot_tensor = np.transpose(plot_tensor.cpu(), [0,1,2,4,3])
        plot_tensor = torch.from_numpy(np.flip(plot_tensor.numpy(), 3).copy())
        cmap = plt.get_cmap("jet")   #trovata su internet, dovrebbe dìandare dai rossi ai blu
        #plt.imshow(plot_tensor[0, channel, depth_level, :, :], cmap=cmap, vmin = 0, vmax = torch.quantile(plot_tensor[0, channel, depth_level, :, :], 0.99, interpolation="linear"))
        plt.imshow(plot_tensor[0, channel, depth_level, :, :], cmap=cmap, vmin = parameters_plots[var][0][depth_level], vmax = parameters_plots[var][1][depth_level])

        plt.colorbar(shrink=0.6, pad=0.01)
        my_xticks= np.arange(0, h, 20)
        my_yticks = np.arange(0, w, 20)
        my_xticks_label = np.array([int((index - 1) * resolution[1] / constant_longitude + 1 + longitude_interval[0]) for index in np.arange(0, h, 20)])
        my_yticks_label = np.array([int((index - 1) * resolution[0] / constant_latitude + 1 + latitude_interval[0]) for index in np.arange(w-1, -1, -20)])   #prima era -1
        plt.xticks(my_xticks, my_xticks_label, fontsize=6)
        plt.yticks(my_yticks, my_yticks_label, fontsize=6)
        plt.xlabel("longitude")
        plt.ylabel("latitude")
        plt.savefig(path_fig_channel + "/depth_" + str(depth_level) + ".png")
        plt.close()


def plot_difference_maps(tensor_input, tensor_output, list_masks, var, path_fig_channel):
    """function that plot the difference between 2 tensors, one containing NN model for a variable, the another the numerical model distribution of the same variable"""
    channel = compute_channel(var)
    depth_levels = np.arange(0, tensor_input.shape[2])
    masked_tensor_input = apply_masks(tensor_input, list_masks)
    masked_tensor_output = apply_masks(tensor_output, list_masks)
    difference_tensor = masked_tensor_output - masked_tensor_input
    for depth_level in depth_levels:
        plot_tensor = torch.clone(difference_tensor)   #difference_tensor.copy()
        plot_tensor = np.transpose(plot_tensor.cpu(), [0,1,2,4,3])
        plot_tensor = torch.from_numpy(np.flip(plot_tensor.numpy(), 3).copy())
        cmap = plt.get_cmap("Greens")
        plt.imshow(plot_tensor[0, channel, depth_level, :, :], cmap=cmap, 
                   vmin=torch.quantile(plot_tensor[0, channel, depth_level, :, :], 0.05, interpolation="linear"), 
                   vmax=torch.quantile(plot_tensor[0, channel, depth_level, :, :], 0.95, interpolation="linear"))
        plt.colorbar(shrink=0.6, pad=0.01)
        my_xticks= np.arange(0, h, 20)
        my_yticks = np.arange(0, w, 20)
        my_xticks_label = np.array([int((index - 1) * resolution[1] / constant_longitude + 1 + longitude_interval[0]) for index in np.arange(0, h, 20)])
        my_yticks_label = np.array([int((index - 1) * resolution[0] / constant_latitude + 1 + latitude_interval[0]) for index in np.arange(w-1, -1, -20)])   #prima era -1
        plt.xticks(my_xticks, my_xticks_label, fontsize=6)
        plt.yticks(my_yticks, my_yticks_label, fontsize=6)
        plt.xlabel("longitude")
        plt.ylabel("latitude")
        plt.savefig(path_fig_channel + "/depth_" + str(depth_level) + ".png")
        plt.close()



def plot_difference_NN_phases(tensor_NN_1, tensor_NN_2, list_masks, var, path_fig_channel, float_locations_coord):
    """plot the difference tensor between the output of the 1 and the 2 phase. In this way wwe hope tho see a difference in the ngh of float locations"""
    channel = compute_channel(var)
    depth_levels = np.arange(0, tensor_NN_1.shape[2])
    masked_tensor_NN_1 = apply_masks(tensor_NN_1, list_masks)
    masked_tensor_NN_2 = apply_masks(tensor_NN_2, list_masks)
    difference_tensor = masked_tensor_NN_2 - masked_tensor_NN_1
    for depth_level in depth_levels[0:10]:
        plot_tensor = torch.clone(difference_tensor)   #difference_tensor.copy()
        plot_tensor = np.transpose(plot_tensor.cpu(), [0,1,2,4,3])
        plot_tensor = torch.from_numpy(np.flip(plot_tensor.numpy(), 3).copy())
        cmap = plt.get_cmap("viridis")
        plt.imshow(plot_tensor[0, channel, depth_level, :, :], cmap=cmap, vmin= parameters_plots[var][0][depth_level], vmax = parameters_plots[var][1][depth_level])
                   #vmin=torch.quantile(plot_tensor[0, channel, depth_level, :, :], 0.05, interpolation="linear"), 
                   #vmax=torch.quantile(plot_tensor[0, channel, depth_level, :, :], 0.95, interpolation="linear"))
        plt.colorbar(shrink=0.6, pad=0.01)
        my_xticks= np.arange(0, h, 20)
        my_yticks = np.arange(0, w, 20)
        my_xticks_label = np.array([int((index - 1) * resolution[1] / constant_longitude + 1 + longitude_interval[0]) for index in np.arange(0, h, 20)])
        my_yticks_label = np.array([int((index - 1) * resolution[0] / constant_latitude + 1 + latitude_interval[0]) for index in np.arange(w-1, -1, -20)])   #prima era -1
        plt.xticks(my_xticks, my_xticks_label, fontsize=6)
        plt.yticks(my_yticks, my_yticks_label, fontsize=6)
        plt.scatter(np.array([float_locations_coord[i][0] for i in range(len(float_locations_coord))]), np.array([(w- float_locations_coord[i][1]) for i in range(len(float_locations_coord))]), s=1, c="red")  #controllare che longitudine e latitudien siano messe correttamente negli assi, e non invertite
        plt.xlabel("longitude")
        plt.ylabel("latitude")
        plt.savefig(path_fig_channel + "/depth_" + str(depth_level) + ".png")
        plt.close()



def plot_models_profiles_1p(tensor_input_NN, tensor_output_NN_model, tensor_output_num_model, var, path_fig_channel, list_to_plot_coordinates):
    channel = compute_channel(var)
    depth_levels = resolution [2] * np.arange(tensor_input_NN.shape[2]-1, -1, -1)  #20, non resolution[2] prima
    for plot_coordinate in list_to_plot_coordinates:
        profile_tensor_num_model = tensor_output_num_model[0, channel, :, plot_coordinate[0], plot_coordinate[1]] 
        profile_tensor_input_NN = tensor_input_NN[0, channel, :, plot_coordinate[0], plot_coordinate[1]] 
        #print("num sum ", torch.sum(profile_tensor_num_model))
        #se lui è effettivamente un profilo, vado a generarmi anche quelle dei due modelli per le stesse coordinate
        profile_tensor_NN_model = tensor_output_NN_model[0, channel, :, plot_coordinate[0], plot_coordinate[1]] 
        #print("NN sum", torch.sum(profile_tensor_NN_model))
        path_fig_channel_coordinates = path_fig_channel + "/lat_" + str(plot_coordinate[1]) + "_lon_" + str(plot_coordinate[0])
        plt.yticks(depth_levels, resolution[2] * np.arange(0, tensor_input_NN.shape[2]), fontsize=6)  
        plt.plot(profile_tensor_input_NN.cpu(), depth_levels, color="red", label="input CNN profile")
        plt.plot(profile_tensor_NN_model.cpu(), depth_levels, color="green", label="CNN profile")
        plt.plot(profile_tensor_num_model.cpu(), depth_levels, color="blue", label="BFM profile")
        plt.grid(axis = 'y')
        plt.xlabel(var + " values")
        plt.ylabel("depths values")
        plt.title("profiles of " +  var + " distribution")
        plt.legend(loc="upper left", prop={'size': 6})
        plt.savefig(path_fig_channel_coordinates + ".png")
        plt.close()




def plot_models_profiles(tensor_output_float, tensor_output_NN_model, tensor_output_num_model, var, path_fig_channel, list_to_plot_coordinates):
    channel = compute_channel(var)
    depth_levels = resolution [2] * np.arange(tensor_output_float.shape[2]-1, -1, -1)  #20, non resolution[2] prima
    longitude_indexes = np.arange(0, tensor_output_float.shape[3])
    latitude_indexes = np.arange(0, tensor_output_float.shape[4])
    counter_float = 0
    for latitude_index in latitude_indexes:
        for longitude_index in longitude_indexes:
            profile_tensor_float = tensor_output_float[0, channel, :, longitude_index, latitude_index] 
            if torch.sum(profile_tensor_float) != 0.0 and not torch.equal(profile_tensor_float, profile_tensor_float[0] * torch.ones(tensor_output_float.shape[2])):   #.to(device)):   #profile_tensor_float != profile_tensor_float[0] * torch.ones(tensor_output_float.shape[2]):
                #print("sampled coordi", (longitude_index, latitude_index) )
                counter_float += 1
                #print("counter_float", counter_float)
                #print("float sum ", torch.sum(profile_tensor_float))
                #se lui è effettivamente un profilo, vado a generarmi anche quelle dei due modelli per le stesse coordinate
                profile_tensor_NN_model = tensor_output_NN_model[0, channel, :, longitude_index, latitude_index] 
                #print("NN sum", torch.sum(profile_tensor_NN_model))
                profile_tensor_num_model = tensor_output_num_model[0, channel, :, longitude_index, latitude_index] 
                path_fig_channel_coordinates = path_fig_channel + "/lat_" + str(latitude_index) + "_lon_" + str(longitude_index)
                plt.yticks(depth_levels, resolution[2] * np.arange(0, tensor_output_float.shape[2]), fontsize=6)  
                plt.plot(profile_tensor_float.cpu(), depth_levels, color="red", label="float profile")
                plt.plot(profile_tensor_NN_model.cpu(), depth_levels, color="green", label="CNN profile")
                plt.plot(profile_tensor_num_model.cpu(), depth_levels, color="blue", label="BFM profile")
                plt.grid(axis = 'y')
                plt.xlabel(var + " values")
                plt.ylabel("depths values")
                plt.title("profiles of " +  var + " distribution")
                plt.legend(loc="upper left", prop={'size': 6})
                plt.savefig(path_fig_channel_coordinates + ".png")
                plt.close()




def plot_NN_maps_layer_mean(NN_tensor, list_masks, var, path_fig_channel, list_layers):
    """function that plot the mean computed on different layer of the NN output"""
    channel = compute_channel(var)
    masked_NN_tensor = apply_masks(NN_tensor, list_masks)
    #compute the mean wrt layers and return the new tensor
    masked_NN_tensor_mean_layer = compute_mean_layers(masked_NN_tensor, list_layers, 2, (1, 1, len(list_layers), 181, 73))
    #plot the resulted tensor
    for layers_level in range(len(list_layers)):
        plot_tensor = torch.clone(masked_NN_tensor_mean_layer)   
        plot_tensor = np.transpose(plot_tensor.cpu(), [0,1,2,4,3])
        plot_tensor = torch.from_numpy(np.flip(plot_tensor.numpy(), 3).copy())
        cmap = plt.get_cmap("jet")   #trovata su internet, dovrebbe dìandare dai rossi ai blu
        plt.imshow(plot_tensor[0, channel, layers_level, :, :], cmap=cmap, vmin = 0, vmax = torch.quantile(plot_tensor[0, channel, layers_level, :, :], 0.99, interpolation="linear"))
                   #vmin=torch.quantile(plot_tensor[0, channel, depth_level, :, :], 0.01, interpolation="linear"), 
                   #vmax=torch.quantile(plot_tensor[0, channel, depth_level, :, :], 0.99, interpolation="linear"))
        plt.colorbar()
        my_xticks= np.arange(0, h, 5)
        my_yticks = np.arange(0, w, 5)
        my_xticks_label = np.array([int((index - 1) * resolution[1] / constant_longitude + 1 + longitude_interval[0]) for index in np.arange(0, h, 5)])
        my_yticks_label = np.array([int((index - 1) * resolution[0] / constant_latitude + 1 + latitude_interval[0]) for index in np.arange(w-1, -1, -5)])   #prima era -1
        plt.xticks(my_xticks, my_xticks_label, fontsize=6)
        plt.yticks(my_yticks, my_yticks_label, fontsize=6)
        plt.xlabel("longitude")
        plt.ylabel("latitude")
        plt.savefig(path_fig_channel + "/depth_" + str(layers_level) + ".png")
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
            if torch.sum(profile_tensor_float) != 0.0 and not torch.equal(profile_tensor_float, profile_tensor_float[0] * torch.ones(tensor_output_float.shape[2])):    #.to(device)):   #profile_tensor_float != profile_tensor_float[0] * torch.ones(tensor_output_float.shape[2]):
                #print("sampled coordi", (longitude_index, latitude_index) )
                counter_float += 1
                #print("counter_float", counter_float)
                #print("float sum ", torch.sum(profile_tensor_float))
                #se lui è effettivamente un profilo, vado a generarmi anche quelle dei due modelli per le stesse coordinate
                profile_tensor_NN_model = tensor_output_NN_model[0, channel, :, longitude_index, latitude_index] 
                #print("NN sum", torch.sum(profile_tensor_NN_model))
                profile_tensor_num_model = tensor_output_num_model[0, channel, :, longitude_index, latitude_index] 
                profile_tensor_NN_1_model = tensor_output_NN_1_model[0, channel, :, longitude_index, latitude_index]
                path_fig_channel_coordinates = path_fig_channel + "/lat_" + str(latitude_index) + "_lon_" + str(longitude_index)
                plt.yticks(depth_levels, resolution[2] * np.arange(0, tensor_output_float.shape[2]), fontsize=6)   
                plt.plot(profile_tensor_float.cpu(), depth_levels, color="red", label="float profile")
                plt.plot(profile_tensor_NN_model.cpu(), depth_levels, color="green", label="CNN profile")
                plt.plot(profile_tensor_NN_1_model.cpu(), depth_levels, color="green", linestyle="dashed", label = "CNN profile first phase")
                plt.plot(profile_tensor_num_model.cpu(), depth_levels, color="blue", label="BFM profile")
                plt.grid(axis = 'y')
                plt.xlabel(var + " values")
                plt.ylabel("depths values")
                plt.title("profiles of " +  var + " distribution")
                plt.legend(loc="upper left", prop={'size': 6})
                plt.savefig(path_fig_channel_coordinates + ".png")
                plt.close()


def profiles_2p_mean(tensor_float, tensor_NN, tensor_BFM, var, path_fig_channel, list_to_plot_coordinates):
    """this function compute the mean of profiles wrt depths for each test week"""
    mean_profiles_float = compute_profile_mean(tensor_float, 2, list_to_plot_coordinates)
    mean_profiles_NN = compute_profile_mean(tensor_NN, 2, list_to_plot_coordinates)
    mean_profiles_BFM = compute_profile_mean(tensor_BFM, 2, list_to_plot_coordinates)
    depth_levels = resolution [2] * np.arange(mean_profiles_BFM.shape[2]-1, -1, -1)
    path_fig_channel_coordinates = path_fig_channel
    plt.yticks(depth_levels, resolution[2] * np.arange(0, mean_profiles_BFM.shape[2]), fontsize=6)   #np.arange(tensor_output_float.shape[2]-1, -1, -1))
    plt.plot(mean_profiles_float.cpu(), depth_levels, color="red", label="mean float profile")
    plt.plot(mean_profiles_NN.cpu(), depth_levels, color="green", label="mean CNN profile")
    plt.plot(mean_profiles_BFM.cpu(), depth_levels, color="blue", label="mean BFM profile")
    plt.grid(axis = 'y')
    plt.xlabel(var + " values")
    plt.ylabel("depths values")
    plt.title("profiles of " +  var + " distribution")
    plt.legend(loc="upper left", prop={'size': 6})
    plt.savefig(path_fig_channel_coordinates + ".png")
    plt.close()



def NN_differences_layer_mean(tensor_NN_1, tensor_NN_2, list_masks, var, path_fig_channel, float_locations_coord, list_layers):
    """plot the difference tensor between the output of the 1 and the 2 phase. In this way wwe hope tho see a difference in the ngh of float locations"""
    channel = compute_channel(var)
    depth_levels = np.arange(0, tensor_NN_1.shape[2])
    masked_tensor_NN_1 = apply_masks(tensor_NN_1, list_masks)
    masked_tensor_NN_2 = apply_masks(tensor_NN_2, list_masks)
    #compute mean layers of both tensors
    masked_tensor_NN_1_mean_layers = compute_mean_layers(masked_tensor_NN_1, list_layers, 2, (masked_tensor_NN_1.shape[0], masked_tensor_NN_1.shape[1, len(list_layers), masked_tensor_NN_1.shape[3], masked_tensor_NN_1.shape[4]]))
    masked_tensor_NN_2_mean_layers = compute_mean_layers(masked_tensor_NN_2, list_layers, 2, (masked_tensor_NN_2.shape[0], masked_tensor_NN_2.shape[1, len(list_layers), masked_tensor_NN_2.shape[3], masked_tensor_NN_2.shape[4]]))
    difference_tensor = masked_tensor_NN_2_mean_layers - masked_tensor_NN_1_mean_layers
    for depth_level in depth_levels[0:10]:
        plot_tensor = torch.clone(difference_tensor)   #difference_tensor.copy()
        plot_tensor = np.transpose(plot_tensor.cpu(), [0,1,2,4,3])
        plot_tensor = torch.from_numpy(np.flip(plot_tensor.numpy(), 3).copy())
        cmap = plt.get_cmap("viridis")
        plt.imshow(plot_tensor[0, channel, depth_level, :, :], cmap=cmap, 
                   vmin=torch.quantile(plot_tensor[0, channel, depth_level, :, :], 0.05, interpolation="linear"), 
                   vmax=torch.quantile(plot_tensor[0, channel, depth_level, :, :], 0.95, interpolation="linear"))
        plt.colorbar()
        my_xticks= np.arange(0, h, 5)
        my_yticks = np.arange(0, w, 5)
        my_xticks_label = np.array([int((index - 1) * resolution[1] / constant_longitude + 1 + longitude_interval[0]) for index in np.arange(0, h, 5)])
        my_yticks_label = np.array([int((index - 1) * resolution[0] / constant_latitude + 1 + latitude_interval[0]) for index in np.arange(w-1, -1, -5)])   #prima era -1
        plt.xticks(my_xticks, my_xticks_label, fontsize=6)
        plt.yticks(my_yticks, my_yticks_label, fontsize=6)
        plt.scatter(np.array([float_locations_coord[i][0] for i in range(len(float_locations_coord))]), np.array([(w- float_locations_coord[i][1]) for i in range(len(float_locations_coord))]), s=1, c="red")  #controllare che longitudine e latitudien siano messe correttamente negli assi, e non invertite
        #print("x scatter", np.array([float_locations_coord[i][0] for i in range(len(float_locations_coord))]))
        #print("y scatter", np.array([(w - float_locations_coord[i][1]) for i in range(len(float_locations_coord))]))
        plt.xlabel("longitude")
        plt.ylabel("latitude")
        plt.savefig(path_fig_channel + "/depth_" + str(depth_level) + ".png")
        plt.close()
