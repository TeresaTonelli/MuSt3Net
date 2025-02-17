#In this script we write all the function to generate plots FOR POST-PROCESSING, so the ones that I can recall after the network training 


import numpy as np
import torch 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
#import seaborn as sns
#from mpl_toolkits.basemap import Basemap
import os
import itertools

from convolutional_network import CompletionN
from denormalization import Denormalization
from hyperparameter import * 
from utils_mask import apply_masks
from utils_function import compute_mean_layers, compute_profile_mean


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device", device)


#sns.set_theme(context='paper', style='whitegrid', font='sans-serif', font_scale=1.5,
 #             color_codes=True, rc=None)


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
    newcolors[0, :] = np.array([1, 1, 1, 1])  # RGBA for white
    newcmap = ListedColormap(newcolors)
    return newcmap



def plot_NN_maps_final_1(input_tensor, CNN_model, path_job, list_masks, var, path_mean_std, path_fig_channel):
    """function that plot the tensor resulted from the NN"""
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
        print("chl tensor shape", chl_tensor.shape)
        depth_levels = np.arange(0, chl_tensor.shape[2])
        masked_chl_tensor = apply_masks(chl_tensor, list_masks)
        channel = compute_channel(var)
        for depth_level in depth_levels:
            plot_tensor = torch.clone(masked_chl_tensor)   
            plot_tensor = np.transpose(plot_tensor.cpu(), [0,1,2,4,3])
            plot_tensor = torch.from_numpy(np.flip(plot_tensor.numpy(), 3).copy())
            cmap = plt.get_cmap("jet")   
            newcmap = compute_cmap('jet')
            plt.imshow(plot_tensor[0, channel, depth_level, :, :], cmap=newcmap, vmin = parameters_plots[var][0][depth_level], vmax = parameters_plots[var][1][depth_level], interpolation='spline16')
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


def plot_NN_maps_final_2(input_tensor, CNN_model, path_job, list_masks, var, path_mean_std, path_fig_channel, n_ensemble):
    """function that plot the tensor resulted from the NN"""
    my_mean_tensor = torch.unsqueeze(torch.load(path_mean_std + "/mean_tensor.pt")[:, 6, :, :, :], 1).to(device)
    my_std_tensor = torch.unsqueeze(torch.load(path_mean_std + "/std_tensor.pt")[:, 6, :, :, :], 1).to(device)
    single_chl_tensor_shape = (n_ensemble, input_tensor.shape[0], 1, input_tensor.shape[2], input_tensor.shape[3], input_tensor.shape[4])
    ensemble_chl_tensor = torch.ones(single_chl_tensor_shape)
    for i_ens in range(n_ensemble):
        CNN_checkpoint = torch.load(path_job + "/results_training_2_ensemble/" + var + "/20/lrc_0.001/ensemble_model_" + str(i_ens) + "/model_checkpoint_2_ens_" + str(i_ens) + ".pth", map_location=device)
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
    print("mean chl tensor shape", mean_chl_tensor.shape)
    depth_levels = np.arange(0, mean_chl_tensor.shape[2])
    masked_chl_tensor = apply_masks(mean_chl_tensor, list_masks)
    channel = compute_channel(var)
    for depth_level in depth_levels:
        plot_tensor = torch.clone(masked_chl_tensor)   
        plot_tensor = np.transpose(plot_tensor.cpu(), [0,1,2,4,3])
        plot_tensor = torch.from_numpy(np.flip(plot_tensor.numpy(), 3).copy())
        cmap = plt.get_cmap("jet")   
        newcmap = compute_cmap('jet')
        plt.imshow(plot_tensor[0, channel, depth_level, :, :], cmap=newcmap, vmin = parameters_plots[var][0][depth_level], vmax = parameters_plots[var][1][depth_level], interpolation='spline16')
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



def plot_models_profiles_1(tensor_input_NN, CNN_model, tensor_output_num_model, path_job, var, path_mean_std, path_fig_channel, list_to_plot_coordinates):
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
            print("plot coordinate", plot_coordinate)
            print("tensor num shape", tensor_output_num_model.shape)
            profile_tensor_num_model = tensor_output_num_model[0, channel, :, plot_coordinate[0], plot_coordinate[1]]
            profile_tensor_input_NN = tensor_input_NN_model[0, channel, :, plot_coordinate[0], plot_coordinate[1]]
            profile_tensor_NN_model = tensor_output_NN_model[0, channel, :, plot_coordinate[0], plot_coordinate[1]]
            #addition of moving_average
            profile_tensor_num_model = moving_average(profile_tensor_num_model.detach().cpu().numpy(), 3)
            profile_tensor_input_NN = moving_average(profile_tensor_input_NN.detach().cpu().numpy(), 3)
            profile_tensor_NN_model = moving_average(profile_tensor_NN_model.detach().cpu().numpy(), 3)
            #plot profiles
            path_fig_channel_coordinates = path_fig_channel + "/lat_" + str(plot_coordinate[1]) + "_lon_" + str(plot_coordinate[0])
            plt.yticks(depth_levels, resolution[2] * np.arange(0, tensor_input_NN.shape[2]), fontsize=6)  
            plt.plot(profile_tensor_input_NN, depth_levels, color="red", label="input CNN profile")   #prima era profile_tensor_input_NN.cpu()
            plt.plot(profile_tensor_NN_model, depth_levels, color="green", label="CNN profile")       #prima era profile_tensor_NN_model.cpu()
            plt.plot(profile_tensor_num_model, depth_levels, color="blue", label="BFM profile")       #prima era profile_tensor_num_model.cpu()
            plt.grid(axis = 'y')
            plt.xlabel(var + " values")
            plt.ylabel("depths values")
            plt.legend(loc="lower right", prop={'size': 6})
            plt.savefig(path_fig_channel_coordinates + ".png")
            plt.close()



def plot_models_profiles_2(tensor_input_NN, CNN_model, tensor_output_num_model, path_job, var, path_mean_std, path_fig_channel, list_to_plot_coordinates, n_ensemble):
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
    print("mean chl tensor shape", mean_chl_tensor.shape)
    depth_levels = resolution [2] * np.arange(tensor_input_NN.shape[2]-1, -1, -1)  #20, non resolution[2] prima
    channel = compute_channel(var)
    for plot_coordinate in list_to_plot_coordinates:
        print("plot coordinate", plot_coordinate)
        print("tensor num shape", tensor_output_num_model.shape)
        profile_tensor_num_model = tensor_output_num_model[0, channel, :, plot_coordinate[0], plot_coordinate[1]]
        profile_tensor_input_NN = tensor_input_NN_model[0, channel, :, plot_coordinate[0], plot_coordinate[1]]
        profile_tensor_NN_model = mean_chl_tensor[0, channel, :, plot_coordinate[0], plot_coordinate[1]]
        #addition of moving_average
        profile_tensor_num_model = moving_average(profile_tensor_num_model.detach().cpu().numpy(), 3)
        profile_tensor_input_NN = moving_average(profile_tensor_input_NN.detach().cpu().numpy(), 3)
        profile_tensor_NN_model = moving_average(profile_tensor_NN_model.detach().cpu().numpy(), 3)
        #plot profiles
        path_fig_channel_coordinates = path_fig_channel + "/lat_" + str(plot_coordinate[1]) + "_lon_" + str(plot_coordinate[0])
        plt.yticks(depth_levels, resolution[2] * np.arange(0, tensor_input_NN.shape[2]), fontsize=6)  
        plt.plot(profile_tensor_input_NN, depth_levels, color="red", label="input CNN profile")   #prima era profile_tensor_input_NN.cpu()
        plt.plot(profile_tensor_NN_model, depth_levels, color="green", label="CNN profile")       #prima era profile_tensor_NN_model.cpu()
        plt.plot(profile_tensor_num_model, depth_levels, color="blue", label="BFM profile")       #prima era profile_tensor_num_model.cpu()
        plt.grid(axis = 'y')
        plt.xlabel(var + " values")
        plt.ylabel("depths values")
        plt.legend(loc="lower right", prop={'size': 6})
        plt.savefig(path_fig_channel_coordinates + ".png")
        plt.close()