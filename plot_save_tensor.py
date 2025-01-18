"""
Plotting the information contained in the tensors
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from hyperparameter import *
#from hyperparameter2 import *

#creo il device per le GPU e porto i dati su GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device", device)

path = "fig/"
path_directory_save = "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution" + "/dataset" #"/u/ttonelli/Dottorato/ocean_inpainting_reconstruction/CNN_reconstruction_final_resolution" + "/dataset"     #+ "/MODEL"
path_directory_plot = "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution" + "/plots"   #"/u/ttonelli/Dottorato/ocean_inpainting_reconstruction/CNN_reconstruction_final_resolution" + "/plots"
if not os.path.exists(path_directory_save):
    os.makedirs(path_directory_save)
if not os.path.exists(path_directory_plot):
    os.makedirs(path_directory_plot)


def Plot_Tensor(kindof, tensor, data_time, channel, flag, dict_channel):
    """
    Plotting the tensor's values at different levels of depth
    i.e. plotting along the component d (=depth) of tensor (bs, c,c d, h, w)
    tensor = tensor we want to plot (i.e. parallelepiped at a fixed date time)
    data_time = reference date time associated to the list of tensor
    channel = variable we want to plot
    """
    #dict_channel = {0: 'temperature', 1: 'salinity', 2: 'oxygen', 3: 'chla', 4: 'npp', 5: 'n1p', 6: biogeoch_var}   #questo è da cambiare in modo intelligente 
    if flag == 'w':
        directory = path_directory_plot + '/weight_tensor/' + str(kindof) + '/' + str(channel) + '/' + str(data_time)
    if flag == 't':
        directory = path_directory_plot + '/tensor/' + str(kindof) + '/' + str(channel) + '/' + str(data_time)

    if not os.path.exists(directory):
        os.makedirs(directory)  

    number_depths = len(tensor[0, 0, :, 0, 0])  # number of levels of depth

    for i in range(number_depths):
        cmap = plt.get_cmap('Greens')
        plt.imshow(tensor[0, channel, i, :, :], cmap=cmap)
        if flag == 'w':
            plt.title(dict_channel[channel] + 'weight')
        if flag == 't':
            plt.title(dict_channel[channel])
        plt.colorbar()
        plt.savefig(directory + "/profondity_level_" + str(i) + ".png")
        plt.close()


def plot_routine(kindof, list_parallelepiped, list_data_time, channels, year_interval, flag, dict_channel):
    """
    measurement plot different for each kind of data (float/sat/tensor)
    kindof = requires a str (float, sat or tensor)
    list_parallelepiped = list of tensor we want to plot (i.e. parallelepiped at a fixed date time)
    list_data_time = list of reference date time associated to the list of tensor
    channels = list of variable we want to plot
    flag = we are plotting the tensor or the weight, if no specify the tensor
    """
    year_min, year_max = year_interval
    for j in range(len(list_data_time)):
        print('plotting new data time')
        time_considered = list_data_time[j]
        tensor_considered = list_parallelepiped[j]
        if year_min < time_considered < year_max:
            print('plotting tensor relative to time : ', time_considered)
            for channel in channels:
                Plot_Tensor(kindof, tensor_considered, time_considered, channel, flag, dict_channel)


def Save_Tensor(kindof, tensor, data_time, flag, year, var, path_directory_save):
    """
    Saving the tensor's values at different levels of depth
    i.e. plotting along the component d (=depth) of tensor (bs, c,c d, h, w)
    tensor = tensor we want to plot (i.e. parallelepiped at a fixed date time)
    data_time = reference date time associated to the list of tensor
    channel = variable we want to plot
    """
    if kindof == "model":
        path_directory_save = path_directory_save + "/MODEL"
    if kindof == "float":
        path_directory_save = path_directory_save + "/float"
        
    path_directory_save = path_directory_save + "/" + str(year)
    if flag == 'w':
        directory = path_directory_save + '/weight_tensor'
        if not os.path.exists(directory):
            os.mkdir(directory)
    else:
        directory = path_directory_save + '/tensor' + '/' + str(var)
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    #directory = directory + "/" + str(var)

    torch.save(tensor, directory + "/datetime_" + str(data_time) + ".pt")


def save_routine(kindof, list_parallelepiped, list_data_time, year_interval, flag, var, path_directory_save):
    """
    measurement plot different for each kind of data (float/sat/tensor)
    kindof = requires a str (float, sat or tensor)
    list_parallelepiped = list of tensor we want to plot (i.e. parallelepiped at a fixed date time)
    list_data_time = list of reference date time associated to the list of tensor
    channels = list of variable we want to plot
    flag = we are plotting the tensor or the weight, if no specify the tensor
    """
    year_min, year_max = year_interval
    for j in range(len(list_data_time)):
        print('saving new data time')
        time_considered = list_data_time[j]
        tensor_considered = list_parallelepiped[j]
        if year_min < time_considered < year_max:
            print('saving tensor relative to time : ', time_considered)
            Save_Tensor(kindof, tensor_considered, time_considered, flag, year_min, var, path_directory_save)



def save_tensors_directory(kindof, tensors_directory, year_interval, flag, var, path_directory_save):
    year_min, year_max = year_interval
    list_tensors = os.listdir(tensors_directory)
    for j in range(len(list_tensors)):
        print('saving new tensor')
        time_considered = "".join(c for c in list_tensors[j] if c.isdecimal())
        print("time considered", time_considered)
        print("dir tensors", tensors_directory)
        print("single tensor", list_tensors[j])
        tensor_considered = np.load(tensors_directory + list_tensors[j])
        tensor_considered = torch.from_numpy(tensor_considered)
        print("type tensor",type(tensor_considered))
        print('saving tensor relative to week : ', time_considered)
        Save_Tensor(kindof, tensor_considered, time_considered, flag, year_min, var, path_directory_save)




#save_tensors_directory("model", "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/MODEL/2022/saved_parallelepiped_physics_single/vozocrtx/", (2022, 2023), "t", "vozocrtx", "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/")
