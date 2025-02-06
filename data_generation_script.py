#This script geenrates tensors data for training 1 and 2


import numpy as np
import torch.nn as nn
from torch.optim import Adadelta
import matplotlib.pyplot as plt
import random
import datetime

from convolutional_network import CompletionN
from losses import convolutional_network_weighted_loss, convolutional_network_float_weighted_loss, convolutional_network_exp_weighted_loss, convolutional_network_float_exp_weighted_loss
from mean_pixel_value import MV_pixel
from utils_mask import generate_input_mask, generate_sea_land_mask
from normalization import Normalization
from denormalization import Denormalization
from get_dataset import *
from plot_error import Plot_Error
from plot_results import *
from utils_function import *
from utils_mask import generate_float_mask, compute_weights, compute_exponential_weights
from generation_training_dataset import generate_dataset_phase_1_saving
from utils_training_1 import generate_random_week_indexes, generate_random_week_indexes_winter_weighted

num_channel = number_channel  
name_datetime_folder = str(datetime.datetime.utcnow())

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

#creo il device per le GPU e porto i dati su GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device", device)


biogeoch_var_to_predict = list_biogeoch_vars[0]


#data generation for the 1 phase
#generate_dataset_phase_1_saving(biogeoch_var_to_predict, path_results, 2019, "dataset_training")
#generate_dataset_phase_1_saving(biogeoch_var_to_predict, path_results, 2020, "dataset_training")
#generate_dataset_phase_1_saving(biogeoch_var_to_predict, path_results, 2021, "dataset_training")
#generate_dataset_phase_1_saving(biogeoch_var_to_predict, path_results, 2022, "dataset_training")


compute_weights(d, depth_interval[1], 100)
compute_weights(d, depth_interval[1], 200)
compute_exponential_weights(d, depth_interval[1], 100)
compute_exponential_weights(d, depth_interval[1], 200)

#data generation for the 2 phase