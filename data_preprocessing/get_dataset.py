import torch
import os
from hyperparameter import *
import numpy as np


directory_dataset = "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/"


def concatenate_tensors(physic_tensor, biogeochemical_tensor, axis=1):
    """function that concatenates physic and biogeovìchemical tensor to create a singel data tensor"""
    return torch.cat((physic_tensor, biogeochemical_tensor), axis)


def get_list_float_tensor_year_weeks(var, year):
    """created a list containing the my_tensor representing the FLOAT information uploaded for a specific year"""
    list_float_tensor = []
    directory_float = directory_dataset + 'float/'
    directory_year = directory_float + str(year) + "/"
    directory_tensor = directory_year + "final_tensor/"   
    directory_tensor_var = directory_tensor + str(var) + "/"
    list_ptFIles = os.listdir(directory_tensor_var)
    list_weeks = []
    for ptFiles in list_ptFIles:
        pt_week = "".join(c for c in ptFiles if c.isdecimal())    
        list_weeks.append(pt_week)
        my_tensor = torch.load(directory_tensor_var + ptFiles)    
        list_float_tensor.append(my_tensor[:, :, :-1, :, :])
    return list_float_tensor, list_weeks


def get_list_model_tensor_year_weeks(var, year):
    """created a list containing the my_tensor representing the MODEL information uploaded for a specific year"""
    list_model_tensor = []
    directory_model = directory_dataset + 'MODEL/'
    directory_year = directory_model + str(year) + "/"
    directory_tensor = directory_year + "final_tensor/"   
    directory_tensor_var = directory_tensor + str(var) + "/"
    list_ptFIles = os.listdir(directory_tensor_var)
    list_weeks = []
    for ptFiles in list_ptFIles:
        pt_week = "".join(c for c in ptFiles if c.isdecimal())     
        list_weeks.append(pt_week)
        my_tensor = torch.load(directory_tensor_var + ptFiles, map_location=torch.device('cpu'))
        list_model_tensor.append(my_tensor[:, :, :-1, :, :])     
    return list_model_tensor, list_weeks


