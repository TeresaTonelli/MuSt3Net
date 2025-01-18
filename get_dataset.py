import torch
import os
from hyperparameter import *
import numpy as np

directory_dataset = "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/"


def concatenate_tensors(physic_tensor, biogeochemical_tensor, axis=1):
    """function that concatenates physic and biogeovìchemical tensor to create a singel data tensor"""
    return torch.cat((physic_tensor, biogeochemical_tensor), axis)


def get_list_float_tensor(var):
    """
    created a list containing the my_tensor representing the FLOAT information uploaded
    """
    list_float_tensor = []
    directory_float = directory_dataset + 'float/'
    list_years = os.listdir(directory_float)
    for year in list_years: 
        print("year inside get dataset", year)
        directory_tensor = directory_float + year + "/" + "final_tensor/"   
        directory_tensor_var = directory_tensor + str(var) + "/"
        list_ptFIles = os.listdir(directory_tensor_var)
        for ptFiles in list_ptFIles:
            my_tensor = torch.load(directory_tensor_var + ptFiles)  
            my_tensor = torch.from_numpy(my_tensor)
            list_float_tensor.append(my_tensor[:, :, :-1, :, :])
    return list_float_tensor


def get_list_float_tensor_weeks(var):
    """
    created a list containing the my_tensor representing the FLOAT information uploaded
    """
    list_float_tensor = []
    directory_float = directory_dataset + 'float/'
    list_years = os.listdir(directory_float)
    for year in ["2019"]:   #list_years: #ora perchè voglio cosiderare un solo anno  
        print("year inside get dataset", year)
        directory_tensor = directory_float + year + "/" + "final_tensor/"   
        directory_tensor_var = directory_tensor + str(var) + "/"
        list_ptFIles = os.listdir(directory_tensor_var)
        list_weeks = []
        for ptFiles in list_ptFIles:
            pt_week = "".join(c for c in ptFiles if c.isdecimal())     #str(ptFiles)[14:16]     #DA CONTROLLARE SE LORO SONO ANCORA GLI INDICI DELLA WEEK
            #print("pt week", pt_week)
            list_weeks.append(pt_week)
            my_tensor = torch.load(directory_tensor_var + ptFiles)    
            #my_tensor = torch.from_numpy(my_tensor)
            list_float_tensor.append(my_tensor[:, :, :-1, :, :])
    return list_float_tensor, list_weeks




def get_list_model_tensor(var):
    """
    created a list containing the my_tensor representing the MODEL information uploaded
    """
    list_model_tensor = []
    directory_model = directory_dataset + 'MODEL/'
    list_years = os.listdir(directory_model)
    for year in list_years:
        print("year inside get dataset", year)
        directory_tensor = directory_model + year + "/" + "final_tensor/"  
        directory_tensor_var = directory_tensor + str(var) + "/"
        list_ptFIles = os.listdir(directory_tensor_var)
        for ptFiles in list_ptFIles:
            my_tensor = torch.load(directory_tensor_var + ptFiles)
            my_tensor = torch.from_numpy(my_tensor)
            list_model_tensor.append(my_tensor[:, :, :-1, :, :])     #non so perchè fa sta roba Gloria
    return list_model_tensor 



def get_list_model_tensor_weeks(var):
    """
    created a list containing the my_tensor representing the MODEL information uploaded
    """
    list_model_tensor = []
    directory_model = directory_dataset + 'MODEL/'
    list_years = os.listdir(directory_model)
    print("list years", list_years)
    for year in ["2019"]:    #list_years:
        print("year inside get dataset", year)
        directory_tensor = directory_model + year + "/" + "final_tensor/"   
        directory_tensor_var = directory_tensor + str(var) + "/"
        list_ptFIles = os.listdir(directory_tensor_var)
        list_weeks = []
        for ptFiles in list_ptFIles:
            pt_week = "".join(c for c in ptFiles if c.isdecimal())      #pt_week = str(ptFiles)[14:16]   #CONTROLLARE CHE L INDICE PER LE WEEK SIA RIMASTO LO STESSO
            #print("pt week", pt_week)
            list_weeks.append(pt_week)
            my_tensor = torch.load(directory_tensor_var + ptFiles, map_location=torch.device('cpu'))
            ##my_tensor = torch.from_numpy(my_tensor)        ##cambaito dopo aver cambiato i dati da numpy a torch
            list_model_tensor.append(my_tensor[:, :, :-1, :, :])     
    return list_model_tensor, list_weeks


def get_list_float_tensor_year_weeks(var, year):
    """
    created a list containing the my_tensor representing the FLOAT information uploaded for a specific year
    """
    list_float_tensor = []
    directory_float = directory_dataset + 'float/'
    directory_year = directory_float + str(year) + "/"
    directory_tensor = directory_year + "final_tensor/"   
    directory_tensor_var = directory_tensor + str(var) + "/"
    list_ptFIles = os.listdir(directory_tensor_var)
    list_weeks = []
    for ptFiles in list_ptFIles:
        pt_week = "".join(c for c in ptFiles if c.isdecimal())     #str(ptFiles)[14:16]     #DA CONTROLLARE SE LORO SONO ANCORA GLI INDICI DELLA WEEK
        #print("pt week", pt_week)
        list_weeks.append(pt_week)
        my_tensor = torch.load(directory_tensor_var + ptFiles)    
        #my_tensor = torch.from_numpy(my_tensor)
        list_float_tensor.append(my_tensor[:, :, :-1, :, :])
    return list_float_tensor, list_weeks


def get_list_model_tensor_year_weeks(var, year):
    """
    created a list containing the my_tensor representing the MODEL information uploaded for a specific year
    """
    list_model_tensor = []
    directory_model = directory_dataset + 'MODEL/'
    directory_year = directory_model + str(year) + "/"
    directory_tensor = directory_year + "final_tensor/"   
    directory_tensor_var = directory_tensor + str(var) + "/"
    list_ptFIles = os.listdir(directory_tensor_var)
    list_weeks = []
    for ptFiles in list_ptFIles:
        pt_week = "".join(c for c in ptFiles if c.isdecimal())      #pt_week = str(ptFiles)[14:16]   #CONTROLLARE CHE L INDICE PER LE WEEK SIA RIMASTO LO STESSO
        #print("pt week", pt_week)
        list_weeks.append(pt_week)
        my_tensor = torch.load(directory_tensor_var + ptFiles, map_location=torch.device('cpu'))
        ##my_tensor = torch.from_numpy(my_tensor)        ##cambaito dopo aver cambiato i dati da numpy a torch
        list_model_tensor.append(my_tensor[:, :, :-1, :, :])     
    return list_model_tensor, list_weeks


