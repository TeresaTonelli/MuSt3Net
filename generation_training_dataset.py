#This script contains the functions fro the generatiorn of the dataset that we need to train our netwprk
#USEFUL TO SAVE RAM 


import numpy as np
import torch
import os
import datetime

from get_dataset import concatenate_tensors, get_list_float_tensor_weeks, get_list_model_tensor_weeks, get_list_float_tensor_year_weeks, get_list_model_tensor_year_weeks
from utils_function import *
from utils_mask import *
from normalization import Normalization
from utils_generation_train_1p import write_list, read_list



#def generate_training_dataset_phase_1(biogeoch_var, path_results):
#    """this function generates data that we need to train the CNN for phase 1"""
#    #generate the new train dataset --> merging physical and biogeoch variable
#    physics_train_dataset, list_physics_weeks = get_list_model_tensor_weeks("physics_vars")
#    biogeoch_train_dataset, list_biogeoch_weeks = get_list_model_tensor_weeks(biogeoch_var)
#    #re_order the tensors and the weeks
#    physics_train_dataset, list_physics_weeks = re_order_weeks(list_physics_weeks, physics_train_dataset)
#    biogeoch_train_dataset, list_biogeoch_weeks = re_order_weeks(list_biogeoch_weeks, biogeoch_train_dataset)
#    #Generation of land - sea masks
#    land_sea_masks = []
#    for i_d in range(int((depth_interval[1] - depth_interval[0]) / resolution[2] + 1) - 1):
#        land_sea_mask = generate_sea_land_mask(biogeoch_train_dataset[0], i_d)
#        land_sea_masks.append(land_sea_mask)
#    #copy of biogeoch tensors
#    old_biogeoch_train_dataset = biogeoch_train_dataset.copy()   
#    old_total_dataset = [concatenate_tensors(physics_train_dataset[i], old_biogeoch_train_dataset[i], axis=1) for i in range(len(physics_train_dataset))]
#    #sottocampiono i tensori biogeochimici per arrivare ad una cardinalità simile a quella dei dati float (ovviamente calcolati a livello settimanale)
#    duplicates_biogeoch_train_dataset, duplicates_biogeoch_coordinates = generate_list_sample_tensors(biogeoch_train_dataset, land_sea_masks[0], n_samples_biogeoch, n_duplicates_biogeoch)  
#    duplicates_physics_train_dataset = extend_list(physics_train_dataset, n_duplicates_biogeoch)
#    old_duplicates_total_dataset = [concatenate_tensors(duplicates_physics_train_dataset[i], duplicates_biogeoch_train_dataset[i], axis=1) for i in range(len(duplicates_physics_train_dataset))]
#    old_duplicates_total_dataset = [old_duplicate_tensor[:, :, :-1, :, 1:-1] for old_duplicate_tensor in old_duplicates_total_dataset]
#    transposed_duplicates_biogeoch_coordinates = transpose_latitudes(duplicates_biogeoch_coordinates)
#    list_duplicates_biogeoch_weeks = create_list_duplicates(list_biogeoch_weeks, n_duplicates_biogeoch)
#    #fill missing values of biogeoch tensor with standard values
#    biogeoch_train_dataset_fill = [fill_tensor_opt(biogeoch_tensor, land_sea_masks, standard_mean_values[list_biogeoch_vars[0]]) for biogeoch_tensor in duplicates_biogeoch_train_dataset]
#    print("end opt fill")
#    #merging part
#    total_dataset = [concatenate_tensors(duplicates_physics_train_dataset[i], biogeoch_train_dataset_fill[i], axis=1) for i in range(len(duplicates_physics_train_dataset))]
#    #generate train and test dataset
#    index_testing = random.sample(range(len(total_dataset)), int(len(total_dataset) * 0.2))  #Qua bisogna cambiare il modo in cui fare il test
#    index_internal_testing = random.sample(index_testing, int(len(index_testing) / 2))
#    index_external_testing = [i for i in index_testing if i not in index_internal_testing]
#    index_training = [i for i in range(len(total_dataset)) if i not in index_testing]
#    denormalized_test_dataset = [total_dataset[i] for i in index_testing]
#    denormalized_train_dataset = [total_dataset[i] for i in index_training]
#    #normalization of dataset
#    total_dataset_norm, _, _ = Normalization(total_dataset, "1p", path_results)    
#    #test_dataset = [total_dataset_norm[i] for i in index_testing] 
#    test_dataset = [total_dataset_norm[i] for i in index_external_testing] 
#    internal_test_dataset = [total_dataset_norm[i] for i in index_internal_testing]                       
#    train_dataset = [total_dataset_norm[i] for i in index_training]
#    #return biogeoch_train_dataset, land_sea_masks, old_total_dataset, index_training, index_testing, train_dataset, test_dataset, list_duplicates_biogeoch_weeks, transposed_duplicates_biogeoch_coordinates
#    return biogeoch_train_dataset, land_sea_masks, old_total_dataset, index_training, index_internal_testing, index_external_testing, train_dataset, internal_test_dataset,test_dataset, list_duplicates_biogeoch_weeks, transposed_duplicates_biogeoch_coordinates



#def generate_training_dataset_phase_2(biogeoch_var, land_sea_masks, path_results_2):
#    """this function generates data that we need to train the CNN for phase 2"""
#    physics_train_dataset_2, list_physics_weeks_2 = get_list_model_tensor_weeks("physics_vars")
#    biogeoch_float_train_dataset, float_list_weeks = get_list_float_tensor_weeks(biogeoch_var) 
#    #re_order the tensors and the weeks
#    physics_train_dataset_2, list_physics_weeks_2 = re_order_weeks(list_physics_weeks_2, physics_train_dataset_2)
#    biogeoch_float_train_dataset, float_list_weeks = re_order_weeks(float_list_weeks, biogeoch_float_train_dataset)
#    #pop the last element since it is a void tensor
#    physics_train_dataset_2.pop(-1)  
#    list_physics_weeks_2.pop(-1)  
#    biogeoch_float_train_dataset.pop(-1)
#    float_list_weeks.pop(-1)
#    #concatenation of physical and biogeochemical variables
#    denorm_old_float_train_dataset = [concatenate_tensors(physics_train_dataset_2[i], biogeoch_float_train_dataset[i][:, 0:1, :, :, :], axis=1) for i in range(len(physics_train_dataset_2))]
#    denorm_old_float_train_dataset = [denorm_old_float_data[:, :, :-1, :, 1:-1] for denorm_old_float_data in denorm_old_float_train_dataset]
#    #find the coordinates of profiles of float, for each week 
#    list_float_profiles_coordinates = [compute_profile_coordinates(float_tensor[:, 0:1, :, :, :]) for float_tensor in biogeoch_float_train_dataset]
#    sampled_list_float_profile_coordinates = [random.sample(float_profile_coord, int(0.4 * len(float_profile_coord))) for float_profile_coord in list_float_profiles_coordinates]
#    #remove the float elements that we use as testing ones
#    reduced_biogeoch_float_train_dataset = [remove_float(biogeoch_float_train_dataset[j], sampled_list_float_profile_coordinates[j]) for j in range(len(biogeoch_float_train_dataset))]
#    #fill biogeoch tensor with mean value
#    fill_biogeoch_float_train_dataset = [fill_tensor_opt(reduced_biogeoch_float_train_dataset[i_float][:, 0:1, :, :, :], land_sea_masks, standard_mean_values[list_biogeoch_vars[0]]) for i_float in range(len(reduced_biogeoch_float_train_dataset))]
#    #merging part
#    total_dataset_2 = [concatenate_tensors(physics_train_dataset_2[i], fill_biogeoch_float_train_dataset[i][:, 0:1, :, :, :], axis=1) for i in range(len(physics_train_dataset_2))]
#    #generate test daatset
#    index_testing_2 = random.sample(range(len(total_dataset_2)), int(len(total_dataset_2) * 0.2))
#    index_internal_testing_2 = random.sample(index_testing_2, int(len(index_testing_2) / 2))
#    index_external_testing_2 = [i for i in index_testing_2 if i not in index_internal_testing_2]  
#    index_training_2 = [i for i in range(len(total_dataset_2)) if i not in index_testing_2]
#    #normalization of the dataset
#    total_dataset_2_norm, _, _ = Normalization(total_dataset_2, "2p", path_results_2)    
#    #test_dataset_2 = [total_dataset_2_norm[i] for i in index_testing_2]  
#    test_dataset_2 = [total_dataset_2_norm[i] for i in index_external_testing_2]  
#    internal_test_dataset_2 = [total_dataset_2_norm[i] for i in index_internal_testing_2]                        
#    train_dataset_2 = [total_dataset_2_norm[i] for i in index_training_2]
#    #return float_list_weeks, denorm_old_float_train_dataset, list_float_profiles_coordinates, sampled_list_float_profile_coordinates, index_training_2, index_testing_2, train_dataset_2, test_dataset_2
#    return float_list_weeks, denorm_old_float_train_dataset, list_float_profiles_coordinates, sampled_list_float_profile_coordinates, index_training_2, index_internal_testing_2, index_external_testing_2, train_dataset_2, internal_test_dataset_2, test_dataset_2


#Funzioni per data generation con più anni (non soltanto 2019):
#def generate_multiple_dataset_phase_1(biogeoch_var, path_results, years):
#    """this function generates data that we need to train the CNN for phase 1"""
#    #generate the new train dataset --> merging physical and biogeoch variable
#    years = sorted(years)
#    list_total_physics_dataset = []
#    list_total_biogeoch_dataset = []
#    list_total_weeks = []
#    for year in years:
#        physics_train_dataset, list_physics_weeks = get_list_model_tensor_year_weeks("physics_vars", year)
#        biogeoch_train_dataset, list_biogeoch_weeks = get_list_model_tensor_year_weeks(biogeoch_var, year)
#        #re_order the tensors and the weeks
#        physics_train_dataset, list_physics_weeks = re_order_weeks(list_physics_weeks, physics_train_dataset)
#        biogeoch_train_dataset, list_biogeoch_weeks = re_order_weeks(list_biogeoch_weeks, biogeoch_train_dataset)
#        #save these tensors in the total lists and update the weeks
#        list_total_physics_dataset.extend(physics_train_dataset)
#        list_total_biogeoch_dataset.extend(biogeoch_train_dataset)
#        if len(list_total_weeks) > 0:
#            k = int(list_total_weeks[-1])
#        else:
#            k = 0
#        list_physics_weeks = list_biogeoch_weeks = [str(int(list_physics_weeks[j]) + k) for j in range(len(list_physics_weeks))]
#        list_total_weeks.extend(list_biogeoch_weeks)
#    #Rename the list of tensors and weeks in order not to change the next code
#    physics_train_dataset = list_total_physics_dataset
#    biogeoch_train_dataset = list_total_biogeoch_dataset
#    list_physics_weeks = list_biogeoch_weeks = list_total_weeks
#    del list_total_physics_dataset
#    del list_total_biogeoch_dataset
#    del list_total_weeks
#    #Generation of land - sea masks
#    land_sea_masks = []
#    for i_d in range(int((depth_interval[1] - depth_interval[0]) / resolution[2] + 1) - 1):
#        land_sea_mask = generate_sea_land_mask(biogeoch_train_dataset[0], i_d)
#        land_sea_masks.append(land_sea_mask)
#    #copy of biogeoch tensors
#    old_biogeoch_train_dataset = biogeoch_train_dataset.copy()   
#    old_total_dataset = [concatenate_tensors(physics_train_dataset[i], old_biogeoch_train_dataset[i], axis=1) for i in range(len(physics_train_dataset))]
#    #sottocampiono i tensori biogeochimici per arrivare ad una cardinalità simile a quella dei dati float (ovviamente calcolati a livello settimanale)
#    duplicates_biogeoch_train_dataset, duplicates_biogeoch_coordinates = generate_list_sample_tensors(biogeoch_train_dataset, land_sea_masks[0], n_samples_biogeoch, n_duplicates_biogeoch)  
#    duplicates_physics_train_dataset = extend_list(physics_train_dataset, n_duplicates_biogeoch)
#    old_duplicates_total_dataset = [concatenate_tensors(duplicates_physics_train_dataset[i], duplicates_biogeoch_train_dataset[i], axis=1) for i in range(len(duplicates_physics_train_dataset))]
#    old_duplicates_total_dataset = [old_duplicate_tensor[:, :, :-1, :, 1:-1] for old_duplicate_tensor in old_duplicates_total_dataset]
#    transposed_duplicates_biogeoch_coordinates = transpose_latitudes(duplicates_biogeoch_coordinates)
#    list_duplicates_biogeoch_weeks = create_list_duplicates(list_biogeoch_weeks, n_duplicates_biogeoch)
#    #remove from cpus all the tensors that I don't use anymore
#    del old_duplicates_total_dataset
#    #fill missing values of biogeoch tensor with standard values
#    biogeoch_train_dataset_fill = [fill_tensor_opt(biogeoch_tensor, land_sea_masks, standard_mean_values[list_biogeoch_vars[0]]) for biogeoch_tensor in duplicates_biogeoch_train_dataset]
#    print("end opt fill")
#    #merging part
#    total_dataset = [concatenate_tensors(duplicates_physics_train_dataset[i], biogeoch_train_dataset_fill[i], axis=1) for i in range(len(duplicates_physics_train_dataset))]
#    #remove all teh tensors that I don't use anymore
#    del duplicates_biogeoch_train_dataset
#    del duplicates_biogeoch_coordinates
#    del duplicates_physics_train_dataset
#    del biogeoch_train_dataset_fill
#    #generate train and test dataset
#    index_testing = random.sample(range(len(total_dataset)), int(len(total_dataset) * 0.2))  #Qua bisogna cambiare il modo in cui fare il test
#    index_internal_testing = random.sample(index_testing, int(len(index_testing) / 2))
#    index_external_testing = [i for i in index_testing if i not in index_internal_testing]
#    index_training = [i for i in range(len(total_dataset)) if i not in index_testing]
#    #denormalized_test_dataset = [total_dataset[i] for i in index_testing]
#    #denormalized_train_dataset = [total_dataset[i] for i in index_training]
#    #normalization of dataset
#    total_dataset_norm, _, _ = Normalization(total_dataset, "1p", path_results)   
#    #remove the tensors that I don't use anymore
#    del total_dataset 
#    #test_dataset = [total_dataset_norm[i] for i in index_testing] 
#    test_dataset = [total_dataset_norm[i] for i in index_external_testing] 
#    internal_test_dataset = [total_dataset_norm[i] for i in index_internal_testing]                       
#    train_dataset = [total_dataset_norm[i] for i in index_training]
#    #return biogeoch_train_dataset, land_sea_masks, old_total_dataset, index_training, index_testing, train_dataset, test_dataset, list_duplicates_biogeoch_weeks, transposed_duplicates_biogeoch_coordinates
#    return biogeoch_train_dataset, land_sea_masks, old_total_dataset, index_training, index_internal_testing, index_external_testing, train_dataset, internal_test_dataset,test_dataset, list_duplicates_biogeoch_weeks, transposed_duplicates_biogeoch_coordinates



#def generate_multiple_dataset_phase_2(biogeoch_var, land_sea_masks, path_results_2, years):
#    """this function generates data that we need to train the CNN for phase 2"""
#    years = sorted(years)
#    list_total_physics_dataset = []
#    list_total_biogeoch_dataset = []
#    list_total_weeks = []
#    for year in years:
#        physics_train_dataset_2, list_physics_weeks_2 = get_list_model_tensor_year_weeks("physics_vars", year)
#        biogeoch_float_train_dataset, float_list_weeks = get_list_float_tensor_year_weeks(biogeoch_var, year) 
#        #re_order the tensors and the weeks
#        physics_train_dataset_2, list_physics_weeks_2 = re_order_weeks(list_physics_weeks_2, physics_train_dataset_2)
#        biogeoch_float_train_dataset, float_list_weeks = re_order_weeks(float_list_weeks, biogeoch_float_train_dataset)
#        #pop the last element since it is a void tensor
#        #physics_train_dataset_2.pop(-1)  
#        #list_physics_weeks_2.pop(-1)  
#        biogeoch_float_train_dataset.pop(-1)
#        float_list_weeks.pop(-1)
#        #save these tensors in the total lists and update the weeks
#        list_total_physics_dataset.extend(physics_train_dataset_2)
#        list_total_biogeoch_dataset.extend(biogeoch_float_train_dataset)
#        if len(list_total_weeks) > 0:
#            k = int(list_total_weeks[-1])
#        else:
#           k = 0
#        list_physics_weeks = float_list_weeks = [str(int(list_physics_weeks[j]) + k) for j in range(len(list_physics_weeks))]
#        list_total_weeks.extend(float_list_weeks)
#    #Rename the list of tensors and weeks in order not to change the next code
#    physics_train_dataset_2 = list_total_physics_dataset
#    biogeoch_float_train_dataset = list_total_biogeoch_dataset
#    list_physics_weeks_2 = float_list_weeks = list_total_weeks
#    #concatenation of physical and biogeochemical variables
#    denorm_old_float_train_dataset = [concatenate_tensors(physics_train_dataset_2[i], biogeoch_float_train_dataset[i][:, 0:1, :, :, :], axis=1) for i in range(len(physics_train_dataset_2))]
#    denorm_old_float_train_dataset = [denorm_old_float_data[:, :, :-1, :, 1:-1] for denorm_old_float_data in denorm_old_float_train_dataset]
#    #find the coordinates of profiles of float, for each week 
#    list_float_profiles_coordinates = [compute_profile_coordinates(float_tensor[:, 0:1, :, :, :]) for float_tensor in biogeoch_float_train_dataset]
#    sampled_list_float_profile_coordinates = [random.sample(float_profile_coord, int(0.4 * len(float_profile_coord))) for float_profile_coord in list_float_profiles_coordinates]
#    #remove the float elements that we use as testing ones
#    reduced_biogeoch_float_train_dataset = [remove_float(biogeoch_float_train_dataset[j], sampled_list_float_profile_coordinates[j]) for j in range(len(biogeoch_float_train_dataset))]
#    #fill biogeoch tensor with mean value
#    fill_biogeoch_float_train_dataset = [fill_tensor_opt(reduced_biogeoch_float_train_dataset[i_float][:, 0:1, :, :, :], land_sea_masks, standard_mean_values[list_biogeoch_vars[0]]) for i_float in range(len(reduced_biogeoch_float_train_dataset))]
#    #merging part
#    total_dataset_2 = [concatenate_tensors(physics_train_dataset_2[i], fill_biogeoch_float_train_dataset[i][:, 0:1, :, :, :], axis=1) for i in range(len(physics_train_dataset_2))]
#    #generate test daatset
#    index_testing_2 = random.sample(range(len(total_dataset_2)), int(len(total_dataset_2) * 0.2))
#    index_internal_testing_2 = random.sample(index_testing_2, int(len(index_testing_2) / 2))
#    index_external_testing_2 = [i for i in index_testing_2 if i not in index_internal_testing_2]  
#    index_training_2 = [i for i in range(len(total_dataset_2)) if i not in index_testing_2]
#    #normalization of the dataset
#    total_dataset_2_norm, _, _ = Normalization(total_dataset_2, "2p", path_results_2)    
#    #test_dataset_2 = [total_dataset_2_norm[i] for i in index_testing_2]  
#    test_dataset_2 = [total_dataset_2_norm[i] for i in index_external_testing_2]  
#    internal_test_dataset_2 = [total_dataset_2_norm[i] for i in index_internal_testing_2]                        
#    train_dataset_2 = [total_dataset_2_norm[i] for i in index_training_2]
#    #return float_list_weeks, denorm_old_float_train_dataset, list_float_profiles_coordinates, sampled_list_float_profile_coordinates, index_training_2, index_testing_2, train_dataset_2, test_dataset_2
#    return float_list_weeks, denorm_old_float_train_dataset, list_float_profiles_coordinates, sampled_list_float_profile_coordinates, index_training_2, index_internal_testing_2, index_external_testing_2, train_dataset_2, internal_test_dataset_2, test_dataset_2






def generate_dataset_phase_1_saving(biogeoch_var, path_results, year, path_saving):
    """this function generates data that we need to train the CNN for phase 1"""
    #generate the new train dataset --> merging physical and biogeoch variable
    physics_train_dataset, list_physics_weeks = get_list_model_tensor_year_weeks("physics_vars", year)
    biogeoch_train_dataset, list_biogeoch_weeks = get_list_model_tensor_year_weeks(biogeoch_var, year)
    #re_order the tensors and the weeks
    physics_train_dataset, list_physics_weeks = re_order_weeks(list_physics_weeks, physics_train_dataset)
    biogeoch_train_dataset, list_biogeoch_weeks = re_order_weeks(list_biogeoch_weeks, biogeoch_train_dataset)
    #Generation of land - sea masks
    land_sea_masks = []
    for i_d in range(int((depth_interval[1] - depth_interval[0]) / resolution[2] + 1) - 1):
        land_sea_mask = generate_sea_land_mask(biogeoch_train_dataset[0], i_d)
        land_sea_masks.append(land_sea_mask)
    #subsamle the biogeocehmical tensors to reach a profiles cardinality compatible to the BGC-Argo float ones
    duplicates_biogeoch_train_dataset, duplicates_biogeoch_coordinates = generate_list_sample_tensors(biogeoch_train_dataset, land_sea_masks[0], n_samples_biogeoch, n_duplicates_biogeoch)  
    duplicates_physics_train_dataset = extend_list(physics_train_dataset, n_duplicates_biogeoch)
    old_duplicates_total_dataset = [concatenate_tensors(duplicates_physics_train_dataset[i], duplicates_biogeoch_train_dataset[i], axis=1) for i in range(len(duplicates_physics_train_dataset))]
    old_duplicates_total_dataset = [old_duplicate_tensor[:, :, :-1, :, 1:-1] for old_duplicate_tensor in old_duplicates_total_dataset]
    transposed_duplicates_biogeoch_coordinates = transpose_latitudes(duplicates_biogeoch_coordinates)
    list_duplicates_biogeoch_weeks = create_list_duplicates(list_biogeoch_weeks, n_duplicates_biogeoch)
    #remove from cpus all the tensors that I don't use anymore
    del physics_train_dataset
    del biogeoch_train_dataset
    del old_duplicates_total_dataset
    #fill missing values of biogeoch tensor with standard values
    biogeoch_train_dataset_fill = [fill_tensor_opt(biogeoch_tensor, land_sea_masks, standard_mean_values[list_biogeoch_vars[0]]) for biogeoch_tensor in duplicates_biogeoch_train_dataset]
    del land_sea_masks
    #merging part
    total_dataset = [concatenate_tensors(duplicates_physics_train_dataset[i], biogeoch_train_dataset_fill[i], axis=1) for i in range(len(duplicates_physics_train_dataset))]
    #remove all teh tensors that I don't use anymore
    del duplicates_biogeoch_train_dataset
    del duplicates_biogeoch_coordinates
    del duplicates_physics_train_dataset
    del biogeoch_train_dataset_fill
    #save total_dataset
    n_dupl_base = len(os.listdir(path_saving + "/total_dataset/" + str(biogeoch_var) + "/" + str(year) + "/" + "week_" + str(1) + "/"))
    for i in range(len(total_dataset)):
        i_week = list_duplicates_biogeoch_weeks[i]
        n_dupl = i % n_duplicates_biogeoch + n_dupl_base
        tot_data_path_sav = path_saving + "/total_dataset/" + str(biogeoch_var) + "/" + str(year) + "/" + "week_" + str(i_week) + "/" + "duplicate_" + str(n_dupl)+ "/"
        if not os.path.exists(tot_data_path_sav):
            os.makedirs(tot_data_path_sav)
        torch.save(total_dataset[i], tot_data_path_sav + "tensor.pt")
        write_list(transposed_duplicates_biogeoch_coordinates[i], tot_data_path_sav + "file_transpose_latitudes.txt")
    return None



def generate_dataset_phase_2_saving(biogeoch_var, path_results_2, years, path_saving_2, land_sea_masks):
    """this function generates data that we need to train the CNN for phase 2"""
    list_physics_tensors = []
    list_biog_float_tensors = []
    list_year_week_indexes = []
    for year in years:
        physics_train_dataset_2, list_physics_weeks_2 = get_list_model_tensor_year_weeks("physics_vars", year)
        biogeoch_float_train_dataset, float_list_weeks = get_list_float_tensor_year_weeks(biogeoch_var, year)
        #re-order tensors and weeks
        physics_train_dataset_2, list_physics_weeks_2 = re_order_weeks(list_physics_weeks_2, physics_train_dataset_2)
        biogeoch_float_train_dataset, float_list_weeks = re_order_weeks(float_list_weeks, biogeoch_float_train_dataset)
        if year != 2020:  
            biogeoch_float_train_dataset.pop(-1)
            float_list_weeks.pop(-1)
        ##extend physycal and biogeochemical tensors list with the all tensors of an year
        list_physics_tensors.extend(physics_train_dataset_2)
        list_biog_float_tensors.extend(biogeoch_float_train_dataset)
        #update and extend the list of week indexes, adding the corresponding year
        float_list_year_week = [[year] + [int(float_list_weeks[i])] for i in range(len(float_list_weeks))]
        list_year_week_indexes.extend(float_list_year_week)
    #concatenation of physical and biogeochemical variables
    old_float_total_dataset = [concatenate_tensors(list_physics_tensors[i], list_biog_float_tensors[i][:, 0:1, :, :, :], axis=1) for i in range(len(list_physics_tensors))]
    old_float_total_dataset = [denorm_old_float_data[:, :, :-1, :, 1:-1] for denorm_old_float_data in old_float_total_dataset]
    #find the coordinates of profiles of float, for each week 
    list_float_profiles_coordinates = [compute_profile_coordinates(float_tensor[:, 0:1, :, :, :]) for float_tensor in list_biog_float_tensors]
    sampled_list_float_profile_coordinates = [random.sample(float_profile_coord, int(0.4 * len(float_profile_coord))) for float_profile_coord in list_float_profiles_coordinates]
    #remove the float elements that we use as testing ones
    reduced_biogeoch_float_total_dataset = [remove_float(list_biog_float_tensors[j], sampled_list_float_profile_coordinates[j]) for j in range(len(list_biog_float_tensors))]
    #fill biogeoch tensor with mean value --> modified with / 2 to transform it to 0.075
    fill_biogeoch_float_total_dataset = [fill_tensor_opt(reduced_biogeoch_float_total_dataset[i_float][:, 0:1, :, :, :], land_sea_masks, standard_mean_values[list_biogeoch_vars[0]]/2) for i_float in range(len(reduced_biogeoch_float_total_dataset))]
    del reduced_biogeoch_float_total_dataset
    #merging part
    total_dataset_2 = [concatenate_tensors(list_physics_tensors[i], fill_biogeoch_float_total_dataset[i][:, 0:1, :, :, :], axis=1) for i in range(len(list_physics_tensors))]
    del fill_biogeoch_float_total_dataset
    #generate test dataset
    index_testing_2 = random.sample(range(len(total_dataset_2)), int(len(total_dataset_2) * 0.2))
    index_internal_testing_2 = random.sample(index_testing_2, int(len(index_testing_2) / 2))
    index_external_testing_2 = [i for i in index_testing_2 if i not in index_internal_testing_2]  
    index_training_2 = [i for i in range(len(total_dataset_2)) if i not in index_testing_2]
    #normalization of the dataset
    total_dataset_2_norm, _, _ = Normalization(total_dataset_2, "2p", path_results_2) 
    del total_dataset_2   
    #test_dataset_2 = [total_dataset_2_norm[i] for i in index_testing_2]  
    test_dataset_2 = [total_dataset_2_norm[i] for i in index_external_testing_2]  
    internal_test_dataset_2 = [total_dataset_2_norm[i] for i in index_internal_testing_2]                        
    train_dataset_2 = [total_dataset_2_norm[i] for i in index_training_2]
    return list_year_week_indexes, old_float_total_dataset, list_float_profiles_coordinates, sampled_list_float_profile_coordinates, index_training_2, index_internal_testing_2, index_external_testing_2, train_dataset_2, internal_test_dataset_2, test_dataset_2