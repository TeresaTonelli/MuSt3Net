"""
Routine for the creation of the parallelepiped that composed the training set
"""
import torch
import netCDF4 as nc
import numpy as np
import pandas as pd
import os
from plot_save_tensor import plot_routine, save_routine, path_directory_save, path_directory_plot
from hyperparameter import *

import datetime 


constant_latitude = 111 
constant_longitude = 111 
float_path = "/leonardo_scratch/large/userexternal/ttonelli/OGS/SUPERFLOAT/SUPERFLOAT/"


def generate_list_float_variables(my_list_data):
  float_var_lists = []
  for i in range(np.size(my_list_data)):
    path_current_float = float_path + my_list_data[i]
    ds = nc.Dataset(path_current_float)
    var_list = []
    for var in ds.variables:
      var_list.append(var)
    float_var_lists.append(var_list)
  return float_var_lists


def read_date_time_sat(date_time):
  """
  Take as input a date-time in str format and decode it in a format considering only year and month
  year + 0.01 * week
  """
  year = int(date_time[0:4])
  month = int(date_time[4:6])
  day = int(date_time[6:8])
  week = datetime.date(year, month, day).isocalendar()[1]
  date_time_decoded = year + 0.01 * week
  return date_time_decoded



def to_depth(press, latitude):
  """
  convert press input in depth one
  press = pressure in decibars
  lat = latitude in deg
  depth = depth in metres
  """
  x = np.sin(latitude / 57.29578)
  x = x * x
  gr = 9.780318 * (1.0 + (5.2788e-3 + 2.36e-5 * x) * x) + 1.092e-6 * press
  depth = (((-1.82e-15 * press + 2.279e-10) * press - 2.2512e-5) * press + 9.72659) * press / gr
  return depth


def create_list_date_time(years_consider):
  """
  Creation of a list containing date_time reference for training dataset
  years_consider = (first year considered, last year considered)
  interval_of_time = intervals within measurement are aggregated
  """
  first_year_considered, last_year_considered = years_consider
  total_list = []
  for year in np.arange(first_year_considered, last_year_considered):
    lists = np.arange(year, year + 0.54, 0.01)   #prima era 0.53, poi portato a 0.54
    for i in range(len(lists)):
      lists[i] = round(lists[i], 2)
    lists = lists.tolist()
    total_list = total_list + lists
  return total_list


def create_box(batch, number_channel, lat, lon, depth, resolution):
  """
  Function that creates the EMPTY tensor that will be filled with data
  batch = batch size/ batch number ?
  number_channel = number of channel (i.e. unknowns we want to predict)
  lat = (lat_min, lat_max)
  lon = (lon_min, lon_max)
  depth = (depth_min, depth_max) in km
  resolution = (w_res, h_res, d_res) dimension of a voxel (in km)
  output = tensor zeros (MB, C, D, H, W)
  """
  lat_min, lat_max = lat
  lon_min, lon_max = lon
  depth_min, depth_max = depth
  w_res, h_res, d_res = resolution
  w = int((lat_max - lat_min) * constant_latitude / w_res + 1)
  h = int((lon_max - lon_min) * constant_longitude / h_res + 1)
  d = int((depth_max - depth_min) / d_res + 1)
  empty_parallelepiped = torch.zeros(batch, number_channel, d, h, w)
  return empty_parallelepiped.numpy()   


def find_index(var, var_limits, var_size):
  """
  Function that given a latitude/longitude/depth as input return the index where to place it in the tensor
  lat = latitude considered
  lat_limits = (lat_min, lat_max)
  lat_size = dimension of latitude dmensin in the tensor
  """
  var_min, var_max = var_limits
  var_res = (var_max - var_min) / var_size
  var_index = int((var - var_min) / var_res)
  return var_index


def dictionary_float_vars_names(list_biogeoch_vars):
  dictionary_names = {}
  for var in list_biogeoch_vars:
    dictionary_names[var] = var
  dictionary_names["N3n"] = "NITRATE"
  dictionary_names["O2o"] = "DOXY"
  dictionary_names["P_l"] = "CHLA" 
  return dictionary_names



def compute_index_mean(list_data_float, list_datetime):
  """this function computes the list of indexes that refers to the same float device and the same week"""
  list_devices = [data.split("/", 1)[0] for data in list_data_float]
  list_weeks = [read_date_time_sat(datet) for datet in list_datetime] 
  list_couples = list(zip(list_devices, list_weeks))
  unique_couples = []
  seen = set()
  for item in list_couples:
    if item not in seen:
        unique_couples.append(item)
        seen.add(item)
  indexes_mean = [list() for i in range(len(unique_couples))]
  for i_u in range(len(unique_couples)):
     for i_l in range(len(list_couples)):
        if unique_couples[i_u] == list_couples[i_l]:
          indexes_mean[i_u].append(i_l)
  return unique_couples, indexes_mean 


def compute_float_mean(total_float_tensor, unique_couples, indexes_mean):
  """this function takes the total float daily profiles and compute the mean between weeks and among the same float"""
  weekly_mean_float_tensor = torch.zeros([d, len(unique_couples)])
  for i in range(len(unique_couples)):
    tmp_tensor = torch.stack([total_float_tensor[:,j] for j in indexes_mean[i]], dim= 1)
    single_mean_tensor = torch.mean(tmp_tensor, axis=1)
    weekly_mean_float_tensor[:,i] = single_mean_tensor
  return unique_couples, weekly_mean_float_tensor
  

def compute_float_mean_no_0(total_float_tensor, unique_couples, indexes_mean):
  """this function takes the total float daily profiles and compute the mean between weeks and among the same float"""
  indexes_mean_no_0 = []
  unique_couples_no_0 = []  
  weekly_mean_float_tensor = torch.zeros([d, len(unique_couples)])
  for i in range(len(unique_couples)):
    indexes_no_0 = [indexes_mean[i][j] for j in range(len(indexes_mean[i])) if torch.count_nonzero(total_float_tensor[:, indexes_mean[i][j]]) > 0]
    if len(indexes_no_0) == 0:
      continue
    tmp_tensor = torch.stack([total_float_tensor[:,j] for j in indexes_no_0], dim= 1)
    single_mean_tensor = torch.mean(tmp_tensor, axis=1)
    weekly_mean_float_tensor[:,i] = single_mean_tensor
    indexes_mean_no_0.append(indexes_no_0)
    unique_couples_no_0.append(unique_couples[i])
  return unique_couples, weekly_mean_float_tensor[:, :len(indexes_mean_no_0)]


def search_corresponding_date_weeks(single_float_device, unique_couples):
   """this function computes the list of weeks whose measures are registered by a specific float device"""
   list_week_measured = []
   for couple in unique_couples:
      if couple[0] == single_float_device:
         list_week_measured.append(couple[1])
   return list_week_measured


def insert_float_values(lat_limits, lon_limits, depth_limits, year_limits, resolution, list_data_times, dict_list_parallelepiped_vars, list_float_vars):
  """
  Function that update the parallelepiped updating the voxel where the float info is available
  lat_limits = (lat_min, lat_max)
  lon_limits = (lon_min, lon_max)
  depth_limits = (depth_min, depth_max) in km
  year_limits = (year_min, year_max)
  resolution = (w_res, h_res, d_res) dimension of a voxel (in km)
  """
  lat_min, lat_max = lat_limits
  lon_min, lon_max = lon_limits
  depth_min, depth_max = depth_limits
  year_min, year_max = year_limits
  w_res, h_res, d_res = resolution
  w = int((lat_max - lat_min) * constant_latitude / w_res + 1)
  h = int((lon_max - lon_min) * constant_longitude / h_res + 1)
  d = int((depth_max - depth_min) / d_res + 1)

  list_data = pd.read_csv(float_path + '/Float_Index.txt', header=None).to_numpy()[:, 0].tolist()
  list_datetime = pd.read_csv(float_path + '/Float_Index.txt', header=None).to_numpy()[:, 3].tolist()
  dict_float_names = dictionary_float_vars_names(list_biogeoch_vars)

  for i in range(np.size(list_data)):  
    path_current_float = float_path + list_data[i]
    ds = nc.Dataset(path_current_float)
    datetime = list_datetime[i]
    time = read_date_time_sat(datetime)
    if not year_min < time < year_max:
      continue
    index = list_data_times[int(time) - int(year_interval[0])].index(time)  
    for biogeoch_var in list_biogeoch_vars:
      if dict_float_names[biogeoch_var] in list_float_vars[i]:   
        list_biog_parallelepiped = dict_list_parallelepiped_vars[biogeoch_var]
        select_parallelepiped = list_biog_parallelepiped[int(time) - year_interval[0]][index]
        counter_coordinates = torch.zeros((1, number_channel_biogeoch, d, h, w))
        lat = float(ds['LATITUDE'][:].data)  
        lon = float(ds['LONGITUDE'][:].data)  
        lat_index = find_index(lat, lat_limits, w)
        lon_index = find_index(lon, lon_limits, h)
        pres_list = ds['PRES' + '_' + dict_float_names[biogeoch_var]][:].data   
        depth_list = []
        for pres in pres_list:
          depth_list.append(to_depth(pres, lat))  
        biog_var = ds[dict_float_names[biogeoch_var]][:].data  

        if lat_max > lat > lat_min:
          if lon_max > lon > lon_min:
            for depth in depth_list:
              if depth_max > depth > depth_min:
                depth_index = find_index(depth, depth_limits, d)
                channel_index = np.where(depth_list == depth)[0][0]    

                biog_var_v = float(biog_var[channel_index])
                if  dict_variables_domain[biogeoch_var][0]< biog_var_v < dict_variables_domain[biogeoch_var][1]:
                  biog_var_v = (select_parallelepiped[0, 0, depth_index, lon_index, lat_index] * counter_coordinates[0, 0, depth_index, lon_index, lat_index] + biog_var_v) / (counter_coordinates[0, 0, depth_index, lon_index, lat_index] + 1)
                  counter_coordinates[0, 0, depth_index, lon_index, lat_index] += 1
                  select_parallelepiped[0, 0, depth_index, lon_index, lat_index] = biog_var_v
        np.save(os.getcwd() + "/dataset/float/" + str(year_min) + "/saved_parallelepiped_biogeoch/" + biogeoch_var + "_parallelepiped.npy", dict_list_parallelepiped_vars[biogeoch_var][int(time)-year_interval[0]])    
  
  return


def routine_insert_float_biog(float_path, resolution):
  """
    routine to run the function that update the parallelepiped updating all the voxel with float information
    it is implemented for the biogeochemical variable for a single year (for computational limits), but it can be used for all the years together adding a for cycle at the beginning
  """
  list_data = pd.read_csv(float_path + '/Float_Index.txt', header=None).to_numpy()[:, 0].tolist()
  list_datetime = pd.read_csv(float_path + '/Float_Index.txt', header=None).to_numpy()[:, 3].tolist()
  float_var_lists = generate_list_float_variables(list_data)
  list_data_times = [create_list_date_time((year_interval[i], year_interval[i+1])) for i in range(len(year_interval) - 1)]
  dict_list_parallelepiped_vars = dict.fromkeys([biog_var for biog_var in list_biogeoch_vars], [])
  dict_names_float_var = dictionary_float_vars_names(list_biogeoch_vars)
  for biogeoch_var in list_biogeoch_vars:
    if biogeoch_var in dict_names_float_var.keys():
      list_parallelepipeds = []
      for i in range(len(year_interval)-1):
        list_parallelepiped = [
          create_box(batch, number_channel_biogeoch, latitude_interval, longitude_interval, depth_interval, resolution) for j in
          range(len(list_data_times[i]))]
        list_parallelepipeds.append(list_parallelepiped)
      dict_list_parallelepiped_vars[biogeoch_var] = list_parallelepipeds
  #Insert float values inside the matrixes
  for i in range(len(year_interval)-1):
    insert_float_values(latitude_interval, longitude_interval, depth_interval, (year_interval[i], year_interval[i+1]), resolution, list_data_times,dict_list_parallelepiped_vars, float_var_lists)
    np.save(os.getcwd() + "/dataset/float/" + str(year_interval[i]) + "/saved_parallelepiped_biogeoch/" + biogeoch_var + "_parallelepiped.npy", dict_list_parallelepiped_vars[biogeoch_var][int(year_interval[i])-year_interval[0]]) 
  return None

if kindof == "float_save":
  list_vars_list_parallelepiped = np.load(os.getcwd() + "/dataset/float/" + str(2022) + "_" + str(2024) + "/saved_parallelepiped_biogeoch/" + "P_l" + "_parallelepiped_2022_2024.npy", allow_pickle=True)
  new_list_vars_list_parallelepiped = [list_vars_list_parallelepiped[j, :, :, :, :, :] for j in range(len(list_vars_list_parallelepiped))]
  print("len new list parallelepiped", len(new_list_vars_list_parallelepiped))
  save_routine(kindof, new_list_vars_list_parallelepiped, list_data_times[-1], (2023, 2024), t, biogeoch_var, path_directory_save)


if kindof == 'float':
    
    ##for i in range(len(year_interval)-1):
      ##insert_float_values(latitude_interval, longitude_interval, depth_interval, (year_interval[i], year_interval[i+1]), resolution, list_data_times,dict_list_parallelepiped_vars, float_var_lists)
      ##np.save(os.getcwd() + "/dataset/float/" + str(year_interval[i]) + "/saved_parallelepiped_biogeoch/" + biogeoch_var + "_parallelepiped.npy", dict_list_parallelepiped_vars[biogeoch_var][int(year_interval[i])-year_interval[0]])   #prima era np.save
    #implement save routine for float
    list_vars_list_parallelepiped = np.load(os.getcwd() + "/dataset/float/" + str(year_interval[0]) + "/saved_parallelepiped_biogeoch/" + "P_l" + "_parallelepiped.npy", allow_pickle=True)
    print("type", type(list_vars_list_parallelepiped))
    print("type old old", type(list_vars_list_parallelepiped[2]))
    print("shape", list_vars_list_parallelepiped.shape)

    new_list_vars_list_parallelepiped = [list_vars_list_parallelepiped[j, :, :, :, :, :] for j in range(len(list_vars_list_parallelepiped))]
    new_list_vars_list_parallelepiped_np = [list_vars_list_parallelepiped[j, :, :, :, :, :] for j in range(list_vars_list_parallelepiped.shape[0])]
    print("len", len(new_list_vars_list_parallelepiped))
    print("single shape", new_list_vars_list_parallelepiped[0].shape)
    for j in range(len(new_list_vars_list_parallelepiped)-1):
       print("shape single list",new_list_vars_list_parallelepiped[j].shape)
       print("EQUALITY", np.array_equal(new_list_vars_list_parallelepiped[j], new_list_vars_list_parallelepiped[j+1]))
       print("type old", type(list_vars_list_parallelepiped[j]))
       print("EQUALITY old", np.array_equal(list_vars_list_parallelepiped[j], new_list_vars_list_parallelepiped[j]))
       print("EQUALITY new", np.array_equal(new_list_vars_list_parallelepiped_np[j], new_list_vars_list_parallelepiped[j]))


    for biogeoch_var in list_biogeoch_vars:
       list_biog_parallelepiped = dict_list_parallelepiped_vars[biogeoch_var]
       #print("type ", list_biog_parallelepiped)
       for i in range(len(year_interval)-1):
        print("year", year_interval[i])
        list_vars_list_parallelepiped = np.load(os.getcwd() + "/dataset/float/" + str(year_interval[i]) + "/saved_parallelepiped_biogeoch/" + "P_l" + "_parallelepiped.npy", allow_pickle=True)

        new_list_vars_list_parallelepiped = [list_vars_list_parallelepiped[j, :, :, :, :, :] for j in range(len(list_vars_list_parallelepiped))]

        save_routine(kindof, new_list_vars_list_parallelepiped, list_data_times[i], (year_interval[i], year_interval[i+1]), t, biogeoch_var, path_directory_save)
        #save_routine(kindof, list_biog_parallelepiped[i], list_data_times[i], (year_interval[i], year_interval[i+1]), t, biogeoch_var, path_directory_save)
        ###prima in list_biog_parallelepiped era list_biog_parallelepiped[i]  (dentro save routine)










