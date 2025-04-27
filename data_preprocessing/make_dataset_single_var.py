"""
Routine for the creation of the parallelepiped that composed the training set
"""
import torch
import netCDF4 as nc
import numpy as np
import pandas as pd
import os
from hyperparameter import *
import datetime 


constant_latitude = 111 
constant_longitude = 111 
float_path = "../FLOAT_BIO/"


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
    lists = np.arange(year, year + 0.54, 0.01)  
    for i in range(len(lists)):
      lists[i] = round(lists[i], 2)
    lists = lists.tolist()
    total_list = total_list + lists
  return total_list


def create_box(batch, number_channel, lat, lon, depth, resolution):
  """
  Function that creates the EMPTY tensor that will be filled with data
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
  var = variable considered
  var_limits = (var_min, var_max)
  var_size = tensor dimension relative to the current variable
  """
  var_min, var_max = var_limits
  var_res = (var_max - var_min) / var_size
  var_index = int((var - var_min) / var_res)
  return var_index



def insert_model_physic_values_single(year, lat_limits, lon_limits, depth_limits, year_limits, resolution, physic_var, model_file, ds_phys_var):
  """
    function that update the parallelepiped updating all the voxel with MODEL information
    year = folder of the year we are considering
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

  var_tensor = torch.zeros((1, 1, d, h, w))
  counter_coordinates = torch.zeros((1, number_channel_physics, d, h, w))

  latitude_list = ds_phys_var['latitude'][:].data
  longitude_list = ds_phys_var['longitude'][:].data
  depth_list = ds_phys_var['depth'][:].data

  phys_tens = torch.tensor(ds_phys_var[physic_var][:].data)[:, :, :]    

  print(model_file + ' analysis started')
  for i in range(len(latitude_list)): 
    for j in range(len(longitude_list)): 
        for k in range(len(depth_list)): 
            latitude = latitude_list[i]
            longitude = longitude_list[j]
            depth = depth_list[k]
            phys = phys_tens[k, i, j]   
            if lat_max > latitude > lat_min:
                if lon_max > longitude > lon_min:
                    if depth_max > depth > depth_min:
                        latitude_index = find_index(latitude, lat_limits, w)
                        longitude_index = find_index(longitude, lon_limits, h)
                        depth_index = find_index(depth, depth_limits, d)
                    if dict_variables_domain[physic_var][0] < phys < dict_variables_domain[physic_var][1]:
                        phys = (var_tensor[0, 0, depth_index, longitude_index, latitude_index] * counter_coordinates[0, 0, depth_index, longitude_index, latitude_index] + phys) / (counter_coordinates[0, 0, depth_index, longitude_index, latitude_index] + 1)
                        counter_coordinates[0, 0, depth_index, longitude_index, latitude_index] += 1
                        var_tensor[0, 0, depth_index, longitude_index, latitude_index] = phys
  print(model_file + ' analysis completed')
  return var_tensor
  

def insert_model_biogeoch_values_single(year, lat_limits, lon_limits, depth_limits, year_limits, resolution, biogeoch_var, model_file, ds_biogeoch):
  """
    function that update the parallelepiped updating all the voxel with MODEL information
    year = folder of the year we are considering
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

  var_tensor = torch.zeros((1, number_channel_biogeoch, d, h, w))
  counter_coordinates = torch.zeros((1, number_channel_biogeoch, d, h, w))

  latitude_list = ds_biogeoch['lat'][:].data
  longitude_list = ds_biogeoch['lon'][:].data
  depth_list = ds_biogeoch['depth'][:].data

  biogeoch_tens = torch.tensor(ds_biogeoch[biogeoch_var][:].data)[0, :, :, :]   

  print(model_file + ' analysis started')
  for i in range(len(latitude_list)): 
    for j in range(len(longitude_list)): 
      for k in range(len(depth_list)): 
        latitude = latitude_list[i]
        longitude = longitude_list[j]
        depth = depth_list[k]
        biogeoch = biogeoch_tens[k, i, j] 
        if lat_max > latitude > lat_min:
          if lon_max > longitude > lon_min:
            if depth_max > depth > depth_min:
              latitude_index = find_index(latitude, lat_limits, w)
              longitude_index = find_index(longitude, lon_limits, h)
              depth_index = find_index(depth, depth_limits, d)
              if dict_variables_domain[biogeoch_var][0] < biogeoch < dict_variables_domain[biogeoch_var][1]:
                biogeoch = (var_tensor[0, 0, depth_index, longitude_index, latitude_index] * counter_coordinates[0, 0, depth_index, longitude_index, latitude_index] + biogeoch) / (counter_coordinates[0, 0, depth_index, longitude_index, latitude_index] + 1)
                counter_coordinates[0, 0, depth_index, longitude_index, latitude_index] += 1
                var_tensor[0, 0, depth_index, longitude_index, latitude_index] = biogeoch
  print(model_file + ' analysis completed')
  return var_tensor


def routine_insert_model_physics(physic_var, path_physic_var, year, resolution):
  """
    routine to run the function that update the parallelepiped updating all the voxel with MODEL information
    it is implemented for a single physycal variable and for a single year (for computational limits), but  it can be used for all the physycal variables together adding a for cycle at the beginning
  """
  physics_var_files = os.listdir(path_physic_var)   
  for model_file in physics_var_files:
    if model_file[0:3] != 'ave':
      continue
    file_phys_var = path_physic_var + model_file
    ds_phys_var = nc.Dataset(file_phys_var)
    time = model_file[4:12]
    time = read_date_time_sat(time)
    if not int(year) < time < int(year) + 1:
      continue   
    phys_matrix = insert_model_physic_values_single(int(year), latitude_interval, longitude_interval, depth_interval, (int(year), int(year)+1), resolution, physic_var, model_file, ds_phys_var)
    week = datetime.date(int(model_file[4:12][0:4]), int(model_file[4:12][4:6]), int(model_file[4:12][6:8])).isocalendar()[1]
    np.save(os.getcwd() + "/dataset/MODEL/" + str(year) + "/saved_parallelepiped_physics_single/" + physic_var + "/" + str(week)+ "_parallelepiped.npy", phys_matrix)
  return None

def routine_insert_model_biog(biogeoch_var, path_biogeoch, year, resolution):
  """
    routine to run the function that update the parallelepiped updating all the voxel with MODEL information
    it is implemented for the biogeochemical variable for a single year (for computational limits), but it can be used for all the years together adding a for cycle at the beginning
  """
  biogeoch_files = os.listdir(path_biogeoch)
  for model_file in biogeoch_files:
    if model_file[0:3] != 'ave':
      continue
    file_biogeoch = path_biogeoch + model_file
    ds_biogeoch = nc.Dataset(file_biogeoch)
    time = model_file[4:12]
    time = read_date_time_sat(time)
    if not int(year) <= time < int(year) + 1:    
      continue  
    var_tensor = insert_model_biogeoch_values_single(int(year), latitude_interval, longitude_interval, depth_interval, (int(year), int(year)+1), resolution, biogeoch_var, model_file, ds_biogeoch)
    week = datetime.date(int(model_file[4:12][0:4]), int(model_file[4:12][4:6]), int(model_file[4:12][6:8])).isocalendar()[1]
    np.save(os.getcwd() + "/dataset/MODEL/" + str(year) + "/saved_parallelepiped_biogeoch_single/" + biogeoch_var + "/" + str(week)+ "_parallelepiped.npy", var_tensor)
  return None