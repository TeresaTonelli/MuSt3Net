#this script contains all the functions to identify float devices starting from data


import numpy as np
import torch
import os
from datetime import datetime

from utils.utils_general import transform_latitudes_longitudes
from utils.utils_dataset_generation import read_txt_file


def transform_float_data(time):
    """this function tranforms the time in the form day - month - year"""
    data_time = time[0:8]
    year = data_time[0:4]
    month = data_time[4:6]
    day = data_time[6:8]
    return [year, month, day]


def compute_list_float_devices(file_dir):
    """this function computes the list of all the float devides contained in the Float_Index file"""
    list_float_devices = []
    with open(file_dir, 'r') as file:
        lines = file.readlines()
    for i_l in range(len(lines)):
        line = lines[i_l]
        line_elements = line.split()
        float_device = line_elements[0].split(",")[0]
        list_float_devices.append(float_device)
    return list(dict.fromkeys(list_float_devices))


def select_float_measures(file_dir, float_device_id, var='CHLA'):
    """this function returns all the index lines in which there are measures of that specific float"""
    list_indexes_float_data = []
    with open(file_dir, 'r') as file:
        lines = file.readlines()
    for i_l in range(len(lines)):
        line = lines[i_l]
        line_elements = line.split()
        list_variables = line_elements[1:]
        if var in list_variables:
            float_device_i_l = lines[i_l].split()[0].split(",")[0]    
            if float_device_i_l == float_device_id:
                list_indexes_float_data.append(i_l)
    return list_indexes_float_data


def search_unseen_float(file_dir, list_float_devices, list_seen_years, var='CHLA'):
    """this function returns the float id of a float whose measures where never seen in a specific year interval (and in its past)"""
    list_unseen_float_devices = []
    list_unseen_indexes = []
    with open(file_dir, 'r') as file:
        lines = file.readlines()
    for float_device in list_float_devices:
        float_devices_line_indexes = select_float_measures(file_dir, float_device, var)
        for line_index in float_devices_line_indexes:
            time = lines[line_index].split()[0].split(",")[3]
            year, month, day = transform_float_data(time)
            if int(year) > list_seen_years[-1]:
                list_unseen_indexes.append(line_index + 1)
                list_unseen_float_devices.append(float_device)
    return list_unseen_float_devices, list_unseen_indexes



def create_float_txt(list_floats_indexes, file_total_floats):
    """this function takes the list of selected float devices and creates a txt file which contains only the data refeering to those specific float devices"""
    file_selected_floats = open("/leonardo_scratch/large/userexternal/ttonelli/OGS/SUPERFLOAT/SUPERFLOAT/Float_2022_Index.txt", "w")
    with open(file_total_floats, 'r') as file:
        float_lines = file.readlines()
    for index in list_floats_indexes:
        file_selected_floats.write(float_lines[index-1])
    return None




def single_float_device_identifier(file_dir, list_year_week, list_lat_long, epsilons):
    list_lat_long = transform_latitudes_longitudes(list_lat_long)
    with open(file_dir, 'r') as file:
        lines = file.readlines()
    for i_l in range(len(lines)):
        line = lines[i_l]
        line_elements = line.split()
        list_variables = line_elements[1:]
        if 'CHLA' in list_variables:
            line_info_data = line_elements[0]
            line_info_data = line_info_data.split(",")
            float_device = line_info_data[0]
            float_device = float_device.split(".", 1)[0]   
            latitude = float(line_info_data[1])
            longitude = float(line_info_data[2])
            time = line_info_data[3]
            year, month, day = transform_float_data(time)
            day = int(day)
            month = int(month)
            year = int(year)
            week = datetime(int(year), int(month), int(day)).isocalendar()[1]
            my_year, my_week = list_year_week[0], list_year_week[1]
            my_lat, my_long = list_lat_long[0], list_lat_long[1]
            if my_year == year and my_week == week:
                if abs(my_lat - latitude) < epsilons[0] and abs(my_long - longitude) < epsilons[1]:
                    return float_device
    return None


def float_device_identifier(file_dir, list_year_week, list_lat_long, epsilons):
    """this function iterates the behavior of the previous function but for multiple latitudes and longitudes, like it will happens while plotting images""" 
    list_float_devices = []
    for i in range(len(list_lat_long)):
        my_lat, my_long = list_lat_long[i][0], list_lat_long[i][1]
        float_device = single_float_device_identifier(file_dir, list_year_week, [my_lat, my_long], epsilons)
        list_float_devices.append(float_device)
    return list_float_devices



