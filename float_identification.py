#this script contains all the functions to identify float devices starting from data


import numpy as np
import torch
import os
from datetime import datetime

from utils_function import transform_latitudes_longitudes
from utils_generation_train_1p import read_txt_file


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
        float_device = float_device.split("_", 1)[0]
        list_float_devices.append(float_device)
    print("list without duplicates", list(dict.fromkeys(list_float_devices)))
    return list(dict.fromkeys(list_float_devices))


def select_float_measures(file_dir, float_device_id):
    """this function returns all the index lines in which there are measures of that specific float"""
    list_indexes_float_data = []
    with open(file_dir, 'r') as file:
        lines = file.readlines()
    for i_l in range(len(lines)):
        float_device_i_l = lines[i_l].split()[0].split(",")[0].split("_", 1)[0]
        if float_device_i_l == float_device_id:
            list_indexes_float_data.append(i_l)
    print("list_indexes_float_data", list_indexes_float_data)
    return list_indexes_float_data


def search_unseen_float(file_dir, list_float_devices, list_seen_years):
    """this function returns the float id of a float whose measures where never seen in a specific year interval (and in its past)"""
    list_unseen_float_devices = []
    with open(file_dir, 'r') as file:
        lines = file.readlines()
    for float_device in list_float_devices:
        print("float device", float_device)
        float_counter = 0
        float_devices_line_indexes = select_float_measures(file_dir, float_device)
        for line_index in float_devices_line_indexes:
            time = lines[line_index].split()[0].split(",")[3]
            year, month, day = transform_float_data(time)
            if int(year) not in list_seen_years and int(year) <= list_seen_years[0]:
                break
            else:
                float_counter += 1
        if float_counter == len(float_devices_line_indexes):
            list_unseen_float_devices.append(float_device)
    print("list unseen float devices", list_unseen_float_devices)
    return list_unseen_float_devices



def single_float_device_identifier(file_dir, list_year_week, list_lat_long, epsilon):
    list_lat_long = transform_latitudes_longitudes(list_lat_long)
    with open(file_dir, 'r') as file:
        lines = file.readlines()
    for i_l in range(len(lines)):
        line = lines[i_l]
        line_elements = line.split()
        line_info_data = line_elements[0]
        #line_info_data = information_data.split()
        #print("elemtns of info data", [line_info_data[i] for i in range(len(line_info_data))])
        line_info_data = line_info_data.split(",")
        float_device = line_info_data[0]
        float_device = float_device.split("_", 1)[0]
        latitude = float(line_info_data[1])
        longitude = float(line_info_data[2])
        time = line_info_data[3]
        #print("time", time)
        year, month, day = transform_float_data(time)
        day = int(day)
        month = int(month)
        year = int(year)
        week = datetime(int(year), int(month), int(day)).isocalendar()[1]
        #print("week", week)
        my_year, my_week = list_year_week[0], list_year_week[1]
        my_lat, my_long = list_lat_long[0], list_lat_long[1]
        if my_year == year and my_week == week and abs(my_lat - latitude) < epsilon and abs(my_long - longitude) < epsilon:
            print("line index", i_l)
            print("float device", float_device)
            return float_device
    return None


def float_device_identifier(file_dir, list_year_week, list_lat_long, epsilon):
    """this function iterates the behavior of the previous function but for multiple latitudes and longitudes, like it will happens while plotting images""" 
    list_float_devices = []
    list_lat_long = [transform_latitudes_longitudes(list_lat_long[i]) for i in range(len(list_lat_long))]
    for i in range(len(list_lat_long)):
        my_lat, my_long = list_lat_long[i][0], list_lat_long[i][1]
        float_device = single_float_device_identifier(file_dir, list_year_week, [my_lat, my_long], epsilon)
        list_float_devices.append(float_device)
    return list_float_devices


#single_float_device_identifier("/leonardo_scratch/large/userexternal/ttonelli/OGS/SUPERFLOAT/SUPERFLOAT/Float_Index.txt",[2022,47], [34.1, 33.1], 0.1)
#float_device_identifier("/leonardo_scratch/large/userexternal/ttonelli/OGS/SUPERFLOAT/SUPERFLOAT/Float_Index.txt",[2021,47], [[20, 443], [22, 304], [27, 495]], 0.1)
list_float_devices = compute_list_float_devices("/leonardo_scratch/large/userexternal/ttonelli/OGS/SUPERFLOAT/SUPERFLOAT/Float_Index.txt")
select_float_measures("/leonardo_scratch/large/userexternal/ttonelli/OGS/SUPERFLOAT/SUPERFLOAT/Float_Index.txt", '6902875/SD6902875')
search_unseen_float("/leonardo_scratch/large/userexternal/ttonelli/OGS/SUPERFLOAT/SUPERFLOAT/Float_Index.txt", list_float_devices, [2019, 2020, 2021])
print("end")




def unseen_float_identifier(file_dir, list_seen_years):
    """this function identifies a new float devices whose measures were never used and wrote inside float tensors"""
    return None
