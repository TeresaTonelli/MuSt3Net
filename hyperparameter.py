"""
hyperparameter for the implementation
"""

batch = 1
number_channel = 7     # 6 var physic + 1 var biogeoch 
number_channel_physics = 6
number_channel_biogeoch = 1
list_physics_vars = ["soshfldo", "sossheig", "vomecrty", "vosaline", "votemper", "vozocrtx"]
list_biogeoch_vars = ["P_l"]  #["N3n", "O2o", "P_l"] 
dict_variables_domain = {"soshfldo" : (0, 1000), "sossheig": (-10, 10), "vomecrty":(-10, 10), "vosaline":(10, 41), "votemper":(-3, 40), "vozocrtx":(-10, 10), 
                            "ALK":(0, 3500), "DIC":(0, 3500), "N1p":(0, 2), "N3n":(0, 20), "O2o":(0, 400), "P_c":(), "pCO2":(0, 1000), "pH":(0, 14), "ppn":(-10, 1000), "R6c":(0, 60), "P_l":(0,10)}
dict_channel_physics = {0 : "soshfldo", 1 : "sossheig", 2 : "vomecrty", 3 : "vosaline", 4 : "votemper", 5 : "vozocrtx"}
latitude_interval = (30, 46)       
longitude_interval = (-3, 36)         
depth_interval = (0, 300)    
year_interval = (2019, 2020, 2021, 2022, 2023, 2024)       

resolution = (8, 8, 10)  
constant_latitude = 111
constant_longitude = 111
standard_mean_values = {"P_l": 0.15, "O2o": 200, "N3n" : 4, "N1p": 0.2}  #Manca il POC da inserire

h = int((longitude_interval[1] - longitude_interval[0]) * constant_longitude / resolution[1] + 1)
w = int((latitude_interval[1] - latitude_interval[0]) * constant_latitude / resolution[0] + 1)
d = int((depth_interval[1] - depth_interval[0]) / resolution[2] + 1)


n_samples_biogeoch = 100
n_duplicates_biogeoch = 10

deep_sea_good_count = 20   #30
superficial_bound_depth = 200

parameters_plots = {"P_l": [[0.0 for i in range(d)], [-0.0175*i + 0.5 for i in range(d)]], "O2o": [], "N3n": [], "POC": [], 
                    "soshfldo":[[0.0 for i in range(d)], [200 for i in range(d)]], "sossheig":[[0.0 for i in range(d)], [2.0 for i in range(d)]], "vomecrty":[[0.0 for i in range(d)], [2.0 for i in range(d)]], 
                    "vosaline":[[32.0 for i in range(d)], [40.0 for i in range(d)]], "votemper":[[0.0 for i in range(d)], [-(12/29)*i + 30.0 for i in range(d)]], "vozocrtx":[[0.0 for i in range(d)], [2.0 for i in range(d)]]}

kindof = '-'


if kindof == 'float':
    channels = [6]      #TEORICAMENTE: solo la var biogeochimica avrà dati float
if kindof == 'sat':
    channels = [6]      #TEORICAMENTE: solo la var biogeochimica avrà dati satellitari
if kindof == 'model':
    channels = [0, 1, 2, 3, 4, 5, 6]

