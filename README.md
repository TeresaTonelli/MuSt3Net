# MuSt3Net

Python library for the implementation of MuSt3Net (Multiple Steps 3D Convolutional Neural Network for data integration (or model data fusion)). 
This project developes a convolutional neural network (CNN) to perform model data fusion through an innovative 2-step training pocedure. 

```bash
pip install -r requirements.txt 
```

## Code's structure
The code is organized as follows:
* `--MuSt3Net` contains the functions for the implementation of the CNN architecture and for the 2-step training procedure
* `--plots` contains the functions to generate the plots
* `--data_preprocessing` contains the functions to generate the input data (3D tensors for the CNN trainings)
* `--RMSE_computation` contain the funtions to compute the RMSE to assess the model results
* `--utils` contains all the auxiliary functions for models' training and validation


### Training MuSt3Net
To train the MuSt3Net model, it is sufficient to run the 2_step_train_ensemble.py script:

```bash
sbatch ./2_step_train_ensemble.sh 
```

The .sh script run the .py script, which is organized as follows: 
* it downloads the data from the dataset_training directory and it uses the function training_1p (`--MuSt3Net/training_testing_function.training_1p`) to emulate the BFM model 
* it uses the function testing_1p (`--MuSt3Net/training_testing_function.testing_1p`) to test the quality of the BFM emulation on the test dataset
* it downloads the data from the dataset_training directory and the model_1p results and it uses the function training_2p (`--MuSt3Net/training_testing_function.training_2p`) to integrate Argo-floats
*  it uses the function testing_2p_ensemble (`--MuSt3Net/training_testing_function.testing_2p_ensemble`) to test the quality of the integration on the test dataset

To manage the runtime, which could exceed 24 hours, the code flow is ruled by 3 binary parameters
*  `--first_run_id` is the parameter which identifies the first run of the job; it is useful to correctly download data.
*  `--end_train_1p` is the parameter which identifies if the 1<sup>st</sup> training step is ended or not.
*  `--end_1p` is the parameter which identifies if the 1<sup>st</sup> step (training and testing) is ended or not. if is true, the code passes to execute the 2<sup>nd</sup> step. 



### Data pre-processing
The input data are concatenation od 3D tensors, representing the 3D distribution of physical variables and chlorophyll. 
To obtain these tensors starting from netCDF files (.nc), it is sufficient to run the `data_preprocessing/run_data_preprocessing.py` script, in which, for each single variable:
*  `--make_dataset_single_var` creates the .npy file in which the 3D distribution of a single variable is saved. The file `--make_dataset_float`does the same for the float measures. 
*  `--plot_save_tensor` transforms the .npy in a .pt file
*  `--interpolation` interpolates the .pt file in order to reach an higher spatial resolution; different interpolation functions are implemented for numerical model data and float data. 
The final tensors are stored in `final_tensor` folder, inside the `dataset` folder, and they are simply a copy of the `interp_tensor` folder. 
Due to high dimensionality of data and lower computational resources, the tensor data for the 1<sup>st</sup> training step can be generated through the script `generation_training_dataset` and therefore stored in `dataset_training` folder. 


### Results' plots 
The results, along with the trained models, will be automatically saved in the directory referred to the current job; the `plots` directory contain all the plots used to see and evaluate the prediction quality. 

Different plots are used for this purpose, in particular: 
* `--maps 1p` are the maps which compares the 1<sup>st</sup> training step results with the BFM predictions
* `--maps 2p` are the maps which shows the chlorophyll 3D maps after the data integration with ARGO floats
* `--profiles 1p` are the plots which compares the same profiles computed by the MuSt3Net after the 1<sup>st</sup> training step and the BFM one
* `--profiles 2p` are the plots which compares the same profiles computed by the MuSt3Net after the 2<sup>nd</sup> training step and the real BGC-Argo float measure
* `--hovmoller` are the plots which show the temporal behavior of the predicted chlorophyll
* `--hovmoller_external` are the plots which show the temporal behavior of the predicted chlorophyll on unseen float data



### RMSE computation
To asses the result quality, the plots are coupled with the computation of the root mean square error (RMSE). 

The `RMSE_computation/rmse_function_test.sh` runs the `RMSE_computation/rmse_function_test.py`script, which computes the RMSE with respect to different geographical areas (`RMSE_compiutation/rmse_functions.RMSE_ensemble_ga`) and the RMSE with respect to different seasons (`RMSE_computation/rmse_functions.RMSE_ensemble_season`). 



### Modifying the Model Architecture
To apply the same architecture for the prediction of other biogeochemical variables, it could be useful to modify the model architecture, for example, adding or removing some convolutional layers. The default architectures are located in:
* `MuSt3Net/convolutional_network.py`.


### Dataset
The dimension of the training dataset exceeds the available memory space of github. A portion of the dataset is available in Zenodo (); for the whole training dataset, ask the codeowner and it will be sent. 