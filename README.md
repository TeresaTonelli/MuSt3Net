# CNN-3DMedSea

Python library for the implementation of CNN-3DMedSea (Convolutional Neural Network for 3D data integration in the Mediterranean Sea). 
This project developes a CNN to perform model data fusion thorugh an innovative 2-step training pocedure. 

```bash
pip install -r requirements.txt 
```

## Code's structure
The code is organized as follows:
* `--CNN_3DMedSea` contains the functions for the implementation of the CNN architecture and for the 2-step training procedure
* `--CNN_plots` contains the functions to generate the plots
* `--data_preprocessing` contains the functions to generate the input data (3D tensors for training the CNN)
* `--posterior_analysis` contain the funtions to assess the model results (i.e. RMSE computation)
* `--utils` contains all the auxiliary functions for models' training and validation


### Training CNN-3DMedSea
To train the CNN-3DMedSea model, it is sufficient to run the 2_step_train_ensemble.py script:

```bash
sbatch ./2_step_train_ensemble.sh 
```

The .sh script run the .py script, which is organized as follows: 
* it downloads the data from the dataset_training directory and it uses the function training_1p (training_testing_function.training_1p) to emulate the BFM model 
* it uses the function testing_1p (training_testing_function.testing_1p) to test the quality of the BFM emulation on the test dataset
* it downloads the data from the dataset_training directory and the model_1p results and it uses the function training_2p (training_testing_function.training_2p) to integrate Argo-floats
*  it uses the function testing_2p_ensemble (training_testing_function.testing_2p_ensemble) to test the quality of the integration on the test dataset

To manage the runtime, whoch could overcome 24 hours, the code flow is ruled by 3 binary parameters
*  `--first_run_id` is the parameter which identifies the first run of the job; it is useful to correctly download data.
*  `--end_train_1p` is the parameter which identifies if the 1^{st} training step is ended or not.
*  `--end_1p` is the parameter which identifies if the 1^{st} step (training and testing) is ended or not. if is true, the code passes to execute the 2^{nd} step. 



### Data pre-processing
The input data are concatenation od 3D tensors, representing the 3D distribution of physical variables and chlorophyll. 
To obtain these tensors starting from netCDF files (.nc), it is sufficient to run the `data_preprocessing/run_data_preprocessing.py` script, in which, for each single variable:
*  `--make_dataset_single_var` creates the .npy file in which the 3D distribution of a single variable is saved. The file `--make_dataset_float`does the same for the float measures. 
*  `--plot_save_tensor` transforms the .npy in a .pt file
*  `--interpolation` interpolates the .pt file in order to reach an higher spatial resolution; different interpolation functions are implemented for numerical model data and float data. 
The final tensors are stored in `final_tensor` folder, inside the `dataset` folder, and they are simply a copy of the `interp_tensor` folder. 
Due to high dimensionality of data and lower computational resources, the tensor data for the 1^{st} training step can be generated through the script `generation_training_dataset` and therefore stored in `dataset_training` folder. 


### Results' plots 
The results, along with the trained models, will be automatically saved in the directory referred to the current job; inside it, the `plots` directory contain all the plots ised to see and evaluate the prediction quality. 

Different plots are used for this purpose, in particular: 
* `--maps 1p` are the maps which compares the 1^{st} training step results with the BFM predictions
* `--maps 2p` are the maps which shows the chlorophyll 3D maps after the data integration with ARGO floats
* `--profiles 1p` are the plots which compares the same profiles computed by the CNN-3DMedSea after the 1^{st} training step and the BFM one
* `--profiles 2p` are the plots which compares the same profiles computed by the CNN-3DMedSea after the 2^{nd} training step and the real ARGO float measure
* `--hovmoller` are the plots which show the temporal behavior of the predicted chlorophyll
* `--hovmoller_external` are the plots which show the temporal behavior of the predicted chlorophyll on unseen float data



### Posterior analysis
To asses the result quality, the plots are coupled with the computation of the root mean square error (RMSE) and the float identification (for the Hovmoller plots). 

The `posterior_analysis/rmse_function_test.sh` runs the `posterior_analysis/rmse_function_test.py`script, which computes the RMSE with respect to different geographical areas (`posterior_analysis/rmse_functions.RMSE_ensemble_ga`) and the RMSE with respect to different seasons (`posterior_analysis/rmse_functions.RMSE_ensemble_season`), together with the functions to identify the float devices which generates a specific profile (`posterior_analysis/float_identification.single_float_device_identifier` and `posterior_analysis/float_identification.float_device_identifier`). 



### Modifying the Model Architecture
To apply the same architecture for the prediction of other biogeocehmical variables, it could be useful to also modify the model architecture, for example, adding or removing some convolutional layers. The default architectures are located in:
* `CNN_3DMedSea/convolutional_network.py`.