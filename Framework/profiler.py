import numpy as np
import pandas as pd
import os
import importlib
import sys

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

# save for profiling
import builtins
try:
    builtins.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func): return func
    builtins.profile = profile

# Enforce synchronious cuda execution
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# NOTE: Here still local path is assumed, so it might be useful to adapt that later
path = '/'.join(os.path.dirname(__file__).split('\\'))

data_set_path = os.path.join(path, 'Data_sets', '')
if not data_set_path in sys.path:
    sys.path.insert(0, data_set_path)

metrics_path = os.path.join(path, 'Evaluation_metrics', '')
if not metrics_path in sys.path:
    sys.path.insert(0, metrics_path)
       
split_path = os.path.join(path, 'Splitting_methods', '')
if not split_path in sys.path:
    sys.path.insert(0, split_path)

model_path = os.path.join(path, 'Models', '')
if not model_path in sys.path:
    sys.path.insert(0, model_path)
    
##### Select modules
n_I_max = 8

Data_sets = ['CoR_left_turns']
Metrics = ['ROC_curve_col']
Data_params = [{'dt': 0.2, 'num_timesteps_in': (2, n_I_max)}] 
Splitters = ['Random_split']
Models = ['commotions_markkula_prof']

##### Unavailable combinations
Not_allowed_combinations = [['commotions_markkula', {'dt': 0.2, 'num_timesteps_in': (3, n_I_max)}],
                            ['commotions_markkula', {'dt': 0.2, 'num_timesteps_in': n_I_max}],
                            ['commotions_markkula', 'Commotions_crossing'],]


##### Select method for D.T
model_to_path_name = 'trajectron_salzmann'
model_to_path_module = importlib.import_module(model_to_path_name)
model_class_to_path = getattr(model_to_path_module, model_to_path_name)

##### Prepare run
Results = np.empty((len(Data_sets), len(Metrics), len(Data_params), len(Splitters), len(Models)),
                   object)

# Number of different paths predicted by trajectory prediction models.
num_samples_path_pred = 100

# Plotting results if the metric allows it (e.g., ROC curves)
create_plot_if_possible = False

# Deciding wether to allways use strict metrics
always_strict_prediction_times = True

for i, data_set_name in enumerate(Data_sets):
    # Get data set
    data_set_module = importlib.import_module(data_set_name)
    data_set_class = getattr(data_set_module, data_set_name)
    data_set = data_set_class(model_class_to_path = model_class_to_path, 
                              num_samples_path_pred = num_samples_path_pred)
    
    
    for j, metric_name in enumerate(Metrics):
        # Get metric used
        metric_module = importlib.import_module(metric_name)
        metric_class = getattr(metric_module, metric_name)
        # get the required output type for metric
        metric_type = metric_class.get_output_type_class()
        # Get the required t0 type
        metric_t0_type = metric_class.get_t0_type()
        
        for k, data_param in enumerate(Data_params):
            # Enforce strict prediction times if necessary
            if always_strict_prediction_times and metric_t0_type[-6:] != 'strict':
                if (data_set_name == 'CoR_left_turns' and 
                    metric_t0_type == 'start' and 
                    data_param['num_timesteps_in'][0] == 2):
                    import copy
                    data_param = copy.deepcopy(data_param)
                    data_param['num_timesteps_in'] = (2,2)
                else:
                    metric_t0_type = metric_t0_type + '_strict'
            
            data_set.reset()
            # Select training data
            data = data_set.get_data(t0_type = metric_t0_type,
                                     exclude_post_crit = True,
                                     **data_param)
            
            # Check if a usable dataset was produced
            if data != None:
                [Input_prediction, Input_path, Input_T, 
                 Output_path, Output_T, Output_T_pred, Output_A, Output_T_E,
                 Domain, data_set_save_file] = data
                
                for l, splitter_name in enumerate(Splitters):
                    # Get data set
                    splitter_module = importlib.import_module(splitter_name)
                    splitter_class = getattr(splitter_module, splitter_name)
                    splitter = splitter_class()
                    [Train_index, Test_index, 
                     Test_sim_all, Test_Sim_any,
                     splitter_save_file] = splitter.split_data(Input_prediction, Input_path, Input_T,
                                                               Output_path, Output_T, Output_T_pred, 
                                                               Output_A, Output_T_E,
                                                               Domain, data_set_save_file)
                                                               
                    # initialize estimator output
                    # all possible output data used to allow for calculation of
                    # more complex weighting methods
                    metric = metric_class(Output_path           = Output_path.iloc[Test_index], 
                                          Output_T              = Output_T[Test_index], 
                                          Output_A              = Output_A[Test_index], 
                                          Output_T_E            = Output_T_E[Test_index], 
                                          Domain                = Domain.iloc[Test_index],
                                          splitter_save_file    = splitter_save_file)
                    
                     
                    for m, model_name in enumerate(Models):
                        # Print current state
                        print('')
                        print('Applying data-set ' + 
                              format(i + 1, '.0f').rjust(len(str(len(Data_sets)))) +
                              '/{}'.format(len(Data_sets)) + 
                              ' and metric ' + 
                              format(j + 1, '.0f').rjust(len(str(len(Metrics)))) +
                              '/{}'.format(len(Metrics)) + 
                              ' and input params ' + 
                              format(k + 1, '.0f').rjust(len(str(len(Data_params)))) +
                              '/{}'.format(len(Data_params)) +
                              ' and split ' + 
                              format(l + 1, '.0f').rjust(len(str(len(Splitters)))) +
                              '/{}'.format(len(Splitters)) +
                              ' to model ' + 
                              format(m + 1, '.0f').rjust(len(str(len(Models)))) +
                              '/{}'.format(len(Models)))
                        
                        model_module = importlib.import_module(model_name)
                        model_class = getattr(model_module, model_name)        
                        model_type = model_class.get_output_type_class()
                        
                        
                        # Check if this combinations should be applied
                        names = [data_set_name, metric_name, data_param, splitter_name, model_name]
                        names_not_allowed = False
                        for not_allowed_combination in Not_allowed_combinations:
                            if all(not_allowed_element in names for not_allowed_element in not_allowed_combination):
                                names_not_allowed = True
                        
                        if not names_not_allowed:
                            model = model_class(Input_prediction_train  = Input_prediction.iloc[Train_index],
                                                Input_path_train        = Input_path.iloc[Train_index],
                                                Input_T_train           = Input_T[Train_index],
                                                Output_path_train       = Output_path.iloc[Train_index], 
                                                Output_T_train          = Output_T[Train_index],  
                                                Output_T_pred_train     = Output_T_pred[Train_index],
                                                Output_A_train          = Output_A[Train_index], 
                                                Output_T_E_train        = Output_T_E[Train_index], 
                                                Domain_train            = Domain.iloc[Train_index], 
                                                splitter_save_file      = splitter_save_file,
                                                num_samples_path_pred   = num_samples_path_pred)
                            
                            model_save_file = model.train()