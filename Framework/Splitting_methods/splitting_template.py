import pandas as pd
import numpy as np
import os
import scipy as sp




class splitting_template():
       
    def split_data(self, Input_prediction, Input_path, Input_T,
                   Output_path, Output_T, Output_T_pred, Output_A, Output_T_E,
                   Domain, data_set_save_file):
        
        path = os.path.dirname(data_set_save_file)[:-4]
        file = os.path.basename(data_set_save_file)[:-4]
        
        self.split_file = (path +
                           'Splitting/' + 
                           file +
                           '-' + 
                           self.get_name() + 
                           '.npy')
        
        if os.path.isfile(self.split_file):
            [self.Train_index, 
             self.Test_index,
             sim_all,
             Sim_any, _] = np.load(self.split_file, allow_pickle = True)
        else:
            self.split_data_method(Input_prediction, Input_path, Input_T,
                                   Output_path, Output_T, Output_T_pred, 
                                   Output_A, Output_T_E, Domain)
             
            # check if split was successful
            if not all([hasattr(self, attr) for attr in ['Train_index', 'Test_index']]):
                 raise AttributeError("The splitting into train and test set was unsuccesful")
            
            np.random.shuffle(self.Train_index)
            np.random.shuffle(self.Test_index)
            # Determine similarity
            sim_all, Sim_any = self.get_similarity_method(Input_prediction, Input_path, Input_T,
                                                          Output_path, Output_T, Output_T_pred, 
                                                          Output_A, Output_T_E, Domain)
            
            save_data = np.array([self.Train_index, 
                                  self.Test_index,
                                  sim_all,
                                  Sim_any, 0], object) #0 is there to avoid some numpy load and save errros
            np.save(self.split_file, save_data)
 
        return self.Train_index, self.Test_index, sim_all, Sim_any, self.split_file

        
    
    #########################################################################################
    #########################################################################################
    ###                                                                                   ###
    ###                      Splitting method dependend functions                         ###
    ###                                                                                   ###
    #########################################################################################
    #########################################################################################
    
    
    
    def __init__(self):
        # Important: Only internal matters, no additional inputs possible
        raise AttributeError('Has to be overridden in actual method.')
        

    def split_data_method(self, Input_prediction, Input_path, Input_T,
                          Output_path, Output_T, Output_T_pred, 
                          Output_A, Output_T_E, Domain):
        # this function takes the given input and then creates a 
        # split according to a desied method
        # creates:
            # self.Train_index -    A 1D numpy including the samples IDs of the training set
            # self.Test_index -     A 1D numpy including the samples IDs of the test set
        raise AttributeError('Has to be overridden in actual method.')
        
    
    def get_similarity_method(self, Input_prediction, Input_path, Input_T,
                              Output_path, Output_T, Output_T_pred, 
                              Output_A, Output_T_E, Domain):
        sim_all = 1
        Sim_any = np.ones(len(self.Test_index))
        return sim_all, Sim_any
        
    
    def get_name(self):
        # Returns a string with the name of the class
        # IMPORTANT: No '-' in the name
        raise AttributeError('Has to be overridden in actual method')
        
        
    
        



