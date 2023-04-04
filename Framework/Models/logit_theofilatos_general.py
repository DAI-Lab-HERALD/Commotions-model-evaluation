import numpy as np
import pandas as pd
from logit_theofilatos import logit_theofilatos
from sklearn.linear_model import LogisticRegression as LR

class logit_theofilatos_general(logit_theofilatos):
    
    def setup_method(self):
        self.timesteps = max([len(T) for T in self.Input_T_train])
        
        self.Parameter = np.zeros(len(self.Input_prediction_train.columns) * self.timesteps + 1)

    def train_method(self, l2_regulization = 0.01):
        # Multiple timesteps have to be flattened
        L_help = self.Input_prediction_train.to_numpy()
        L = np.ones([L_help.shape[0], L_help.shape[1], self.timesteps]) * np.nan
        for i in range(len(L)):
            for j in range(L_help.shape[1]):
                L[i, j, (self.timesteps - len(L_help[i,j])):] =  L_help[i,j]
        L = L.reshape(L.shape[0], -1)
        
        # Normalize data, so no input can be set to zero
        self.mean = np.nanmean(L, axis = 0, keepdims = True)
        L = L - self.mean
        L[np.isnan(L)] = 0
        
        # get parameters of linear model
        Log_reg = LR(C = 1/(l2_regulization+1e-7), max_iter = 100000)
        param = Log_reg.fit(L, self.Output_A_train)
        self.Parameter = np.concatenate((param.coef_[0], param.intercept_))
        
        self.weights_saved = [self.mean, self.Parameter]
        
    def predict_method(self):
        # Multiple timesteps have to be flattened
        L_help = self.Input_prediction_test.to_numpy()
        L = np.ones([L_help.shape[0], L_help.shape[1], self.timesteps]) * np.nan
        for i in range(len(L)):
            for j in range(L_help.shape[1]):
                L[i, j, max(0,(self.timesteps - len(L_help[i,j]))):] =  L_help[i,j][max(0,(len(L_help[i,j]) - self.timesteps)):]
        L = L.reshape(L.shape[0], -1)
        
        # Normalize data, so no input can be set to zero
        L = L - self.mean
        # set missing values to zero
        L[np.isnan(L)] = 0
        
        
        Linear = np.sum(np.array(L * self.Parameter[:-1]), axis=1) + self.Parameter[-1]
        #prevent overflow
        Prob = np.zeros_like(Linear)
        pos = Linear > 0
        neg = np.invert(pos)
        Prob[pos] = 1 / (1 + np.exp(-Linear[pos]))
        Prob[neg] = np.exp(Linear[neg]) / (np.exp(Linear[neg]) + 1)
        return [Prob]

    def check_input_names_method(self, names, train = True):
        return True
     
    def get_input_type_class():
        return 'general'
    
    def get_name(self):
        return 'logit_1D'
        

        
        
        
        
        
    
        
        
        