import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 

class Time_CDF(evaluation_template):
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self):
        t_A_quantile = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        num_samples = len(self.Output_A)
        
        # get true values and predictions
        A_true = np.copy(self.Output_A)
        A_pred = np.copy(self.Output_A_pred)
        
        T_C = np.array([T[-1] for T in self.Output_T])[:, np.newaxis]
        T_A_true = self.Output_T_E[:,np.newaxis]
        T_A_true[~A_true] = T_C[~A_true]
        T_A_pred = np.array(tuple(self.Output_T_E_pred), dtype = float)
        
        # zero A_pred
        Useful = A_pred > 0
        
        # Get predicted cummulative density prediction
        CDF_T  = np.concatenate((np.zeros((num_samples, 1)), T_A_pred, T_C), axis = 1)[Useful]
        CDF_P  = np.array([0.0, *t_A_quantile, 1.0])[np.newaxis, :] * A_pred[Useful, np.newaxis]
        
        # Overwrite CDF_T
        Rep = np.isnan(CDF_T[:,1:-1]).all(-1) 
        CDF_T[Rep, :] = np.linspace(0, 1, len(t_A_quantile) + 2)[np.newaxis] * CDF_T[Rep, -1][:,np.newaxis]
        
        
        Ind_T_A = (CDF_T[:,:-1] < T_A_true[Useful]) & (T_A_true[Useful] <= CDF_T[:,1:])
        assert np.all(Ind_T_A.sum(1) == 1)
        Ind_insert = np.where(Ind_T_A)[1] + 1
        
        # Get error on left side
        # Find insert location
        Eps_index = np.tile(np.arange(len(t_A_quantile) + 4)[np.newaxis,:], (Useful.sum(), 1))
        Eps_old = (Eps_index != Ind_insert[:, np.newaxis]) & (Eps_index != Ind_insert[:, np.newaxis] + 1)
        
        # Expand the time vector
        Eps_T = np.ones((1, len(t_A_quantile) + 4)) * T_A_true[Useful]
        Eps_T[Eps_old] = CDF_T[np.ones_like(CDF_T, dtype = bool)]  
        
        # Expand the probability vector
        Eps_P = np.ones((Useful.sum(), len(t_A_quantile) + 4), dtype = float)
        Eps_P[Eps_old] = CDF_P[np.ones_like(CDF_T, dtype = bool)]
        
        # Fill intermediate probability value
        I = np.arange(Useful.sum())
        
        Eps_P_T_A = (CDF_P[I, Ind_insert - 1] + (T_A_true[Useful, 0] - CDF_T[I, Ind_insert - 1]) *
                     (CDF_P[I, Ind_insert] - CDF_P[I, Ind_insert - 1]) / (CDF_T[I, Ind_insert] - CDF_T[I, Ind_insert - 1]))
        
        Eps_P[I, Ind_insert] = Eps_P_T_A
        Eps_P[I, Ind_insert + 1] = Eps_P_T_A
        
        # Transform to error
        Eps_invert = Eps_index > Ind_insert[:, np.newaxis]
        Eps_P[Eps_invert] = 1 - Eps_P[Eps_invert]
        
        E_L = np.trapz(Eps_P, Eps_T, axis = -1) / T_C[Useful, 0]
        
        # Get error of right half
        Error = (1 - A_pred) * A_true 
        
        # Combine errors
        Error[Useful] += E_L
        
        # Get mean value between left and right
        Error *= 0.5
        
        return [Error.mean(), Error]
    
    def main_result_idx():
        return 0
    
    def get_output_type_class(self = None):
        return 'binary_and_time'
    
    def get_t0_type():
        return 'col'
    
    def get_opt_goal():
        return 'minimize'
    
    def get_name(self):
        return 'CDF_TA'
    
    def requires_preprocessing(self):
        return False
    
    def allows_plot(self):
        return False