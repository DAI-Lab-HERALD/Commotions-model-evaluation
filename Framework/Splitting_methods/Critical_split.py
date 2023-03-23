import pandas as pd
import numpy as np
import os
import importlib
from splitting_template import splitting_template



class Critical_split(splitting_template):
    
    def __init__(self):
        self.test_part = 0.2
        
        
    def split_data_method(self, Input_predictions, Input_path, Input_T,
                          Output_path, Output_T, Output_T_pred, 
                          Output_A, Output_T_E, Domain):
        
        ### REdo
        Index = np.arange(len(Output_A))
        Index_accepted = Index[Output_A]
        Index_rejected = Index[np.invert(Output_A)]
        
        num_test_accepted = int(self.test_part * len(Index_accepted))
        num_test_rejected = int(self.test_part * len(Index_rejected))
        
        # Test: longest reejcted gaps
        Tc_rejected = Output_T_E[Index_rejected]
        Sorted_rejected = np.argsort(- Tc_rejected) 
        Index_rejected_test = Index_rejected[Sorted_rejected][:num_test_rejected]
        Index_rejected_train = Index_rejected[Sorted_rejected][num_test_rejected:]
        
        # Load specific time data:
        path = os.path.dirname(self.split_file)[:-9]
        data_set = os.path.basename(self.split_file)[:-4].split('-')[0]
        
        data_file = path + 'Data/' + data_set + '-time_points.npy'
        
        [Path_id, Path_t, _, Path_tcpre, _, _, 
         _, _, _,  _, _, _, _, Path_ta, _, _] = np.load(data_file, allow_pickle = True)
        
        # Accepted: shortest gap size at acceptance
        Path_id_accepted = Domain.iloc[Index_accepted].Path_ID.to_numpy()
        DT_accepted = np.zeros(len(Index_accepted))
        for i in range(len(Index_accepted)):
            i_old = np.where(Path_id_accepted[i] == Path_id)[0]
            assert len(i_old == 1), "Something went wrong"
            
            # get relevant data
            ta = Path_ta[i_old[0]]
            t = Path_t[i_old[0]]
            tcpre = Path_tcpre[i_old[0]]
            
            # find gap size at ta
            dta = t - ta
            ind = np.where(dta <= 0)[0][-1]
            tcpre_ta = (tcpre[ind] * dta[ind + 1] - tcpre[ind + 1] * dta[ind]) / (dta[ind + 1] - dta[ind])
            
            assert tcpre_ta > ta, "Gap should be open when accepted"
            DT_accepted[i] = tcpre_ta - ta
            
        Sorted_accepted = np.argsort(DT_accepted)
        Index_accepted_test = Index_accepted[Sorted_accepted][:num_test_accepted]
        Index_accepted_train = Index_accepted[Sorted_accepted][num_test_accepted:]
        
        
        self.Train_index= np.sort(np.concatenate((Index_accepted_train, Index_rejected_train), axis = 0))
        self.Test_index= np.sort(np.concatenate((Index_accepted_test, Index_rejected_test), axis = 0))
        
    def get_name(self):
        return 'critical_split'
        
        
    
        



