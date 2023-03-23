import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from data_set_template import data_set_template
import os
from scipy import interpolate


def rotate_track(track, angle, center):
    Rot_matrix = np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])
    tar_tr = track[['x','y']].to_numpy()
    track[['x','y']] = np.dot(Rot_matrix,(tar_tr - center).T).T
    return track


class RounD_round_about_reduced(data_set_template):   
    def create_path_samples(self): 
        test_file = self.path + '/Results/Data/' + self.get_name()[:-8] + '-processed_paths.npy'
        if os.path.isfile(test_file):
            [Path_full, 
             T_full, 
             Domain_old_full, 
             path_names_full, 
             num_samples_full] = np.load(test_file, allow_pickle = True) 
                
                
        
        self.Path = Path_full[['V_ego_x', 'V_ego_y', 'V_tar_x', 'V_tar_y']]
        self.T = T_full
        self.Domain_old = Domain_old_full
        self.num_samples = num_samples_full
         
        
     
    def extract_time_points(self):
        # here for analyizing the dataset
        self.data_time_file = (self.path + 
                               '/Results/Data/' + 
                               self.get_name() + 
                               '-time_points.npy')
        
        if os.path.isfile(self.data_time_file):
            [self.id,
             self.t,
             self.Dc,
             self.tcpre,
             self.Da,
             self.tapre,
             self.D1,
             self.D2,
             self.D3,
             self.Le,
             self.Lt,
             self.ts,
             self.tc,
             self.ta,
             self.tcrit, _] = np.load(self.data_time_file, allow_pickle = True)
        
        else:
            
            load_data_file = (self.path + 
                              '/Results/Data/' + 
                              self.get_name()[:-8] + 
                              '-time_points.npy')
            [self.id,
             self.t,
             self.Dc,
             self.tcpre,
             self.Da,
             self.tapre,
             self.D1,
             self.D2,
             self.D3,
             self.Le,
             self.Lt,
             self.ts,
             self.tc,
             self.ta,
             self.tcrit, _] = np.load(load_data_file, allow_pickle = True)
            
            for i_sample in range(len(self.id)):
                self.D1[i_sample][:] = 500
                self.D2[i_sample][:] = 500
                self.D3[i_sample][:] = 500
            
            save_data_time = np.array([self.id,
                                       self.t,
                                       self.Dc,
                                       self.tcpre,
                                       self.Da,
                                       self.tapre,
                                       self.D1,
                                       self.D2,
                                       self.D3,
                                       self.Le,
                                       self.Lt,
                                       self.ts,
                                       self.tc,
                                       self.ta,
                                       self.tcrit, 0], object) #0 is there to avoid some numpy load and save errros
            np.save(self.data_time_file, save_data_time)
    
    def path_to_binary_and_time_sample(self, output_path, output_t, domain):
        # Assume that t_A is when vehicle crosses into street,
        # One could also assume reaction as defined by Zgonnikov et al. (tar_ax > 0)
        # but that might be stupid
        tar_x = output_path.V_tar_x
        tar_y = output_path.V_tar_y
        
        t = output_t
        
        R_dic = {0: 25.146, 1: 14.528, 2: 13.633}
        R = R_dic[domain.location]
        
        tar_r = np.sqrt(tar_x ** 2 + tar_y ** 2)
        
        tar_rh = tar_r[np.invert(np.isnan(tar_r))]
        th = t[np.invert(np.isnan(tar_r))]
        try:
            #interpolate here
            ind_2 = np.where(tar_rh < R)[0][0]
            if ind_2 == 0:
                output_t_e = th[ind_2] - 0.5 * self.dt
                output_a = True
            else:
                ind_1 = ind_2 - 1
                tar_r_1 = tar_rh[ind_1] - R
                tar_r_2 = tar_rh[ind_2] - R
                fac = np.abs(tar_r_1) / (np.abs(tar_r_1) + np.abs(tar_r_2))
                output_t_e = th[ind_1] * (1 - fac) + th[ind_2] * fac
                output_a = True
            
        except:
            output_t_e = None
            output_a  = False
        
        return output_a, output_t_e
    
    def fill_empty_input_path(self):
        pass
    
    def provide_map_drawing(self, domain):
        
        R_dic = {0: 25.146, 1: 14.528, 2: 13.633}
        R = R_dic[domain.location]
        
        r_dic = {0: 15.646, 1: 6.679, 2: 6.544}
        r = r_dic[domain.location]
        
        x = np.arange(-1,1,501)[:,np.newaxis]
        unicircle_upper = np.concatenate((x, np.sqrt(1 - x ** 2)), axis = 1)
        unicircle_lower = np.concatenate((- x, - np.sqrt(1 - x ** 2)), axis = 1)
        
        unicircle = np.concatenate((unicircle_upper, unicircle_lower[1:, :]))
        
        lines_solid = []
        lines_solid.append(unicircle * r)
        
        lines_dashed = []
        lines_dashed.append(np.array([[R, 0],[300, 0]]))
        lines_dashed.append(unicircle * R)
        
        
        return lines_solid, lines_dashed
