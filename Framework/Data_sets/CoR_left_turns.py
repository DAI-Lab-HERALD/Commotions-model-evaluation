import numpy as np
import pandas as pd
from data_set_template import data_set_template
import os
class CoR_left_turns(data_set_template):
   
    def create_path_samples(self): 
        # Load raw data
        self.Data = pd.read_pickle(self.path + '/Data_raw/CoR - Unprotected left turns/CoR_processed.pkl')
        # analize raw dara 
        self.num_samples = len(self.Data)
        self.Path = []
        self.T = []
        self.Domain_old = []
        names = ['V_ego_x', 'V_ego_y', 'V_tar_x', 'V_tar_y']
        # extract raw samples
        for i in range(self.num_samples):
            path = pd.Series(np.empty(len(names), tuple), index = names)
            
            t_index = self.Data.bot_track.iloc[i].index
            t = tuple(self.Data.bot_track.iloc[i].t[t_index])
            path.V_ego_x = tuple(self.Data.bot_track.iloc[i].x[t_index])
            path.V_ego_y = tuple(self.Data.bot_track.iloc[i].y[t_index])
            path.V_tar_x = tuple(self.Data.ego_track.iloc[i].x[t_index])
            path.V_tar_y = tuple(self.Data.ego_track.iloc[i].y[t_index])

            domain = pd.Series(np.ones(1, int) * self.Data.subj_id.iloc[i], index = ['Subj_ID'])
            
            self.Path.append(path)
            self.T.append(t)
            self.Domain_old.append(domain)
        
        self.Path = pd.DataFrame(self.Path)
        self.T = np.array(self.T+[()], tuple)[:-1]
        self.Domain_old = pd.DataFrame(self.Domain_old)
     
    
    def position_states_in_path(self, path, t, domain):
        #################################################
        # vehicle  position is center                   #
        #################################################
        ego_x = np.array(path.V_ego_x)
        tar_x = np.array(path.V_tar_x)
        tar_y = np.array(path.V_tar_y)
        
        lane_width = 3.5
        vehicle_length = 5
        
        in_position = tar_x > - 3 
        
        # Get Dc and Le
        Dc = (-ego_x) - (lane_width + 0.5 * vehicle_length)
        
        Le = np.ones_like(Dc) * 2 * lane_width
        
        
        # get Da and Lt
        X = np.concatenate((tar_x[:,np.newaxis], tar_y[:,np.newaxis]), -1)
        
        X0 = X[:-5]
        X1 = X[5:]
        
        Xm = 0.5 * (X0 + X1)
        DX = (X1 - X0)
        DX[:, 0] = np.sign(DX[:, 0]) * np.maximum(np.abs(DX[:, 0]), 0.01)
        DX = DX / (np.linalg.norm(DX, axis = -1, keepdims = True) + 1e-6)
        DX[np.linalg.norm(DX, axis = -1) < 0.1, 0] = -1
        
        N = Xm[:,1] / (DX[:, 1] + 1e-5 + 3 * 1e-5 * np.sign(DX[:,1]))
        Dx = N * DX[:, 0] 
        Dx = np.concatenate((np.ones(5) * Dx[0], Dx))
        Dx[(Dx < 0) & (tar_y > 0)] = np.max(Dx)
        Dx[tar_y > 0] = np.minimum(Dx[tar_y > 0], (tar_x[tar_y > 0] + 0.5 * lane_width))
        Da = np.sign(tar_y) * np.sqrt(Dx ** 2 + tar_y ** 2)
        
        Lt = np.ones_like(Da) * lane_width
        
        D1 = np.ones_like(Dc) * 500
        
        D2 = np.ones_like(Dc) * 500
        
        D3 = np.ones_like(Dc) * 500
        
        return [in_position, Dc, Da, D1, D2, D3, Le, Lt]
    
    
    
    def path_to_binary_and_time_sample(self, output_path, output_t, domain):
        # Assume that t_A is when vehicle crosses into street,
        # One could also assume reaction as defined by Zgonnikov et al. (tar_ax > 0)
        # but that might be stupid
        tar_y = output_path.V_tar_y
        t = output_t
        
        tar_yh = tar_y[np.invert(np.isnan(tar_y))]
        th = t[np.invert(np.isnan(tar_y))]
        try:
            # assume width of car of 1 m
            # interpolate here 
            ind_2 = np.where(tar_yh < 0)[0][0]
            if ind_2 == 0:
                output_t_e = th[ind_2] - 0.5 * self.dt
                output_a = True
            else:
                ind_1 = ind_2 - 1
                tar_y_1 = tar_yh[ind_1]
                tar_y_2 = tar_yh[ind_2]
                fac = np.abs(tar_y_1) / (np.abs(tar_y_1) + np.abs(tar_y_2))
                output_t_e = th[ind_1] * (1 - fac) + th[ind_2] * fac
                output_a = True
        except:
            output_t_e = None
            output_a  = False
        
        return output_a, output_t_e
    
    def fill_empty_input_path(self):
        # No processing necessary
        pass
    
    def provide_map_drawing(self, domain):
        lines_solid = []
        lines_solid.append(np.array([[-300, -4],[-4, -4],[-4, -300]]))
        lines_solid.append(np.array([[300, -4],[4, -4],[4, -300]]))
        lines_solid.append(np.array([[-300, 4],[-4, 4],[-4, 300]]))
        lines_solid.append(np.array([[300, 4],[4, 4],[4, 300]]))
        
        lines_solid.append(np.array([[-4, -4],[-4, 0],[-6, 0]]))
        lines_solid.append(np.array([[4, 4],[4, 0],[6, 0]]))
        lines_solid.append(np.array([[4, -4],[0, -4],[0, -6]]))
        lines_solid.append(np.array([[-4, 4],[0, 4],[0, 6]]))
        
        lines_dashed = []
        lines_dashed.append(np.array([[0, 6],[0, 300]]))
        lines_dashed.append(np.array([[0, -6],[0, -300]]))
        lines_dashed.append(np.array([[6, 0],[300, 0]]))
        lines_dashed.append(np.array([[-6, 0],[-300, 0]]))
        
        
        return lines_solid, lines_dashed

