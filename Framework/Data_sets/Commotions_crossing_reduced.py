import numpy as np
import pandas as pd
from data_set_template import data_set_template
import os
import importlib

def rotate_track(track, angle, center):
    Rot_matrix = np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])
    tar_tr = track[['x','y']].to_numpy()
    track[['x','y']] = np.dot(Rot_matrix,(tar_tr - center).T).T
    return track


class Commotions_crossing_reduced(data_set_template):
    def create_path_samples(self): 
        # Load raw data
        self.Data = pd.read_pickle(self.path + '/Data_raw/Commotions - Crossing/Commotions_processed.pkl')
        # analize raw dara 
        num_tars = len(self.Data)
        self.num_samples = 0 
        self.Path = []
        self.T = []
        self.Domain_old = []
        # v_1 is the vehicle in front of the ego vehicle
        # v_2 is the vehicle behind the ego vehicle
        names = ['V_ego_x', 'V_ego_y', 
                 'V_tar_x', 'V_tar_y',]
        # extract raw samples
        # If false, set every y value to zero
        Driving_left = True
        
        for i in range(num_tars):
            data_i = self.Data.iloc[i]
            # find crossing point
            tar_track_all = data_i.participant_track.copy(deep = True)
            # target vehicle is to come along x axis, towards origin
            # find crossing point
            drones = np.sort(self.Data.columns[2:])
            drones = [drone for drone in drones if type(self.Data.iloc[i][drone]) == type(self.Data)]
            if len(drones) > 0:
                test_track = self.Data.iloc[i][drones[0]].copy(deep = True)
                
                tar_arr  = tar_track_all.to_numpy()[np.arange(0,len(tar_track_all), 12), 1:3]
                test_arr = test_track.to_numpy()[np.arange(0,len(test_track), 12), 1:3]
                
                dist = np.sum((tar_arr[:,np.newaxis,:] - test_arr[np.newaxis,:,:]) ** 2, -1)
                _, test_ind = np.unravel_index(np.argmin(dist), dist.shape)
                
                test_dx = test_arr[test_ind] - test_arr[test_ind - 1]
                
                test_i = test_ind * 12
                
                angle = np.angle(test_dx[0] + 1j * test_dx[1])
                angle_des = np.pi / 2
                
                rot_angle = (angle_des - angle)
                
                tar_track_all  = rotate_track(tar_track_all, - rot_angle, np.zeros((1,2)))
                test_track = rotate_track(test_track, - rot_angle, np.zeros((1,2)))
                
                x_center = test_track.iloc[test_i].x + np.abs(test_track.offset.iloc[test_i])
                
                ind = test_track.index[test_i]
                
                y_center = tar_track_all.loc[ind].y + np.abs(tar_track_all.offset.loc[ind])
                
                center = np.array([[x_center, y_center]])
                
                tar_track_all = rotate_track(tar_track_all, 0, center)
                
                
                # lane_width = 3.65 m
                
                for j, drone in enumerate(drones):
                    ego_track = self.Data.iloc[i][drone].copy(deep = True)
                    ego_track = rotate_track(ego_track, -rot_angle, np.zeros((1,2)))
                    ego_track = rotate_track(ego_track, 0, center)
                    
                    tar_track = tar_track_all.loc[ego_track.index].copy(deep = True)
                    t = tuple(tar_track.t)
                    path = pd.Series(np.empty(len(names), tuple), index = names)
                    
                    path.V_ego_x = tuple(ego_track.x)
                    path.V_tar_x = tuple(tar_track.x)
                    if Driving_left:
                        path.V_ego_y = tuple(ego_track.y)
                        path.V_tar_y = tuple(tar_track.y)
                        domain = pd.Series(np.array([data_i.participant, 'left']), index = ['Subj_ID', 'Driving'])
                    else:
                        path.V_ego_y = tuple(-ego_track.y)
                        path.V_tar_y = tuple(-tar_track.y)
                        domain = pd.Series(np.array([data_i.participant, 'right']), index = ['Subj_ID', 'Driving'])
                        
                    self.Path.append(path)
                    self.T.append(t)
                    self.Domain_old.append(domain)
                    self.num_samples = self.num_samples + 1
        
        self.Path = pd.DataFrame(self.Path)
        self.T = np.array(self.T+[()], tuple)[:-1]
        self.Domain_old = pd.DataFrame(self.Domain_old)
     
    
    def position_states_in_path(self, path, t, domain):
        #################################################
        # vehicle  position is center                   #
        #################################################
        ego_x = np.array(path.V_ego_x)
        tar_x = np.array(path.V_tar_x)
        
        if domain.Driving == 'left':
            ego_y = np.array(path.V_ego_y)
            tar_y = np.array(path.V_tar_y)
        else:
            ego_y = - np.array(path.V_ego_y)
            tar_y = - np.array(path.V_tar_y)
        
        lane_width = 3.65
        vehicle_length = 5
        
        help_file = self.path + '/Results/Data/Commotions_crossing-time_points.npy'
        if not os.path.isfile(help_file):
            data_set_name = 'Commotions_crossing'
            data_set_module = importlib.import_module(data_set_name)
            data_set_class = getattr(data_set_module, data_set_name)
            data_set = data_set_class(None, 100)
            data_set.extract_time_points()
        
        helper = np.load(help_file, allow_pickle = True)
        path_id = path.name
        try:
            ts = helper[11][np.where(path.name == helper[0])[0][0]]
        
            in_position = (-(lane_width + 1) < tar_y) & (tar_y < 0) & (t >= ts)
        except:
            in_position = np.zeros(len(t), bool)
            
        
        Dc = (-ego_y) - (lane_width + 1 + 0.5 * vehicle_length) # 1 m for traffic island
        Da = tar_x - 0.5 * vehicle_length
        
        D1 = np.ones(len(Dc)) * 1000
        
        D2 = np.ones(len(Dc)) * 1000
        
        D3 = np.ones(len(Dc)) * 1000
        
        Le = np.ones_like(Dc) * (lane_width + 1)
        Lt = np.ones_like(Dc) * lane_width
        
        
        return [in_position, Dc, Da, D1, D2, D3, Le, Lt]
    
    
    
    def path_to_binary_and_time_sample(self, output_path, output_t, domain):
        # Assume that t_A is when vehicle crosses into street,
        # One could also assume reaction as defined by Zgonnikov et al. (tar_ax > 0)
        # but that might be stupid
        tar_x = output_path.V_tar_x
        t = output_t
        
        lane_width = 3.65
        vehicle_length = 5
        
        tar_xh = tar_x[np.isfinite(tar_x)]
        th = t[np.isfinite(tar_x)]
        try:
            # assume width of car of 1 m
            # interpolate here 
            ind_2 = np.where(tar_xh - 0.5 * vehicle_length < 0)[0][0]
            if ind_2 == 0:
                output_t_e = th[ind_2] - 0.5 * self.dt
                output_a = True
            else:
                ind_1 = ind_2 - 1
                tar_y_1 = tar_xh[ind_1] - 0.5 * vehicle_length
                tar_y_2 = tar_xh[ind_2] - 0.5 * vehicle_length
                fac = np.abs(tar_y_1) / (np.abs(tar_y_1) + np.abs(tar_y_2))
                output_t_e = th[ind_1] * (1 - fac) + th[ind_2] * fac
                output_a = True
        except:
            output_t_e = None
            output_a  = False
        
        return output_a, output_t_e
    
    def fill_empty_input_path(self):
        pass
    
    def provide_map_drawing(self, domain):
        lines_solid = []
        lines_solid.append(np.array([[-300, -4],[-4, -4],[-4, -300]]))
        lines_solid.append(np.array([[300, -4],[4, -4],[4, -300]]))
        lines_solid.append(np.array([[-300, 4],[-4, 4],[-4, 300]]))
        lines_solid.append(np.array([[300, 4],[4, 4],[4, 300]]))
        
        lines_dashed = []
        lines_dashed.append(np.array([[0, -300],[0, 300]]))
        lines_dashed.append(np.array([[4, -4],[4, 0],[300, 0]]))
        lines_dashed.append(np.array([[-4, 4],[-4, 0],[-300,0]]))
        
        return lines_solid, lines_dashed

