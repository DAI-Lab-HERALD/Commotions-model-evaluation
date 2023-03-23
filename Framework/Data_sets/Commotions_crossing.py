import numpy as np
import pandas as pd
from data_set_template import data_set_template
import os

def rotate_track(track, angle, center):
    Rot_matrix = np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])
    tar_tr = track[['x','y']].to_numpy()
    track[['x','y']] = np.dot(Rot_matrix,(tar_tr - center).T).T
    return track


class Commotions_crossing(data_set_template):
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
                 'V_tar_x', 'V_tar_y', 
                 'V_v_1_x', 'V_v_1_y', 
                 'V_v_2_x', 'V_v_2_y']
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
                    
                    
                    if ego_track.leaderID.iloc[0] == -1:
                        v_1_x = tuple(np.ones(len(ego_track.index)) * np.nan)
                        v_1_y = tuple(np.ones(len(ego_track.index)) * np.nan)
                    
                    else:
                        v_1_name = drones[ego_track.leaderID.iloc[0]] 
                        v_1_track = self.Data.iloc[i][v_1_name].copy(deep = True)
                        v_1_track = rotate_track(v_1_track, -rot_angle, np.zeros((1,2)))
                        v_1_track = rotate_track(v_1_track, 0, center)
                        
                        v_1_x = np.ones(len(ego_track.index)) * np.nan
                        v_1_y = np.ones(len(ego_track.index)) * np.nan
                        
                        frame_ego_min = ego_track.index[0]
                        frame_ego_max = ego_track.index[-1]
                        
                        frame_v_1_min = v_1_track.index[0]
                        frame_v_1_max = v_1_track.index[-1]
                        
                        v_1_x[max(0, frame_v_1_min - frame_ego_min) : 
                              min(len(v_1_x), 1 + frame_v_1_max - frame_ego_min)] = v_1_track.x.loc[max(frame_ego_min, frame_v_1_min):
                                                                                                    min(frame_ego_max, frame_v_1_max)]
                        v_1_y[max(0, frame_v_1_min - frame_ego_min) : 
                              min(len(v_1_x), 1 + frame_v_1_max - frame_ego_min)] = v_1_track.y.loc[max(frame_ego_min, frame_v_1_min):
                                                                                                    min(frame_ego_max, frame_v_1_max)]
                        
                        v_1_x = tuple(v_1_x)
                        v_1_y = tuple(v_1_y)
                        
                        
                    if ego_track.followerID.iloc[0] == -1:
                        v_2_x = tuple(np.ones(len(ego_track.index)) * np.nan)
                        v_2_y = tuple(np.ones(len(ego_track.index)) * np.nan)
                        
                    else:
                        v_2_name = drones[ego_track.followerID.iloc[0]] 
                        v_2_track = self.Data.iloc[i][v_2_name].copy(deep = True)
                        v_2_track = rotate_track(v_2_track, -rot_angle, np.zeros((1,2)))
                        v_2_track = rotate_track(v_2_track, 0, center)
                        
                        v_2_x = np.ones(len(ego_track.index)) * np.nan
                        v_2_y = np.ones(len(ego_track.index)) * np.nan
                        
                        frame_ego_min = ego_track.index[0]
                        frame_ego_max = ego_track.index[-1]
                        
                        frame_v_2_min = v_2_track.index[0]
                        frame_v_2_max = v_2_track.index[-1]
                        
                        v_2_x[max(0, frame_v_2_min - frame_ego_min) : 
                              min(len(v_2_x), 1 + frame_v_2_max - frame_ego_min)] = v_2_track.x.loc[max(frame_ego_min, frame_v_2_min):
                                                                                                    min(frame_ego_max, frame_v_2_max)]
                        v_2_y[max(0, frame_v_2_min - frame_ego_min) : 
                              min(len(v_2_x), 1 + frame_v_2_max - frame_ego_min)] = v_2_track.y.loc[max(frame_ego_min, frame_v_2_min):
                                                                                                    min(frame_ego_max, frame_v_2_max)]
                        
                        v_2_x = tuple(v_2_x)
                        v_2_y = tuple(v_2_y)
                    
                    path.V_ego_x = tuple(ego_track.x)
                    path.V_tar_x = tuple(tar_track.x)
                    path.V_v_1_x = v_1_x
                    path.V_v_2_x = v_2_x
                    if Driving_left:
                        path.V_ego_y = tuple(ego_track.y)
                        path.V_tar_y = tuple(tar_track.y)
                        path.V_v_1_y = v_1_y
                        path.V_v_2_y = v_2_y
                        domain = pd.Series(np.array([data_i.participant, 'left']), index = ['Subj_ID', 'Driving'])
                    else:
                        path.V_ego_y = tuple(-ego_track.y)
                        path.V_tar_y = tuple(-tar_track.y)
                        path.V_v_1_y = tuple(-np.array(v_1_y))
                        path.V_v_2_y = tuple(-np.array(v_2_y))
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
        v_1_x = np.array(path.V_v_1_x)
        v_2_x = np.array(path.V_v_2_x)
        
        if domain.Driving == 'left':
            ego_y = np.array(path.V_ego_y)
            tar_y = np.array(path.V_tar_y)
            v_1_y = np.array(path.V_v_1_y)
            v_2_y = np.array(path.V_v_2_y)
        else:
            ego_y = - np.array(path.V_ego_y)
            tar_y = - np.array(path.V_tar_y)
            v_1_y = - np.array(path.V_v_1_y)
            v_2_y = - np.array(path.V_v_2_y)
        
        lane_width = 3.65
        vehicle_length = 5
        
        Dc = (-ego_y) - (lane_width + 1 + 0.5 * vehicle_length) # 1 m for traffic island
        Da = tar_x - 0.5 * vehicle_length
        
        D1 = v_1_y - ego_y - vehicle_length
        D1_good = np.isfinite(D1)
        if not all(D1_good):
            if any(D1_good):
                index = np.arange(len(D1))
                D1 = np.interp(index, index[D1_good], D1[D1_good], left = D1[D1_good][0], right = D1[D1_good][-1])
            else:
                D1 = np.ones(len(Dc)) * 1000
        
        D2 = ego_y - v_2_y - vehicle_length
        D2_good = np.isfinite(D2)
        if not all(D2_good):
            if any(D2_good):
                index = np.arange(len(D2))
                D2 = np.interp(index, index[D2_good], D2[D2_good], left = D2[D2_good][0], right = D2[D2_good][-1])
            else:
                D2 = np.ones(len(Dc)) * 1000
        D3 = np.ones(len(Dc)) * 1000
        
        Le = np.ones_like(Dc) * (lane_width + 1)
        Lt = np.ones_like(Dc) * lane_width
        
        in_position = (-(lane_width + 1) < tar_y) & (tar_y < 0)
        
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
        for i_sample in range(len(self.Input_path)):
            path = self.Input_path.iloc[i_sample]
            t = self.Input_T[i_sample]
            domain = self.Domain.iloc[i_sample]
            
            # check vehicle v_1 (in front of ego)
            v_1_x = path.V_v_1_x
            v_1_x_good = np.isfinite(v_1_x)
            if not all(v_1_x_good):
                if any(v_1_x_good):
                    D1 = path.V_v_1_y - path.V_ego_y
                    index = np.arange(len(D1))
                    D1 = np.interp(index, index[v_1_x_good], D1[v_1_x_good], left = D1[v_1_x_good][0], right = D1[v_1_x_good][-1])
                    v_1_y = path.V_ego_y + D1
                    v_1_x = np.interp(index, index[v_1_x_good], v_1_x[v_1_x_good], left = v_1_x[v_1_x_good][0], right = v_1_x[v_1_x_good][-1])
                else:
                    v_1_x = path.V_ego_x
                    v_1_y = path.V_ego_y + 1000
                    
                path.V_v_1_x = v_1_x
                path.V_v_1_y = v_1_y
            
            # check vehicle v_2 (behind ego)
            v_2_x = path.V_v_2_x
            v_2_x_good = np.isfinite(v_2_x)
            if not all(v_2_x_good):
                if any(v_2_x_good):
                    D2 = path.V_v_2_y - path.V_ego_y
                    index = np.arange(len(D1))
                    D2 = np.interp(index, index[v_2_x_good], D2[v_2_x_good], left = D2[v_2_x_good][0], right = D2[v_2_x_good][-1])
                    v_2_y = path.V_ego_y + D2
                    v_2_x = np.interp(index, index[v_2_x_good], v_2_x[v_2_x_good], left = v_2_x[v_2_x_good][0], right = v_2_x[v_2_x_good][-1])
                else:
                    v_2_x = path.V_ego_x
                    v_2_y = path.V_ego_y - 1000
                    
                path.V_v_2_x = v_2_x
                path.V_v_2_y = v_2_y
            
            self.Input_path.iloc[i_sample] = path
    
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

