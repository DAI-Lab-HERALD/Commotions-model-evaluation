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


class RounD_round_about(data_set_template):   
    def create_path_samples(self): 
        # Load raw data
        self.Data = pd.read_pickle(self.path + '/Data_raw/RounD - Round about/RounD_processed.pkl')
        # analize raw dara 
        self.Data = self.Data.reset_index(drop = True)
        num_samples_max = len(self.Data)
        self.Path = []
        self.T = []
        self.Domain_old = []
        # v_1 is the vehicle in front of the ego vehicle
        # v_2 is the vehicle behind the ego vehicle
        # v_3 is the vehicle in front of the target vehicle waiting to move onto the round about
        # v_4 is the pedestrian closest to the contested area
        names = ['V_ego_x', 'V_ego_y',
                 'V_tar_x', 'V_tar_y',
                 'V_v_1_x', 'V_v_1_y',
                 'V_v_2_x', 'V_v_2_y',
                 'V_v_3_x', 'V_v_3_y',
                 'P_v_4_x', 'P_v_4_y']
        
        
        
        
        map_info = self.Data[['locationId','recordingId']].to_numpy()
        
        unique_map = np.unique(map_info, axis = 0)
        unique_map = unique_map[np.unique(unique_map[:,0], return_index = True)[1]]
        
        Loc_data_pix = pd.DataFrame(np.zeros((len(unique_map),4),float), columns = ['xCenter', 'yCenter', 'r', 'R'])
        
        Loc_data_pix.xCenter.iloc[0] = 801.0
        Loc_data_pix.yCenter.iloc[0] = -465.0
        Loc_data_pix.R.iloc[0] = 247.5
        Loc_data_pix.r.iloc[0] = 154.0
        
        Loc_data_pix.xCenter.iloc[1] = 781.0
        Loc_data_pix.yCenter.iloc[1] = -481.0
        Loc_data_pix.R.iloc[1] = 98.1
        Loc_data_pix.r.iloc[1] = 45.1
        
        Loc_data_pix.xCenter.iloc[2] = 1011.0
        Loc_data_pix.yCenter.iloc[2] = -449.0
        Loc_data_pix.R.iloc[2] = 100.0
        Loc_data_pix.r.iloc[2] = 48.0
        
        
        Loc_data = Loc_data_pix.copy(deep=True)
        for [locId, recId] in unique_map:
            Meta_data=pd.read_csv(self.path + '/Data_raw/RounD - Round about/data/{}_recordingMeta.csv'.format(str(recId).zfill(2)))
            orthoPxToMeter=Meta_data['orthoPxToMeter'][0] * 10
            Loc_data.iloc[locId] = Loc_data_pix.iloc[locId] * orthoPxToMeter
            
        # extract raw samples
        self.num_samples = 0
        for i in range(num_samples_max):
            # to keep track:
            if np.mod(i,100) == 0:
                print('trajectory ' + str(i).rjust(len(str(num_samples_max))) + '/{} analized'.format(num_samples_max))
                print('found cases: ' + str(self.num_samples))
                print('')
            data_i = self.Data.iloc[i]
            # assume i is the tar vehicle, which has to be a motor vehicle
            if data_i['class'] in ['bicycle', 'pedestrian', 'trailer']:
                continue
            
            tar_track = data_i.track[['frame','xCenter','yCenter']].rename(columns={"xCenter": "x", "yCenter": "y"}).copy(deep = True)
            Rot_center = np.array([[Loc_data.iloc[data_i.locationId].xCenter, Loc_data.iloc[data_i.locationId].yCenter]])
            
            tar_track['r'] = np.sqrt((tar_track.x - Loc_data.iloc[data_i.locationId].xCenter) ** 2 + 
                                     (tar_track.y - Loc_data.iloc[data_i.locationId].yCenter) ** 2)
            
            # exclude trajectory driving over the middle
            if any(tar_track['r'] < Loc_data.iloc[data_i.locationId].r):
                continue
            
            # check if tar_track goes through round_about or use shortcut around it
            if not any(tar_track['r'] <= Loc_data.iloc[data_i.locationId].R):
                continue
            
            # Exclude vehicles that already startinside the round about
            if tar_track['r'].iloc[0] <= Loc_data.iloc[data_i.locationId].R + 10:
                continue
            
            # frame where target vehicle approaches roundd about
            frame_entry = np.where(tar_track['r'] < Loc_data.iloc[data_i.locationId].R + 10)[0][0]
            
            tar_frame_A = tar_track['frame'].iloc[np.where(tar_track['r'] < Loc_data.iloc[data_i.locationId].R)[0][0]]

            # angle along this route
            original_angle = np.angle((tar_track.x.iloc[0] - tar_track.x.iloc[frame_entry]) + 
                                       (tar_track.y.iloc[0] - tar_track.y.iloc[frame_entry]) * 1j,deg = False)
            
            
            tar_track = rotate_track(tar_track, original_angle, Rot_center)
            
            tar_track['angle'] = np.angle(tar_track.x + tar_track.y * 1j)
            
            tar_track = tar_track.set_index('frame')
            
            other_agents = self.Data[['trackId','class','track']].iloc[data_i.otherVehicles].copy(deep = True)
            
            for j in range(len(other_agents)):
                track_i = other_agents['track'].iloc[j] 
                
                track_i = track_i[['frame','xCenter','yCenter']].rename(columns={"xCenter": "x", "yCenter": "y"}).copy(deep = True)
                
                track_i = rotate_track(track_i, original_angle, Rot_center)
                
                track_i = track_i.set_index('frame').loc[tar_track.index[0]: tar_track.index[-1]]
                
                other_agents['track'].iloc[j] = track_i
                
                other_agents['track'].iloc[j]['r'] = np.sqrt(track_i.x ** 2 + track_i.y ** 2)
                
                other_agents['track'].iloc[j]['angle'] = np.angle(track_i.x + track_i.y * 1j)
            
            # Looking for ego_vehicle. Two conditions:
            # - Actually cross (if ego vehicle leaves before, it has no need for predictions)
            # - Gap is either rejected
            
            rejected_Ego = []
            rejected_frame_C = []
            accepted_Ego = []
            accepted_frame_C = []
            
            for j in range(len(other_agents)):
                tr_j = other_agents['track'].iloc[j] 
                
                if other_agents['class'].iloc[j] in ['bicycle', 'pedestrian', 'trailer']:
                    continue
                # Check if ego vehicle is there to have offered accepted gap
                if tr_j.index[0] > tar_frame_A:
                    continue
                
                contested = ((tr_j.r.to_numpy() <= Loc_data.iloc[data_i.locationId].R) &
                             (tr_j.angle.to_numpy() > 0) & 
                             (tr_j.angle.to_numpy() < np.pi / 6))
                K = np.where(contested[1:] & (contested[:-1] == False))[0] + 1
                
                for k in K:
                    frame_C = tr_j.index[0] + k
                    if tr_j.r.to_numpy()[k - 1] > Loc_data.iloc[data_i.locationId].R:
                        continue
                    
                    # Check if target vehicle was there to have closed the gap
                    if tar_track.index[0] > frame_C:
                        continue
                    
                    if frame_C <= tar_frame_A:
                        rejected_Ego.append(other_agents.trackId.iloc[j])
                        rejected_frame_C.append(frame_C)
                    else:
                        accepted_Ego.append(other_agents.trackId.iloc[j])
                        accepted_frame_C.append(frame_C)
            
            rejected_order = np.argsort(rejected_frame_C)
            rejected_Ego = np.array(rejected_Ego)[rejected_order]  
            rejected_frame_C = np.array(rejected_frame_C)[rejected_order] 
            
            other_ped = other_agents[np.logical_or(other_agents['class'] == 'pedestrian', other_agents['class'] == 'bicycle')]
            
            entered_RA = []
            in_RA = []
            
            for j in range(len(other_agents)):
                tr_j = other_agents['track'].iloc[j] 
                interested = ((tr_j.r.to_numpy() <= Loc_data.iloc[data_i.locationId].R) &
                              (tr_j.angle.to_numpy() > - np.pi / 6) & 
                              (tr_j.angle.to_numpy() < np.pi / 6))
                K = np.where(interested[1:] & (interested[:-1] == False))[0] + 1
                if interested[0]:
                    K = np.concatenate(([0], K))
                
                for k in K:
                    if k == 0:
                        frame_E = tr_j.index[0] + k
                        # Decide if vehicle came into roundabout
                        dx = tr_j.x.iloc[min(5, len(tr_j.x) - 1)] - tr_j.x.iloc[0]
                        dy = tr_j.y.iloc[min(5, len(tr_j.x) - 1)] - tr_j.y.iloc[0]
                        
                        dangle = np.angle(dx + dy * 1j) - 0.5 * np.pi - tr_j.angle.iloc[0]
              
                        if dangle > 0.2 and tr_j.angle.iloc[0] > 0:
                            entered_RA.append([other_agents.trackId.iloc[j], frame_E])
                        else:
                            in_RA.append([other_agents.trackId.iloc[j], frame_E])
                    else:
                        if tr_j.r.to_numpy()[k - 1] > Loc_data.iloc[data_i.locationId].R and tr_j.angle.to_numpy()[k - 1] > 0:
                            # moved into round about
                            frame_A = tr_j.index[0] + k
                            entered_RA.append([other_agents.trackId.iloc[j], frame_A])
                        else:
                            # time of exiting the round about
                            frame_E = tr_j.index[0] + k
                            
                            in_RA.append([other_agents.trackId.iloc[j], frame_E])
                
            entered_RA = np.array(entered_RA)
            in_RA = np.array(in_RA)
            
            if len(in_RA) > 1:
                in_order = np.argsort(in_RA[:,1])
                in_RA = in_RA[in_order,:] 
            
            # Assume accepted gap
            if len(accepted_Ego) > 0:
                ego_id = accepted_Ego[np.argmin(accepted_frame_C)]
                frame_C = np.min(accepted_frame_C)
                
                ego_track = other_agents.loc[ego_id].track.copy(deep = True)
                
                frame_min = max(ego_track.index[0], tar_track.index[0])
                frame_max = min(ego_track.index[-1], tar_track.index[-1])
                
                # find v_1: in_RA directly before ego vehicle
                in_ego = np.where(in_RA[:,0] == ego_id)[0][0]
                
                if in_ego == 0:
                    v_1_id = -1
                else:
                    v_1_id = in_RA[in_ego - 1, 0]
                    
                # find v_2: in_RA directly after ego vehicle
                try:
                    v_2_id = in_RA[in_ego + 1, 0]
                except:
                    v_2_id = -1
                
                # find v_3: entered_RA with the largest tar_frame_A that is still smaller than tar_frame_A
                if len(entered_RA) > 0:
                    feasible = entered_RA[:,1] < tar_frame_A
                else:
                    feasible = [False]
                if np.any(feasible):
                    v_3_id = entered_RA[feasible,0][np.argmax(entered_RA[feasible,1])] 
                else: 
                    v_3_id = -1
                # check for pedestrian 
                if len(other_ped) > 0:
                    distance = []
                    for j in range(len(other_ped)):
                        track_p = other_ped.iloc[j].track.loc[frame_min:frame_max]
                        tar_track_help = tar_track.copy(deep = True)
                        distance_to_cross = np.sqrt((track_p.x - Loc_data.iloc[data_i.locationId].R - 5) ** 2 +
                                                    (track_p.y - tar_track_help.iloc[frame_entry].y) ** 2)
                        distance_to_tar = np.sqrt((track_p.x - tar_track_help.x) ** 2 + (track_p.y - tar_track_help.y) ** 2)
                        
                        if np.min(distance_to_cross) < 10:
                            usable = distance_to_cross < 10
                            distance.append(np.min(distance_to_tar.loc[usable.index[usable]]))
                        else:
                            distance.append(np.min(distance_to_cross) + 1000)
                    v_4_id = other_ped.index[np.argmin(distance)]              
                else:
                    v_4_id = -1
            
            
                # Collect path data
                path = pd.Series(np.empty(len(names), tuple), index = names)
                
                tar_track_l = tar_track.loc[frame_min:frame_max].copy(deep = True)
                ego_track_l = ego_track.loc[frame_min:frame_max].copy(deep = True)
                
                path.V_ego_x = tuple(ego_track_l.x)
                path.V_ego_y = tuple(ego_track_l.y)
                path.V_tar_x = tuple(tar_track_l.x)
                path.V_tar_y = tuple(tar_track_l.y)
            
                if v_1_id >= 0:
                    v_1_track = other_agents.loc[v_1_id].track.loc[frame_min:frame_max]
                    
                    if len(v_1_track) > 0:
                        frame_min_v1 = v_1_track.index.min()
                        frame_max_v1 = v_1_track.index.max()
                        
                        v1x = np.ones(frame_max + 1 - frame_min) * np.nan
                        v1x[frame_min_v1 - frame_min : frame_max_v1 + 1 - frame_min] = v_1_track.x
                        
                        v1y = np.ones(frame_max + 1 - frame_min) * np.nan
                        v1y[frame_min_v1 - frame_min : frame_max_v1 + 1 - frame_min] = v_1_track.y
                        
                        path.V_v_1_x = tuple(v1x)
                        path.V_v_1_y = tuple(v1y)
                    else:
                        path.V_v_1_x = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                        path.V_v_1_y = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                else:
                    path.V_v_1_x = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                    path.V_v_1_y = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                    
                if v_2_id >= 0:
                    v_2_track = other_agents.loc[v_2_id].track.loc[frame_min:frame_max]
                    
                    if len(v_2_track) > 0:
                        frame_min_v2 = v_2_track.index.min()
                        frame_max_v2 = v_2_track.index.max()
                        
                        v2x = np.ones(frame_max + 1 - frame_min) * np.nan
                        v2x[frame_min_v2 - frame_min : frame_max_v2 + 1 - frame_min] = v_2_track.x
                        
                        v2y = np.ones(frame_max + 1 - frame_min) * np.nan
                        v2y[frame_min_v2 - frame_min : frame_max_v2 + 1 - frame_min] = v_2_track.y
                        
                        path.V_v_2_x = tuple(v2x)
                        path.V_v_2_y = tuple(v2y)
                    else:
                        path.V_v_2_x = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                        path.V_v_2_y = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                else:
                    path.V_v_2_x = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                    path.V_v_2_y = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                    
                if v_3_id >= 0:
                    v_3_track = other_agents.loc[v_3_id].track.loc[frame_min:frame_max]
                    
                    if len(v_3_track) > 0:
                        frame_min_v3 = v_3_track.index.min()
                        frame_max_v3 = v_3_track.index.max()
                        
                        v3x = np.ones(frame_max + 1 - frame_min) * np.nan
                        v3x[frame_min_v3 - frame_min : frame_max_v3 + 1 - frame_min] = v_3_track.x
                        
                        v3y = np.ones(frame_max + 1 - frame_min) * np.nan
                        v3y[frame_min_v3 - frame_min : frame_max_v3 + 1 - frame_min] = v_3_track.y
                        
                        path.V_v_3_x = tuple(v3x)
                        path.V_v_3_y = tuple(v3y)
                    else:
                        path.V_v_3_x = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                        path.V_v_3_y = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                        
                else:
                    path.V_v_3_x = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                    path.V_v_3_y = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
            
                if v_4_id >= 0:
                    v_4_track = other_agents.loc[v_4_id].track.loc[frame_min:frame_max]
                    
                    if len(v_4_track) > 0:
                        frame_min_v4 = v_4_track.index.min()
                        frame_max_v4 = v_4_track.index.max()
                        
                        v4x = np.ones(frame_max + 1 - frame_min) * np.nan
                        v4x[frame_min_v4 - frame_min : frame_max_v4 + 1 - frame_min] = v_4_track.x
                        
                        v4y = np.ones(frame_max + 1 - frame_min) * np.nan
                        v4y[frame_min_v4 - frame_min : frame_max_v4 + 1 - frame_min] = v_4_track.y
                        
                        path.P_v_4_x = tuple(v4x)
                        path.P_v_4_y = tuple(v4y)
                    else:
                        path.P_v_4_x = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                        path.P_v_4_y = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                else:
                    path.P_v_4_x = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                    path.P_v_4_y = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)

            
                t = tuple(tar_track_l.index / 25)
                
                # collect domain data
                domain = pd.Series(np.zeros(2, int), index = ['location', 'class'])
                domain.location = data_i.locationId
                domain['class'] = data_i['class']
                
                self.Path.append(path)
                self.T.append(t)
                self.Domain_old.append(domain)
                self.num_samples = self.num_samples + 1
                
                
            # Assume rejected gap
            for [ego_id, frame_C] in zip(rejected_Ego,rejected_frame_C):
                ego_track = other_agents.loc[ego_id].track.copy(deep = True)
                
                frame_min = max(ego_track.index[0], tar_track.index[0])
                frame_max = min(ego_track.index[-1], tar_track.index[-1])
                
                # find v_1: in_RA directly before ego vehicle
                in_ego = np.where(in_RA[:,0] == ego_id)[0][0]
                
                if in_ego == 0:
                    v_1_id = -1
                else:
                    v_1_id = in_RA[in_ego - 1, 0]
                    
                # find v_2: in_RA directly after ego vehicle
                try:
                    v_2_id = in_RA[in_ego + 1, 0]
                except:
                    v_2_id = -1
                
                # find v_3: entered_RA with the largest tar_frame_A that is still smaller than tar_frame_A
                if len(entered_RA) > 0:
                    feasible = entered_RA[:,1] < tar_frame_A
                else:
                    feasible = [False]
                if np.any(feasible):
                    v_3_id = entered_RA[feasible,0][np.argmax(entered_RA[feasible,1])] 
                else: 
                    v_3_id = -1
                # check for pedestrian 
                if len(other_ped) > 0:
                    distance = []
                    for j in range(len(other_ped)):
                        track_p = other_ped.iloc[j].track.loc[frame_min:frame_max]
                        tar_track_help = tar_track.copy(deep = True)
                        distance_to_cross = np.sqrt((track_p.x - Loc_data.iloc[data_i.locationId].R - 5) ** 2 +
                                                    (track_p.y - tar_track_help.iloc[frame_entry].y) ** 2)
                        distance_to_tar = np.sqrt((track_p.x - tar_track_help.x) ** 2 + (track_p.y - tar_track_help.y) ** 2)
                        
                        if np.min(distance_to_cross) < 10:
                            usable = distance_to_cross < 10
                            distance.append(np.min(distance_to_tar.loc[usable.index[usable]]))
                        else:
                            distance.append(np.min(distance_to_cross) + 1000)
                    v_4_id = other_ped.index[np.argmin(distance)] 
                else:
                    v_4_id = -1
            
            
                # Collect path data
                path = pd.Series(np.empty(len(names), tuple), index = names)
                
                tar_track_l = tar_track.loc[frame_min:frame_max].copy(deep = True)
                ego_track_l = ego_track.loc[frame_min:frame_max].copy(deep = True)
                
                path.V_ego_x = tuple(ego_track_l.x)
                path.V_ego_y = tuple(ego_track_l.y)
                path.V_tar_x = tuple(tar_track_l.x)
                path.V_tar_y = tuple(tar_track_l.y)
            
                if v_1_id >= 0:
                    v_1_track = other_agents.loc[v_1_id].track.loc[frame_min:frame_max]
                    
                    if len(v_1_track) > 0:
                        frame_min_v1 = v_1_track.index.min()
                        frame_max_v1 = v_1_track.index.max()
                        
                        v1x = np.ones(frame_max + 1 - frame_min) * np.nan
                        v1x[frame_min_v1 - frame_min : frame_max_v1 + 1 - frame_min] = v_1_track.x
                        
                        v1y = np.ones(frame_max + 1 - frame_min) * np.nan
                        v1y[frame_min_v1 - frame_min : frame_max_v1 + 1 - frame_min] = v_1_track.y
                        
                        path.V_v_1_x = tuple(v1x)
                        path.V_v_1_y = tuple(v1y)
                    else:
                        path.V_v_1_x = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                        path.V_v_1_y = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                else:
                    path.V_v_1_x = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                    path.V_v_1_y = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                    
                if v_2_id >= 0:
                    v_2_track = other_agents.loc[v_2_id].track.loc[frame_min:frame_max]
                    
                    if len(v_2_track) > 0:
                        frame_min_v2 = v_2_track.index.min()
                        frame_max_v2 = v_2_track.index.max()
                        
                        v2x = np.ones(frame_max + 1 - frame_min) * np.nan
                        v2x[frame_min_v2 - frame_min : frame_max_v2 + 1 - frame_min] = v_2_track.x
                        
                        v2y = np.ones(frame_max + 1 - frame_min) * np.nan
                        v2y[frame_min_v2 - frame_min : frame_max_v2 + 1 - frame_min] = v_2_track.y
                        
                        path.V_v_2_x = tuple(v2x)
                        path.V_v_2_y = tuple(v2y)
                    else:
                        path.V_v_2_x = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                        path.V_v_2_y = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                else:
                    path.V_v_2_x = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                    path.V_v_2_y = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                    
                if v_3_id >= 0:
                    v_3_track = other_agents.loc[v_3_id].track.loc[frame_min:frame_max]
                    
                    if len(v_3_track) > 0:
                        frame_min_v3 = v_3_track.index.min()
                        frame_max_v3 = v_3_track.index.max()
                        
                        v3x = np.ones(frame_max + 1 - frame_min) * np.nan
                        v3x[frame_min_v3 - frame_min : frame_max_v3 + 1 - frame_min] = v_3_track.x
                        
                        v3y = np.ones(frame_max + 1 - frame_min) * np.nan
                        v3y[frame_min_v3 - frame_min : frame_max_v3 + 1 - frame_min] = v_3_track.y
                        
                        path.V_v_3_x = tuple(v3x)
                        path.V_v_3_y = tuple(v3y)
                    else:
                        path.V_v_3_x = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                        path.V_v_3_y = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                        
                else:
                    path.V_v_3_x = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                    path.V_v_3_y = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
            
                if v_4_id >= 0:
                    v_4_track = other_agents.loc[v_4_id].track.loc[frame_min:frame_max]
                    
                    if len(v_4_track) > 0:
                        frame_min_v4 = v_4_track.index.min()
                        frame_max_v4 = v_4_track.index.max()
                        
                        v4x = np.ones(frame_max + 1 - frame_min) * np.nan
                        v4x[frame_min_v4 - frame_min : frame_max_v4 + 1 - frame_min] = v_4_track.x
                        
                        v4y = np.ones(frame_max + 1 - frame_min) * np.nan
                        v4y[frame_min_v4 - frame_min : frame_max_v4 + 1 - frame_min] = v_4_track.y
                        
                        path.P_v_4_x = tuple(v4x)
                        path.P_v_4_y = tuple(v4y)
                    else:
                        path.P_v_4_x = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                        path.P_v_4_y = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                else:
                    path.P_v_4_x = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                    path.P_v_4_y = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                    
            
                t = tuple(ego_track_l.index / 25)
                
                # collect domain data
                domain = pd.Series(np.zeros(2, int), index = ['location', 'class'])
                domain.location = data_i.locationId
                domain['class'] = data_i['class']
                
                self.Path.append(path)
                self.T.append(t)
                self.Domain_old.append(domain)
                self.num_samples = self.num_samples + 1
        
        self.Path = pd.DataFrame(self.Path)
        self.T = np.array(self.T+[()], tuple)[:-1]
        self.Domain_old = pd.DataFrame(self.Domain_old)
         
    def position_states_in_path(self, path, t, domain):
        lane_width = 3.5
        vehicle_length = 5 
        
        ego_x = np.array(path.V_ego_x)
        ego_y = np.array(path.V_ego_y)
        tar_x = np.array(path.V_tar_x)
        tar_y = np.array(path.V_tar_y)
        v_1_x = np.array(path.V_v_1_x)
        v_1_y = np.array(path.V_v_1_y)
        v_2_x = np.array(path.V_v_2_x)
        v_2_y = np.array(path.V_v_2_y)
        v_3_x = np.array(path.V_v_3_x)
        v_3_y = np.array(path.V_v_3_y)
        
        ego_r = np.sqrt(ego_x ** 2 + ego_y ** 2)
        ego_a = np.angle(ego_x + ego_y * 1j)
        tar_r = np.sqrt(tar_x ** 2 + tar_y ** 2)
        tar_a = np.angle(tar_x + tar_y * 1j)
        v_1_r = np.sqrt(v_1_x ** 2 + v_1_y ** 2)
        v_1_a = np.angle(v_1_x + v_1_y * 1j)
        v_2_r = np.sqrt(v_2_x ** 2 + v_2_y ** 2)
        v_2_a = np.angle(v_2_x + v_2_y * 1j)
        v_3_r = np.sqrt(v_3_x ** 2 + v_3_y ** 2)
        v_3_a = np.angle(v_3_x + v_3_y * 1j)
        
        # From location data
        R_dic = {0: 25.146, 1: 14.528, 2: 13.633}
        R = R_dic[domain.location]
        
        ego_frame_0 = np.nanargmin(np.abs(ego_a) + (ego_r > R) * 2 * np.pi)
        ego_a_change = np.where((ego_a[1:] < np.pi * 0.5) & (ego_a[:-1] > np.pi * 0.5))[0] + 1
        for i_change in ego_a_change:
            if i_change < ego_frame_0:
                ego_a[:i_change] -= 2 * np.pi
            else:
                ego_a[i_change:] += 2 * np.pi
        
        # This is an error flag designed to catch vehicles, whose roll is wrongly classified
        # Example. What could be v_3 is already inside the roundabout and then classified as 
        # either v_1 or v_2, which is an mistake
        some_error = False
        if not np.isnan(v_1_a).all():
            v_1_frame_0 = np.nanargmin(np.abs(v_1_a) + (v_1_r > R) * 2 * np.pi)
            v_1_a_change = np.where((v_1_a[1:] < np.pi * 0.5) & (v_1_a[:-1] > np.pi * 0.5))[0] + 1
            for i_change in v_1_a_change:
                if i_change < v_1_frame_0:
                    v_1_a[:i_change] -= 2 * np.pi
                else:
                    v_1_a[i_change:] += 2 * np.pi
             
            if not v_1_frame_0 < ego_frame_0:
                some_error = True
        
        if not np.isnan(v_2_a).all():
            v_2_frame_0 = np.nanargmin(np.abs(v_2_a) + (v_2_r > R) * 2 * np.pi)
            v_2_a_change = np.where((v_2_a[1:] < np.pi * 0.5) & (v_2_a[:-1] > np.pi * 0.5))[0] + 1
            for i_change in v_2_a_change:
                if i_change < v_2_frame_0:
                    v_2_a[:i_change] -= 2 * np.pi
                else:
                    v_2_a[i_change:] += 2 * np.pi
                 
            if not v_2_frame_0 > ego_frame_0:
                some_error = True
        
        
        Rl = R - 0.5 * lane_width
        
        D_ego = np.log(1 / (1 + np.exp(ego_r - Rl))) * np.tanh(5 * ego_a) - Rl * ego_a
        D_v_1 = np.log(1 / (1 + np.exp(v_1_r - Rl))) * np.tanh(5 * v_1_a) - Rl * v_1_a
        D_v_2 = np.log(1 / (1 + np.exp(v_2_r - Rl))) * np.tanh(5 * v_2_a) - Rl * v_2_a
        
        Dc = D_ego - 0.5 * vehicle_length
        D1 = D_ego - D_v_1 - vehicle_length
        D2 = D_v_2 - D_ego - vehicle_length
        
        assert False 
        
        Le = np.ones_like(Dc) * lane_width
        
        tar_a_change = np.where((tar_a[1:] < np.pi * 0.5) & (tar_a[:-1] > np.pi * 0.5))[0] + 1
        for i_change in tar_a_change:
            tar_a[i_change:] += 2 * np.pi
            
        if not np.isnan(v_3_a).all():
            v_3_a_change = np.where((v_3_a[1:] < np.pi * 0.5) & (v_3_a[:-1] > np.pi * 0.5))[0] + 1
            for i_change in v_3_a_change:
                v_3_a[i_change:] += 2 * np.pi
        
        
        DR_tar = np.log((1 + np.exp(tar_r - R))) * np.tanh(5 * (np.pi * 0.25 - tar_a)) - np.log(2)
        DR_v_3 = np.log((1 + np.exp(v_3_r - R))) * np.tanh(5 * (np.pi * 0.25 - v_3_a)) - np.log(2)
        
        D_tar = DR_tar - Rl * (tar_a - lane_width / R) * np.maximum(0, 1 - np.exp(5 * DR_tar)) ** 2
        D_v_3 = DR_v_3 - Rl * (v_3_a - lane_width / R) * np.maximum(0, 1 - np.exp(5 * DR_v_3)) ** 2
        
        Da = D_tar
        D3 = D_tar - D_v_3 - vehicle_length
        
        Dadt = np.interp(t, t[5:], (Da[5:] - Da[:-5]) / (t[5:] - t[:-5]))
        Dcdt = np.interp(t, t[5:], (Dc[5:] - Dc[:-5]) / (t[5:] - t[:-5]))
        
        Dv = np.maximum(Dadt - Dcdt, 0)
        Te = Dv / 2 # assume acceleration of up to 2m/s^2
        Lt = 0.5 * 2 * Te ** 2 - Dadt * Te # assume again acceleration of up to 2m/s^2
        Lt = np.clip(Lt, lane_width, None)
        
        out_of_position = ((ego_r > R) | (ego_r < R - 2 * lane_width)  |
                           (v_1_a < 0) | 
                           ((v_1_a < 0.5 * np.pi) & (v_1_y < lane_width + 0.5 * vehicle_length)))
        
        in_position = np.invert(out_of_position)
        
        # repair 
        if np.isnan(D1).any():
            if np.isnan(D1).all():
                D1 = np.ones_like(Dc) * 500
            else:
                D1 = np.interp(t, t[np.isfinite(D1)], D1[np.isfinite(D1)])
        if np.isnan(D2).any():
            if np.isnan(D2).all():
                D2 = np.ones_like(Dc) * 500
            else:
                D2 = np.interp(t, t[np.isfinite(D2)], D2[np.isfinite(D2)])
        if np.isnan(D3).any():
            if np.isnan(D3).all():
                D3 = np.ones_like(Dc) * 500
            else:
                D3 = np.interp(t, t[np.isfinite(D3)], D3[np.isfinite(D3)])
                
        if some_error:
            in_position[:] = False # This will cause the sample to be rejected, as T_s = empty
        return [in_position, Dc, Da, D1, D2, D3, Le, Lt]
     
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
        for i_sample in range(len(self.Input_path)):
            path = self.Input_path.iloc[i_sample]
            t = self.Input_T[i_sample]
            domain = self.Domain.iloc[i_sample]
        
            R_dic = {0: 25.146, 1: 14.528, 2: 13.633}
            R = R_dic[domain.location]
            
            # check vehicle v_1 (in front of ego)
            v_1_x = path.V_v_1_x
            v_1_rewrite = np.isnan(v_1_x)
            if v_1_rewrite.any():
                v_1_y = path.V_v_1_y
                
                v_1_r = np.sqrt(v_1_x ** 2 + v_1_y ** 2)
                v_1_a = np.angle(v_1_x + v_1_y * 1j)
                
                n_1 = len(v_1_rewrite) - v_1_rewrite.sum()
                
                if n_1 == 0:
                    v_1_r = np.arange(len(v_1_rewrite)) * self.dt * 10 + R + 40
                    v_1_a = np.ones(len(v_1_rewrite)) * np.pi * 0.4
                    
                    v_1_x = np.cos(v_1_a) * v_1_r
                    v_1_y = np.sin(v_1_a) * v_1_r
                elif n_1 == 1:
                    if v_1_rewrite[-1]:
                        i_last = np.where(np.invert(v_1_rewrite))[0][-1]
                        r_last = v_1_r[i_last]
                        if r_last > R:
                            dr_last = 10 * self.dt
                            v_1_r[i_last:] = (r_last + np.arange(len(v_1_rewrite) - i_last) * 
                                              max(dr_last, (R - r_last) / (len(v_1_rewrite) - i_last - 1)))
                            v_1_a[i_last:] = v_1_a[i_last]
                        else:
                            a_last = v_1_a[i_last]
                            da_last = 5 * self.dt / r_last
                            v_1_a[i_last:] = v_1_a[i_last] + np.arange(1, len(v_1_rewrite) - i_last) * da_last
                            v_1_r[i_last:] = r_last
                            
                    if v_1_rewrite[0]:
                        i_first = np.where(np.invert(v_1_rewrite))[0][0]
                        r_first = v_1_r[i_first]
                        if r_first > R:
                            dr_first = 10 * self.dt
                            v_1_r[:i_first] = r_first + np.arange(-i_first, 0) * max(dr_first, (R - r_first) / i_first)
                            v_1_a[:i_first] = v_1_a[i_first]
                        else:
                            a_first = v_1_a[i_first]
                            da_first = 5 * self.dt / r_last
                            v_1_a[:i_first] = a_first + np.arange(-i_first, 0) * da_first
                            v_1_r[:i_first] = r_first
                    
                    v_1_x = np.cos(v_1_a) * v_1_r
                    v_1_y = np.sin(v_1_a) * v_1_r
                    v_1_rewrite = np.isnan(v_1_x)
                    v_1_x = np.interp(t,t[np.invert(v_1_rewrite)],v_1_x[np.invert(v_1_rewrite)])
                    v_1_y = np.interp(t,t[np.invert(v_1_rewrite)],v_1_y[np.invert(v_1_rewrite)])
                    
                else:
                    if v_1_rewrite[-1]:
                        i_last = np.where(np.invert(v_1_rewrite))[0][-1]
                        r_last = v_1_r[i_last]
                        if r_last > R:
                            dr_last = v_1_r[i_last] - v_1_r[i_last - 1]
                            v_1_r[i_last:] = (r_last + np.arange(len(v_1_rewrite) - i_last) * 
                                              max(dr_last, (R - r_last) / (len(v_1_rewrite) - i_last - 1)))
                            v_1_a[i_last:] = v_1_a[i_last]
                        else:
                            a_last = v_1_a[i_last]
                            da_last = v_1_a[i_last] - v_1_a[i_last - 1]
                            v_1_a[i_last:] = v_1_a[i_last] + np.arange(1, len(v_1_rewrite) - i_last) * da_last
                            v_1_r[i_last:] = r_last
                            
                    if v_1_rewrite[0]:
                        i_first = np.where(np.invert(v_1_rewrite))[0][0]
                        r_first = v_1_r[i_first]
                        if r_first > R:
                            dr_first = v_1_r[i_first + 1] - v_1_r[i_first - 1]
                            v_1_r[:i_first] = r_first + np.arange(-i_first, 0) * max(dr_first, (R - r_first) / i_first)
                            v_1_a[:i_first] = v_1_a[i_first]
                        else:
                            a_first = v_1_a[i_first]
                            da_first = v_1_a[i_first + 1] - v_1_a[i_first]
                            v_1_a[:i_first] = a_first + np.arange(-i_first, 0) * da_first
                            v_1_r[:i_first] = r_first
                    
                    v_1_x = np.cos(v_1_a) * v_1_r
                    v_1_y = np.sin(v_1_a) * v_1_r
                    v_1_rewrite = np.isnan(v_1_x)
                    v_1_x = np.interp(t,t[np.invert(v_1_rewrite)],v_1_x[np.invert(v_1_rewrite)])
                    v_1_y = np.interp(t,t[np.invert(v_1_rewrite)],v_1_y[np.invert(v_1_rewrite)])
                    
                path.V_v_1_x = v_1_x
                path.V_v_1_y = v_1_y
            
            # check vehicle v_2 (behind ego)
            v_2_x = path.V_v_2_x
            v_2_rewrite = np.isnan(v_2_x)
            if v_2_rewrite.any():
                v_2_y = path.V_v_2_y
                
                v_2_r = np.sqrt(v_2_x ** 2 + v_2_y ** 2)
                v_2_a = np.angle(v_2_x + v_2_y * 1j)
                
                n_2 = len(v_2_rewrite) - v_2_rewrite.sum()
                
                if n_2 == 0:
                    v_2_r = R + 40 + (len(v_2_rewrite) - np.arange(len(v_2_rewrite))) * self.dt * 10
                    v_2_a = np.ones(len(v_2_rewrite)) * np.pi * -0.9
                    
                    v_2_x = np.cos(v_2_a) * v_2_r
                    v_2_y = np.sin(v_2_a) * v_2_r
                elif n_2 == 1:
                    if v_2_rewrite[-1]:
                        i_last = np.where(np.invert(v_2_rewrite))[0][-1]
                        r_last = v_2_r[i_last]
                        if r_last > R:
                            dr_last = 10 * self.dt
                            v_2_r[i_last:] = (r_last + np.arange(len(v_2_rewrite) - i_last) * 
                                              max(dr_last, (R - r_last) / (len(v_2_rewrite) - i_last - 1)))
                            v_2_a[i_last:] = v_2_a[i_last]
                        else:
                            a_last = v_2_a[i_last]
                            da_last = 5 * self.dt / r_last
                            v_2_a[i_last:] = v_2_a[i_last] + np.arange(1, len(v_2_rewrite) - i_last) * da_last
                            v_2_r[i_last:] = r_last
                            
                    if v_2_rewrite[0]:
                        i_first = np.where(np.invert(v_2_rewrite))[0][0]
                        r_first = v_2_r[i_first]
                        if r_first > R:
                            dr_first = 10 * self.dt
                            v_2_r[:i_first] = r_first + np.arange(-i_first, 0) * max(dr_first, (R - r_first) / i_first)
                            v_2_a[:i_first] = v_2_a[i_first]
                        else:
                            a_first = v_2_a[i_first]
                            da_first = 5 * self.dt / r_last
                            v_2_a[:i_first] = a_first + np.arange(-i_first, 0) * da_first
                            v_2_r[:i_first] = r_first
                    
                    v_2_x = np.cos(v_2_a) * v_2_r
                    v_2_y = np.sin(v_2_a) * v_2_r
                    v_2_rewrite = np.isnan(v_2_x)
                    v_2_x = np.interp(t,t[np.invert(v_2_rewrite)],v_2_x[np.invert(v_2_rewrite)])
                    v_2_y = np.interp(t,t[np.invert(v_2_rewrite)],v_2_y[np.invert(v_2_rewrite)])
                    
                else:
                    if v_2_rewrite[-1]:
                        i_last = np.where(np.invert(v_2_rewrite))[0][-1]
                        r_last = v_2_r[i_last]
                        if r_last > R:
                            dr_last = v_2_r[i_last] - v_2_r[i_last - 1]
                            v_2_r[i_last:] = (r_last + np.arange(len(v_2_rewrite) - i_last) * 
                                              max(dr_last, (R - r_last) / (len(v_2_rewrite) - i_last - 1)))
                            v_2_a[i_last:] = v_2_a[i_last]
                        else:
                            a_last = v_2_a[i_last]
                            da_last = a_last - v_2_a[i_last - 1]
                            v_2_a[i_last:] = v_2_a[i_last] + np.arange(1, len(v_2_rewrite) - i_last) * da_last
                            v_2_r[i_last:] = r_last
                            
                    if v_2_rewrite[0]:
                        i_first = np.where(np.invert(v_2_rewrite))[0][0]
                        r_first = v_2_r[i_first]
                        if r_first > R:
                            dr_first = v_2_r[i_first + 1] - v_2_r[i_first - 1]
                            v_2_r[:i_first] = r_first + np.arange(-i_first, 0) * max(dr_first, (R - r_first) / i_first)
                            v_2_a[:i_first] = v_2_a[i_first]
                        else:
                            a_first = v_2_a[i_first]
                            da_first = v_2_a[i_first + 1] - v_2_a[i_first]
                            v_2_a[:i_first] = a_first + np.arange(-i_first, 0) * da_first
                            v_2_r[:i_first] = r_first
                    
                    v_2_x = np.cos(v_2_a) * v_2_r
                    v_2_y = np.sin(v_2_a) * v_2_r
                    v_2_rewrite = np.isnan(v_2_x)
                    v_2_x = np.interp(t,t[np.invert(v_2_rewrite)],v_2_x[np.invert(v_2_rewrite)])
                    v_2_y = np.interp(t,t[np.invert(v_2_rewrite)],v_2_y[np.invert(v_2_rewrite)])
                    
                path.V_v_2_x = v_2_x
                path.V_v_2_y = v_2_y
            
            # check vehicle v_3 (in front of tar)
            v_3_x = path.V_v_3_x
            v_3_rewrite = np.isnan(v_3_x)
            if v_3_rewrite.any():
                v_3_y = path.V_v_3_y
                
                v_3_r = np.sqrt(v_3_x ** 2 + v_3_y ** 2)
                v_3_a = np.angle(v_3_x + v_3_y * 1j)
                
                n_3 = len(v_3_rewrite) - v_3_rewrite.sum()
                
                if n_3 == 0:
                    v_3_r = np.arange(len(v_3_rewrite)) * self.dt * 10 + R + 20
                    v_3_a = np.ones(len(v_3_rewrite)) * np.pi * 0.4
                    
                    v_3_x = np.cos(v_3_a) * v_3_r
                    v_3_y = np.sin(v_3_a) * v_3_r
                elif n_3 == 1:
                    if v_3_rewrite[-1]:
                        i_last = np.where(np.invert(v_3_rewrite))[0][-1]
                        r_last = v_3_r[i_last]
                        if r_last > R:
                            dr_last = 10 * self.dt
                            v_3_r[i_last:] = (r_last + np.arange(len(v_3_rewrite) - i_last) * 
                                              max(dr_last, (R - r_last) / (len(v_3_rewrite) - i_last - 1)))
                            v_3_a[i_last:] = v_3_a[i_last]
                        else:
                            a_last = v_3_a[i_last]
                            da_last = 5 * self.dt / r_last
                            v_3_a[i_last:] = a_last + np.arange(1, len(v_3_rewrite) - i_last) * da_last
                            v_3_r[i_last:] = r_last
                            
                    if v_3_rewrite[0]:
                        i_first = np.where(np.invert(v_3_rewrite))[0][0]
                        r_first = v_3_r[i_first]
                        if r_first > R:
                            dr_first = 10 * self.dt
                            v_3_r[:i_first] = r_first + np.arange(-i_first, 0) * max(dr_first, (R - r_first) / i_first)
                            v_3_a[:i_first] = v_3_a[i_first]
                        else:
                            a_first = v_3_a[i_first]
                            da_first = 5 * self.dt / r_last
                            v_3_a[:i_first] = a_first + np.arange(-i_first, 0) * da_first
                            v_3_r[:i_first] = r_first
                    
                    v_3_x = np.cos(v_3_a) * v_3_r
                    v_3_y = np.sin(v_3_a) * v_3_r
                    v_3_rewrite = np.isnan(v_3_x)
                    v_3_x = np.interp(t,t[np.invert(v_3_rewrite)],v_3_x[np.invert(v_3_rewrite)])
                    v_3_y = np.interp(t,t[np.invert(v_3_rewrite)],v_3_y[np.invert(v_3_rewrite)])
                    
                else:
                    if v_3_rewrite[-1]:
                        i_last = np.where(np.invert(v_3_rewrite))[0][-1]
                        r_last = v_3_r[i_last]
                        if r_last > R:
                            dr_last = v_3_r[i_last] - v_3_r[i_last - 1]
                            v_3_r[i_last:] = (r_last + np.arange(len(v_3_rewrite) - i_last) * 
                                              max(dr_last, (R - r_last) / (len(v_3_rewrite) - i_last - 1)))
                            v_3_a[i_last:] = v_3_a[i_last]
                        else:
                            a_last = v_3_a[i_last]
                            da_last = a_last - v_3_a[i_last - 1]
                            v_3_a[i_last:] = v_3_a[i_last] + np.arange(1, len(v_3_rewrite) - i_last) * da_last
                            v_3_r[i_last:] = r_last
                            
                    if v_3_rewrite[0]:
                        i_first = np.where(np.invert(v_3_rewrite))[0][0]
                        r_first = v_3_r[i_first]
                        if r_first > R:
                            dr_first = v_3_r[i_first + 1] - v_3_r[i_first - 1]
                            v_3_r[:i_first] = r_first + np.arange(-i_first, 0) * max(dr_first, (R - r_first) / i_first)
                            v_3_a[:i_first] = v_3_a[i_first]
                        else:
                            a_first = v_3_a[i_first]
                            da_first = v_3_a[i_first + 1] - v_3_a[i_first]
                            v_3_a[:i_first] = a_first + np.arange(-i_first, 0) * da_first
                            v_3_r[:i_first] = r_first
                    
                    v_3_x = np.cos(v_3_a) * v_3_r
                    v_3_y = np.sin(v_3_a) * v_3_r
                    v_3_rewrite = np.isnan(v_3_x)
                    v_3_x = np.interp(t,t[np.invert(v_3_rewrite)],v_3_x[np.invert(v_3_rewrite)])
                    v_3_y = np.interp(t,t[np.invert(v_3_rewrite)],v_3_y[np.invert(v_3_rewrite)])
                    
                path.V_v_3_x = v_3_x
                path.V_v_3_y = v_3_y
                
            
            # check vehicle v_4 (pedestrian)
            v_4_x = path.P_v_4_x
            v_4_rewrite = np.isnan(v_4_x)
            if v_4_rewrite.any():
                v_4_y = path.P_v_4_y
                
                n_4 = len(v_4_rewrite) - v_4_rewrite.sum()
                if n_4 == 0:
                    v_4_y = np.ones(len(v_4_rewrite)) * 50
                    v_4_x = np.ones(len(v_4_rewrite)) * (- 200)
                else:
                    v_4_y = np.interp(t,t[np.invert(v_4_rewrite)],v_4_y[np.invert(v_4_rewrite)])
                    v_4_x = np.interp(t,t[np.invert(v_4_rewrite)],v_4_y[np.invert(v_4_rewrite)])
                    
                path.P_v_4_x = v_4_x
                path.P_v_4_y = v_4_y     
                
            self.Input_path.iloc[i_sample] = path
            
    
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
