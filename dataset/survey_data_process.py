import torch
import os
import pickle
from pandas import read_pickle
root = 'D:\\Nuscenes\\survey\\processed'

dim = 2
adj_num = 3
veh_feature_dim =5
ped_feature_dim =3
data_ = {}

for each in range(30):
    each =  f'q{each+1}'
    a = os.listdir(os.path.join(root, each))
    for b in a:
        if b.endswith('pkl') and b.split('.', 1)[0] == 'answer':
            gt = read_pickle(os.path.join(root, each, b))
        elif b.endswith('pkl') and b.split('.', 1)[0] =='survey_data':
            pass 
        elif b.endswith('pkl'): 
            data = read_pickle(os.path.join(root, each, b))
    
    ego_vec = data['ego_velocity'][:dim]
    ego_pos = data['ego_position']
    ego_vec = torch.tensor(ego_vec).unsqueeze(0)
    adj_veh_num = torch.tensor(data['adj_vehicle_num']).unsqueeze(0)
    adj_ped_num = torch.tensor(data['adj_pedestrain_num']).unsqueeze(0)
    
    adj_veh_feat = []
    for each_veh_idx in range(min(adj_veh_num, adj_num)):
        tmp = torch.cat([torch.tensor(data['adj_vehicles'][f'vehicle_{each_veh_idx}']['distance']).unsqueeze(0),
                         torch.tensor(data['adj_vehicles'][f'vehicle_{each_veh_idx}']['relative_position'][:dim]),
                         torch.tensor(data['adj_vehicles'][f'vehicle_{each_veh_idx}']['relative_velocity'][:dim])],
                         dim=0)
        adj_veh_feat.append(tmp)
    if len(adj_veh_feat) != 0:
        adj_veh_feat = torch.stack(adj_veh_feat, dim=0)
    else:
        adj_veh_feat = torch.zeros(adj_veh_num, veh_feature_dim)
    if adj_veh_feat.shape[0] < 3 and adj_veh_feat.shape[0] !=0:
        adj_veh_feat = torch.cat((adj_veh_feat, torch.zeros(adj_num - adj_veh_feat.shape[0], adj_veh_feat.shape[1])), dim=0)
    elif adj_veh_feat.shape[0] ==0:
        adj_veh_feat = torch.zeros(adj_veh_num, veh_feature_dim)

    adj_ped_feat = []
    for each_ped_idx in range(min(adj_ped_num, adj_num)):
        tmp = torch.cat([
            torch.tensor(data['adj_pedestrains'][f'pedestrain_{each_ped_idx}']['distance']).unsqueeze(0),
            torch.tensor(data['adj_pedestrains'][f'pedestrain_{each_ped_idx}']['relative_position'][:dim])
        ], dim=0)
        adj_ped_feat.append(tmp)
    if len(adj_ped_feat) != 0:
        adj_ped_feat = torch.stack(adj_ped_feat, dim=0)
    else:
        adj_ped_feat = torch.zeros(adj_ped_num, ped_feature_dim)
    if adj_ped_feat.shape[0] < 3 and adj_ped_feat.shape[0] !=0:
        adj_ped_feat = torch.cat((adj_ped_feat, torch.zeros(adj_num - adj_ped_feat.shape[0], adj_ped_feat.shape[1])), dim=0)

    prob_cls_num = gt['prob'].keys().__len__()
    gt_prob = torch.zeros(prob_cls_num)
    gt_prob[0] = gt['prob'][' 0']
    gt_prob[1] = gt['prob'][' 20']
    gt_prob[2] = gt['prob'][' 40']
    gt_prob[3] = gt['prob'][' 60']
    gt_prob[4] = gt['prob'][' 80']
    gt_prob[5] = gt['prob'][' 90']

    prob_act_num = gt['action'].keys().__len__()
    gt_act = torch.zeros(prob_act_num)
    gt_act[0] = gt['action'][' 直线减速']
    gt_act[1] = gt['action'][' 原速度向左转向']
    gt_act[2] = gt['action'][' 原速度向右转向']
    gt_act[3] = gt['action'][' 保持当前的速度和方向']
    gt_act[4] = gt['action'][' 直线加速']

    data_['ego_vec'] = ego_vec
    data_['adj_veh_num'] = adj_veh_num
    data_['adj_ped_num'] = adj_ped_num
    data_['adj_veh_feat'] = adj_veh_feat
    data_['adj_ped_feat'] = adj_ped_feat
    data_['gt_prob'] = gt_prob
    data_['gt_act'] = gt_act

    file_path = os.path.join(root, each,'survey_data.pkl')
    if os.path.isfile(file_path):
        os.remove(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(data_, f)