from nuscenes.nuscenes import NuScenes
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

import os
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import transform_matrix, view_points, BoxVisibility
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def get_ego_velocity(nusc, sample_data_token):
    sd = nusc.get('sample_data', sample_data_token)
    current_pose = nusc.get('ego_pose', sd['ego_pose_token'])
    current_time = sd['timestamp'] / 1e6
    prev_sd_token = sd['prev']
    if prev_sd_token == '':
        return np.array([0.0, 0.0, 0.0])
    
    prev_sd = nusc.get('sample_data', prev_sd_token)
    prev_pose = nusc.get('ego_pose', prev_sd['ego_pose_token'])
    prev_time = prev_sd['timestamp'] / 1e6
    delta_position = np.array(current_pose['translation']) - np.array(prev_pose['translation'])
    delta_time = current_time - prev_time
    velocity = delta_position / delta_time
    return velocity
def get_velocity(nusc, annotation_token):
    try:
        velocity = nusc.box_velocity(annotation_token)
        return np.array(velocity)
    except:
        return np.array([0.0, 0.0, 0.0])

def nuScenes_other_agent(nusc, sample, search_range = 150, dump = False, dump_path = None, idx = None, q_idx=None, version='v1.0-mini', data_root=None):
    ego_pose = nusc.get('ego_pose', sample['data']['CAM_FRONT'])
    ego_position = ego_pose['translation']
    ego_velocity = get_ego_velocity(nusc, sample['data']['CAM_FRONT'])
    #print(f"Ego车辆的速度: {ego_velocity}")
    #print(f"Ego车辆的位置: {ego_position}")
    vehicles = []
    # =================================== vehicles =================================== #
    for annotation_token in sample['anns']:
        annotation = nusc.get('sample_annotation', annotation_token)
        instance = nusc.get('instance', annotation['instance_token'])

        # 只考虑车辆
        if 'vehicle' in annotation['category_name']:
            position = np.array(annotation['translation'])
            distance = np.linalg.norm(position - ego_position)

            if distance <= search_range:  # 筛选距离在150米以内的车辆
                velocity = get_velocity(nusc, annotation_token)
                relative_position = position - ego_position
                relative_velocity = velocity - ego_velocity
                vehicles.append((distance, annotation_token, relative_position, velocity,relative_velocity))
    veh_num = len(vehicles)
    #print(f"{search_range}m范围内所有车辆的数量为{len(vehicles)}")
    vehicles = sorted(vehicles, key=lambda x: x[0])[:3]
    adj_vehicles = dict()
    for i, (distance, annotation_token, relative_position, velocity,relative_velocity) in enumerate(vehicles):
        annotation = nusc.get('sample_annotation', annotation_token)
        instance = nusc.get('instance', annotation['instance_token'])
        category = annotation['category_name']
        #print(f"车辆 {i+1}:")
        #print(f"  类别: {category}")
        #print(f"  距离: {distance:.2f} 米")
        #print(f"  相对位置: {relative_position}")
        #print(f"  相对速度: {relative_velocity}")
        #print(f"  速度: {velocity}")
        adj_vehicles[f'vehicle_{i}'] = dict()
        adj_vehicles[f'vehicle_{i}']['distance']=distance
        adj_vehicles[f'vehicle_{i}']['velocity']=velocity
        adj_vehicles[f'vehicle_{i}']['relative_position']=relative_position
        adj_vehicles[f'vehicle_{i}']['relative_velocity']=relative_velocity

    # =================================================================================== #

    # =================================== pedestrains =================================== #
    pedestrians = []
    for annotation_token in sample['anns']:
        annotation = nusc.get('sample_annotation', annotation_token)
        instance = nusc.get('instance', annotation['instance_token'])

        # 只考虑行人
        if 'pedestrian' in annotation['category_name']:
            position = np.array(annotation['translation'])
            distance = np.linalg.norm(position - ego_position)

            if distance <= search_range:  # 筛选距离在150米以内的行人
                relative_position = position - ego_position
                pedestrians.append((distance, annotation_token, relative_position))

    ped_num = len(pedestrians)
    #print(f"{search_range}m范围内所有行人的数量为{len(pedestrians)}")
    pedestrians = sorted(pedestrians, key=lambda x: x[0])[:3]
    adj_pedestrains = dict()
    for i, (distance, annotation_token, relative_position) in enumerate(pedestrians):
        annotation = nusc.get('sample_annotation', annotation_token)
        instance = nusc.get('instance', annotation['instance_token'])
        category = annotation['category_name']
        #print(f"行人 {i}:")
        #print(f"  类别: {category}")
        #print(f"  距离: {distance:.2f} 米")
        #print(f"  相对位置: {relative_position}")
        adj_pedestrains[f'pedestrain_{i}'] = dict()
        adj_pedestrains[f'pedestrain_{i}']['distance']=distance
        adj_pedestrains[f'pedestrain_{i}']['relative_position']=relative_position
    # =================================================================================== #


    sensor = 'CAM_FRONT'
    cam_data = nusc.get('sample_data', sample['data'][sensor])
    image_path = os.path.join(data_root, cam_data['filename'])
    #print('image_path : ',image_path)
    image = Image.open(image_path)
    #plt.imshow(image)
    #plt.show()

    if dump:
        data = dict(ego_velocity=ego_velocity,
                    ego_position=ego_position,
                    adj_vehicle_num = veh_num,
                    adj_vehicles = adj_vehicles,
                    adj_pedestrain_num = ped_num,
                    adj_pedestrains = adj_pedestrains,
                    image_path=image_path)
        if idx == None:
            raise ValueError("idx have not been setting")
        pickle_file = f'{idx}.pkl'
        dump_path = os.path.join(dump_path, 'processed',f'q{q_idx+1}',pickle_file)
        if os.path.isfile(dump_path):
            os.remove(dump_path)
        with open(dump_path, 'wb') as f:
            pickle.dump(data, f)


def main():
    data_root = 'D:\\Nuscenes'  # 设置你的NuScenes数据集路径
    nusc = NuScenes(version='v1.0-trainval', dataroot=data_root, verbose=True)
    #idx = 2700

    """for _ in range(20):
        idx = idx + 30
        sample = nusc.sample[idx]
        print('idx : ', idx)
        nuScenes_other_agent(nusc ,sample, dump=False, dump_path='D:\\Nuscenes\\survey',idx=idx, data_root=data_root, version='v1.0-trainval')"""

    idx_list = [666, 121, 211, 301,405,435,745,805,1015,637,967,1207,1427,1607,1847,1648,1708,1998,2028,2058,2388,2418,2508,2311,2161,2491,2551,2581,2731,2791]
    """idx_list = [666, 1207, 2508, 2161, 2491, 2551, 2581, 2731, 2791,
                121, 211, 301, 405, 435, 745, 805, 1015, 637, 967, 
                1427, 1607, 1847, 1648, 1708, 1998, 2028, 2058, 2388, 2418, 2311]"""
    for idx_,q_idx in zip(idx_list, range(len(idx_list))):
        sample = nusc.sample[idx_]
        #print(idx_)
        nuScenes_other_agent(nusc, sample, dump=True, dump_path='D:\\Nuscenes\\survey',idx=idx_, q_idx=q_idx,data_root=data_root, version='v1.0-trainval')

if __name__ == '__main__':
    main()