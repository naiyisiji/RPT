import os
import pickle
import pandas 
from tqdm import tqdm
from pandas import read_csv

path = 'D:\\Nuscenes\\survey\\Pollfish_Survey_.csv'
file = read_csv(path)

question_feature_path_root = 'D:\\Nuscenes\\survey\\processed'

ques_1 = '观察图像中展示的车辆主视角，选择该场景下发生危险的可能性。'
ques_2 = '观察上一问题中的图像中展示的车辆主视角，选择该场景应该做出的行为'
result_1 = []
result_2 = []
for q_1 in range(30):
    if q_1 ==  0:
        ques_1_ = ques_1
        ques_2_ = ques_2
    else:
        ques_1_ = ques_1 + '.' + str(q_1)
        ques_2_ = ques_2 + '.' + str(q_1)
    result_1.append(ques_1_)
    result_2.append(ques_2_)

count = 0
prob_list = [' 0', ' 20', ' 40', ' 60', ' 80', ' 90']
action_list = [' 直线加速', ' 保持当前的速度和方向', ' 原速度向右转向', ' 原速度向左转向', ' 直线减速']
for each_1, each_2, idx in zip(result_1, result_2, range(30)):
    tmp_1 = file[each_1]
    tmp_2 = file[each_2]
    each_stat_result_1 = []
    each_stat_result_2 = []
    for e_1,e_2 in zip(tmp_1,tmp_2):
        e_1 = e_1.split('.', 1)[-1].split('%',1)[0]
        e_2 = e_2.split('.', 1)[-1]

        each_stat_result_1.append(e_1)
        each_stat_result_2.append(e_2)
    element_counts_1 = {item: float(f'{each_stat_result_1.count(item)/63 :.4f}') for item in set(each_stat_result_1)}
    element_counts_2 = {item: float(f'{each_stat_result_2.count(item)/63 :.4f}') for item in set(each_stat_result_2)}
    for each_prob_choose in prob_list:
        if each_prob_choose not in element_counts_1.keys():
            element_counts_1[each_prob_choose] = 0.000
    for each_action_choose in action_list:
        if each_action_choose not in element_counts_2.keys():
            element_counts_2[each_action_choose] = 0.000

    data = {'prob':element_counts_1,'action':element_counts_2}
    file_name = 'answer.pkl'
    file_path = os.path.join(question_feature_path_root, f'q{idx+1}',file_name)
    if os.path.isfile(file_path):
        os.remove(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)