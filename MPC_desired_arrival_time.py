# 此程序为DAT方法的主程序 - 需调用GUROBI求解器

import pandas as pd
from MPC_old_model import mpc_own
from draw_single_veh import draw_single_veh

predict_horizon = 10
time_step = 0.5
green_time = 30
yellow_time = 3
red_time = 30
vfree = 20
u_max = 3
u_min = -3

# 创建红绿灯
light = []
for i in range(int(green_time / time_step)):
    light.append(0)
for i in range(int(yellow_time / time_step)):
    light.append(1)
for i in range(int(red_time / time_step)):
    light.append(2)

light_all = light
for i in range(11):
    light_all = light_all + light

# 单车运动矩阵
single_veh_trajectory = pd.DataFrame(columns=['No', 'VehType', 'Speed', 'Pos', 'Acc', 'recent_time'])

for time in range(40, 200):
    # 获取当前信号状态
    current_light = light_all[time]
    # print(time, current_light)
    if time == 40:
        tra_inf = [0, 'classical', 10, 0, 0, time]
        #在末尾插入新的一行
        single_veh_trajectory.loc[single_veh_trajectory.shape[0]] = tra_inf
    else:
        if single_veh_trajectory['Pos'].iloc[-1] < 300:
            print(single_veh_trajectory['Speed'].iloc[-1], single_veh_trajectory['Acc'].iloc[-1], time_step)
            speed1 = single_veh_trajectory['Speed'].iloc[-1] + single_veh_trajectory['Acc'].iloc[-1] * time_step
            pos1 = single_veh_trajectory['Pos'].iloc[-1] + single_veh_trajectory['Speed'].iloc[-1] * time_step + 0.5 * single_veh_trajectory['Acc'].iloc[-1] * time_step * time_step
            desired_arrivaltime = 126 - time  # 注意，此处由于是特殊案例，期望到达时间可以直接获得，其等于下一次绿灯的开始时间
            acc1,runtime = mpc_own(desired_arrivaltime, desired_arrivaltime, pos1, speed1)  # 调用DAT优化模型
            tra_inf = [0, 'classical', speed1, pos1, acc1, time]
            single_veh_trajectory.loc[single_veh_trajectory.shape[0]] = tra_inf

        elif single_veh_trajectory['Pos'].iloc[-1] >= 300 and single_veh_trajectory['Pos'].iloc[-1] < 400:
            speed1 = single_veh_trajectory['Speed'].iloc[-1] + single_veh_trajectory['Acc'].iloc[-1] * time_step
            pos1 = single_veh_trajectory['Pos'].iloc[-1] + single_veh_trajectory['Speed'].iloc[-1] * time_step + 0.5 * \
                   single_veh_trajectory['Acc'].iloc[-1] * time_step * time_step
            acc1 = min(u_max, (vfree - speed1) / time_step)
            tra_inf = [0, 'classical', speed1, pos1, acc1, time]
            single_veh_trajectory.loc[single_veh_trajectory.shape[0]] = tra_inf

        elif single_veh_trajectory['Pos'].iloc[-1] >= 400:
            break

fig = draw_single_veh(single_veh_trajectory, 'IDM')
