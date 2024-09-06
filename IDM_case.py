# 此为IDM模型的运行程序
'''
IDM对应到本篇paper就是HV,本代码更新了HV的所有轨迹。通过时间的推移,HV的速度、加速度、位置等信息不断发生变化。注意到300米的时候,HV会停下等红绿灯,
本文要做的内容就是查看HV能不能比较慢得收敛到300米处的位置,
'''

import pandas as pd
from IDM_model import IDM_model
from draw_single_veh import draw_single_veh
#from main import veh_trajectory
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
    
class IDM():
    # 单车运动矩阵
    def __init__(self,v2,x2):
        veh_trajectory = pd.DataFrame(columns=['No', 'VehType', 'Speed', 'Pos', 'Acc', 'recent_time'])
        self.v2=v2
        self.x2=x2
        start_time =40

    def update_state(self):
        for time in range(40, 200):
            # 获取当前信号状态
            current_light = light_all[time]
            #print(time, current_light)
            if time == 40:
                tra_inf = [1, 'Human_drive', 10, 0, 0, time]   #给IDM一个初始速度，为10，初始位置为0？
                self.veh_trajectory.loc[self.veh_trajectory.shape[0]] = tra_inf #用这个方法可以将数据添加到最后一行
                # .shape属性会返回一个包含数据的行数和列数的元组
            else:
                if self.veh_trajectory['Pos'].iloc[-1] < 300:
                    speed1 = self.veh_trajectory['Speed'].iloc[-1] + self.veh_trajectory['Acc'].iloc[-1] * time_step
                    pos1 = self.veh_trajectory['Pos'].iloc[-1] + self.veh_trajectory['Speed'].iloc[-1] * time_step + 0.5 * self.veh_trajectory['Acc'].iloc[-1] * time_step * time_step
                    if current_light == 0: # 绿灯
                        acc1 = min(u_max, (vfree - speed1) / time_step)
                    else: # 黄灯和红灯
                        # TODO在这里进行了修改，没有默认IDM的前方车辆速度为0且在300m处，而是input了前车的位置和速度
                        acc1 = IDM_model(self.v2, speed1, self.x2, pos1)   #假设前车在300的位置处已经停下来  
                    tra_inf = [1, 'Human_drive', speed1, pos1, acc1, time]
                    self.veh_trajectory.loc[self.veh_trajectory.shape[0]] = tra_inf
        
                elif self.veh_trajectory['Pos'].iloc[-1] >= 300 and self.veh_trajectory['Pos'].iloc[-1] < 400:
                    speed1 = self.veh_trajectory['Speed'].iloc[-1] + self.veh_trajectory['Acc'].iloc[-1] * time_step    #匀加速运动
                    pos1 = self.veh_trajectory['Pos'].iloc[-1] + self.veh_trajectory['Speed'].iloc[-1] * time_step + 0.5 * \
                        self.veh_trajectory['Acc'].iloc[-1] * time_step * time_step
                    acc1 = min(u_max, (vfree - speed1) / time_step)
                    tra_inf = [1, 'Human_drive', speed1, pos1, acc1, time]
                    self.veh_trajectory.loc[self.veh_trajectory.shape[0]] = tra_inf

                elif self.veh_trajectory['Pos'].iloc[-1] >= 400:
                    return self.veh_trajectory['Acc'].iloc[-1]
#veh_trajectory.loc[]
idm=IDM()
idm.update_state()
print(idm.veh_trajectory[idm.veh_trajectory['Pos']== 300])
fig = draw_single_veh(idm.veh_trajectory, idm.start_time, 'IDM')

#开始的标准是
