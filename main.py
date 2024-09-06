# 此程序为该方法主程序 （注意：内部参数以及势函数形式被多次调整，同学需要仔细检查势函数形式以及重新调试参数，以达到最佳效果）
import pandas as pd
from optimization_single import nmpc
import CAV as CAV
from IDM_model import IDM_model
from draw_single_veh import draw_single_veh
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'FreeSans']  # 替换为系统上可用的字体列表
# 相关系数
predict_horizon = 50
time_step = 0.5
green_time = 30
yellow_time = 3
red_time = 30

# 创建红绿灯
light = []
for i in range(int(green_time / time_step)): #60
    light.append(0)
for i in range(int(yellow_time / time_step)): #6
    light.append(1)
for i in range(int(red_time / time_step)):  #60
    light.append(2)

light_all = light
#下面的代码相当于在原来的light列表中添加light列表
for i in range(11):
    light_all = light_all + light

##截取一个predict_horizon的长度作为红绿灯，每次predict长度都是50

def light_predict(predict_horizon, predict_origin_list, start_index):
    # 原始列表
    original_list = predict_origin_list

    # 初始化一个空列表，用于存储拼接后的结果
    concatenated_list = []

    # 设置目标列表长度
    target_length = predict_horizon

    # 从指定位置开始循环直到达到目标长度
    while len(concatenated_list) < target_length:
        concatenated_list.extend(original_list[start_index:])
        start_index = 0  # 之后的循环从列表开头开始

    # 如果列表长度超过目标长度，可以截取前面的部分以满足要求
    concatenated_list = concatenated_list[:target_length]

    return concatenated_list

start_time=40
mean=0  
std_dev=1

class Multi_vehicle():
    def __init__(self):
        self.length=5
        self.width=2
        self.veh_trajectory=pd.DataFrame(columns=['No', 'VehType', 'Speed', 'Pos', 'Acc', 'recent_time', 'light', 'predict_light'])
        #self.distance=2
        self.distance=5 #两辆车之间的距离最好设置为5，距离要大一些
        self.number=7 #车辆的数量
        self.vehicle_type=['Auto_vehicle','Human_vehicle','Auto_vehicle','Human_vehicle','Auto_vehicle','Auto_vehicle','Human_vehicle']

    def order(self):
        for i in range(self.number):
            # FIXME:
            pass

    def initial_state(self):
        #第一辆车是AV，一共5辆车，AV与HV混合排列成一队
        predict_light = light_predict(predict_horizon = 50,predict_origin_list = light_all,start_index=40)
        predict_light = light_predict(predict_horizon = 50,predict_origin_list = light_all,start_index=40)
        for i in range(len(self.vehicle_type)):
            if self.vehicle_type[i] == 'Auto_vehicle':
                self.veh_trajectory.loc[self.veh_trajectory.shape[0]] = [i,'Auto_vehicle',10,i*(self.length + self.distance),0,40,0,predict_light]
            elif self.vehicle_type[i] == 'Human_vehicle':
                self.veh_trajectory.loc[self.veh_trajectory.shape[0]] = [i,'Human_vehicle',10,i*(self.length + self.distance),0,40,0,predict_light]

        #veh_inf1=[0,'Auto_vehicle',10,0,0,40,0,predict_light]
        #veh_trajectory.loc[veh_trajectory.shape[0]]==veh_inf1
    def update(self):
        # 每辆车之间的间距，每辆车的速度，每辆车的灯状态
        #max_index=row_with_last.index.max()
        #for time in range(start_time,400):
        for time in range(start_time,50):
            if (time-start_time) == 0:
                self.initial_state()
            elif (time-start_time) > 0 and self.veh_trajectory['Pos'].iloc[0] < 300:
                for i in range(len(self.vehicle_type)):
                    if self.vehicle_type[i] == 'Auto_vehicle':
                        acc1=nmpc(state_all_start = self.veh_trajectory.iloc[self.veh_trajectory.shape[0] - 5 + i], T=time_step, N=predict_horizon)
                        print('veh_trajectory',self.veh_trajectory.iloc[self.veh_trajectory.shape[0] - 5 + i])
                        #print('lens_acc',len(acc1))
                        #因为mpc会有horizon，这里只取预测的第一个值
                        speed1 = self.veh_trajectory['Speed'].iloc[self.veh_trajectory.shape[0] - 5 + i] + acc1[0] * time_step
                        Pos1 = self.veh_trajectory['Pos'].iloc[-1] + self.veh_trajectory['Speed'].iloc[-1] * time_step + 0.5 * acc1[0] * time_step * time_step
                        light_state=light_all[time]
                        predict_light=light_predict(predict_horizon = 50,predict_origin_list = light_all,start_index = time)
                        self.veh_trajectory.loc[self.veh_trajectory.shape[0]] = [i,'Auto_vehicle',speed1,Pos1,acc1,time,light_state,predict_light]
                        #print('300米之前的值',self.veh_trajectory)
                    elif self.vehicle_type[i] == 'Human_vehicle':
                        #v1=IDM_model(v1, v2, x1, x2)
                        x1=self.veh_trajectory['Pos'].iloc[-1] 
                        v1=self.veh_trajectory['Speed'].iloc[-1]
                        veh_tmp=self.veh_trajectory.loc[self.veh_trajectory['No'] == i]
                        #print('veh_tmp',veh_tmp)
                        veh_index=veh_tmp.index.max()
                        #print('veh_index',veh_index)
                        #print('veh_trajectory[]',self.veh_trajectory['Speed'].iloc[veh_index])
                        #print('veh_trajectory[',self.veh_trajectory['Acc'].iloc[veh_index])
                        #print('at',self.veh_trajectory['Acc'].iloc[veh_index]*time_step)
                        v2=self.veh_trajectory['Speed'].iloc[veh_index] + self.veh_trajectory['Acc'].iloc[veh_index] * time_step
                        #print('v2',v2)
                        x2=self.veh_trajectory['Pos'].iloc[veh_index] + v2 * time_step + 0.5 *\
                            self.veh_trajectory['Acc'].iloc[veh_index] * time_step * time_step
                        acc = IDM_model(v1, v2, x1, x2);  # 2代表本车，1代表前车
                        light_state = light_all[time]
                        predict_light = light_predict(predict_horizon = 50,predict_origin_list = light_all,start_index = time)
                        self.veh_trajectory.loc[self.veh_trajectory.shape[0]] = [i,'Human_vehicle',v2,x2,acc,time,light_state,predict_light]
                    else:
                        pass
            elif Pos1 > 400 and (time - start_time) > 0:
                break
            else:
                pass
multi_vehicle=Multi_vehicle()
multi_vehicle.update()
multi_vehicle.veh_trajectory.to_csv(r'2.csv', index = False)
fig = draw_single_veh(multi_vehicle.veh_trajectory, start_time, 'Time-space Map')  # 调用绘图函数
