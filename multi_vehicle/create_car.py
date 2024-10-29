import random
import pandas as pd

# 构建车辆矩阵
def create_volume(volume, file):
    route = 'E01'
    t = 0  # 初始时刻
    num = 0  # 车辆编号
    rate = volume  # 设置交通流量
    end_time = 3600
    # 1.1：753  1:685   0.9：616   0.8:548    0.7:479    0.6:411    0.5:342
    # 构建pandas矩阵
    veh_inf = pd.DataFrame(columns=['id', 'departLane', 'type', 'route', 'depart'])
    N = 0 # 车辆计数
    while t < end_time:
        random1 = random.expovariate(rate/3600)  # 参数为每小时交通量/3600 = 每秒到达的车辆数
        data = [num, '0', 'Auto_vehicle', route, t]
        veh_inf.loc[veh_inf.shape[0]] = data
        t = int(t + random1)  #这里取整的方式是直接截断小数部分
        N += 1
        num += 1

    veh_inf.sort_values(by='depart') #按照出发时间排序
    veh_inf.to_csv(file + '.csv')  #
    veh1 = veh_inf[veh_inf['depart'] <= end_time/2]  #

    return veh1.shape[0] #

def create_MPR(MPR, file, file1):  # MPR : CAV渗透率
    v = []
    veh_inf = pd.read_csv(file)
    num = veh_inf.shape[0]
    cav_num = int(num * MPR)
    for i in range(cav_num):
        v.append(i)  # 1为自动车
    for i in range(num - cav_num):
        v.append(cav_num + i)
    for i in range(num):
        a = random.choice(v)
        if a < cav_num:
            type = 'Auto_vehicle'
            v.remove(a)
        else:
            type = 'Human_vehicle'
            v.remove(a)
        veh_inf['type'].loc[i] = type
    veh_inf.to_csv(file1 + '.csv')
    auto = veh_inf[veh_inf['type'] == 'Auto_vehicle']
    return auto.shape[0]/num



if __name__ == "__main__":
    num = create_volume(1000, 'volume1')
    '''
    饱和度：
    1.1：1100  1:1000   0.9：900   0.8:800    0.7:700    0.6:600    0.5:500 
    '''
    mpr = create_MPR(0.6, 'volume1', 'MPR1')
    print(num, mpr)
