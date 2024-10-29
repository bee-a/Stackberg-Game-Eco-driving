import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# 导入字体属性相关的包或类
from matplotlib.font_manager import FontProperties
# 预设字体类型、大小
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


def single_acc(veh_trajectory):
    mintime = veh_trajectory['time'].min()
    maxtime = veh_trajectory['time'].max()
    maxX = veh_trajectory['Pos'].max()
    minX = veh_trajectory['Pos'].min()

    # 设置画布
    plt.figure(figsize=(6, 8))

    # 提取车辆编号
    x_vehID = veh_trajectory.drop_duplicates(['No'])
    q = x_vehID.shape[0]
    # 按照时间先后顺序排序
    x_vehID = x_vehID.sort_values(by='time')
    # 对排序后的车辆 ID 的索引进行重置，方便索引
    x_vehID = x_vehID.reset_index(drop=True)

    i = 0
    while i <= (len(x_vehID) - 1):
        # 循环绘制轨迹图
        cardata = veh_trajectory[veh_trajectory.No == x_vehID['No'][i]]
        # 将时间赋值给变量 x
        x = cardata['time']
        # 位置并赋值给变量 y
        y = cardata['Pos']
        # 将速度赋值给变量 v，同时定义速度为颜色映射
        # v = cardata['Speed']
        # 设定每个图的colormap和colorbar所表示范围是一样的，即归一化
        # norm = matplotlib.colors.Normalize(vmin=0, vmax=25)
        # 绘制散点图
        if cardata['VehType'].iloc[0] == 'traditionCar':
            ax = plt.scatter(x, y, marker='.', s=1, c='b', alpha=0.7)
        else:
            ax = plt.scatter(x, y, marker='.', s=1, c='r', alpha=0.7)
        i = i + 1
    # plt.show()

    # 添加颜色条
    plt.clim(0, 100)

    plt.xticks([0, 300, 600, 900, 1200, 1500, 1800])
    plt.yticks([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300])

    plt.axis([mintime, maxtime, minX, maxX])

    ax1 = plt.gca()
    ax1.set_title('Time-Space Figure')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Distance (m)')
    plt.show()
    return 1
