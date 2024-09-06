import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# 导入字体属性相关的包或类
from matplotlib.font_manager import FontProperties
# 预设字体类型、大小
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def draw_single_veh(lanedata, start_time, map_name):
    # 设置画布
    plt.figure(figsize=(6, 8))
    for i in range(lanedata.shape[1]):
        # 提取车辆编号
        x_vehID = lanedata.drop_duplicates(['No'])
        
        #提取编号的行数
        q = x_vehID.shape[0]
        print('q的大小',q)
        # 按照时间先后顺序排序
        x_vehID = x_vehID.sort_values(by='recent_time')
        # 对排序后的车辆 ID 的索引进行重置，方便索引
        x_vehID = x_vehID.reset_index(drop=True)
        #print('x_vehID',len(x_vehID))
        mintime = lanedata['recent_time'].min()
        maxtime = lanedata['recent_time'].max()
        maxX = lanedata['Pos'].max()
        minX = lanedata['Pos'].min()
###############################################################
    i = 0
    time=[]
    axis=[]
    velocity=[]
    while i <= (len(x_vehID) - 1):
        # 循环绘制轨迹图
        cardata = lanedata[lanedata.No == x_vehID['No'][i]]
        # 将时间赋值给变量 x
        x = cardata['recent_time']
        time.append(x)
        # 位置并赋值给变量 y
        y = cardata['Pos']
        axis.append(y)
        #x1=cardata['recent_time']
        y1=cardata['Speed']
        velocity.append(y1)
        # 将速度赋值给变量 v，同时定义速度为颜色映射
        # v = cardata['Speed']
        # 设定每个图的colormap和colorbar所表示范围是一样的，即归一化
        # norm = matplotlib.colors.Normalize(vmin=0, vmax=25)
        # 绘制散点图
        
        '''
                if cardata['VehType'].iloc[0] == 'traditionCar':
            ax = plt.plot(x, y, marker='.', s=1, c='b', alpha=0.7)  #绘制散点图 
        else:
            ax = plt.plot(x, y, marker='.', s=1, c='r', alpha=0.7)
        i = i + 1
        '''
        i=i+1
    #print('axis',axis)
    #print('velocity',velocity)
    # plt.show()

    # 添加颜色条
    #plt.clim(0, 100)  #颜色映射范围显示数据

    '''
    plt.axhline(xmin=20/126,xmax=26/126,y=300,color='y',linewidth='1') #
    plt.axhline(xmin=26/126,xmax=86/126,y=300,color='r')
    #plt.axline(xy1=(60,300),xy2=(66,300),slope=0)
    #plt.axline(xy1=(66,300),xy2=(126,300),slope=0)   
    
    '''
    point1=[60,300]
    point2=[66,300]
    point3=[126,300]
    x1_values=[point1[0],point2[0]]
    y1_values=[point1[1],point2[1]]
    x2_values=[point2[0],point3[0]]
    y2_values=[point2[1],point3[1]]

    ''' 画黄灯和红灯'''
    plt.plot(x1_values,y1_values,color='gold') 
    plt.plot(x2_values,y2_values,color='red')
    plt.plot(time,velocity,color='b')
    plt.xticks([40, 60, 66, 126])  # 设置x轴刻度
    plt.yticks([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]) #设置y轴刻度
    plt.axis([mintime, maxtime, minX, maxX])

    ax1 = plt.gca()#挪动坐标轴
    ax1.set_title('Time-Space Figure')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Distance (m)')
    plt.show()
    return 1
