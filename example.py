# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 17:50:51 2021

@author: erfa_zhang
"""

# -*- coding: UTF-8 -*- 

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']   
mpl.rcParams['axes.unicode_minus'] = False           

import matplotlib.pyplot as plt
import random
import numpy as np
import xlwt

class Car:
    def _init_(self,length,width):
        self.length = length #车长
        self.width = width #车宽
        
    length = 3
    width = 1.6
    
    # 静态参数
    S = 2 #静态车间距
    T = 1.6 #车头时距
    aMax = 0.73 #最大期望加速度
    bMax = 1.67 #最大期望减速度
    V0 = 33.3 #m/s  
    par = 0.6
    
    # 更新车间距
    def updateD(self):
        if self.X > self.front.X: 
            self.D =(self.X - - Road.length) - self.front.X + self.front.length 

            print('self.X - self.front.X + self.front.length - Road.length', self.X ,self.front.X , self.front.length , Road.length)
            print('Road',Road.length)
        else:
            self.D = self.X - self.front.X + self.front.length
    # 更新速度差
    def updatedV(self):
        self.dV = self.V - self.front.V
    # 更新期望跟驰距离
    def updateD_des(self):
        self.updatedV()
        self.D_des = self.S + self.V*self.T +self.V*self.dV/(2*(self.aMax*self.bMax)**0.5)
    # 更新加速度
    def updateA(self):
        self.updateD()
        self.updateD_des()
        self.A = self.par*(1-(self.V/self.V0)**4-(self.D_des/self.D)**2)

    # 更新速度
    def updateV(self):
        self.updateA()
        self.V = max(self.V + self.A,0)
    # 更新位置
    def updateX(self):
        self.updateV()
        self.X = (self.X + self.V)%Road.length
    
class Road:
    def _init_(self,motorPercent,length,width):
        self.motorPercent = motorPercent #道路占有率
        self.length = length #车道长度
        self.width = width #车道宽度
        
    motorPercent = 0.9
    length = 1000
    width = 3.5

    # 车辆加载
    def initCars(self):
        # 计算车辆数量
        self.numOfCar = (int)(self.length/3 * self.motorPercent)
        # 生成车辆
        self.ls = []
        gas = (self.length / self.numOfCar)
        for i in range(self.numOfCar):
            c = Car()
            # 等间距放置车辆
            c.X = i*gas
            c.Y = 1
            c.V = 0
            c.index = i
            self.ls.append(c)
            # 新增车辆为上一辆车的前车
            if i>0:
                c.back = self.ls[self.ls.index(c)-1]
                c.back.front = c
            # 设置车辆颜色
            c.color = [random.random(),random.random(),random.random()]
        #头车的前车是尾车
        self.ls[len(self.ls)-1].front = self.ls[0] 
        self.ls[0].back = self.ls[len(self.ls)-1]
        
    # 画图函数
    def draw(self,ax):
        # 绘制道路
        rect = plt.Rectangle([0,0],self.length,self.width, fill=False,facecolor = [0,0,1])
        ax.add_patch(rect)
        plt.xlim(0,1000)
        plt.ylim(-50,50)
        # 绘制车辆
        for i in range(self.numOfCar):
            c = self.ls[i]
            rect = plt.Rectangle([c.X,c.Y],c.length,c.width,facecolor=c.color)
            ax.add_patch(rect)
        
    # 运行效果展示函数
    def show(self,timeMax):
        # 初始化道路        
        self.initCars()
        # 创建画布
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # 开始仿真
        for time in range(timeMax):
            # 车辆位置更新
            for c in self.ls:
                c.updateX()
            # 汇出更新后车辆位置
            plt.cla()
            self.draw(ax)
            plt.pause(0.1)
        # 删除画布
        plt.close(fig)

# 获取输入道路的平均车速
def get_vMean(r,timeMax):
    # 初始化道路        
    r.initCars()
    # 开始仿真
    ls = []
    vSum = 0
    for time in range(timeMax):
        # 车辆位置更新
        for c in r.ls:
            c.updateX()
            vSum = vSum + c.V
        # 计算当前时间步平均车速
        ls.append(vSum / r.numOfCar)
        vSum = 0
    return sum(ls)/len(ls)

def get_K_V():
    V_ls = np.zeros((1,20))
    K_ls = np.zeros((1,20))
    timeMax = 1000
    j = 0
    for i in np.arange(0.05,1,0.05):
        r = Road()
        r.motorPercent = i
        V_ls[0,j] = get_vMean(r, timeMax)
        K_ls[0,j] = r.numOfCar/r.length*1000
        j = j + 1
    return (K_ls,V_ls)

def plot_K_V(K_V):
    # 输入元组第0个元素为密度，第1个元素为平均速度
    ls_K = []
    ls_V = []
    for i in range(19):
        ls_K.append(K_V[0][0,i])
        ls_V.append(K_V[1][0,i])
    plt.plot(ls_K,ls_V)
    plt.title('密度-速度图',fontsize = 35)        
    plt.xlabel('密度',fontsize = 30)             
    plt.ylabel('速度',fontsize = 30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

def plot_K_Q(K_V):
    # 输入元组第0个元素为密度，第1个元素为平均速度
    ls_K = []
    ls_Q = []
    for i in range(19):
        ls_K.append(K_V[0][0,i])
        ls_Q.append(K_V[1][0,i]*K_V[0][0,i])
    plt.plot(ls_K,ls_Q)
    plt.title('密度-流量图',fontsize = 35)        
    plt.xlabel('密度',fontsize = 30)             
    plt.ylabel('流量',fontsize = 30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
   
def plot_V_Q(K_V):
    # 输入元组第0个元素为密度，第1个元素为平均速度
    ls_V = []
    ls_Q = []
    for i in range(19):
        ls_V.append(K_V[1][0,i])
        ls_Q.append(K_V[1][0,i]*K_V[0][0,i])
    plt.plot(ls_Q,ls_V)
    plt.title('速度-流量图',fontsize = 35)        
    plt.xlabel('流量',fontsize = 30)             
    plt.ylabel('速度',fontsize = 30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
def xlsWrite(K_V):
    workbook = xlwt.Workbook(encoding= 'ascii')
    worksheet = workbook.add_sheet("三参数")
    worksheet.write(0,0, "密度")
    worksheet.write(0,1, "速度")
    worksheet.write(0,2, "流量")
    for i in range(20):
        worksheet.write(i+1,0,K_V[0][0,i])
        worksheet.write(i+1,1,K_V[1][0,i])
        worksheet.write(i+1,2,K_V[0][0,i]*K_V[1][0,i])
    fname = "C:\OneDrive\文档\文本仓库\课程ppt\学习\交通仿真\IDM_python实现\三参数数据.xls"
    workbook.save(fname)
     
#################### main
# 创建道路
r = Road()
# 展示动画
################### 动画展示
r.motorPercent = 0.3
timeMax = 100
r.show(timeMax)
################### 三参数图
# 获取占有率-速度
#K_V = get_K_V()
# 汇出密度-速度图
#plot_K_V(K_V)
# 汇出密度-流量图
#plot_K_Q(K_V)
# 汇出速度-流量图
################### 导出数据
#plot_V_Q(K_V)
# 写入excel文件
#xlsWrite(K_V)
################### V0变化
# for i in range(1,7):
#     Car.V0 = 5*i
#     K_V = get_K_V()
#     plot_K_Q(K_V)
#     #plot_V_Q(K_V)
#     #plot_K_V(K_V)
# plt.legend(labels=['5','10','15','20','25','30'],fontsize = 20)
################### S变化
# for i in range(1,6):
#     Car.S = i
#     K_V = get_K_V()
#     #plot_K_Q(K_V)
#     #plot_V_Q(K_V)
#     plot_K_V(K_V)
# plt.legend(labels=['1','2','3','4','5'],fontsize = 20)
################### par变化
# for i in range(1,6):
#     Car.par = i*0.2
#     K_V = get_K_V()
#     plot_K_Q(K_V)
#     #plot_V_Q(K_V)
#     #plot_K_V(K_V)
# plt.legend(labels=['0.2','0.4','0.6','0.8','1'],fontsize = 20)

