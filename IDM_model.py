'''
class IDM_model():
    def __init__(self,length,width,v1,v2,x1,x2):
        self.length=3
        self.width=2
        self.v1=v1
        self.v2=v2
        self.x1=x1
        self.x2=x2
    def update_D_dis(self):
        dV=self.v1-self.v2
        s = self.s0 + self.T * self.v2 + (self.v2 * dV) / (2 * (self.A * self.B) ** 0.5)  #跟车距离
    def 
    ##静态参数
    A = 2  #自车最大加速度
    dta = 2 ##derta上标
    s0 = 2  #最小距离
    T = 1.5    
    B = 2  #舒适减速度
    Vf = 11.11  #期望速度
    def update_model(self,v1,v2,x1,x2)
        s = self.s0 + self.T * v2 + (v2 * (v2 - v1)) / (2 * (self.A * self.B) ** 0.5)  #跟车距离
        a = self.A * (1 - (self.v2 / self.Vf) ** self.dta - (s / (self.x1 - self.x2)) ** 2)  #
        return round(a, 5)
'''


'''在IDM模型中加入了uncertainty'''
import numpy as np
import pandas as pd
def IDM_model(v1, v2, x1, x2):  # 2代表本车，1代表前车
    A = 2  #自车最大加速度
    dta = 2 ##derta上标
    s0 = 5  #最小距离
    T = 0.5 #   
    B = 2  #舒适减速度
    Vf = 11.11  #期望速度
    mean=0 #均值
    std=1 #标准差
    size=1000 #随机数数量
    noise=np.random.normal(mean,std,size)
    s = s0 + T * v2 + (v2 * (v2 - v1)) / (2 * (A * B) ** 0.5)  #跟车距离
    print('x1-x2',x1-x2)
    print('Vf',Vf)
    a = A * (1 - (v2 / Vf) ** dta - (s / (x1 - x2)) ** 2) #
    
    return round(a, 5)


