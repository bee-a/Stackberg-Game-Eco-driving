import casadi as ca
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from casadi import Function, if_else
from datetime import datetime 

class NMPC():
    def __init__(self):
        self.L = 10000
        self.D = 300
        self.G0 = 1 #引力场的权重
        self.G1 = 10 #黄灯斥力场
        self.G2 = 0.3 #生态驾驶
        self.G3 = 1.0 #速度场
        self.G4 = 1.0 #交互项
        self.G5 = 1.0
        self.a1 = 1
        self.a2 = 1000
        self.a3 = 1000
        self.b1 = 100
        self.max_iter = 126
        self.u_max = 5
        self.u_min = -5
        self.w_influence = 1.0
        self.T = 0.5
        self.N = 20
        self.v_free = 20
        self.r1 = self.v_free*200
        self.r2 = (self.u_max*3*3)/2
        self.vehicle_type = ['Auto_vehicle', 'Human_vehicle', 'Auto_vehicle', 'Human_vehicle', 'Auto_vehicle', 'Human_vehicle', 'Auto_vehicle', 'Human_vehicle', 'Auto_vehicle', 'Human_vehicle']
        #self.vehicle_type = ['Auto_vehicle', 'Human_vehicle']
        self.num_l=self.vehicle_type.count('Auto_vehicle')
        self.num_f=self.vehicle_type.count('Human_vehicle')
        self.state_l = np.ones((1,2 * self.num_l))
        self.state_f = np.ones((1,2 * self.num_f))
        self.control_l = np.ones((1, self.num_l))
        self.control_f = np.ones((1, self.num_f))
        self.IDM_state = np.ones((1,2 * self.num_f))

    def traffic_light(self,time):
        light = []
        green_time = 30
        red_time = 30
        yellow_time=4
        time_step=0.5
        real_light = []
        
        for i in range(int(green_time/time_step)): #60
            light.append(0)
        for i in range(int(yellow_time / time_step)): #6
            light.append(1)
        for i in range(int(red_time / time_step)):  #60
            light.append(2)

        period = int((green_time + yellow_time+red_time) /time_step)

        current_t=int((time % period) )
        #light = light[current_time_in_period:]\
        print(f"current_t is {period}")
        for i in range(current_t,current_t + 126):
            if i >= period:
                real_light.append(light[i % period])
            else:
                real_light.append(light[i])


        # if 0 < current_time_in_period < green_time:
        #     light[current_time_in_period] = 0
        # elif green_time <= current_time_in_period < green_time + yellow_time:
        #     light[current_time_in_period] = 1
        # elif green_time+yellow_time<=current_time_in_period< green_time + yellow_time + red_time:
        #     light[current_time_in_period]=2
        # else:
        #     pass
        return real_light, current_t
        
        
    def predict_light(self,horizon,predict_origin_list,start_point):
        light = predict_origin_list
        predict_list = []
        period = len(light)
        trans_point = start_point%period

        for i in range(trans_point , trans_point + horizon):
            if i >= period:
                predict_list.append(light[i%period])
            else:
                predict_list.append(light[i])
        return predict_list

    def leader(self,pred_state_f,pred_state_l,pred_control_l,predict_list):
        current_state_l = pred_state_l[0,:]
        self.opti_l = ca.Opti()
        self.opti_controls_l = self.opti_l.variable(self.N,self.num_l)
        self.opti_states_l = self.opti_l.variable(self.N+1,2)

        ''' TODO: 这个约束添加的意义是什么 '''

        self.opt_l_x0 = self.opti_l.parameter(1,2*self.num_l)
        
        for i in range(self.num_l-1):
            new_var=self.opti_l.variable(self.N+1, 2)
            self.opti_states_l = ca.horzcat(self.opti_states_l,new_var)
        
        self.opti_position_l = self.opti_states_l[:, 0]
        self.opti_velocity_l = self.opti_states_l[:, 1]

        for i in range(2,self.num_l,2):
            self.opti_position_l = ca.horzcat(self.opti_position_l,self.opti_states_l[:, i])
            self.opti_velocity_l = ca.horzcat(self.opti_velocity_l,self.opti_states_l[:, i+1])

        self.obj_l = 0
        for i in range(self.num_l):
            for t in range(self.N):
                j = 2*i
                if j>= self.opti_states_l.size()[1]:
                    raise IndexError(f"Index out of bounds: j={j},max index={self.opti_states_l.size()[1]-1}")
                if t == 0:
                    if predict_list[t] == 0.0: #绿灯
                        self.obj_l += self.G0 * self.a1 * (self.L-self.opti_states_l[t, j])\
                        + self.G2 * self.opti_controls_l[t, i] * self.opti_controls_l[t, i]+\
                        if_else(self.state_l[-1,j] < self.D, self.G3 * np.exp(- (((self.D - self.opti_states_l[t, j]) ** 2) / self.r1)) * (self.v_free - self.opti_states_l[t,j+1]),self.G3 * (self.v_free - self.opti_states_l[t,j+1]))\
                        + self.G4 * (self.opti_states_l[t, j+1]-pred_state_f[t, j+1]) **2
                    
                    elif predict_list[t] == 1.0:    #黄灯
                        self.obj_l +=  self.G0 * self.a1 * (self.L - self.opti_states_l[t, j])+\
                        if_else(self.state_l[-1,j] < self.D,(self.G1 * self.a2 /(self.D + self.r2 - self.opti_states_l[t, j])),0)\
                        + self.G2 * self.opti_controls_l[t, i] * self.opti_controls_l[t, i]+\
                        if_else(self.state_l[-1,j] < self.D, self.G3 * np.exp(-(((self.D - self.opti_states_l[t, j]) ** 2) / self.r1)) * (self.v_free - self.opti_states_l[t, j+1]),self.G3 * (self.v_free - self.opti_states_l[t, j+1]) )\
                        + self.G4 * (self.opti_states_l[t, j+1] - pred_state_f[t, j+1])**2

                    elif predict_list[t] == 2.0:  #红灯
                        self.obj_l += self.G0 * self.a1 * (self.L - self.opti_states_l[t, j])+\
                        if_else(self.state_l[-1,j] < self.D,(self.G1 * self.a3 /(self.D - self.opti_states_l[t, j])),0)\
                        + self.G2 *self.opti_controls_l[t, i] *self.opti_controls_l[t, i]+\
                        if_else(self.state_l[-1,j] < self.D, self.G3 * np.exp(-(((self.D - self.opti_states_l[t, j]) ** 2) / self.r1)) * (self.v_free - self.opti_states_l[t, j+1]),self.G3 * (self.v_free - self.opti_states_l[t, j+1]))\
                        + self.G4*(self.opti_states_l[t, j+1]-pred_state_f[t, j+1])**2
                    else:
                        print('worry')
                else:
                    if predict_list[t] == 0.0: #绿灯
                        self.obj_l += self.G0 * self.a1 * (self.L-self.opti_states_l[t, j])
                        + self.G2 * self.opti_controls_l[t, i] * self.opti_controls_l[t, i]\
                        + self.G3 * (self.v_free-self.opti_states_l[t, j+1]) * (self.v_free-self.opti_states_l[t, j+1])\
                        + self.G4 * (self.opti_states_l[t, j+1]-pred_state_f[t, j+1]) **2
                    
                    elif predict_list[t] == 1.0:    #黄灯
                        if self.state_l[-1,j] < self.D:
                            self.obj_l +=  self.G0 * self.a1 * (self.L - self.opti_states_l[t, j])\
                            + self.G1 * self.a2 /(self.D + self.r2 - self.opti_states_l[t, j])\
                            + self.G2 * self.opti_controls_l[t, i] * self.opti_controls_l[t, i]
                            + self.G3 * np.exp(- (((self.D + self.r2 - self.opti_states_l[t, j]) ) / self.r1)** 2) * (self.v_free - self.opti_states_l[t,j+1])\
                            + self.G4 * (self.opti_states_l[t, j+1] - pred_state_f[t, j+1])**2
                        else:
                            self.obj_l +=  self.G0 * self.a1 * (self.L - self.opti_states_l[t, j])\
                            + self.G2 * self.opti_controls_l[t, i] * self.opti_controls_l[t, i]\
                            + self.G3 * (self.v_free - self.opti_states_l[t, j+1]) * (self.v_free - self.opti_states_l[t, j+1])\
                            + self.G4 * (self.opti_states_l[t, j+1] - pred_state_f[t, j+1])**2                        

                    elif predict_list[t] == 2.0:  #红灯
                        if self.state_l[-1,j] < self.D:
                            self.obj_l += self.G0 * self.a1 * (self.L - self.opti_states_l[t, j])\
                            +self.G1 * self.a3 /(self.D - self.opti_states_l[t, j])\
                            + self.G2 *self.opti_controls_l[t, i] *self.opti_controls_l[t, i]
                            + self.G3 * np.exp(- (((self.D - self.opti_states_l[t, j]) ) / self.r1)** 2) * (self.v_free - self.opti_states_l[t,j+1])\
                            + self.G4*(self.opti_states_l[t, j+1]-pred_state_f[t, j+1])**2
                        else:
                            self.obj_l += self.G0 * self.a1 * (self.L - self.opti_states_l[t, j])\
                            + self.G2 *self.opti_controls_l[t, i] *self.opti_controls_l[t, i]\
                            + self.G3 * (self.v_free - self.opti_states_l[t, j+1]) * (self.v_free - self.opti_states_l[t, j+1])\
                            + self.G4*(self.opti_states_l[t, j+1]-pred_state_f[t, j+1])**2  
                    else:
                        print("worry")                      


        self.opti_l.minimize(self.obj_l)
        '''添加车辆约束'''

        #self.opti_l.subject_to(self.opti_l.bounded(self.u_min,))
        
        for i in range(self.num_l):
            for t in range(self.N):
                j = 2*i
                p_next_1 = self.opti_states_l[t,j] + self.opti_states_l[t,j+1] * self.T + 0.5 * self.T * self.T *self.opti_controls_l[t,i] #p_next是指它的位置
                v_next_1 = self.opti_states_l[t,j+1] + self.T *self.opti_controls_l[t,i] #v是指它的速度
                x_next_1 = ca.horzcat(p_next_1, v_next_1) #水平方向连接矩阵，竖直方向连接矩阵需要ca.vertcat()函数
                self.opti_l.subject_to(self.opti_states_l[t+1, j:j+2] == x_next_1)
        
        self.opti_l.subject_to(self.opti_states_l[0,:] == self.opt_l_x0)
        
        for i in range(self.num_l):

            self.opti_l.subject_to(self.opti_l.bounded(self.u_min, self.opti_controls_l[:,i], self.u_max))
            self.opti_l.subject_to(self.opti_l.bounded(4, self.opti_states_l[:,2*i+1], self.v_free))        

        self.opti_l.set_initial(self.opti_controls_l, pred_control_l)
        self.opti_l.set_initial(self.opti_states_l, pred_state_l)
        self.opti_l.set_value(self.opt_l_x0, current_state_l)

        
        opts_setting = {'ipopt.max_iter': 1000, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
                    'ipopt.acceptable_obj_change_tol': 1e-6}

        self.opti_l.solver('ipopt',opts_setting)
        

        solution_l = self.opti_l.solve()

        pred_controls_l = solution_l.value(self.opti_controls_l)
        pred_states_l = solution_l.value(self.opti_states_l)
        pred_control_l = np.concatenate((pred_controls_l[1:],pred_controls_l[-1:]))
        pred_state_l = np.concatenate((pred_states_l[1:],pred_states_l[-1:]))
        #cur_control_l = pred_controls_l[0,:].reshape(1,-1)
        cur_control_l = pred_controls_l[0]
        cur_v_l = np.ones((self.N,self.num_l))
        self.control_l = np.vstack((self.control_l,cur_control_l))
        for i in range(self.num_l):
            cur_v_l = pred_state_l[1,2*i+1]
            cur_x_l = pred_state_l[1,2*i]

        #print(f"cur_control_l  is  {cur_control_l}")
        new_element = np.array([[cur_x_l,cur_v_l]])
        cur_state = pred_state_l[0,:].reshape(1,-1)
        self.state_l = np.vstack((self.state_l, cur_state))
        return pred_control_l,pred_state_l

    def follower(self,pred_state_l,pred_state_f,pred_control_f):
        current_state_f = pred_state_f[0,:]
        current_state_IDM = self.IDM_state[-1,0]
        self.K_s = 1
        self.K_e = 1
        self.sigma = 2
        self.LV = 5
        self.K_s_log = 10
        self.K_v_log = 1

        self.opti_f = ca.Opti()
        self.opti_controls_f = self.opti_f.variable(self.N,self.num_f)
        self.opti_states_f=self.opti_f.variable(self.N+1,2*self.num_f)
        self.opt_f_x0 = self.opti_f.parameter(1,2*self.num_f)
        self.opti_f.subject_to(self.opti_states_f[0,:] == self.opt_f_x0)

        self.opti_position_f = self.opti_states_f[:, 0]
        self.opti_velocity_f = self.opti_states_f[:, 1]

        for i in range(2,self.num_f,2):
            self.opti_position_f = ca.horzcat(self.opti_position_f,self.opti_states_f[:, i])
            self.opti_velocity_f = ca.horzcat(self.opti_velocity_f,self.opti_states_f[:, i+1])
        
        self.obj_f=0        
        for i in range(self.num_f):
            for t in range(self.N+1):
                j=2*i
                self.obj_f += self.K_v_log * ((self.opti_states_f[t,j+1] - pred_state_l[t,j+1])**2 + self.sigma)\
                            + self.K_s_log / ((self.opti_states_f[t,j] - pred_state_l[t,j])**2 + self.LV)\
                            + self.K_e * ((self.opti_states_f[t,j+1] - self.v_free)**2 + self.sigma)

        self.opti_f.minimize(self.obj_f)

        ''' 添加车辆约束 '''
        for i in range(self.num_f):   
            for t in range(self.N):
                j=2*i
                p_next_f = self.opti_states_f[t,j] + self.opti_states_f[t,j+1] * self.T + 0.5 * self.T * self.T * self.opti_controls_f[t,i] #p_next是指它的位置
                v_next_f = self.opti_states_f[t,j+1] + self.T * self.opti_controls_f[t,i] #v是指它的速度
                x_next_f = ca.horzcat(p_next_f, v_next_f) #水平方向连接矩阵，竖直方向连接矩阵需要ca.vertcat()函数
                self.opti_f.subject_to(self.opti_states_f[t+1, j:j+2] == x_next_f)

        """ 注意这里的约束不能太多 """

        for i in range(self.num_f):
            self.opti_f.subject_to(self.opti_f.bounded(self.u_min, self.opti_controls_f[:,i], self.u_max))  
            self.opti_f.subject_to(self.opti_f.bounded(4, self.opti_states_f[:,2*i+1], self.v_free))
        #self.opti_f.subject_to(self.opti_f.bounded(0, self.opti_velocity_f, self.v_free+10))
                                                                                                                                        
        #pred_control_f=np.zeros((self.N, self.num_f))
        #pred_state_f = np.ones((self.N+1, 2*self.num_f))
    
        self.opti_f.set_initial(self.opti_controls_f, pred_control_f)
        self.opti_f.set_initial(self.opti_states_f, pred_state_f)
        self.opti_f.set_value(self.opt_f_x0, current_state_f)        
        
        opts_setting = {'ipopt.max_iter': 1000, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
                    'ipopt.acceptable_obj_change_tol': 1e-6}
        self.opti_f.solver('ipopt',opts_setting)

        # num_constraint = self.opti_f.g.size1()
        # print("Number of constraints: ", num_constraint)
        #try:
        solution_f = self.opti_f.solve()
        #except RuntimeError as e:
            # print("Solver failed. Debugging values:")
            # opti_controls_f_values = self.opti_f.debug.value(self.opti_controls_f)
            # opti_states_f_values = self.opti_f.debug.value(self.opti_states_f)
            # print("opti_controls_f:", opti_controls_f_values)
            # print("opti_states_f:", opti_states_f_values)
        pred_controls_f = solution_f.value(self.opti_controls_f)
        pred_states_f = solution_f.value(self.opti_states_f)    
        pred_control_f = np.concatenate((pred_controls_f[1:],pred_controls_f[-1:]))
        pred_state_f = np.concatenate((pred_states_f[1:],pred_states_f[-1:]))
        #cur_control_f=pred_control_f[0,:].reshape(1,-1)
        cur_control_f=pred_control_f[0]
        #cur_control_f = pd.DataFrame(cur_control_f, columns = ["u1","u2","u3","u4","u5"])
        self.control_f = np.vstack((self.control_f,cur_control_f))
        cur_state_f = pred_state_f[0,:].reshape(1,-1)
        #cur_state_f = pd.DataFrame(cur_state_f, columns = ["x1","v1","x2","v2","x3","v3","x4","v4","x5","v5"])
        #print(f"cur_control_f  {cur_control_f}")
        return pred_control_f, pred_state_f

    def IDM(self,v1,v2,x1,x2):# 2代表本车，1代表前车
        A = 2  #自车最大加速度
        dta = 2 ##derta上标
        s0 = 5  #最小距离
        T = 0.5 #   
        B = 2  #舒适减速度
        Vf = 20  #期望速度
        mean=0 #均值
        std=1 #标准差
        size=1 #随机数数量
        noise=np.random.normal(mean,std,size)
        noise = noise[0]
        s = s0 + T * v2 + (v2 * (v2 - v1)) / (2 * (A * B) ** 0.5)  #跟车距离
        a = A * (1 - (v2 / Vf) ** dta - (s / (x1 - x2)) ** 2) +noise #
        return a

    # TODO: 这里的IBR迭代的过程中，follower是有问题的 
    ''' IBR是针对L-F博弈，但是MPC的滚动优化是针对MPC，二者不冲突 '''

    ''' IBR要收敛到stackberg，先证明收敛性，再求 '''
    def IBR(self):

        random_int = np.random.randint(0,126)
        self.light, self.start_time = self.traffic_light(random_int)
        predict_list = self.predict_light(self.N, self.light, random_int)
        print(f"predict_list { predict_list}")
        ''' 初始化leader的state,follower的state '''
        pred_state_l=np.ones((self.N+1,2*self.num_l))
        for i in range(self.num_l):
            pred_state_l[:, 2*i]=12.5 + i*15
            pred_state_l[:, 2*i+1]=5 
            self.state_l[0,2*i] = 12.5 + i*15
            self.state_l[0,2*i+1] = 5

        pred_state_f=np.ones((self.N+1,2*self.num_f))
        for i in range(self.num_f):
            pred_state_f[:,2*i] = 5 +15*i
            pred_state_f[:,2*i+1] = 5
            self.state_f[0,2*i] = 5+15*i
            self.state_f[0,2*i+1] = 5

        pred_control_f = np.zeros((self.N, self.num_f))
        pred_control_f[0,:] = 5
        pred_control_l = np.zeros((self.N, self.num_l))

        for t in range(self.start_time, self.start_time + self.max_iter):
            new_element_f = np.ones((1,2*self.num_f)).reshape(1,-1)
            if t == self.start_time:
                predict_list = self.predict_light(self.N, self.light, t)
                ''' 这里输入值和输出值的命名一致'''
                pred_control_f,pred_state_f = self.follower(pred_state_l,pred_state_f,pred_control_f)
                # IDM的计算
                for i in range(self.num_f):
                    j = 2*i
                    v1 = pred_state_l[0,j+1]
                    x1 = pred_state_l[0,j] 
                    v2 = pred_state_f[0,j+1]
                    x2 = pred_state_f[0,j]
                    a_f = self.IDM(v1,v2,x1,x2)
                    v_f = v2 + self.T * a_f
                    x_f = x2 + v2 * self.T + 0.5 * a_f * self.T * self.T
                    #pred_state_f[1,j] = x_f
                    new_element_f[0,j] = x_f
                    new_element_f[0,j+1] = v_f
                    print(f"new_element_f is {new_element_f}")
                self.IDM_state = np.vstack((self.IDM_state,new_element_f)) 
                self.state_f = np.vstack((self.state_f,pred_state_f[1,:]))  
                
                #self.state_f = np.vstack((self.state_f,pred_state_f[0,:]))
                pred_control_l, pred_state_l = self.leader(pred_state_f, pred_state_l, pred_control_l,predict_list)

            else:
                predict_list = self.predict_light(self.N, self.light, t)
                pred_control_f,pred_state_f = self.follower(pred_state_l, pred_state_f,pred_control_f)
                for i in range(self.num_f):
                    j = 2*i
                    v1 = pred_state_l[0,j+1]
                    x1 = pred_state_l[0,j] 
                    v2 = pred_state_f[0,j+1]
                    x2 = pred_state_f[0,j]
                    a_f = self.IDM(v1,v2,x1,x2)
                    v_f = v_f + self.T * a_f
                    x_f = x_f + v2 * self.T + 0.5 * a_f * self.T * self.T
                    #pred_state_f[1,j] = x_f
                    new_element_f[0,j] = x_f
                    new_element_f[0,j+1] = v_f
                self.IDM_state = np.vstack((self.IDM_state,new_element_f)) 
                self.state_f = np.vstack((self.state_f,pred_state_f[1,:]))                  
                cur_control_l, pred_state_l = self.leader(pred_state_f, pred_state_l, pred_control_l,predict_list)

                if (self.control_l[-1] < self.control_l[-2]).all() and (self.control_f[-1] < self.control_f[-2]).all():
                    print("leader算法收敛")
                    print()
                    #return self.control_l, self.control_f, t
                else:
                    pass
        pd.DataFrame(self.state_l).to_csv("/home/fanjx/project/FJX_project/Code_main/multi_vehicle/results/state_l.csv",index =False)
        pd.DataFrame(self.control_l).to_csv("/home/fanjx/project/FJX_project/Code_main/multi_vehicle/results/control_l.csv",index =False)
        pd.DataFrame(self.state_f).to_csv("/home/fanjx/project/FJX_project/Code_main/multi_vehicle/results/state_f.csv",index =False)
        pd.DataFrame(self.control_f).to_csv("/home/fanjx/project/FJX_project/Code_main/multi_vehicle/results/control_f.csv",index =False)
        pd.DataFrame(self.IDM_state).to_csv("/home/fanjx/project/FJX_project/Code_main/multi_vehicle/results/IDM_state.csv",index =False)
        return self.state_l, self.state_f, self.light, self.IDM_state
# TODO: 这里MPC的设置有问题，我的输入的current state有问题，应该实时更新，记得以后去修改一下

class VIS():
    def __init__(self,state_l,state_IDM,light,num_l,num_f):
        self.state_l = state_l
        self.state_f = state_IDM
        self.leader_patches = [[]]
        self.follower_patches = [[]]
        self.traffic_patches = []
        self.fig,self.ax = plt.subplots(figsize=(100,50))
        self.length = 5
        self.width = 3
        self.light =light
        self.num_l = num_l
        self.num_f = num_f
        self.max_iter = 126
    def Road(self):
        # 创建横向的路
        #fig, ax = plt.subplots(figsize=(100,50))
        x_lat_start1 = 0
        x_lat_start2 = 310
        x_lat_end1 = 300
        x_lat_end2 =610
        y_lat_start1 = -10
        y_lat_end1 = -100
        y_long_start_1 = 10
        y_long_end = 100
        #x_long_1 = -10
        #x_long_2 = 10
        self.ax.plot([x_lat_start1,x_lat_end1],[y_lat_start1,y_lat_start1],'k-',lw=2)
        self.ax.plot([x_lat_start1,x_lat_end1],[y_long_start_1,y_long_start_1],'k-',lw=2)
        self.ax.plot([x_lat_start2,x_lat_end2],[y_long_start_1,y_long_start_1],'k-',lw=2)
        self.ax.plot([x_lat_start2,x_lat_end2],[y_lat_start1,y_lat_start1],'k-',lw=2)
        # 创建纵向的路
        self.ax.plot([x_lat_end1, x_lat_end1],[y_long_start_1,y_long_end],'k-',lw=2)
        self.ax.plot([x_lat_start2, x_lat_start2],[y_long_start_1, y_long_end],'k-',lw=2)
        self.ax.plot([x_lat_end1, x_lat_end1],[y_lat_start1,y_lat_end1],'k-',lw=2)
        self.ax.plot([x_lat_start2, x_lat_start2],[y_lat_start1,y_lat_end1],'k-',lw=2)
            #plt.pause(0.5)
        # 设置坐标轴的范围和比例
        self.ax.set_xlim(0, 600)
        self.ax.set_ylim(-50, 100)
    
    def update_patches(self,i,leader_patch,follower_patch,current_light):

        for j in range(self.num_l):
            current_patch = patches.Rectangle((self.state_l[i][j],-9),self.length,self.width,edgecolor='black',facecolor='black')
            leader_patch.append(current_patch)
        for j in range(self.num_f):
            current_patch = patches.Rectangle((self.state_f[i][j],-9),self.length,self.width,edgecolor='blue',facecolor='blue')
            follower_patch.append(current_patch)
        x_tangle = 320
        y_tangle = 20
        x_circle = 330
        red_circle = 70
        yellow_circle = 50
        green_circle = 30
        background = patches.Rectangle((x_tangle,y_tangle),20,60,edgecolor='black',facecolor='black')
        self.ax.add_patch(background)

        current_light_patch = None
        if current_light_patch is not None:
            self.ax.patches.remove(current_light_patch)  #移除了之前的红灯
        if current_light == 2.0:
            current_light_patch = patches.Circle((x_circle, red_circle), 10, facecolor='red')
            self.traffic_patches.append(current_light_patch)
        elif current_light == 1.0:
            current_light_patch = patches.Circle((x_circle, yellow_circle), 10, facecolor='yellow')
            self.traffic_patches.append(current_light_patch)
        elif current_light == 0.0:
            current_light_patch = patches.Circle((x_circle, green_circle), 10, facecolor='green')
            self.traffic_patches.append(current_light_patch)
            
        self.ax.add_patch(current_light_patch)
        return self.leader_patches, self.follower_patches,self.traffic_patches
    
    def show(self):
        self.Road()
        for i in range(self.max_iter-1):
            list= []
            self.leader_patches.append(list)
            self.follower_patches.append(list)

        i = 0
        for leader_patch,follower_patch in zip(self.leader_patches,self.follower_patches):
            self.leader_patches,self.follower_patches,self.traffic_patches = self.update_patches(i,leader_patch,follower_patch,self.light[i])
            i += 1

        for leader_patch,follower_patch,traffic_patch in zip(self.leader_patches,self.follower_patches,self.traffic_patches):
            plt.cla()
            self.Road()
            for leader,follower in zip(leader_patch,follower_patch):
                patch_l = leader
                patch_f = follower
                self.ax.add_patch(patch_l)
                self.ax.add_patch(patch_f)
            patch_light = traffic_patch
            self.ax.add_patch(patch_light)
            self.ax.set_aspect(1)
            plt.pause(1)           
        plt.show()

if __name__ == '__main__':
    nmpc = NMPC()
    state_l,state_f,light,IDM_state = nmpc.IBR()
    leader_path = '/home/fanjx/project/FJX_project/Code_main/multi_vehicle/results/state_l.csv'
    follower_path = '/home/fanjx/project/FJX_project/Code_main/multi_vehicle/results/state_f.csv'
    IDM_path = '/home/fanjx/project/FJX_project/Code_main/multi_vehicle/results/IDM_state.csv'
    light_path = '/home/fanjx/project/FJX_project/Code_main/multi_vehicle/results/light.csv'
    light = np.loadtxt(light_path, delimiter=',',skiprows=1)
    print(f"light is {light}")
    print(f"type_light is {light.dtype}")
    state_l = np.loadtxt(leader_path, delimiter=',',skiprows = 2)
    state_IDM = np.loadtxt(IDM_path, delimiter=',',skiprows = 2)
    state_f = np.loadtxt(follower_path, delimiter=',',skiprows = 2)
    light = np.loadtxt(light_path, delimiter=',',skiprows = 1)
    vis = VIS(state_l,state_f,light,nmpc.num_l,nmpc.num_f)
    vis.show()