## TODO: 多个CAV是leader，多个IDM车是follower，计算结果有问题

import casadi as ca
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from casadi import Function, if_else
from datetime import datetime 
import os

class NMPC():
    def __init__(self, random_int,current_file):
        self.L = 1000
        #self.L1 = -1000
        self.D = 300
        self.G0 = 0.1 #引力场的权重
        self.G1 = 1.0 #黄灯斥力场
        self.G2 = 0.3 #生态驾驶
        self.G3 = 1 #速度场
        self.G4 = 1#交互项
        self.G5 = 5

        self.a2 = 5
        self.a3 = 5
        self.b1 = 100

        self.max_iter = 450
        self.u_max = 3
        self.u_min = -3
        self.random_int = random_int
        self.current_file = current_file

        self.T = 0.1
        self.N = 60
        self.v_free = 15
        self.r1 = self.v_free * 200
        self.r2 = (self.u_max * 3 * 3) / 2
        #self.vehicle_type = ['Auto_vehicle', 'Human_vehicle', 'Auto_vehicle', 'Human_vehicle', 'Auto_vehicle', 'Human_vehicle', 'Auto_vehicle', 'Human_vehicle', 'Auto_vehicle', 'Human_vehicle']
        self.vehicle_type = ['Auto_vehicle', 'Human_vehicle', 'Auto_vehicle', 'Human_vehicle']
        self.vehicle1_type = ['Auto_vehicle', 'Human_vehicle', 'Auto_vehicle', 'Human_vehicle','Auto_vehicle', 'Human_vehicle']
        self.num_l = self.vehicle_type.count('Auto_vehicle')
        self.num_f= self.vehicle_type.count('Human_vehicle')
        self.num_l1 = self.vehicle1_type.count('Auto_vehicle')
        self.num_f1 = self.vehicle1_type.count('Human_vehicle')
        self.state_l = np.ones((1,2 * self.num_l))
        self.state_f = np.ones((1,2 * self.num_f))
        self.control_l = np.ones((1, self.num_l))
        self.control_f = np.ones((1, self.num_f))
        self.IDM_state = np.array((30,10,10,10)).reshape(1,-1)

        self.state_l1 = np.ones((1, 2*self.num_l1))
        self.state_f1 = np.ones((1, 2*self.num_f1 ))
        
        self.control_l1 = np.ones((1, self.num_l1))
        self.control_f1 = np.ones((1, self.num_f1))
        self.IDM_state1 = np.array((50,5,30,5,10,5)).reshape(1,-1)
    def traffic_light(self,time):
        light = []
        green_time = 16
        red_time = 16
        yellow_time = 1.6
        time_step = 0.1
        real_light = []
        
        for i in range(int(green_time / time_step)): #16
            light.append(0)
        for i in range(int(yellow_time / time_step)): #1.6
            light.append(1)
        for i in range(int(red_time / time_step)):   #16
            light.append(2)
        
        period = int((green_time + yellow_time+red_time) /time_step)
        current_t=int((time % period) )
        print(f"current_light is {current_t}")
        for i in range(current_t,current_t + 378):
            if i >= period:
                real_light.append(light[i % period])
            else:
                real_light.append(light[i])
        
        return light, real_light, current_t
        
    def predict_light(self, horizon, light,start_point):
        predict_list = []
        period = len(light)
        trans_point = start_point % period
        #print(f"trans_point {trans_point}")
        
        for i in range(trans_point , trans_point + horizon):
            if i >= period:
                predict_list.append(light[i % period])
            else:
                predict_list.append(light[i])

        return predict_list

    def leader(self,pred_state_f, pred_state_l,  pred_control_l, predict_list):
        current_state_l = pred_state_l[0,:]

        self.opti_l = ca.Opti()
        self.num_l = self.vehicle_type.count('Auto_vehicle')
        self.opti_controls_l = self.opti_l.variable(self.N,self.num_l)
        self.opti_states_l = self.opti_l.variable(self.N+1,2 * self.num_l)
        self.opt_l_x0 = self.opti_l.parameter(1,2 * self.num_l)
        self.obj_l = 0

        for i in range(self.num_l):
             for t in range(self.N):
                j = 2*i

                if j>= self.opti_states_l.size()[1]:
                    raise IndexError(f"Index out of bounds: j={j},max index={self.opti_states_l.size()[1]-1}")

                if predict_list[t] == 0.0: #绿灯

                    self.obj_l += self.G0 *   (self.L - self.opti_states_l[t,j]) + self.G2 * self.opti_controls_l[t,i] * self.opti_controls_l[t,i] + self.G3 * (self.v_free - self.opti_states_l[t,j+1]) * (self.v_free - self.opti_states_l[t,j+1]) + self.G4 * (self.opti_states_l[t,j+1] - pred_state_f[t,j+1])**2 +\
                    if_else(j > 0, self.G5 *(pred_state_f[t,j-2] - self.opti_states_l[t,j] -10)**2,0)
                    
                elif predict_list[t] == 1.0:    #黄灯
                    if self.state_l[-1,j] <= self.D:
                        #self.obj_l +=  self.G0 *  * (self.L - self.opti_states_l[t,j])+ self.G1 * self.a2 /(self.D - self.r2 - self.opti_states_l[t, j])**2 +self.G2 * self.opti_controls_l[t,i] * self.opti_controls_l[t, i]+ self.G3 * (self.v_free - self.opti_states_l[t, j+1]) * (self.v_free - self.opti_states_l[t, j+1]) + self.G4 * (self.opti_states_l[t,j+1] - pred_state_f[t,j+1])**2 +\
                        self.obj_l +=  self.G0 *   (self.L - self.opti_states_l[t,j]) + self.G1 * self.a2 *\
                             (self.opti_states_l[t,j] - 500) + self.G2 * self.opti_controls_l[t,i] * self.opti_controls_l[t, i] + self.G3 * (
                                self.v_free - self.opti_states_l[t, j+1]) * (self.v_free - self.opti_states_l[t, j+1]) + self.G4 * (
                                self.opti_states_l[t,j+1] - pred_state_f[t,j+1])**2 +\
                            if_else(j > 0, self.G5 *(pred_state_f[t,j-2] - self.opti_states_l[t,j] - 10)**2,0)
                        
                    else:
                        self.obj_l +=  self.G0 *  (self.L - self.opti_states_l[t,j]) +\
                        self.G2 * self.opti_controls_l[t,i] *self.opti_controls_l[t, i]+ self.G3 * (self.v_free - self.opti_states_l[t, j+1]) * (self.v_free - self.opti_states_l[t, j+1])+ self.G4 * (self.opti_states_l[t,j+1] - pred_state_f[t,j+1])**2 +\
                        if_else(j > 0, self.G5 *(pred_state_f[t,j-2] - self.opti_states_l[t,j] -10)**2,0) 
        

                elif predict_list[t] == 2.0:  #红灯
                    if self.state_l[-1,j] <= self.D:
                        self.obj_l += self.G0 *  (self.L - self.opti_states_l[t, j]) + self.G1 * self.a3 *\
                        (self.opti_states_l[t,j]-500) + self.G2 * self.opti_controls_l[t, i] * self.opti_controls_l[t, i]+ self.G3 * (self.v_free - self.opti_states_l[t, j+1]) * (self.v_free - self.opti_states_l[t, j+1])+ self.G4 * (self.opti_states_l[t,j+1] - pred_state_f[t,j+1])**2 +\
                        if_else(j > 0, self.G5 *(pred_state_f[t,j-2] - self.opti_states_l[t,j] -10)**2,0)

                    else:
                        self.obj_l += self.G0 *   (self.L - self.opti_states_l[t, j]) +\
                        self.G2 * self.opti_controls_l[t, i] * self.opti_controls_l[t, i]+ self.G3 * (self.v_free - self.opti_states_l[t, j+1]) * (self.v_free - self.opti_states_l[t, j+1])+ self.G4 * (self.opti_states_l[t,j+1] - pred_state_f[t,j+1])**2 +\
                        if_else(j > 0, self.G5 *(pred_state_f[t,j-2] - self.opti_states_l[t,j] -10)**2,0)
                        
                else:
                    pass

        self.opti_l.minimize(self.obj_l)
        '''添加车辆约束'''

        for i in range(self.num_l):
            for t in range(self.N):
                j = 2 * i
                p_next_1 = self.opti_states_l[t,j] + self.opti_states_l[t,j+1] * self.T + 0.5 * self.T * self.T *self.opti_controls_l[t,i] #p_next是指它的位置
                v_next_1 = self.opti_states_l[t,j+1] + self.T *self.opti_controls_l[t,i] #v是指它的速度
                x_next_1 = ca.horzcat(p_next_1, v_next_1) #水平方向连接矩阵，竖直方向连接矩阵需要ca.vertcat()函数
                self.opti_l.subject_to(self.opti_states_l[t+1, j:j+2] == x_next_1)

        self.opti_l.subject_to(self.opti_states_l[0,:] == self.opt_l_x0)

        for i in range(self.num_l):

            self.opti_l.subject_to(self.opti_l.bounded(self.u_min, self.opti_controls_l[:,i], self.u_max))
            self.opti_l.subject_to(self.opti_l.bounded(2, self.opti_states_l[:,2 * i+1], self.v_free))        

        self.opti_l.set_initial(self.opti_controls_l, pred_control_l)
        self.opti_l.set_initial(self.opti_states_l, pred_state_l)
        self.opti_l.set_value(self.opt_l_x0, current_state_l)

        opts_setting = {'ipopt.max_iter': 200, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
                    'ipopt.acceptable_obj_change_tol': 1e-6}

        self.opti_l.solver('ipopt',opts_setting)

        try:
            solution_l = self.opti_l.solve()

        except RuntimeError as e:
            print("Solver failed. Debugging values:")
            opti_controls_l_values = self.opti_l.debug.value(self.opti_controls_l)
            opti_states_l_values = self.opti_l.debug.value(self.opti_states_l)
            print("opti_controls_l:",opti_controls_l_values)
            print("opti_states_l:",opti_states_l_values)     

        pred_controls_l = solution_l.value(self.opti_controls_l)
        pred_states_l = solution_l.value(self.opti_states_l)
        pred_state_l = np.concatenate((pred_states_l[1:],pred_states_l[-1:]))
        pred_control_l = np.concatenate((pred_controls_l[1:],pred_controls_l[-1:]))
        cur_control_l = pred_control_l[0,:]
        #self.control_l.append(pred_control_l[0])
        self.control_l = np.vstack((self.control_l,cur_control_l))
        #cur_v_l = pred_state_l[1,1]
        #cur_x_l = pred_state_l[1,0]
        #new_element = np.array([[cur_x_l,cur_v_l]])
        new_element = pred_state_l[0,:]
        self.state_l = np.vstack((self.state_l,new_element))

        return pred_control_l,pred_state_l
      
    def leader_1(self,pred_state_f, pred_state_l,  pred_control_l, predict_list):
        current_state_l = pred_state_l[0]

        self.opti_l1 = ca.Opti()
        self.num_l1 = self.vehicle1_type.count('Auto_vehicle')

        self.opti_controls_l1 = self.opti_l1.variable(self.N, self.num_l1)
        self.opti_states_l1 = self.opti_l1.variable(self.N+1, 2 * self.num_l1)

        ''' TODO: revise 1'''
        self.opt_l1_x0 = self.opti_l1.parameter(1,2 * self.num_l1)
        ''' TODO: revise 2'''
        # for i in range(self.num_l1-1):
        #     new_var = self.opti_l1.variable(self.N+1,2 * self.num_l1)
        #     self.opti_states_l1 = ca.horzcat(self.opti_states_l1,new_var)
        
        self.obj_l1 = 0
        for i in range(self.num_l1):
             for t in range(self.N):
                j = 2*i
                if j>= self.opti_states_l1.size()[1]:
                    raise IndexError(f"Index out of bounds: j={j},max index={self.opti_states_l1.size()[1]-1}")

                if predict_list[t] == 0.0: #绿灯
                    self.obj_l1 += self.G0 *   (self.L - self.opti_states_l1[t,j])+ self.G2 * self.opti_controls_l1[t,i] * self.opti_controls_l1[t,i]+ self.G3 * (self.v_free - self.opti_states_l1[t,j+1])*(self.v_free - self.opti_states_l1[t,j+1])+ self.G4 * (self.opti_states_l1[t,j+1]-pred_state_f[t,j+1])**2 +\
                    if_else(j > 0, self.G5 *(pred_state_f[t,j-2] - self.opti_states_l1[t,j] -12)**2,0)
                    #if_else(j>0, self.G5 *(self.opti_states_l1[t,j-2]-self.opti_states_l1[t,j]-25)**2,0)
                elif predict_list[t] == 1.0:    #黄灯
                    if self.state_l1[-1,j] <= self.D:
                        self.obj_l1 +=  self.G0 *  (self.L - self.opti_states_l1[t,j])+self.G1 * self.a2 /(self.D - self.r2 - self.opti_states_l1[t, j])**2+ self.G2 * self.opti_controls_l1[t,i] * self.opti_controls_l1[t, i]+ self.G3 * (self.v_free - self.opti_states_l1[t, j+1]) * (self.v_free - self.opti_states_l1[t, j+1])+ self.G4 * (self.opti_states_l1[t,j+1] - pred_state_f[t,j+1])**2 +\
                        if_else(j > 0, self.G5 *(pred_state_f[t,j-2] - self.opti_states_l1[t,j] -12)**2,0)
                        #if_else(j>0, self.G5 *(self.opti_states_l1[t,j-2]-self.opti_states_l1[t,j]-25)**2,0)

                    else:
                        self.obj_l1 +=  self.G0 *   (self.L - self.opti_states_l1[t,j])+ self.G2 * self.opti_controls_l1[t,i] * self.opti_controls_l1[t, i]+ self.G3 * (self.v_free - self.opti_states_l1[t, j+1]) * (self.v_free - self.opti_states_l1[t, j+1])+ self.G4 * (self.opti_states_l1[t,j+1]-pred_state_f[t,j+1])**2+\
                        if_else(j > 0, self.G5 *(pred_state_f[t,j-2] - self.opti_states_l1[t,j] -12)**2,0)
                        #if_else(j>0, self.G5 *(self.opti_states_l1[t,j-2]-self.opti_states_l1[t,j]-25)**2,0)
                elif predict_list[t] == 2.0:  #红灯
                    if  self.state_l1[-1,j] <= self.D:
                        self.obj_l1 += self.G0 *   (self.L - self.opti_states_l1[t, j])+self.G1 * self.a2 /(self.D - self.opti_states_l1[t, j])**2+ self.G2 * self.opti_controls_l1[t, i] * self.opti_controls_l1[t, i]+ self.G3 * (self.v_free - self.opti_states_l1[t, j+1])**2 + self.G4 * (self.opti_states_l1[t,j+1] - pred_state_f[t,j+1])**2+\
                        if_else(j > 0, self.G5 *(pred_state_f[t,j-2] - self.opti_states_l1[t,j] -12)**2,0)
                        #if_else(j>0, self.G5 *(self.opti_states_l1[t,j-2]-self.opti_states_l1[t,j]-25)**2,0)

                    else:
                        self.obj_l1 += self.G0 *   (self.L - self.opti_states_l1[t, j])+ self.G2 * self.opti_controls_l1[t, i] * self.opti_controls_l1[t, i]+ self.G3 * (self.v_free - self.opti_states_l1[t, j+1]) **2+ self.G4 * (self.opti_states_l1[t, j+1]-pred_state_f[t,j+1])**2 +\
                        if_else(j > 0, self.G5 *(pred_state_f[t,j-2] - self.opti_states_l1[t,j] -12)**2,0)
                        #if_else(j>0, self.G5 *(self.opti_states_l1[t,j-2]-self.opti_states_l1[t,j]-25)**2,0)

                    pass

        self.opti_l1.minimize(self.obj_l1)
        '''添加车辆约束'''
        
        for i in range(self.num_l1):
            for t in range(self.N):
                j = 2*i
                p_next_l = self.opti_states_l1[t,j] + self.opti_states_l1[t,j+1] * self.T + 0.5 * self.T * self.T *self.opti_controls_l1[t,i] #p_next是指它的位置
                v_next_l = self.opti_states_l1[t,j+1] + self.T *self.opti_controls_l1[t,i] #v是指它的速度
                x_next_l = ca.horzcat(p_next_l, v_next_l) #水平方向连接矩阵，竖直方向连接矩阵需要ca.vertcat()函数
                self.opti_l1.subject_to(self.opti_states_l1[t+1, j:j+2] == x_next_l)
        
        self.opti_l1.subject_to(self.opti_states_l1[0,:] == self.opt_l1_x0)
        
        for i in range(self.num_l1):
        
            self.opti_l1.subject_to(self.opti_l1.bounded(self.u_min, self.opti_controls_l1[:,i], self.u_max))
            self.opti_l1.subject_to(self.opti_l1.bounded(0, self.opti_states_l1[:,2*i+1], self.v_free))        

        self.opti_l1.set_initial(self.opti_controls_l1, pred_control_l)
        self.opti_l1.set_initial(self.opti_states_l1, pred_state_l)
        self.opti_l1.set_value(self.opt_l1_x0, current_state_l)

        opts_setting = {'ipopt.max_iter': 200, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
                    'ipopt.acceptable_obj_change_tol': 1e-6}

        self.opti_l1.solver('ipopt',opts_setting)
        
        try:
            solution_l = self.opti_l1.solve()

        except RuntimeError as e:
            print("Solver failed. Debugging values:")
            opti_controls_l_values=self.opti_l1.debug.value(self.opti_controls_l1)
            opti_states_l_values = self.opti_l1.debug.value(self.opti_states_l1)
            print("opti_controls_l:",opti_controls_l_values)
            print("opti_states_l:",opti_states_l_values)     

        pred_controls_l = solution_l.value(self.opti_controls_l1)
        pred_states_l = solution_l.value(self.opti_states_l1)

        pred_state_l = np.concatenate((pred_states_l[1:], pred_states_l[-1:]))
        pred_control_l = np.concatenate((pred_controls_l[1:], pred_controls_l[-1:]))
        cur_control_l = pred_control_l[0,:]
        self.control_l1 = np.vstack((self.control_l1,pred_control_l[0,:]))
        new_element = pred_state_l[0,:]
        self.state_l1 = np.vstack((self.state_l1,new_element))

        return pred_control_l,pred_state_l 
    def follower(self, pred_state_l, pred_state_f, pred_control_f):
        current_f_state = pred_state_f[0, :]
        current_state_IDM = self.IDM_state[-1, :]
        #print(f"current_state_IDM {current_state_IDM}")

        for i in range(self.num_f):
            current_f_state[2 * i] = current_state_IDM[2 * i]
        print(f"current_f_state {current_f_state}")

        self.sigma = 2
        self.K_e = 1
        self.LV = 15
        self.K_s_log = 500
        self.K_v_log = 1

        self.opti_f = ca.Opti()
        self.opti_controls_f = self.opti_f.variable(self.N, self.num_f)
        self.opti_states_f=self.opti_f.variable(self.N+1, 2 * self.num_f)
        self.opt_f_x0 = self.opti_f.parameter(1, 2 * self.num_f)
        self.opti_f.subject_to(self.opti_states_f[0,:] == self.opt_f_x0)

        # for i in range(2,self.num_f,2):
        #     self.opti_position_f = ca.horzcat(self.opti_position_f,self.opti_states_f[:, i])
        #     self.opti_velocity_f = ca.horzcat(self.opti_velocity_f,self.opti_states_f[:, i+1])
        
        self.obj_f=0        
        for i in range(self.num_f):
            for t in range(self.N):
                j = 2*i
                self.obj_f += self.K_v_log * ((self.opti_states_f[t,j+1] - pred_state_l[t,j+1])**2 + self.sigma)+ self.K_s_log * ((pred_state_l[t,j] - self.opti_states_f[t,j] - self.LV)**2 )+ self.K_e * ((self.opti_states_f[t,j+1] - self.v_free)**2 + self.sigma)

        self.opti_f.minimize(self.obj_f)

        ''' 添加车辆约束 '''
        for i in range(self.num_f):   
            for t in range(self.N):
                j = 2 * i
                p_next_f = self.opti_states_f[t,j] + self.opti_states_f[t,j+1] * self.T + 0.5 * self.T * self.T * self.opti_controls_f[t,i] #p_next是指它的位置
                v_next_f = self.opti_states_f[t,j+1] + self.T * self.opti_controls_f[t,i] #v是指它的速度
                x_next_f = ca.horzcat(p_next_f, v_next_f) #水平方向连接矩阵，竖直方向连接矩阵需要ca.vertcat()函数
                self.opti_f.subject_to(self.opti_states_f[t+1, j:j+2] == x_next_f)

        """ 注意这里的约束不能太多 """

        for i in range(self.num_f):
            for t in range(self.N):
                self.opti_f.subject_to(self.opti_f.bounded(self.u_min, self.opti_controls_f[t,i], self.u_max))  
                self.opti_f.subject_to(self.opti_f.bounded(2, self.opti_states_f[t,2*i+1], self.v_free))
        
        self.opti_f.set_initial(self.opti_controls_f, pred_control_f)
        self.opti_f.set_initial(self.opti_states_f, pred_state_f)
        self.opti_f.set_value(self.opt_f_x0, current_f_state)
        
        opts_setting = {'ipopt.max_iter': 200, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
                    'ipopt.acceptable_obj_change_tol': 1e-4}
        self.opti_f.solver('ipopt',opts_setting)

        # num_constraint = self.opti_f.g.size1()
        # print("Number of constraints: ", num_constraint)
        try:
            solution_f = self.opti_f.solve()
        except RuntimeError as e:
            print("Solver failed. Debugging values:")
            opti_controls_f_values = self.opti_f.debug.value(self.opti_controls_f)
            opti_states_f_values = self.opti_f.debug.value(self.opti_states_f)
            print("opti_controls_f:", opti_controls_f_values)
            print("opti_states_f:", opti_states_f_values)
        pred_controls_f = solution_f.value(self.opti_controls_f)
        pred_states_f = solution_f.value(self.opti_states_f)    
        pred_control_f = np.concatenate((pred_controls_f[1:],pred_controls_f[-1:]))
        pred_state_f = np.concatenate((pred_states_f[1:],pred_states_f[-1:]))
        #cur_control_f=pred_control_f[0,:].reshape(1,-1)
        cur_control_f = pred_control_f[0,:]
        #cur_control_f = pd.DataFrame(cur_control_f, columns = ["u1","u2","u3","u4","u5"])
        self.control_f = np.vstack((self.control_f,cur_control_f))

        return pred_control_f, pred_state_f


    def follower_1(self, pred_state_l, pred_state_f, pred_control_f):
        current_f_state = pred_state_f[0, :]
        current_state_IDM = self.IDM_state1[-1, :]

        for i in range(self.num_f1):
            current_f_state[2 * i] = current_state_IDM[2 * i]

        self.K_s = 1
        self.K_e = 1
        self.sigma = 2
        self.LV = 12
        self.K_s_log = 500
        self.K_v_log = 1

        self.opti_f1 = ca.Opti()
        self.opti_controls_f1 = self.opti_f1.variable(self.N, self.num_f1)
        self.opti_states_f1 = self.opti_f1.variable(self.N+1, 2 * self.num_f1)
        self.opt_f1_x0 = self.opti_f1.parameter(1, 2 * self.num_f1)
        self.opti_f1.subject_to(self.opti_states_f1[0,:] == self.opt_f1_x0)

        self.obj_f1 = 0        
        for i in range(self.num_f1):
            for t in range(self.N):
                j = 2*i
                self.obj_f1 += self.K_v_log * ((self.opti_states_f1[t,j+1] - pred_state_l[t,j+1])**2 + self.sigma)+ self.K_s_log * (pred_state_l[t,j] - self.opti_states_f1[t,j] - self.LV)**2 + self.K_e * ((self.opti_states_f1[t,j+1] - self.v_free)**2 + self.sigma)

        self.opti_f1.minimize(self.obj_f1)

        ''' 添加车辆约束 '''
        for i in range(self.num_f1):   
            for t in range(self.N):
                j = 2 * i
                p_next_f = self.opti_states_f1[t,j] + self.opti_states_f1[t,j+1] * self.T + 0.5 * self.T * self.T * self.opti_controls_f1[t,i] #p_next是指它的位置
                v_next_f = self.opti_states_f1[t,j+1] + self.T * self.opti_controls_f1[t,i] #v是指它的速度
                x_next_f = ca.horzcat(p_next_f, v_next_f) #水平方向连接矩阵，竖直方向连接矩阵需要ca.vertcat()函数
                self.opti_f1.subject_to(self.opti_states_f1[t+1, j:j+2] == x_next_f)

        """ 注意这里的约束不能太多 """

        for i in range(self.num_f1):
            self.opti_f1.subject_to(self.opti_f1.bounded(self.u_min, self.opti_controls_f1[:,i], self.u_max))  
            self.opti_f1.subject_to(self.opti_f1.bounded(0, self.opti_states_f1[:,2*i+1], self.v_free))
    
        self.opti_f1.set_initial(self.opti_controls_f1, pred_control_f)
        self.opti_f1.set_initial(self.opti_states_f1, pred_state_f)
        self.opti_f1.set_value(self.opt_f1_x0, current_f_state)    
        
        opts_setting = {'ipopt.max_iter': 200, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
                    'ipopt.acceptable_obj_change_tol': 1e-4}
        self.opti_f1.solver('ipopt',opts_setting)

        # num_constraint = self.opti_f.g.size1()
        # print("Number of constraints: ", num_constraint)
        # try:
        solution_f = self.opti_f1.solve()
        # except RuntimeError as e:
            # print("Solver failed. Debugging values:")
            # opti_controls_f_values = self.opti_f.debug.value(self.opti_controls_f)
            # opti_states_f_values = self.opti_f.debug.value(self.opti_states_f)
            # print("opti_controls_f:", opti_controls_f_values)
            # print("opti_states_f:", opti_states_f_values)
        pred_controls_f = solution_f.value(self.opti_controls_f1)
        pred_states_f = solution_f.value(self.opti_states_f1)    
        pred_control_f = np.concatenate((pred_controls_f[1:],pred_controls_f[-1:]))
        pred_state_f = np.concatenate((pred_states_f[1:],pred_states_f[-1:]))
        #cur_control_f=pred_control_f[0,:].reshape(1,-1)
        cur_control_f=pred_control_f[0,:]
        #cur_control_f = pd.DataFrame(cur_control_f, columns = ["u1","u2","u3","u4","u5"])
        self.control_f1 = np.vstack((self.control_f1,cur_control_f))
        cur_state_f = pred_state_f[0,:].reshape(1,-1)
        print(f"cur_state_f is {cur_state_f}")
        #cur_state_f = pd.DataFrame(cur_state_f, columns = ["x1","v1","x2","v2","x3","v3","x4","v4","x5","v5"])
        #print(f"cur_control_f  {cur_control_f}")
        return pred_control_f, pred_state_f

    def IDM(self,v1,v2,x1,x2):# 2代表本车，1代表前车
        A = 10  #自车最大加速度
        dta = 2 ##derta上标
        s0 = 7  #最小距离
        T = 0.1 #   
        B = 5  #舒适减速度
        Vf = 40  #期望速度
        mean = 0 #均值
        std = 0.2 #标准差
        size = 1 #随机数数量
        noise = np.random.normal(mean,std,size)
        noise = noise[0]
        s = s0 + T * v2 + (v2 * (v2 - v1)) / (2 * (A * B) ** 0.5)  #跟车距离
        a = A * (1 - (v2 / Vf) ** dta - (s / (x1 - x2)) ** 2) +noise #
        return a
    
    def divide(self):
        self.vehicle_pairwise = []
        temp = []
        for vehicle in self.vehicle_type:
            if vehicle == 'Auto_vehicle':
                    if temp:
                        self.vehicle_pairwise.append(temp)
                        temp = []  # 清空temp
            temp.append(vehicle)
        # add the last temp
        if temp:
            self.vehicle_pairwise.append(temp)
        return self.vehicle_pairwise

    # TODO: 这里的IBR迭代的过程中，follower是有问题的 
    ''' IBR是针对L-F博弈，但是MPC的滚动优化是针对MPC，二者不冲突 '''

    ''' IBR要收敛到stackberg，先证明收敛性，再求 '''
    def IBR(self):
        #df.loc[:,'position'] = 
        self.light, self.real_light, self.start_time = self.traffic_light(self.random_int)
        predict_list = self.predict_light(self.N, self.light, self.random_int)
        predict_list_fin = np.ones((1,self.N))
        print(f"predict_list { predict_list}")
        ''' 初始化leader的state,follower的state '''
        pred_state_l = np.ones((self.N+1 ,2 * self.num_l))
        #state_f_all = np.ones((1,2 * self.num_f))
        #state_l_all = np.ones((1,2 * self.num_l))

        ''' set_initial states of the leader and follower   '''
        for i in range(self.num_l):
            pred_state_l[:, 2 * i] = 40 - i * 21
            pred_state_l[:, 2 * i + 1] = 5 
            self.state_l[0, 2 * i] = 40 - i * 21
            self.state_l[0, 2 * i + 1] = 5

        pred_state_f = np.ones((self.N + 1,2 * self.num_f))
        for i in range(self.num_f):
            pred_state_f[:,2 * i] = 30 - 20 * i
            pred_state_f[:,2 * i + 1] = 5
            self.state_f[0,2 * i] = 30 - 20 * i
            self.state_f[0,2 * i + 1] = 5

        pred_control_f = np.zeros((self.N, self.num_f))
        pred_control_f[0,:] = np.array((5))
        pred_control_l = np.zeros((self.N, self.num_l))
        pred_control_l[0,:] = np.array((5))

        '''calculate the leader and follower '''
        for t in range(self.start_time, self.start_time + self.max_iter):
            new_element_IDM = np.ones((1,2 * self.num_f)).reshape(1,-1)
            new_element_f = np.ones((1, 2 * self.num_f)).reshape(1,-1)
            predict_list = self.predict_light(self.N, self.light, t)
            predict_list_fin = np.vstack((predict_list_fin,predict_list))
            pred_control_f,pred_state_f = self.follower(pred_state_l, pred_state_f,pred_control_f)

            for i in range(self.num_f):
                j = 2 * i
                v1 = self.state_l[-1,j+1]
                x1 = self.state_l[-1,j] 
                v2 = self.IDM_state[-1,j+1]
                x2 = self.IDM_state[-1,j]
                a_f = self.IDM(v1,v2,x1,x2)
                v_f = v2 + self.T * a_f
                x_f = x2 + v2 * self.T + 0.5 * a_f * self.T * self.T
                new_element_IDM[0,j] = x_f
                new_element_IDM[0,j+1] = v_f
                v = pred_state_f[0,j+1]
                x = pred_state_f[0,j]
                new_element_f[0,j+1] = v
                new_element_f[0,j] = x

            self.IDM_state = np.vstack((self.IDM_state,new_element_IDM)) 
            self.state_f = np.vstack((self.state_f,new_element_f))                  
            pred_control_l, pred_state_l = self.leader(pred_state_f, pred_state_l, pred_control_l,predict_list)

            if (self.control_l[-1] < self.control_l[-2]).all() and (self.control_f[-1] < self.control_f[-2]).all():
                #print("leader算法收敛")
                print()

            else:
                pass

        pd.DataFrame(self.state_l).to_csv(f"{current_file}/results/state_l.csv",index =False)
        pd.DataFrame(self.control_l).to_csv(f"{current_file}/results/control_l.csv",index =False)
        light = pd.DataFrame(self.real_light)
        light.to_csv(f"{current_file}/results/light.csv",index =False)
        pd.DataFrame(self.state_f).to_csv(f"{current_file}/results/state_f.csv",index =False)
        pd.DataFrame(self.control_f).to_csv(f"{current_file}/results/control_f.csv",index =False)
        pd.DataFrame(self.IDM_state).to_csv(f"{current_file}/results/IDM_state.csv",index =False)
        pd.DataFrame(predict_list_fin).to_csv(f"{current_file}/results/predict_list.csv",index =False)
        return self.state_l, self.state_f, self.light, self.IDM_state
    
    def IBR_1(self):
        self.light1, self.real_light1,self.start_time1 = self.traffic_light(self.random_int)
        predict_list = self.predict_light(self.N, self.light, self.random_int)
        predict_list_fin = np.ones((1,self.N))
        #print(f"predict_list { predict_list}")
        ''' 初始化leader的state,follower的state '''
        pred_state_l=np.ones((self.N+1,2*self.num_l1))

        for i in range(self.num_l1):
            pred_state_l[:, 2 * i] = 60 - 20 * i
            pred_state_l[:, 2 * i + 1] = 5 
            self.state_l1[0,2 * i] = 60 - 20 * i
            self.state_l1[0,2 * i + 1] = 5

        pred_state_f=np.ones((self.N + 1,2 * self.num_f1))
        for i in range(self.num_f1):
            pred_state_f[:,2 * i] = 50 - 20 * i
            pred_state_f[:,2 * i + 1] = 5
            self.state_f1[0,2 * i] = 50 - 20 * i
            self.state_f1[0,2 * i + 1] = 5

        pred_control_f = np.zeros((self.N, self.num_f1))
        pred_control_f[0,:] = 5
        pred_control_l = np.zeros((self.N, self.num_l1))
        pred_control_l[0,:] = 5

        for t in range(self.start_time, self.start_time + self.max_iter):
            new_element_IDM = np.ones((1,2 * self.num_f1)).reshape(1,-1)
            new_element_f = np.ones((1,2 *self.num_f1)).reshape(1,-1)
            predict_list = self.predict_light(self.N, self.light, t)
            predict_list_fin = np.vstack((predict_list_fin,predict_list))
            pred_control_f,pred_state_f = self.follower_1(pred_state_l, pred_state_f,pred_control_f)
            for i in range(self.num_f1):
                j = 2*i
                v1 = self.state_l1[-1,j+1]
                x1 = self.state_l1[-1,j] 
                v2 = self.IDM_state1[-1,j+1]
                x2 = self.IDM_state1[-1,j]
                a_f = self.IDM(v1,v2,x1,x2)
                v_f = v2 + self.T * a_f
                x_f = x2 + v2 * self.T + 0.5 * a_f * self.T * self.T
                new_element_IDM[0,j] = x_f
                new_element_IDM[0,j+1] = v_f
                v = pred_state_f[0,j+1]
                x = pred_state_f[0,j]
                new_element_f[0,j+1] = v
                new_element_f[0,j] = x                    
            self.IDM_state1 = np.vstack((self.IDM_state1,new_element_IDM)) 
            self.state_f1 = np.vstack((self.state_f1,new_element_f))                  
            pred_control_l, pred_state_l = self.leader_1(pred_state_f, pred_state_l, pred_control_l,predict_list)

            if (self.control_l1[-1] < self.control_l1[-2]).all() and (self.control_f1[-1] < self.control_f1[-2]).all():
                print("leader算法收敛")
                print()

            else:
                pass

        pd.DataFrame(self.state_l1).to_csv("/home/fanjx/project/FJX_project/Code_main/multi_vehicle/single_result/state_l.csv",index =False)
        pd.DataFrame(self.control_l1).to_csv("/home/fanjx/project/FJX_project/Code_main/multi_vehicle/single_result/control_l.csv",index =False)
        pd.DataFrame(self.state_f1).to_csv("/home/fanjx/project/FJX_project/Code_main/multi_vehicle/single_result/state_f.csv",index =False)
        pd.DataFrame(self.control_f1).to_csv("/home/fanjx/project/FJX_project/Code_main/multi_vehicle/single_result/control_f.csv",index =False)
        pd.DataFrame(self.IDM_state1).to_csv("/home/fanjx/project/FJX_project/Code_main/multi_vehicle/single_result/IDM_state.csv",index =False)
        pd.DataFrame(predict_list_fin).to_csv("/home/fanjx/project/FJX_project/Code_main/multi_vehicle/single_result/predict_list.csv",index =False)
        
        return self.state_l, self.state_f, self.light, self.IDM_state, predict_list_fin
# TODO: 这里MPC的设置有问题，我的输入的current state有问题，应该实时更新，记得以后去修改一下


if __name__ == '__main__':
    #random_int = np.random.randint(0, 336)
    random_int = 226
    current_file = os.path.dirname(os.path.abspath(__file__))
    nmpc = NMPC(random_int,current_file)
    state_l,state_f,light,IDM_state = nmpc.IBR()
    #state_l = pd.read_csv('./results/state_l.csv')
    #state_l = state_l.to_numpy()
    #state_f = pd.read_csv('./results/IDM_state.csv')
    #state_f = state_f.to_numpy()
    fig,ax = plt.subplots()
    ax.plot([0,16],[300,300],color = 'g',lw = 2)
    ax.plot([16,17.6],[300,300],color = 'y',lw = 2)
    ax.plot([17.6,33.6],[300,300],color = 'r',lw = 2)
    ax.plot([33.6,49.6],[300,300],color = 'g',lw = 2)
    ax.plot([49.6,51.2],[300,300],color = 'y',lw = 2)
    ax.plot([51.2,67.2],[300,300],color = 'r',lw = 2)
    ax.plot([67.2,83.2],[300,300],color = 'g',lw = 2)

    rows,cols = state_l.shape
    even_colums_l = state_l[:,::2]
    even_colums_f = state_f[:,::2]
    #print(f"rows is {rows},cols is {cols},even_colums is {even_colums}")
    real_time = random_int/10
    array = np.arange(real_time,real_time+45.1,0.1)
    ax.plot(array,even_colums_l,'b',lw = 1)
    ax.plot(array,even_colums_f,'r',lw = 1)
    # for i in range(rows):
    #     for j in range(even_colums_l.shape[1]):
    #         temp_l = even_colums_l[i,j]
    #         print(f"temp_l is {temp_l}")
    #         temp_f = even_colums_f[i,j]
    #         #ax.scatter(i,temp_l,marker='.',c='b',alpha=0.7)
    #         ax.plot(i,temp_l,'b',lw = 2)
    #         #ax.scatter(i,temp_f,marker='.',color='r',alpha=0.7)
    #         ax.plot(i,temp_f,'r',lw = 2)
    ax.set_xlabel("time")
    ax.set_ylabel("position")
    ax.set_xlim(0,90)
    ax.set_ylim(0,400)
    plt.savefig('eco1.png', dpi=300, bbox_inches='tight')
    plt.show()







