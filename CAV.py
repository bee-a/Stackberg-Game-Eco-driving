import casadi as ca
import numpy as np
import time
import pandas as pd
#from draw_multi_veh import Road
import sys
sys.path.append("..") 
#from untils.draw import Arrow, draw_car, Road
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from casadi import Function, if_else
from datetime import datetime
import imageio 
import time

class NMPC():
    def __init__(self,random_int):
        self.L = 10000
        self.L1 =-10000
        self.D = 300
        self.D1 = 320
        self.G0 = 1
        self.G1 = 10
        self.G2 = 0.3
        self.G3 = 1
        self.G4 = 1.0
        self.G5 = 1.0
        self.a1 = 1
        self.a2 = 100
        self.a3 = 100
        self.b1 = 100
        self.max_iter = 126
        self.u_max = 3
        self.u_min = -3
        self.w_influence = 1.0
        self.T = 0.5
        self.N = 20
        self.v_free = 20
        self.r1 = self.v_free*200
        self.r2 = (self.u_max*3*3)/2
        self.vehicle_type = ['Auto_vehicle', 'Human_vehicle']
        self.vehicle1_type = ['Auto_vehicle', 'Human_vehicle']
        self.state_l = np.empty((1,2))
        self.state_f = np.ones((1,2))
        self.control_l = [1,1]
        self.control_f = [1,1]
        self.IDM_state = np.array((5,5)).reshape(1,-1)
        self.state_l1 = np.empty((1,2))
        self.state_f1 = np.empty((1,2))
        self.IDM_state1 = np.array((615,5)).reshape(1,-1)
        self.control_l1 = [1,1]
        self.control_f1 = [1,1]
        self.random_int = random_int
    @staticmethod
    def traffic_light(time):
        light = []
        green_time = 30
        red_time = 30
        yellow_time=3
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
        
    
    def predict_light(self,horizon,light,start_point):
        #light = light
        predict_list = []
        period = len(light)
        trans_point = start_point%period
        print(f"trans_point {trans_point}")

        for i in range(trans_point , trans_point + horizon):
            if i >= period:
                predict_list.append(light[i%period])
            else:
                predict_list.append(light[i])

        return predict_list

    def leader(self,pred_state_f, pred_state_l,  pred_control_l, predict_list):
        current_state_l = pred_state_l[0]

        self.opti_l = ca.Opti()
        self.num_l = self.vehicle_type.count('Auto_vehicle')

        self.opti_controls_l = self.opti_l.variable(self.N,self.num_l)
        self.opti_states_l = self.opti_l.variable(self.N+1,2*self.num_l)


        ''' TODO: 这个约束添加的意义是什么 '''

        self.opt_l_x0 = self.opti_l.parameter(1,2)
        
        for i in range(self.num_l-1):
            new_var = self.opti_l.variable(self.N+1,2*self.num_l)
            self.opti_states_l = ca.horzcat(self.opti_states_l,new_var)
        
        self.obj_l = 0
        for i in range(self.num_l):
             for t in range(self.N):
                j = 2*i
                #f =Function('f',[t,j],[self.opti_states_l[t,j-2]])
                #print(f"t: {t}")
                #print(f"predict_list_leader {predict_list}")self.num_l
                if j>= self.opti_states_l.size()[1]:
                    raise IndexError(f"Index out of bounds: j={j},max index={self.opti_states_l.size()[1]-1}")
                #value = f(t,j)
                if t==0:
                    if predict_list[t] == 0.0: #绿灯
                        self.obj_l += self.G0 * self.a1 * (self.L-self.opti_states_l[t,j])
                        + self.G2 * self.opti_controls_l[t,i] * self.opti_controls_l[t,i]\
                        + self.G3 * (self.v_free-self.opti_states_l[t,j+1])*(self.v_free-self.opti_states_l[t,j+1])\
                        + self.G4 * (self.opti_states_l[t,j+1]-pred_state_f[t,j+1])**2
                        
                    elif predict_list[t] == 1.0:    #黄灯
                        self.obj_l +=  self.G0 * self.a1 * (self.L - self.opti_states_l[t,j])+\
                        if_else(self.state_l[-1,j] < self.D, (self.G1 * self.a2 /(self.D + self.r2 - self.opti_states_l[t, j])), 0)
                        + self.G2 * self.opti_controls_l[t,i] *self.opti_controls_l[t, i]\
                        + self.G3 * (self.v_free - self.opti_states_l[t, j+1]) * (self.v_free - self.opti_states_l[t, j+1])\
                        + self.G4 * (self.opti_states_l[t,j+1]-pred_state_f[t,j+1])**2
                        print('YEL LIGHT')

                    elif predict_list[t] == 2.0:  #红灯
                        self.obj_l += self.G0 * self.a1 * (self.L - self.opti_states_l[t, j])+\
                        if_else(self.state_l[-1,j] < self.D, self.G1 * self.a3 /(self.D - self.opti_states_l[t, j]), 0)
                        + self.G2 * self.opti_controls_l[t, i] * self.opti_controls_l[t, i]\
                        + self.G3 * (self.v_free - self.opti_states_l[t, j+1]) * (self.v_free - self.opti_states_l[t, j+1])\
                        + self.G4 * (self.opti_states_l[t,j+1]-pred_state_f[t,j+1])**2
                        print('RED LIGHT')

                    else:
                        print('worry')                    
                # TODO: 如果是绿灯就可以，如果是红灯、黄灯，就不行
                else:
                    if predict_list[t] == 0.0: #绿灯
                        self.obj_l += self.G0 * self.a1 * (self.L-self.opti_states_l[t,j])
                        + self.G2 * self.opti_controls_l[t,i] * self.opti_controls_l[t,i]\
                        + self.G3 * (self.v_free-self.opti_states_l[t,j+1])*(self.v_free-self.opti_states_l[t,j+1])\
                        + self.G4 * (self.opti_states_l[t,j+1]-pred_state_f[t,j+1])**2
                        
                    elif predict_list[t] == 1.0:    #黄灯
                        if self.state_l[-1,j] < self.D:
                            # self.G0 * self.a1 * (self.L - self.opti_states_l[t-1,j])+\
                            self.obj_l +=  self.G0 * self.a1 * (self.L - self.opti_states_l[t,j])+\
                            (self.G1 * self.a2 /(self.D + self.r2 - self.opti_states_l[t, j])**2)
                            + self.G2 * self.opti_controls_l[t,i] *self.opti_controls_l[t, i]\
                            + self.G3 * (self.v_free - self.opti_states_l[t, j+1]) * (self.v_free - self.opti_states_l[t, j+1])\
                            + self.G4 * (self.opti_states_l[t,j+1]-pred_state_f[t,j+1])**2
                            print('YEL LIGHT')
                        else:
                            # self.G0 * self.a1 * (self.L - self.opti_states_l[t-1,j])+\
                            self.obj_l +=  self.G0 * self.a1 * (self.L - self.opti_states_l[t,j])
                            + self.G2 * self.opti_controls_l[t,i] *self.opti_controls_l[t, i]\
                            + self.G3 * (self.v_free - self.opti_states_l[t, j+1]) * (self.v_free - self.opti_states_l[t, j+1])\
                            + self.G4 * (self.opti_states_l[t,j+1]-pred_state_f[t,j+1])**2
                            print('YEL LIGHT')

                    elif predict_list[t] == 2.0:  #红灯
                        if self.state_l[-1,j] < self.D:
                            self.obj_l += self.G0 * self.a1 * (self.L - self.opti_states_l[t, j])+\
                            self.G0 * self.a1 * (self.L - self.opti_states_l[t-1,j])+\
                            (self.G1 * self.a2 /(self.D - self.opti_states_l[t, j])**2)
                            + self.G2 * self.opti_controls_l[t, i] * self.opti_controls_l[t, i]\
                            + self.G3 * (self.v_free - self.opti_states_l[t, j+1]) * (self.v_free - self.opti_states_l[t, j+1])\
                            + self.G4 * (self.opti_states_l[t,j+1]-pred_state_f[t,j+1])**2
                            print('BEHIND RED LIGHT')
                        else:
                            self.obj_l += self.G0 * self.a1 * (self.L - self.opti_states_l[t, j])+\
                            self.G0 * self.a1 * (self.L - self.opti_states_l[t-1,j])
                            + self.G2 * self.opti_controls_l[t, i] * self.opti_controls_l[t, i]\
                            + self.G3 * (self.v_free - self.opti_states_l[t, j+1]) * (self.v_free - self.opti_states_l[t, j+1])\
                            + self.G4 * (self.opti_states_l[t,j+1]-pred_state_f[t,j+1])**2
                            print('Ifornt RED LIGHT')
                    else:
                        print('worry')

        self.opti_l.minimize(self.obj_l)
        '''添加车辆约束'''
        
        for i in range(self.num_l):
            for t in range(self.N):
                j = 2*i
                p_next_1 = self.opti_states_l[t,j] + self.opti_states_l[t,j+1] * self.T + 0.5 * self.T * self.T *self.opti_controls_l[t,:] #p_next是指它的位置
                v_next_1 = self.opti_states_l[t,j+1] + self.T *self.opti_controls_l[t,:] #v是指它的速度
                x_next_1 = ca.horzcat(p_next_1, v_next_1) #水平方向连接矩阵，竖直方向连接矩阵需要ca.vertcat()函数
                self.opti_l.subject_to(self.opti_states_l[t+1, :] == x_next_1)
        
        self.opti_l.subject_to(self.opti_states_l[0,:] == self.opt_l_x0)
        
        for i in range(self.num_l):
        
            self.opti_l.subject_to(self.opti_l.bounded(self.u_min, self.opti_controls_l[:,i], self.u_max))
            self.opti_l.subject_to(self.opti_l.bounded(4, self.opti_states_l[:,2*i+1], self.v_free))        

        #pred_control_l = np.zeros((self.N, 1))
        #pred_state_l = np.ones((self.N+1,2))
        #self.opti_l.set_initial(self.opti_controls_l, pred_control_l)
        #self.opti_l.set_initial(self.opti_states_l, pred_state_l)
        self.opti_l.set_initial(self.opti_controls_l, pred_control_l)
        self.opti_l.set_initial(self.opti_states_l, pred_state_l)
        self.opti_l.set_value(self.opt_l_x0, current_state_l)

        opts_setting = {'ipopt.max_iter': 1000, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
                    'ipopt.acceptable_obj_change_tol': 1e-6}

        self.opti_l.solver('ipopt',opts_setting)
        
        try:
            solution_l = self.opti_l.solve()

        except RuntimeError as e:
            print("Solver failed. Debugging values:")
            opti_controls_l_values=self.opti_l.debug.value(self.opti_controls_l)
            opti_states_l_values = self.opti_l.debug.value(self.opti_states_l)
            print("opti_controls_l:",opti_controls_l_values)
            print("opti_states_l:",opti_states_l_values)     

        pred_controls_l = solution_l.value(self.opti_controls_l)
        pred_states_l = solution_l.value(self.opti_states_l)
        #for i in range(self.N):
        pred_state_l=np.concatenate((pred_states_l[1:],pred_states_l[-1:]))
        pred_control_l=np.concatenate((pred_controls_l[1:],pred_controls_l[-1:]))
        cur_control_l=pred_control_l[0]
        self.control_l.append(pred_control_l[0])
        cur_v_l=pred_state_l[1,1]
        cur_x_l = pred_state_l[1,0]
        print(f"cur_control_l  is  {cur_control_l}")
        new_element=np.array([[cur_x_l,cur_v_l]])
        self.state_l=np.vstack((self.state_l,new_element))

        return pred_control_l,pred_state_l

    def leader_1(self,pred_state_f, pred_state_l,  pred_control_l, predict_list):
        current_state_l = pred_state_l[0]

        self.opti_l1 = ca.Opti()
        self.num_l1 = self.vehicle1_type.count('Auto_vehicle')

        self.opti_controls_l1 = self.opti_l1.variable(self.N,self.num_l1)
        self.opti_states_l1 = self.opti_l1.variable(self.N+1,2*self.num_l1)


        ''' TODO: 这个约束添加的意义是什么 '''

        self.opt_l1_x0 = self.opti_l1.parameter(1,2)
        
        for i in range(self.num_l1-1):
            new_var = self.opti_l1.variable(self.N+1,2*self.num_l1)
            self.opti_states_l1 = ca.horzcat(self.opti_states_l1,new_var)
        
        self.obj_l1 = 0
        for i in range(self.num_l1):
             for t in range(self.N):
                j = 2*i
                #f =Function('f',[t,j],[self.opti_states_l[t,j-2]])
                #print(f"t: {t}")
                #print(f"predict_list_leader {predict_list}")self.num_l
                if j>= self.opti_states_l1.size()[1]:
                    raise IndexError(f"Index out of bounds: j={j},max index={self.opti_states_l1.size()[1]-1}")
                #value = f(t,j)
                if t==0:
                    if predict_list[t] == 0.0: #绿灯
                        self.obj_l1 += self.G0 * self.a1 * (self.L - self.opti_states_l1[t,j])
                        + self.G2 * self.opti_controls_l1[t,i] * self.opti_controls_l1[t,i]\
                        + self.G3 * (self.v_free-self.opti_states_l1[t,j+1])*(self.v_free-self.opti_states_l1[t,j+1])\
                        + self.G4 * (self.opti_states_l1[t,j+1]-pred_state_f[t,j+1])**2
                        
                    elif predict_list[t] == 1.0:    #黄灯
                        self.obj_l1 +=  self.G0 * self.a1 * (self.L - self.opti_states_l1[t,j])+\
                        if_else(self.state_l1[-1,j] < self.D, (self.G1 * self.a2 /(self.D - self.r2 - self.opti_states_l1[t, j])), 0)
                        + self.G2 * self.opti_controls_l1[t,i] *self.opti_controls_l1[t, i]\
                        + self.G3 * (self.v_free - self.opti_states_l1[t, j+1]) * (self.v_free - self.opti_states_l1[t, j+1])\
                        + self.G4 * (self.opti_states_l1[t,j+1]-pred_state_f[t,j+1])**2
                        print('YEL LIGHT')

                    elif predict_list[t] == 2.0:  #红灯
                        self.obj_l1 += self.G0 * self.a1 * (self.L - self.opti_states_l1[t, j])+\
                        if_else(self.state_l1[-1,j] < self.D, self.G1 * self.a3 /(self.D1 - self.opti_states_l1[t, j]), 0)
                        + self.G2 * self.opti_controls_l1[t, i] * self.opti_controls_l1[t, i]\
                        + self.G3 * (self.v_free - self.opti_states_l1[t, j+1]) * (self.v_free - self.opti_states_l1[t, j+1])\
                        + self.G4 * (self.opti_states_l1[t,j+1]-pred_state_f[t,j+1])**2
                        print('RED LIGHT')

                    else:
                        print('worry')                    
                # TODO: 如果是绿灯就可以，如果是红灯、黄灯，就不行
                else:
                    if predict_list[t] == 0.0: #绿灯
                        self.obj_l1 += self.G0 * self.a1 * (self.L - self.opti_states_l1[t,j])
                        + self.G2 * self.opti_controls_l1[t,i] * self.opti_controls_l1[t,i]\
                        + self.G3 * (self.v_free-self.opti_states_l1[t,j+1])*(self.v_free-self.opti_states_l1[t,j+1])\
                        + self.G4 * (self.opti_states_l1[t,j+1]-pred_state_f[t,j+1])**2
                        
                    elif predict_list[t] == 1.0:    #黄灯
                        if  self.state_l1[-1,j] < self.D:
                            # self.G0 * self.a1 * (self.L - self.opti_states_l[t-1,j])+\
                            self.obj_l1 +=  self.G0 * self.a1 * (self.L - self.opti_states_l1[t,j])+\
                            (self.G1 * self.a2 /(self.D + self.r2 - self.opti_states_l1[t, j])**2)
                            + self.G2 * self.opti_controls_l1[t,i] *self.opti_controls_l1[t, i]\
                            + self.G3 * (self.v_free - self.opti_states_l1[t, j+1]) * (self.v_free - self.opti_states_l1[t, j+1])\
                            + self.G4 * (self.opti_states_l1[t,j+1]-pred_state_f[t,j+1])**2
                            print('YEL LIGHT')
                        else:
                            # self.G0 * self.a1 * (self.L - self.opti_states_l[t-1,j])+\
                            self.obj_l1 +=  self.G0 * self.a1 * (self.L - self.opti_states_l1[t,j])
                            + self.G2 * self.opti_controls_l1[t,i] *self.opti_controls_l1[t, i]\
                            + self.G3 * (self.v_free - self.opti_states_l1[t, j+1]) * (self.v_free - self.opti_states_l1[t, j+1])\
                            + self.G4 * (self.opti_states_l1[t,j+1]-pred_state_f[t,j+1])**2
                            print('YEL LIGHT')

                    elif predict_list[t] == 2.0:  #红灯
                        if  self.state_l1[-1,j] < self.D:
                            self.obj_l1 += self.G0 * self.a1 * (self.L - self.opti_states_l1[t, j])+\
                            self.G1 * self.a2 /(self.D - self.opti_states_l1[t, j])**2
                            + self.G2 * self.opti_controls_l1[t, i] * self.opti_controls_l1[t, i]\
                            + self.G3 * (self.v_free - self.opti_states_l1[t, j+1]) * (self.v_free - self.opti_states_l1[t, j+1])\
                            + self.G4 * (self.opti_states_l1[t,j+1]-pred_state_f[t,j+1])**2
                            print('BEHIND RED LIGHT')
                        else:
                            self.obj_l1 += self.G0 * self.a1 * (self.L - self.opti_states_l1[t, j])+\
                            + self.G2 * self.opti_controls_l1[t, i] * self.opti_controls_l1[t, i]\
                            + self.G3 * (self.v_free - self.opti_states_l1[t, j+1]) * (self.v_free - self.opti_states_l1[t, j+1])\
                            + self.G4 * (self.opti_states_l1[t, j+1]-pred_state_f[t,j+1])**2
                            print('Ifornt RED LIGHT')
                    else:
                        print('worry')

        self.opti_l1.minimize(self.obj_l1)
        '''添加车辆约束'''
        
        for i in range(self.num_l1):
            for t in range(self.N):
                j = 2*i
                p_next_1 = self.opti_states_l1[t,j] + self.opti_states_l1[t,j+1] * self.T + 0.5 * self.T * self.T *self.opti_controls_l1[t,:] #p_next是指它的位置
                v_next_1 = self.opti_states_l1[t,j+1] + self.T *self.opti_controls_l1[t,:] #v是指它的速度
                x_next_1 = ca.horzcat(p_next_1, v_next_1) #水平方向连接矩阵，竖直方向连接矩阵需要ca.vertcat()函数
                self.opti_l1.subject_to(self.opti_states_l1[t+1, :] == x_next_1)
        
        self.opti_l1.subject_to(self.opti_states_l1[0,:] == self.opt_l1_x0)
        
        for i in range(self.num_l):
        
            self.opti_l1.subject_to(self.opti_l1.bounded(self.u_min, self.opti_controls_l1[:,i], self.u_max))
            self.opti_l1.subject_to(self.opti_l1.bounded(4, self.opti_states_l1[:,2*i+1], self.v_free))        

        #pred_control_l = np.zeros((self.N, 1))
        #pred_state_l = np.ones((self.N+1,2))
        #self.opti_l.set_initial(self.opti_controls_l, pred_control_l)
        #self.opti_l.set_initial(self.opti_states_l, pred_state_l)
        self.opti_l1.set_initial(self.opti_controls_l1, pred_control_l)
        self.opti_l1.set_initial(self.opti_states_l1, pred_state_l)
        self.opti_l1.set_value(self.opt_l1_x0, current_state_l)

        opts_setting = {'ipopt.max_iter': 1000, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
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
        #for i in range(self.N):
        pred_state_l=np.concatenate((pred_states_l[1:],pred_states_l[-1:]))
        pred_control_l=np.concatenate((pred_controls_l[1:],pred_controls_l[-1:]))
        cur_control_l=pred_control_l[0]
        self.control_l1.append(pred_control_l[0])
        cur_v_l=pred_state_l[1,1]
        cur_x_l = pred_state_l[1,0]
        print(f"cur_control_l  is  {cur_control_l}")
        new_element=np.array([[cur_x_l,cur_v_l]])
        self.state_l1=np.vstack((self.state_l1,new_element))

        return pred_control_l,pred_state_l
    def follower(self,pred_state_l,pred_state_f,pred_control_f):

        current_f_v = pred_state_f[0,1]
        current_f_x = self.IDM_state[-1,0]
        current_state_f = np.array([current_f_x,current_f_v])   

        self.K_s=1
        self.K_e=1
        self.sigma=2
        
        self.LV=5
        self.K_s_log=100
        self.K_v_log=1

        self.opti_f = ca.Opti()
        self.num_f = self.vehicle1_type.count('Human_vehicle')
        self.opti_controls_f = self.opti_f.variable(self.N,self.num_f)
        self.opti_states_f = self.opti_f.variable(self.N+1,2*self.num_f)

        self.opt_f_x0 = self.opti_f.parameter(1,2)
        self.opti_f.subject_to(self.opti_states_f[0,:] == self.opt_f_x0)
        for i in range(self.num_f-1):
            new_variable = self.opti_f.variable(self.N+1,2*self.num_f)
            self.opti_states_f = ca.horzcat(self.opti_states_f,new_variable)
                    
        self.obj_f=0        
        for i in range(self.num_f):
            for t in range(self.N+1):
                j=2*i
                self.obj_f += self.K_s*(self.K_v_log*((self.opti_states_f[t,j+1]-pred_state_l[t,j+1])**2 +self.sigma)\
                                                    +self.K_s_log/((self.opti_states_f[t,j]-pred_state_l[t,j])**2+self.LV)\
                                                        )\
                +self.K_e *((self.opti_states_f[t,j+1]-self.v_free)**2+self.sigma)

        self.opti_f.minimize(self.obj_f)

        '''添加车辆约束'''
        for i in range(self.num_f):   
            for t in range(self.N):
                j=2*i
                p_next_f = self.opti_states_f[t,j] + self.opti_states_f[t,j+1] * self.T + 0.5 * self.T * self.T * self.opti_controls_f[t,:] #p_next是指它的位置
                v_next_f = self.opti_states_f[t,j+1] + self.T * self.opti_controls_f[t,:] #v是指它的速度
                x_next_f = ca.horzcat(p_next_f, v_next_f) #水平方向连接矩阵，竖直方向连接矩阵需要ca.vertcat()函数
                self.opti_f.subject_to(self.opti_states_f[t+1, :] == x_next_f)
               
        for i in range(self.num_f):
            self.opti_f.subject_to(self.opti_f.bounded(self.u_min, self.opti_controls_f[:,i], self.u_max))
            self.opti_f.subject_to(self.opti_f.bounded(4, self.opti_states_f[:,2*i+1], self.v_free))

        self.opti_f.set_initial(self.opti_controls_f, pred_control_f)
        self.opti_f.set_initial(self.opti_states_f, pred_state_f)
        self.opti_f.set_value(self.opt_f_x0, current_state_f)        
        opts_setting = {'ipopt.max_iter': 1000, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
                    'ipopt.acceptable_obj_change_tol': 1e-6}
        self.opti_f.solver('ipopt',opts_setting)
        solution_f = self.opti_f.solve()

        pred_controls_f = solution_f.value(self.opti_controls_f)
        pred_states_f = solution_f.value(self.opti_states_f) 
        pred_state_f=np.concatenate((pred_states_f[1:],pred_states_f[-1:]))
        pred_control_f=np.concatenate((pred_controls_f[1:],pred_controls_f[-1:]))

        cur_control_f = pred_control_f[0]
        self.control_f.append(cur_control_f)
        #new_element=np.array([[next_x,next_v]])
        #self.state_f=np.vstack((self.state_f,new_element))
        print(f"cur_control_f  {cur_control_f}")
        return pred_control_f,pred_state_f
    
    # TODO: 这里的IBR迭代的过程中，follower是有问题的 
    ''' IBR是针对L-F博弈，但是MPC的滚动优化是针对MPC，二者不冲突 '''

    ''' IBR要收敛到stackberg，先证明收敛性，再求 '''

    def follower_1(self,pred_state_l,pred_state_f,pred_control_f):

        current_f_v = pred_state_f[0,1]
        current_f_x = self.IDM_state[-1,0]
        current_state_f = np.array([current_f_x,current_f_v])   

        self.K_s=1
        self.K_e=1
        self.sigma=2
        
        self.LV=5
        self.K_s_log=100
        self.K_v_log=1

        self.opti_f1=ca.Opti()
        self.num_f1= self.vehicle_type.count('Human_vehicle')
        self.opti_controls_f1 = self.opti_f1.variable(self.N,self.num_f1)
        self.opti_states_f1 = self.opti_f1.variable(self.N+1,2*self.num_f1)

        self.opt_f1_x0 = self.opti_f1.parameter(1,2)
        self.opti_f1.subject_to(self.opti_states_f1[0,:] == self.opt_f1_x0)
        for i in range(self.num_f1-1):
            new_variable = self.opti_f1.variable(self.N+1,2*self.num_f1)
            self.opti_states_f1 = ca.horzcat(self.opti_states_f1,new_variable)
                    
        self.obj_f1 = 0         
        for i in range(self.num_f1):
            for t in range(self.N+1):
                j=2*i
                self.obj_f1 += self.K_s*(self.K_v_log*((self.opti_states_f1[t,j+1]-pred_state_l[t,j+1])**2 +self.sigma)\
                                                    +self.K_s_log/((self.opti_states_f1[t,j]-pred_state_l[t,j])**2+self.LV)\
                                                        )\
                +self.K_e *((self.opti_states_f1[t,j+1]-self.v_free)**2+self.sigma)

        self.opti_f1.minimize(self.obj_f1)

        '''添加车辆约束'''
        for i in range(self.num_f1):   
            for t in range(self.N):
                j=2*i
                p_next_f = self.opti_states_f1[t,j] + self.opti_states_f1[t,j+1] * self.T + 0.5 * self.T * self.T * self.opti_controls_f1[t,:] #p_next是指它的位置
                v_next_f = self.opti_states_f1[t,j+1] + self.T * self.opti_controls_f1[t,:] #v是指它的速度
                x_next_f = ca.horzcat(p_next_f, v_next_f) #水平方向连接矩阵，竖直方向连接矩阵需要ca.vertcat()函数
                self.opti_f1.subject_to(self.opti_states_f1[t+1, :] == x_next_f)
               
        for i in range(self.num_f1):
            self.opti_f1.subject_to(self.opti_f1.bounded(self.u_min, self.opti_controls_f1[:,i], self.u_max))
            self.opti_f1.subject_to(self.opti_f1.bounded(4, self.opti_states_f1[:,2*i+1], self.v_free))

        self.opti_f1.set_initial(self.opti_controls_f1, pred_control_f)
        self.opti_f1.set_initial(self.opti_states_f1, pred_state_f)
        self.opti_f1.set_value(self.opt_f1_x0, current_state_f)        
        opts_setting = {'ipopt.max_iter': 1000, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
                    'ipopt.acceptable_obj_change_tol': 1e-6}
        self.opti_f1.solver('ipopt',opts_setting)
        solution_f = self.opti_f1.solve()

        pred_controls_f = solution_f.value(self.opti_controls_f1)
        pred_states_f = solution_f.value(self.opti_states_f1) 
        pred_state_f=np.concatenate((pred_states_f[1:],pred_states_f[-1:]))
        pred_control_f=np.concatenate((pred_controls_f[1:],pred_controls_f[-1:]))


        #print(f"pred_state_f is {pred_state_f}")

        cur_control_f = pred_control_f[0]
        self.control_f1.append(cur_control_f)
        #new_element=np.array([[next_x,next_v]])
        #self.state_f=np.vstack((self.state_f,new_element))
        print(f"cur_control_f  {cur_control_f}")
        return pred_control_f,pred_state_f
    
    # TODO: 这里的IBR迭代的过程中，follower是有问题的 
    ''' IBR是针对L-F博弈，但是MPC的滚动优化是针对MPC，二者不冲突 '''

    ''' IBR要收敛到stackberg，先证明收敛性，再求 '''

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


    def IBR(self):
        self.light, self.start_time = self.traffic_light(self.random_int)
        predict_list_fin = np.ones((1,self.N))
        pred_state_l = np.ones((self.N+1,2))
        pred_state_l[:,0] = 35
        pred_state_l[:,1] = 5
        self.state_l[0,:] = [35,5]
        pred_state_f = np.ones((self.N+1,2))
        pred_state_f[:,0] = 15
        pred_state_f[:,1] = 5
        self.state_f[0,:] = [15,5]
        pred_control_f = np.zeros((self.N, 1))
        pred_control_f[0] = 5 
        pred_control_l = np.zeros((self.N, 1))

        for t in range(self.start_time, self.start_time + self.max_iter):
            if t == self.start_time:

                predict_list = self.predict_light(self.N, self.light, t)
                predict_list_fin = np.vstack((predict_list_fin,predict_list))
                pred_control_f, pred_state_f = self.follower(pred_state_l,pred_state_f, pred_control_f)
                v_follower = self.state_f[-1,1]
                v1 = pred_state_l[0,1]
                x1 = pred_state_l[0,0]
                v2 = 5
                x2 = 5
                a_f = self.IDM(v1,v2,x1,x2)
                v_f = v2 + a_f * self.T
                x_f = x2 + v2 * self.T + 0.5 * a_f * self.T * self.T
                v = pred_state_f[1,1]
                pred_state_f[1,0] = x_f

                new_element=np.array([[x_f,v_f]])

                self.state_f = np.vstack((self.state_f,new_element))
                self.IDM_state = np.vstack((self.IDM_state,new_element))
                pred_control_l, pred_state_l = self.leader(pred_state_f, pred_state_l, pred_control_l, predict_list)

            else:

                predict_list = self.predict_light(self.N, self.light, t)
                predict_list_fin = np.vstack((predict_list_fin,predict_list))
                pred_control_f,pred_state_f = self.follower(pred_state_l, pred_state_f, pred_control_f)
                v1 = self.state_l[-1,1]
                x1 = self.state_l[-1,0]
                v2 = self.IDM_state[-1,1]
                x2 = self.IDM_state[-1,0]
                a= self.IDM(v1,v2,x1,x2)
                v_f = v2 + a * self.T
                x_f = (x2 + v2 * self.T + 0.5 * a * self.T * self.T)
                v = pred_state_f[1,1]
                new_element=np.array([[x_f,v_f]])
                self.state_f = np.vstack((self.state_f,new_element))
                self.IDM_state = np.vstack((self.IDM_state,new_element))
                pred_control_l, pred_state_l = self.leader(pred_state_f, pred_state_l, pred_control_l, predict_list)
                v_leader = self.state_l[-1,1]
                x_leader = self.state_l[-1,0]


            if self.control_l[-1] < self.control_l[-2] and self.control_f[-1] < self.control_f[-2]:
                print("算法收敛")
                print()
                #return self.control_l, self.control_f, t
            else:
                pass

        #np.save(f"/home/fanjx/project/FJX_project/Code_main/result/f_state.npy", np.array(self.state_f))
        state_f = pd.DataFrame(self.state_f,columns =["x1","v1"])
        state_f.to_csv("/home/fanjx/project/FJX_project/Code_main/result/single/f_state.csv",index = False)

        state_l = pd.DataFrame(self.state_l,columns =["x1","v1"])
        state_l.to_csv("/home/fanjx/project/FJX_project/Code_main/result/single/l_state.csv",index = False)
        
        light = pd.DataFrame(self.light,columns =["light"])
        light.to_csv("/home/fanjx/project/FJX_project/Code_main/result/single/light.csv",index = False)

        predict_list_fin = pd.DataFrame(predict_list_fin)
        predict_list_fin.to_csv("/home/fanjx/project/FJX_project/Code_main/result/single/predict_list_fin.csv",index = False)

        IDM_state = pd.DataFrame(self.IDM_state,columns = ["x1","v1"])
        IDM_state.to_csv("/home/fanjx/project/FJX_project/Code_main/result/single/IDM_state.csv",index = False)
        return self.state_l, self.state_f, self.light,self.IDM_state


    def IBR_1(self):
        
        #random_int = 72 
        self.light, self.start_time = self.traffic_light(self.random_int)
        predict_list_fin = np.ones((1,self.N))
        #print(f"predict_list { predict_list}")
        pred_state_l = np.ones((self.N+1,2))
        pred_state_l[:,0] = 15
        pred_state_l[:,1] = 5
        self.state_l1[0,:] = [15,5]
        pred_state_f = np.ones((self.N+1,2))
        pred_state_f[:,0] = 5
        pred_state_f[:,1] = 5
        self.state_f1[0,:] = [5,5]
        pred_control_f = np.zeros((self.N, 1))
        pred_control_f[0] = 5 
        pred_control_l = np.zeros((self.N, 1))

        for t in range(self.start_time, self.start_time + self.max_iter):
            if t == self.start_time:

                predict_list = self.predict_light(self.N, self.light, t)
                predict_list_fin = np.vstack((predict_list_fin,predict_list))
                pred_control_f, pred_state_f = self.follower_1(pred_state_l,pred_state_f, pred_control_f)
                v_follower = self.state_f1[-1,1]
                v1 = pred_state_l[0,1]
                x1 = pred_state_l[0,0]
                v2 = 5
                x2 = 5
                a_f = self.IDM(v1,v2,x1,x2)
                v_f = v2 + a_f * self.T
                x_f = x2 + v2 * self.T + 0.5 * a_f * self.T * self.T
                v = pred_state_f[1,1]
                pred_state_f[1,0] = x_f
                new_element=np.array([[x_f,v_f]])
                self.state_f1 = np.vstack((self.state_f1,new_element))
                self.IDM_state1 = np.vstack((self.IDM_state1,new_element))
                pred_control_l, pred_state_l = self.leader_1(pred_state_f, pred_state_l, pred_control_l, predict_list)

            else:
                predict_list = self.predict_light(self.N, self.light, t)
                predict_list_fin = np.vstack((predict_list_fin,predict_list))
                pred_control_f,pred_state_f = self.follower_1(pred_state_l, pred_state_f, pred_control_f)
                v1 = self.state_l1[-1,1]
                x1 = self.state_l1[-1,0]
                v2 = self.IDM_state1[-1,1]
                x2 = self.IDM_state1[-1,0]
                a= self.IDM(v1,v2,x1,x2)
                v_f = v2 + a * self.T
                x_f = (x2 + v2 * self.T + 0.5 * a * self.T * self.T)
                v = pred_state_f[1,1]
                new_element=np.array([[x_f,v_f]])
                self.state_f1 = np.vstack((self.state_f1,new_element))
                self.IDM_state1 = np.vstack((self.IDM_state1,new_element))
                pred_control_l, pred_state_l = self.leader_1(pred_state_f, pred_state_l, pred_control_l, predict_list)
                v_leader = self.state_l1[-1,1]
                x_leader = self.state_l1[-1,0]                    

            if self.control_l[-1] < self.control_l[-2] and self.control_f[-1] < self.control_f[-2]:
                print("算法收敛")
                print()

            else:
                pass
        
        light = pd.DataFrame(self.light,columns =["light"])
        light.to_csv("/home/fanjx/project/FJX_project/Code_main/result/multi_leader/light.csv",index = False)

        state_f1 = pd.DataFrame(self.state_f1,columns =["x1","v1"])
        state_f1.to_csv("/home/fanjx/project/FJX_project/Code_main/result/multi_leader/f1_state.csv",index = False)

        state_l1 = pd.DataFrame(self.state_l1,columns =["x1","v1"])
        state_l1.to_csv("/home/fanjx/project/FJX_project/Code_main/result/multi_leader/l1_state.csv",index = False)

        IDM_state1 = pd.DataFrame(self.IDM_state1,columns = ["x1","v1"])
        IDM_state1.to_csv("/home/fanjx/project/FJX_project/Code_main/result/multi_leader/IDM1_state.csv",index = False)                
        return self.state_l1,self.state_f1,self.IDM_state1

class VIS():
    def __init__ (self,state_l, state_l1, state_IDM, state_IDM1,light):
        self.state_l = state_l
        self.state_f = state_IDM
        self.state_l1 = state_l1
        self.state_f1 = state_IDM1
        self.leader_patches = []
        self.follower_patches = []
        self.leader_patches1 = []
        self.follower_patches1 = []
        self.traffic_patches = []
        self.fig,self.ax = plt.subplots(figsize=(60,10))
        self.length = 5
        self.width = 3
        self.light =light

    def Road(self):
        # 创建横向的路
        #fig, ax = plt.subplots(figsize=(100,50))
        x_lat_start1 = 0
        x_lat_start2 = 320
        x_lat_end1 = 300
        x_lat_end2 =620
        y_lat_start1 = -10
        y_lat_end1 = -50
        y_long_start_1 = 10
        y_long_end = 50
        #x_long_1 = -10
        #x_long_2 = 10
        self.ax.plot([x_lat_start1,x_lat_end1],[y_lat_start1,y_lat_start1],'k-',lw=2)
        self.ax.plot([x_lat_start1,x_lat_end1],[y_long_start_1,y_long_start_1],'k-',lw=2)
        self.ax.plot([x_lat_start2,x_lat_end2],[y_long_start_1,y_long_start_1],'k-',lw=2)
        self.ax.plot([x_lat_start2,x_lat_end2],[y_lat_start1,y_lat_start1],'k-',lw=2)
        self.ax.plot([x_lat_start1,x_lat_end1],[x_lat_start1,x_lat_start1],'k--',lw=2)
        self.ax.plot([x_lat_start2,x_lat_end2],[x_lat_start1,x_lat_start1],'k--',lw=2)
        # 创建纵向的路
        self.ax.plot([x_lat_end1, x_lat_end1],[y_long_start_1,y_long_end],'k-',lw=2)
        self.ax.plot([x_lat_start2, x_lat_start2],[y_long_start_1, y_long_end],'k-',lw=2)
        self.ax.plot([x_lat_end1, x_lat_end1],[y_lat_start1,y_lat_end1],'k-',lw=2)
        self.ax.plot([x_lat_start2, x_lat_start2],[y_lat_start1,y_lat_end1],'k-',lw=2)
        self.ax.plot([x_lat_start2-10,x_lat_start2-10],[y_long_start_1,y_long_end],'k--',lw=2)
        self.ax.plot([x_lat_start2-10,x_lat_start2-10],[y_lat_start1,y_lat_end1],'k--',lw=2)        
            #plt.pause(0.5)
        # 设置坐标轴的范围和比例
        self.ax.set_xlim(0, 620)
        self.ax.set_ylim(-50, 50)
    def init_patches(self):
        patch_l = patches.Rectangle((self.state_l[0,0],-9),self.length,self.width,edgecolor='black',facecolor='black')
        patch_f = patches.Rectangle((self.state_f[0,0],-9),self.length,self.width,edgecolor='blue',facecolor='blue')
        patch_l1 = patches.Rectangle((self.state_l1[0,0],1),self.length,self.width,edgecolor='black',facecolor='black')
        patch_f1 = patches.Rectangle((self.state_f1[0,0],1),self.length,self.width,edgecolor='blue',facecolor='blue')         
        self.ax.add_patch((patch_l))
        self.ax.add_patch((patch_f))
        self.ax.add_patch((patch_l1))
        self.ax.add_patch((patch_f1))
        self.leader_patches.append(patch_l)  
        self.follower_patches.append(patch_f)
        self.leader_patches1.append(patch_l1)
        self.follower_patches1.append(patch_f1)
        return self.leader_patches, self.follower_patches
    
    def update_patches(self,frame,current_light):
        #print(f"current_light is {current_light}")
        
        patch_l = patches.Rectangle((self.state_l[frame,0],-8),self.length,self.width,edgecolor='black',facecolor='black')
        patch_f = patches.Rectangle((self.state_f[frame,0],-8),self.length,self.width,edgecolor='blue',facecolor='blue')
        patch_l1 = patches.Rectangle((self.state_l1[frame,0],2),self.length,self.width,edgecolor='black',facecolor='black')
        patch_f1 = patches.Rectangle((self.state_f1[frame,0],2),self.length,self.width,edgecolor='blue',facecolor='blue')                        
        #交通灯
        x_tangle = 330
        y_tangle = 20
        x_circle = 340
        red_circle = 35
        yellow_circle = 25
        green_circle = 15
        background = patches.Rectangle((x_tangle,y_tangle),20,60,edgecolor='black',facecolor='black')
        self.ax.add_patch(background)
        current_light_patch = None
        if current_light_patch is not None:
            self.ax.patches.remove(current_light_patch)  #移除了之前的红灯
        if current_light == 2.0:
            current_light_patch = patches.Circle((x_circle, red_circle), 3, facecolor='red')
            self.traffic_patches.append(current_light_patch)
        elif current_light == 1.0:
            current_light_patch = patches.Circle((x_circle, yellow_circle), 3, facecolor='yellow')
            self.traffic_patches.append(current_light_patch)
        elif current_light == 0.0:
            current_light_patch = patches.Circle((x_circle, green_circle), 3, facecolor='green')
            self.traffic_patches.append(current_light_patch)
        self.ax.add_patch(current_light_patch)
        self.ax.add_patch((patch_l))
        self.ax.add_patch((patch_f))
        self.ax.add_patch((patch_l1))
        self.ax.add_patch((patch_f1))
        self.leader_patches.append(patch_l)  
        self.follower_patches.append(patch_f)
        self.leader_patches1.append(patch_l1)
        self.follower_patches1.append(patch_f1)

        return self.leader_patches, self.follower_patches,self.leader_patches1,self.follower_patches1,self.traffic_patches

    def show(self):
        self.Road()
        self.leader_patches, self.follower_patches, self.leader_patches1,self.follower_patches1,self.traffic_patches = self.update_patches(0,self.light[0])
        
        for frame in range(1,125):

            current_light = self.light[frame]
            self.update_patches(frame, current_light)
        self.ax.set_aspect(1)

        i = 0
        for lea,foll,leader,follower,traf in zip(self.leader_patches,self.follower_patches,self.leader_patches1,self.follower_patches1,self.traffic_patches):
            i += 1
            patch_l = lea
            patch_f = foll
            patch_light = traf
            patch_l1 = leader
            patch_f1 = follower
            plt.cla()
            self.Road()
            self.ax.add_patch(patch_l)
                #for patch_f_son in patch_f:
            self.ax.add_patch(patch_f)
            self.ax.add_patch(patch_l1)
            self.ax.add_patch(patch_f1)
            self.ax.add_patch(patch_light)
            plt.pause(1)
            plt.tight_layout()
            plt.savefig(f'./figure1/frame_{i}.png')
        plt.show()


# TODO: 这里MPC的设置有问题，我的输入的current state有问题，应该实时更新，记得以后去修改一下
if __name__ == '__main__':
    random_int = np.random.randint(0,126)
    #nmpc = NMPC(random_int)
    #state_l, state_f, light, IDM_state= nmpc.IBR()
    print(f"这里开始镜像")
    #state_l_1, state_f_1, IDM_state_1= nmpc.IBR_1()
    leader_path = '/home/fanjx/project/FJX_project/Code_main/final_result/l_state.csv'
    follower_path = '/home/fanjx/project/FJX_project/Code_main/final_result/f_state.csv'
    IDM_path = '/home/fanjx/project/FJX_project/Code_main/final_result/IDM_state.csv'
    light_path = '/home/fanjx/project/FJX_project/Code_main/final_result/light.csv'
    leader_path_1 = '/home/fanjx/project/FJX_project/Code_main/final_result/multi_leader/l1_state.csv'
    follower_path_1 = '/home/fanjx/project/FJX_project/Code_main/final_result/multi_leader/f1_state.csv'
    IDM_path_1 = '/home/fanjx/project/FJX_project/Code_main/final_result/multi_leader/IDM1_state.csv'
    #light_path_1 = '/home/fanjx/project/FJX_project/Code_main/result1/light1.csv'
    light = np.loadtxt(light_path, delimiter=',',skiprows=1)
    #print(f"light is {light}")
    #print(f"type_light is {light.dtype}")
    state_l = np.loadtxt(leader_path, delimiter=',',skiprows = 2)
    #print(f"state_l is {state_l}")
    state_IDM = np.loadtxt(IDM_path, delimiter=',',skiprows = 2)
    df1 = pd.read_csv(leader_path_1)
    df1.iloc[:,0] = -df1.iloc[:,0] +620
    df1.iloc[:,1] = -df1.iloc[:,1]
    df1.to_csv(leader_path_1,index = False)
    state_l1 = np.loadtxt(leader_path_1, delimiter=',',skiprows = 2)
    df2 = pd.read_csv(IDM_path_1)
    df2.iloc[:,0] = -df2.iloc[:,0] +620
    df2.iloc[:,1] = -df2.iloc[:,1]
    df2.to_csv(IDM_path_1,index = False)
    state_IDM1 = np.loadtxt(IDM_path_1, delimiter=',',skiprows = 2)
    #light_1 = np.loadtxt(light_path_1, delimiter=',',skiprows=1)

    #print(f"state_IDM is {state_IDM}")
    vis = VIS(state_l, state_l1, state_IDM, state_IDM1, light)
    vis.show()
