import numpy as np


cost_array=np.random.randint(0,100,size=(3,2),dtype=int)
cost_array_l=np.random.randint(low=0,high=100,size=(3,2),dtype=int)
cost_array_f=np.random.randint(0,100,(3,2),int)
print(f"cost: {cost_array}")
cost=cost_array.reshape(2,3)
print(f"cost:{cost_array_l}")
print(f"cost:{cost_array_f}")
print(f"cost:{cost}")

a = np.eye(4) #4*4矩阵

#生成随机数
abs=np.random.random((2,4))
abs=np.random.rand(2,4) 
abs=np.random.randint(1,100,size=(2,4))
#abs=np.random.randint()

''' '''
a=np.arange(10)
print(f"a:{a}")

''' '''
# print(a.shape[0]) #行数
# print(a.shape[1]) #列数

''' Note: 数组叠加的时候，要保证维度相同 '''
# np.vstack(a,b) #垂直叠加
# np.hstack(a,b) #水平叠加

''' np.concatnate  '''

a = np.array([[1,2],[3,4],[5,6]])
b = np.array([[7,8],[9,10],[11,12]])
c = np.concatenate((a,b), axis = 1)
d = np.concatenate((a,b),axis = 0)
print(f"c{c}")
print(f"d{d}")

''' '''
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

        return pred_control_l,pred_state


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



    def IBR(self):
        random_int = np.random.randint(0,126)
        #random_int = 72 
        self.light, self.start_time = self.traffic_light(random_int)
        predict_list_fin = np.ones((1,self.N))
        #print(f"predict_list { predict_list}")
        pred_state_l = np.ones((self.N+1,2))
        pred_state_l[:,0] = 15
        pred_state_l[:,1] = 5
        self.state_l[0,:] = [15,5]
        pred_state_f = np.ones((self.N+1,2))
        pred_state_f[:,0] = 5
        pred_state_f[:,1] = 5
        self.state_f[0,:] = [5,5]
        pred_control_f = np.zeros((self.N, 1))
        pred_control_f[0] = 5 
        pred_control_l = np.zeros((self.N, 1))

        for t in range(self.start_time, self.start_time + self.max_iter):
            if t == self.start_time:
                #print(f"light[t] is {self.light[t]}")
                predict_list = self.predict_light(self.N, self.light, t)
                predict_list_fin = np.vstack((predict_list_fin,predict_list))
                #print(f"predict_list_init {predict_list}")
                #print(f"pred_state_f_init is {pred_state_f}")
                pred_control_f, pred_state_f = self.follower(pred_state_l,pred_state_f, pred_control_f)
                v_follower = self.state_f[-1,1]
                v1 = pred_state_l[0,1]
                x1 = pred_state_l[0,0]
                v2 = 5
                x2 = 5
                a_f = self.IDM(v1,v2,x1,x2)
                v_f = v2 + a_f * self.T
                x_f = x2 + v2 * self.T + 0.5 * a_f * self.T * self.T
                #print(f"x_f is {x_f}")
                v = pred_state_f[1,1]
                pred_state_f[1,0] = x_f
                #x_follower = self.state_f[-1,0]
                #print(f"v_follower {pred_state_f[0,1]}")
                #print(f"x_follower {x_f}")
                new_element=np.array([[x_f,v_f]])
                #self.state_f=np.vstack((self.state_f,new_element))
                #这里的代码有问题
                self.state_f = np.vstack((self.state_f,new_element))
                self.IDM_state = np.vstack((self.IDM_state,new_element))
                pred_control_l, pred_state_l = self.leader(pred_state_f, pred_state_l, pred_control_l, predict_list)
                #print(f"v_leader {pred_state_l[0,1]}")
                #print(f"x_leader {pred_state_l[0,0]}")
            else:
                #print(f"t is {t}")
                #print(f"light[t] is {self.light[t]}")
                predict_list = self.predict_light(self.N, self.light, t)
                predict_list_fin = np.vstack((predict_list_fin,predict_list))
                #print(f"pred_control_f is {pred_control_f}")
                #print(f"pred_state_f is {pred_state_f}")
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
        state_f.to_csv("/home/fanjx/project/FJX_project/Code_main/result/f_state.csv",index = False)

        state_l = pd.DataFrame(self.state_l,columns =["x1","v1"])
        state_l.to_csv("/home/fanjx/project/FJX_project/Code_main/result/l_state.csv",index = False)
        
        light = pd.DataFrame(self.light,columns =["light"])
        light.to_csv("/home/fanjx/project/FJX_project/Code_main/result/light.csv",index = False)

        predict_list_fin = pd.DataFrame(predict_list_fin)
        predict_list_fin.to_csv("/home/fanjx/project/FJX_project/Code_main/result/predict_list_fin.csv",index = False)

        IDM_state = pd.DataFrame(self.IDM_state,columns = ["x1","v1"])
        IDM_state.to_csv("/home/fanjx/project/FJX_project/Code_main/result/IDM_state.csv",index = False)
        return self.state_l, self.state_f, self.light,self.IDM_state

