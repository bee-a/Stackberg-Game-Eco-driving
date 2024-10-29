import casadi as ca
import numpy as np
import time
from Code_main.single_vehicle.draw_single_veh import draw_single_veh
import pandas as pd
from scipy.stats import norm

#TODO: 在代码中加入uncertainty的部分 
##这部分代码给出了求解一次MPC的过程，最终得出的结果是uc,表示加速度
'''
mean=0 #均值
std=1 #标准差
size=1000 #随机数数量
noise=np.random.normal(mean,std,size)
'''

def movement(current_state, u, T, N):
# 这部分代码是MPC的原理，基于当前的state对horizon为N的时域进行预测
    state_ = np.zeros((N+1, 2))
    state_[0, :] = current_state
    for i in range(N):
        state_[i + 1, 0] = state_[i, 0] + state_[i, 1] * T + 0.5 * T * T * u[i]
        state_[i + 1, 1] = state_[i, 1] + T * u[i]
    return state_

def shift_movement(T, t0, x0, u, x_f):
    p_ = x0[0] + x0[1] * T + 0.5 * T * T * u[0]
    v_ = x0[1] + T * u[0]
    st = np.array([p_, v_])
    t = t0 + T
    u_end = np.concatenate((u[1:], u[-1:])) #需要拼接的向量
    x_f = np.concatenate((x_f[1:], x_f[-1:])) 
    return t, st, u_end, x_f

def nmpc(state_all_start, T, N):
    """
    state_all_start 为上一时刻dataframe中的状态
    T 为时间步长
    N 为预测时域
    """
    L = 100000    #目的地离我的距离
    D = 300     #距离交通灯的距离
    G0 = 1  # 目的地引力
    G1 = 1  # 信号灯
    G2 = 0.3  # 生态驾驶
    G3 = 0.0  # 与期望速度的差
    G4 = 1.0  # Jeck
    a1 = 1
    a2 = 1000  # 黄灯
    a3 = 1000  # 红灯
    b1 = 100
    u_max = 5
    u_min = -5
    vfree = 20
    r1 = vfree * 200
    r2 = (u_max * 3 * 3) / 2

    opti = ca.Opti() #创建优化问题的实例
    opti_controls = opti.variable(N, 5) #使用.variable方法创建一个优化数组，用于存储控制量，是优化问题的直接求解对象
    #TODO：这里有问题
    u_0 = opti_controls[:, 0]

    """初始状态矩阵"""
    opt_x0 = opti.parameter(2, 5) #创建初始矩阵，
    """状态更新矩阵"""
    opt_coefficient_1 = opti.parameter(10, 10)
    opt_coefficient_2 = opti.parameter(10, 1)
    opt_states_1 = opti.variable(N+1, 2)
    opt_states_2 = opti.variable(N+1, 2)
    opt_states_3 = opti.variable(N+1, 2)
    opt_states_4 = opti.variable(N+1,2)
    opt_states_5 = opti.variable(N+1,2)
    opt_states=ca.horzcat(opt_states_1,opt_states_2,opt_states_3,opt_states_4,opt_states_5)
    #这里也有问题
    #v_0 = opt_states[:, 1]
    opti.subject_to(opt_states[0:2,:] == opt_x0) #添加优化条件

    """状态更新函数"""
    ''' 拼接状态向量'''
    for i in range(5):
        j=0
        for t in range(N):
        # x_next = opt_coefficient_1 @ opt_states[t, :].T + opt_coefficient_2 @ opti_controls[t, :]
            p_next_1 = opt_states[t,j] + opt_states[t,j+1] * T + 0.5 * T * T * opti_controls[t,i] #p_next是指它的位置
            v_next_1 = opt_states[t,j+1] + T * opti_controls[t,i] #v是指它的速度
            x_next_1 = ca.horzcat(p_next_1, v_next_1) #水平方向连接矩阵，竖直方向连接矩阵需要ca.vertcat()函数
            opti.subject_to(opt_states[t+1, j:j+2] == x_next_1)
        j+=2          

    """目标函数 predict_light是指我自己预测的红绿灯的颜色"""
    obj = 0
    #light_col_index = state_all_start.columns.get_loc('light')
    for i in range(5):
        j=0
        for t in range(N): 
            i=0
            #print(len(state_all_start['light']))
            #print('这里是代码',state_all_start.at[0,'light'])
            print('state_all_start',state_all_start.iloc[t]['light'])
            print('i',i)
            print('j',j)
            if state_all_start.iloc[t]['light'] == 0:   #绿灯
                obj = obj + G0 * a1 * (L - opt_states[t, j]) + G2 * opti_controls[t, i] * opti_controls[t, i] - G3 * (vfree - opt_states[t, j+1]) * (vfree - opt_states[t, j+1])
            elif state_all_start.iloc[t]['light'] == 1:    #黄灯
                obj = obj + G0 * a1 * (L - opt_states[t, j]) + G1 * a2 * np.exp(
                    -(((D - r2 - opt_states[t, j]) **2) / r1)) + G2 * opti_controls[t,i] * opti_controls[t, i] - G3 * (vfree - opt_states[t, j+1]) * (vfree - opt_states[t, j+1])
            elif state_all_start.iloc[t]['light'] == 2:  #红灯
                obj = obj + G0 * a1 * (L - opt_states[t, j]) + G1 * a3 * np.exp(
                    - (((D - opt_states[t, j]) ** 2) / r1)) + G2 * opti_controls[t, i] * opti_controls[t, i] - G3 * (vfree - opt_states[t, j+1]) * (vfree - opt_states[t, j+1])
            else:
                print('worry')
        j+=2
    for i in range(5):
        for t in range(N):
            if t > 0:
                obj = obj + G4 * (opti_controls[t, i] - opti_controls[t - 1, i]) * (opti_controls[t, i] - opti_controls[t - 1, i])

    opti.minimize(obj)
    opti.subject_to(opti.bounded(u_min, u_0, u_max)) #u的取值范围是u_min到u_max之间，u的初始值是u_0
    for i in range(5):
        opti.subject_to(opti.bounded(0, v_0, vfree))

    mean=0
    std_dev=1
    normal_distribution=norm(loc=mean,scale=std_dev)
    z_value=normal_distribution.ppf(0.95)
    opti.subject_to(w<=z_value) 
    opts_setting = {'ipopt.max_iter': 1000, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
                    'ipopt.acceptable_obj_change_tol': 1e-6}
    
    """
    ipopt.acceptable_tol:是指求解器可接受容忍度 
    ipopt.acceptable_obj_change_tol:是指目标函数的变化小于1e-6时,IPOPT会将当前的解作为最优解
    ipopt.print_level是指打印输出等级为0
    print_time 设置是否打印优化过程中的时间信息
    """
    
    opti.solver('ipopt', opts_setting)
###########################################################################
    # 创建初始输入量将state初始化，将A与B两个系数初始化，
    states_ini = np.ones((N+1, 10))

    '''
    pos_ini = []
    pos0 = 0
    speed0 = 0
    for t in range(N+1):
        states_ini[t, 1] = pos0
        pos0 = pos0 + speed0 * T
    '''
    
    t0 = 0
    u0 = np.zeros((N,5))
    # next_states = np.zeros((N+1, 2))
    next_states = states_ini
    current_ori_state = np.array([state_all_start['Pos'], state_all_start['Speed']])
    coefficient_1 = np.array([[1, T], [0, 1]])
    coefficient_2 = np.array([[0.5*T*T], [T]])

    x_c = []  # contains for the history of the state
    u_c = []
    t_c = [t0]  # for the time

    xx = []
    mpciter = 0
    index_t = []

    opti.set_value(opt_x0, current_ori_state) #将current_ori_state赋值给opt_x0
    opti.set_value(opt_coefficient_1, coefficient_1)
    opti.set_value(opt_coefficient_2, coefficient_2)

    opti.set_initial(opti_controls, u0)
    opti.set_initial(opt_states, next_states)
    t_ = time.time()
    sol = opti.solve()  #
    
    index_t.append(time.time() - t_)
    u = sol.value(opti_controls)
    st = sol.value(opt_states)
    u_c.append(u[0])    #将求解出来的一串u只选择第一个值
    t_c.append(t0)

    next_state = movement(N=N, T=T, u=u, current_state=current_ori_state) #current_ori_state有n+1行2列。N是预测时间
    x_c.append(next_state)
    t0, current_ori_state, u0, next_states = shift_movement(T, t0, current_ori_state, u, next_state) #next_state有n+1行2列。N是预测时间
    # print((time.time() - start_time) / (mpciter))
    t_v = np.array(index_t)
    print('solve time:', t_v.mean())   # 求解一个所需要的时间
    return u_c

if __name__ == '__main__':
     start_all_start = pd.read_csv('C:/Users/LEN/Desktop/STUDY/project/ziliao_zou/ziliao_zou/Code_main/2.csv',sep=',',index_col=0)
     u_c = nmpc(state_all_start = start_all_start, T=0.5, N=50)
     print(u_c)

#############################################################################################

