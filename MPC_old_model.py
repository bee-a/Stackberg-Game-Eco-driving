# DAT优化模型

from gurobipy import *

def mpc_own(predict_horizon1, control_horizon1, pos1, speed1):
    predict_horizon = int(predict_horizon1) + 1
    control_horizon = int(control_horizon1) + 1
    arrive_time = predict_horizon1
    print(arrive_time)

    # 参数
    xishu1 = 10  # 位移比重
    xishu2 = 1  # 速度比重
    xishu3 = 1  # 终端比重
    xishu4 = 20  # 过程比重
    T = 0.5  # 时间步长
    speed_min = 1
    speed_max = 11.11
    stop_pos = 300  # 停车线位置

    try:
        # 创建一个模型
        m = Model('optimization')
        acc = m.addVars(predict_horizon, lb=-2, ub=2, vtype=GRB.CONTINUOUS, name="acc")
        speed = m.addVars(predict_horizon, vtype=GRB.CONTINUOUS, name="speed")
        jeck = m.addVars(predict_horizon, lb=-2, ub=2, vtype=GRB.CONTINUOUS, name="jeck")
        pos = m.addVars(predict_horizon, vtype=GRB.CONTINUOUS, name="pos")

        # 创建目标函数
        Obj = LinExpr(0)  # 终端约束-速度 + 位置
        Obj2 = LinExpr(0)  # 过程约束-jeck
        ##这里的objective在PPC的基础上添加了一些改变
        Obj = Obj + xishu2 * (speed[arrive_time] - 11.11) * (speed[arrive_time] - 11.11) + xishu1 * (
                pos[arrive_time] - stop_pos) * (pos[arrive_time] - stop_pos)
        for time in range(predict_horizon):
            Obj2 = Obj2 + jeck[time] * jeck[time]  # 过程约束-jeck

        Obj = Obj * xishu3 + Obj2 * xishu4
        m.setObjective(Obj, GRB.MINIMIZE)

        ### 添加约束
        ## 运动学结构
        # 写入初始状态 - 位置，速度
        m.addConstr(speed[0] == speed1)
        m.addConstr(pos[0] == pos1)
        # 运动学
        for time in range(predict_horizon):
            if time > 0:
                m.addConstr(speed[time] == speed[time - 1] + acc[time - 1] * T)
                m.addConstr(pos[time] == pos[time - 1] + speed[time - 1] * T + 0.5 * acc[time - 1] * T * T)

        # 加速度，速度约束
        for time in range(predict_horizon):
            if time > 0:
                m.addConstr(speed[time] >= speed_min)
                m.addConstr(speed[time] <= speed_max)


        # 控制时域后加速度为0
        #     time1 = predict_horizon - veh_information_1['expected_time'].iloc[veh]
        #     if time1 > 0:
        #         for time in range(time1 - 1):
        #             m.addConstr(jeck[veh_information_1['expected_time'].iloc[veh] + time] == 0)
        m.write('model5.lp')

        # m.params.TimeLimit = 10
        # m.Params.MIPGap = 1
        # m.params.DualReductions = 0
        m.params.LogToConsole = 0
        # m.params.NonConvex = 2
        m.optimize()
        # print("Runtime:{}".format(m.Runtime))

        if m.status == GRB.Status.INFEASIBLE:  # 判断优化问题是否无解，
            print('Optimization was stopped with status %d' % m.status)
            # do IIS, find infeasible constraints
            m.computeIIS()# 若无可行解，计算导致无解的冲突
            for c in m.getConstrs():  #遍历所有约束
                if c.IISConstr: 
                    print('%s' % c.constrName)  #输出被标记的约束的名称

        # 输出所有变量
        '''
        for v in m.getVars():
            print('%s %g' % (v.varName, round(v.x, 5)))
        print('Obj: %g' % m.objVal)
        '''
        # temp = input("确定评价指标:")

    except GurobiError as e:
        print("Error code" + str(e.errno) + ": " + str(e))

    except AttributeError:
        print('Encountered an attribute error')


    return round(acc[0].x, 5), m.Runtime
