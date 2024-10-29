import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as patches
from Multi_CAV import NMPC

class VIS():
    def __init__(self,state_l,state_IDM,state_l1,state_f1,light,num_l,num_f,num_l1,num_f1):
        self.state_l = state_l
        self.state_f = state_IDM
        self.state_l1 = state_l1
        self.state_f1 = state_f1
        self.leader_patches = [[]]
        self.follower_patches = [[]]
        self.leader_patches1 = [[]]
        self.follower_patches1 = [[]]
        self.traffic_patches = []
        self.fig,self.ax = plt.subplots(figsize=(60,10))
        self.length = 4.5
        self.width = 2
        self.light = light
        self.num_l = num_l    #从左向右行驶的车的数量
        self.num_f = num_f
        self.num_l1 = num_l1  #从右向左行驶的车的数量
        self.num_f1 = num_f1
        self.max_iter = 378

    def Road(self):
        # 创建横向的路
        x_lat_start1 = 0
        x_lat_start2 = 314
        x_lat_start3 = 303.5
        x_lat_start4 =307
        x_lat_start5 = 310.5
        x_lat_end1 = 300

        x_lat_end2 = 614
        y_lat_start1 = -7
        y_lat_start2 = -3.5
        y_lat_start3 = 0
        y_lat_start4 = 3.5
        y_lat_start5 = 7

        y_lat_end1 = -50
        y_long_end = 50

        #创建横向的路
        self.ax.plot([x_lat_start1,x_lat_end1],[y_lat_start1,y_lat_start1],'k-',lw = 2)
        self.ax.plot([x_lat_start1,x_lat_end1],[y_lat_start2,y_lat_start2],'k--',lw = 2)
        self.ax.plot([x_lat_start1,x_lat_end1],[y_lat_start3,y_lat_start3],'k-',lw = 2)
        self.ax.plot([x_lat_start1,x_lat_end1],[y_lat_start4,y_lat_start4],'k--',lw = 2)
        self.ax.plot([x_lat_start1,x_lat_end1],[y_lat_start5,y_lat_start5],'k-',lw = 2)
        
        self.ax.plot([x_lat_start2,x_lat_end2],[y_lat_start1,y_lat_start1],'k-',lw = 2)
        self.ax.plot([x_lat_start2,x_lat_end2],[y_lat_start2,y_lat_start2],'k--',lw = 2)
        self.ax.plot([x_lat_start2,x_lat_end2],[y_lat_start3,y_lat_start3],'k-',lw = 2)
        self.ax.plot([x_lat_start2,x_lat_end2],[y_lat_start4,y_lat_start4],'k--',lw = 2)
        self.ax.plot([x_lat_start2,x_lat_end2],[y_lat_start5,y_lat_start5],'k-',lw = 2)
        
        # 创建纵向的路
        self.ax.plot([x_lat_end1, x_lat_end1],[y_lat_start5, y_long_end],'k-',lw = 2)
        self.ax.plot([x_lat_start3, x_lat_start3],[y_lat_start5, y_long_end],'k--',lw = 2)
        self.ax.plot([x_lat_start4, x_lat_start4],[y_lat_start5, y_long_end],'k-',lw = 2)
        self.ax.plot([x_lat_start5, x_lat_start5],[y_lat_start5, y_long_end],'k--',lw = 2)
        self.ax.plot([x_lat_start2, x_lat_start2],[y_lat_start5, y_long_end],'k-',lw = 2)
        
        self.ax.plot([x_lat_end1,x_lat_end1],[y_lat_start1,y_lat_end1],'k-',lw = 2)
        self.ax.plot([x_lat_start3, x_lat_start3],[y_lat_start1, y_lat_end1],'k--',lw = 2)
        self.ax.plot([x_lat_start4, x_lat_start4],[y_lat_start1, y_lat_end1],'k-',lw = 2)
        self.ax.plot([x_lat_start5, x_lat_start5],[y_lat_start1, y_lat_end1],'k--',lw = 2)
        self.ax.plot([x_lat_start2, x_lat_start2],[y_lat_start1, y_lat_end1],'k-',lw = 2)

        # 设置坐标轴的范围和比例
        self.ax.set_xlim(0, 614)
        self.ax.set_ylim(-50, 50)

    def update_patches(self,frame,current_light):
        current_patch_l = []
        current_patch_f = []
        current_patch_l1 = []
        current_patch_f1 = []
        
        for j in range(self.num_l):
            current_patch = patches.Rectangle((self.state_l[frame][2 * j],-3),self.length,self.width,edgecolor='black',facecolor='black')
            current_patch_l.append(current_patch)
        self.leader_patches.append(current_patch_l)

        for j in range(self.num_f):
            current_patch = patches.Rectangle((self.state_f[frame][2 * j],-3),self.length,self.width,edgecolor='blue',facecolor='blue')
            current_patch_f.append(current_patch)
        self.follower_patches.append(current_patch_f)
        
        for j in range(self.num_l1):
            current_patch = patches.Rectangle((self.state_l1[frame][2 * j],0.5),self.length,self.width,edgecolor='black',facecolor='black')
            current_patch_l1.append(current_patch)
        self.leader_patches1.append(current_patch_l1)
        
        for j in range(self.num_f1):
            current_patch = patches.Rectangle((self.state_f1[frame][2 * j],0.5),self.length,self.width,edgecolor='blue',facecolor='blue')
            current_patch_f1.append(current_patch)
        self.follower_patches1.append(current_patch_f1)
        
        x_tangle = 320
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
        for i in range(len(current_patch_l)):
            patch_l = current_patch_l[i]
            patch_f = current_patch_f[i]
            self.ax.add_patch((patch_l))
            self.ax.add_patch((patch_f))

        for i in range(len(current_patch_l1)):
            patch_l1 = current_patch_l1[i]
            patch_f1 = current_patch_f1[i]
            self.ax.add_patch((patch_l1))
            self.ax.add_patch((patch_f1))

        print(f"currnet_patch_l1 is :{len(current_patch_l1)}")
        print(f"currnet_patch_f1 is :{len(current_patch_f1)}")
        

        return self.leader_patches, self.follower_patches,self.leader_patches1,self.follower_patches1,self.traffic_patches
    
    def show(self):
        self.Road()
        self.leader_patches, self.follower_patches, self.leader_patches1,self.follower_patches1,self.traffic_patches = self.update_patches(0,self.light[0])
        for i in range(1,self.max_iter-1):
            current_light = self.light[i]
            self.update_patches(i,current_light)
        self.ax.set_aspect(1)

        for leader_patch,follower_patch,leader1_patch,follower1_patch,traffic_patch in zip(self.leader_patches,self.follower_patches,self.leader_patches1,self.follower_patches1,self.traffic_patches):
            plt.cla()
            self.Road()
            for leader,follower in zip(leader_patch,follower_patch):
                patch_l = leader
                patch_f = follower
                self.ax.add_patch(patch_l)
                self.ax.add_patch(patch_f)

            for leader_1,follower_1 in zip(leader1_patch,follower1_patch):
                patch_l1 = leader_1
                patch_f1 = follower_1
                self.ax.add_patch(patch_l1)
                self.ax.add_patch(patch_f1)
                
            patch_light = traffic_patch
            self.ax.add_patch(patch_light)
            self.ax.set_aspect(1)
            plt.pause(0.1)           
        plt.show()


if __name__ == '__main__':

###############################
#
#     leader是从左向右，leader1是从右向左
#
###############################
    leader_path = '/home/fanjx/project/FJX_project/Code_main/multi_vehicle/results/state_l.csv'
    IDM_path = '/home/fanjx/project/FJX_project/Code_main/multi_vehicle/results/IDM_state.csv'
    light_path = '/home/fanjx/project/FJX_project/Code_main/multi_vehicle/results/light.csv'
    leader1_path = '/home/fanjx/project/FJX_project/Code_main/multi_vehicle/single_result/state_l.csv'
    IDM1_path = '/home/fanjx/project/FJX_project/Code_main/multi_vehicle/single_result/IDM_state.csv'
    
    light = np.loadtxt(light_path, delimiter = ',',skiprows = 1)
    state_l = np.loadtxt(leader_path, delimiter = ',',skiprows = 2)
    state_IDM = np.loadtxt(IDM_path, delimiter = ',',skiprows = 2)
    state_l1 = np.loadtxt(leader1_path,delimiter = ',',skiprows = 2)
    state_IDM1 = np.loadtxt(IDM1_path,delimiter = ',',skiprows = 2)
    nmpc = NMPC(1)

    df1 = pd.read_csv(leader1_path)
    for i in range(nmpc.num_l1):
        j = 2 * i
        df1.iloc[:,j] = -df1.iloc[:,j] + 614
        df1.iloc[:,j + 1] = -df1.iloc[:,j + 1]
    df1.to_csv(leader1_path,index = False)
    df2 = pd.read_csv(IDM1_path)
    for i in range(nmpc.num_f1):
        j = 2 * i
        df2.iloc[:,j] = -df2.iloc[:,j] + 614
        df2.iloc[:,j + 1] = -df2.iloc[:,j + 1]
    df2.to_csv(IDM1_path,index = False) 
    vis = VIS(state_l, state_IDM, state_l1, state_IDM1, light, nmpc.num_l, nmpc.num_f, nmpc.num_l1, nmpc.num_f1)
    vis.show()




