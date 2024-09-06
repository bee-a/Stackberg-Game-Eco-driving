
import numpy as np
#from CAV import NMPC
#from example import Car, Road
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import sys
sys.path.append("..") 
from untils.draw import Arrow, draw_car
import pandas as pd




leader = pd.read_csv('/home/fanjx/project/FJX_project/Code_main/final_result/l_state.csv')
follower = pd.read_csv('/home/fanjx/project/FJX_project/Code_main/final_result/IDM_state.csv')
leader_o =pd.read_csv('/home/fanjx/project/FJX_project/Code_main/final_result/multi_leader/l1_state.csv')
follower_o = pd.read_csv('/home/fanjx/project/FJX_project/Code_main/final_result/multi_leader/IDM1_state.csv')
light = pd.read_csv('/home/fanjx/project/FJX_project/Code_main/final_result/multi_leader/light.csv')
#plt.figure()
fig1,ax1= plt.subplots(1,figsize=(12,6))
x = np.linspace(1,126,127)
x1 = light.iloc[:,0]
v1 = leader.iloc[:,1]
v2 = follower.iloc[:,1]
v3 = leader_o.iloc[:,1]
v4 = follower_o.iloc[:,1]
print(x1)
j=0
ax1.plot([0,36],[25,25],'k-',color='r',lw=2)
ax1.plot([37,96],[25,25],'k-',color='g',lw=2)
ax1.plot([97,102],[25,25],'k-',color='y',lw=2)
ax1.plot([103,126],[25,25],'k-',color='r',lw=2)                         
ax1.plot(x,v1,label='leader')
ax1.plot(x,v2,label='follower')  
ax1.set_xlim(0,126)
#ax1.set_title('sin(x)')# 显示图形
ax1.set_xlabel('Time')# 显示x轴标签
ax1.set_ylabel('Velocity')# 显示y轴标签
ax1.legend(loc='lower right')


fig2,ax2= plt.subplots(1,figsize=(12,6))
ax2.plot([0,36],[1250,1250],'k-',color='r',lw=2)
ax2.plot([37,96],[1250,1250],'k-',color='g',lw=2)
ax2.plot([97,102],[1250,1250],'k-',color='y',lw=2)
ax2.plot([103,126],[1250,1250],'k-',color='r',lw=2)
x1 = leader.iloc[:,0]
x2 = follower.iloc[:,0]
ax2.plot(x,x1,label='leader')
ax2.plot(x,x2,label='follower')
ax2.set_xlim(0,126)
ax2.set_xlabel('Time')# 显示x轴标签
ax2.set_ylabel('Position')# 显示y轴标签
ax2.legend(loc='lower right')

#plt.tight_layout()  # 自动调整子图的位置，避免重叠
fig3,ax3= plt.subplots(1,figsize=(12,6))
ax3.plot([0,36],[25,25],'k-',color='r',lw=2)
ax3.plot([37,96],[25,25],'k-',color='g',lw=2)
ax3.plot([97,102],[25,25],'k-',color='y',lw=2)
ax3.plot([103,126],[25,25],'k-',color='r',lw=2)
ax3.set_xlim(0,126)                         
ax3.plot(x,v3,label='leader')
ax3.plot(x,v4,label='follower')  
ax3.set_xlabel('Time')# 显示x轴标签
ax3.set_ylabel('Veloity')# 显示y轴标签
ax3.legend(loc='upper right',bbox_to_anchor=(1,0.9))


fig4,ax4= plt.subplots(1,figsize=(12,6))
ax4.plot([0,36],[725,725],'k-',color='r',lw=2)
ax4.plot([37,96],[725,725],'k-',color='g',lw=2)
ax4.plot([97,102],[725,725],'k-',color='y',lw=2)
ax4.plot([103,126],[725,725],'k-',color='r',lw=2)    
x1 = leader_o.iloc[:,0]
x2 = follower_o.iloc[:,0]     
ax4.set_xlim(0,126)     
ax4.plot(x,x1,label='leader')
ax4.plot(x,x2,label='follower')  
ax4.set_xlabel('Time')# 显示x轴标签
ax4.set_ylabel('Position')# 显示y轴标签
ax4.legend(loc='upper right',bbox_to_anchor=(1,0.9))
#plt.tight_layout()

fig1.savefig('/home/fanjx/project/FJX_project/Code_main/final_result/multi_leader/velocity.png',bbox_inches='tight', pad_inches=0.2)
fig2.savefig('/home/fanjx/project/FJX_project/Code_main/final_result/multi_leader/position.png',bbox_inches='tight', pad_inches=0.2)
fig3.savefig('/home/fanjx/project/FJX_project/Code_main/final_result/multi_leader/velocity_o.png',bbox_inches='tight', pad_inches=0.2)
fig4.savefig('/home/fanjx/project/FJX_project/Code_main/final_result/multi_leader/position_o.png',bbox_inches='tight', pad_inches=0.2)
plt.show()