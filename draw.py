import pandas as pd
from draw_single_veh import draw_single_veh

import matplotlib.pyplot as plt 
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']
img = plt.imread('/home/fanjx/project/FJX_project/Code_main/final_result/figure1/frame_1.png')
fig,ax = plt.subplots()
ax.imshow(img)
plt.tight_layout()
plt.savefig('/home/fanjx/project/FJX_project/Code_main/final_result/multi_leader/figure1.png',bbox_inches='tight', pad_inches=0)
