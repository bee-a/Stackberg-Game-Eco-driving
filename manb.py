import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
x=np.linspace(0,10,30)
y=np.arange(0,10,0.5)
df=pd.read_csv('1.csv')
a=df.loc[:,'predict_light']
print(a)
print('index{:.2f}\n,column{:.2f}\n'.format(df.shape[0],df.shape[1]))
abc=pd.DataFrame(columns=['a','b','c',],index=[0,1,2,3],size=(4,3))
print(abc)
'''
plt.plot(x,np.sin(x),'-o')
plt.scatter(x,np.sin(x))
plt.xlabel('sb')
plt.ylabel('sb')
plt.ylim(-1.5,1.5)
plt.grid()
#plt.figure(figsize=(12,8))
#plt.subplots(2,2)
plt.show()
'''
