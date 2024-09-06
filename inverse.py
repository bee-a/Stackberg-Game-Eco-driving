import numpy as np
'''
A=np.array([[2,0,0],[1,2,0],[7,5,2]])
U=np.array([[0,1,7],[0,0,5],[0,0,0]])
'''
A=np.array([[7,0,0],[1,2,0],[2,7,-11]])
U=np.array([[0,5,2],[0,0,1],[0,0,0]])
A_=np.linalg.inv(A)
print(A, "/n" ,A_)
B=-1*A_
print(B)
B_=B*U
print(B_)
c,d= np.linalg.eig(B)
print("特征值向量是",c)
C=np.max(c)
print("最大特征值是",C)