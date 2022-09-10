from threading import main_thread
from unittest import result
import torch
import pybullet as p
import numpy as np

def rotation_matrix(theta1, theta2, theta3, order='xyz'):
    """
    input
        theta1, theta2, theta3 = rotation angles in rotation order (degrees)
        oreder = rotation order of x,y,z　e.g. XZY rotation -- 'xzy'
    output
        3x3 rotation matrix (numpy array)
    """
    print(theta1, theta2, theta3)
    c1 = np.cos(theta1)
    s1 = np.sin(theta1)
    c2 = np.cos(theta2)
    s2 = np.sin(theta2)
    c3 = np.cos(theta3)
    s3 = np.sin(theta3)

    if order == 'xzx':
        matrix=np.array([[c2, -c3*s2, s2*s3],
                         [c1*s2, c1*c2*c3-s1*s3, -c3*s1-c1*c2*s3],
                         [s1*s2, c1*s3+c2*c3*s1, c1*c3-c2*s1*s3]])
    elif order=='xyx':
        matrix=np.array([[c2, s2*s3, c3*s2],
                         [s1*s2, c1*c3-c2*s1*s3, -c1*s3-c2*c3*s1],
                         [-c1*s2, c3*s1+c1*c2*s3, c1*c2*c3-s1*s3]])
    elif order=='yxy':
        matrix=np.array([[c1*c3-c2*s1*s3, s1*s2, c1*s3+c2*c3*s1],
                         [s2*s3, c2, -c3*s2],
                         [-c3*s1-c1*c2*s3, c1*s2, c1*c2*c3-s1*s3]])
    elif order=='yzy':
        matrix=np.array([[c1*c2*c3-s1*s3, -c1*s2, c3*s1+c1*c2*s3],
                         [c3*s2, c2, s2*s3],
                         [-c1*s3-c2*c3*s1, s1*s2, c1*c3-c2*s1*s3]])
    elif order=='zyz':
        matrix=np.array([[c1*c2*c3-s1*s3, -c3*s1-c1*c2*s3, c1*s2],
                         [c1*s3+c2*c3*s1, c1*c3-c2*s1*s3, s1*s2],
                         [-c3*s2, s2*s3, c2]])
    elif order=='zxz':
        matrix=np.array([[c1*c3-c2*s1*s3, -c1*s3-c2*c3*s1, s1*s2],
                         [c3*s1+c1*c2*s3, c1*c2*c3-s1*s3, -c1*s2],
                         [s2*s3, c3*s2, c2]])
    elif order=='xyz':
        matrix=np.array([[c2*c3, -c2*s3, s2],
                         [c1*s3+c3*s1*s2, c1*c3-s1*s2*s3, -c2*s1],
                         [s1*s3-c1*c3*s2, c3*s1+c1*s2*s3, c1*c2]])
    elif order=='xzy':
        matrix=np.array([[c2*c3, -s2, c2*s3],
                         [s1*s3+c1*c3*s2, c1*c2, c1*s2*s3-c3*s1],
                         [c3*s1*s2-c1*s3, c2*s1, c1*c3+s1*s2*s3]])
    elif order=='yxz':
        matrix=np.array([[c1*c3+s1*s2*s3, c3*s1*s2-c1*s3, c2*s1],
                         [c2*s3, c2*c3, -s2],
                         [c1*s2*s3-c3*s1, c1*c3*s2+s1*s3, c1*c2]])
    elif order=='yzx':
        matrix=np.array([[c1*c2, s1*s3-c1*c3*s2, c3*s1+c1*s2*s3],
                         [s2, c2*c3, -c2*s3],
                         [-c2*s1, c1*s3+c3*s1*s2, c1*c3-s1*s2*s3]])
    elif order=='zyx':
        print('hello')
        matrix=np.array([[c1*c2, c1*s2*s3-c3*s1, s1*s3+c1*c3*s2],
                         [c2*s1, c1*c3+s1*s2*s3, c3*s1*s2-c1*s3],
                         [-s2, c2*s3, c2*c3]])
    elif order=='zxy':
        matrix=np.array([[c1*c3-s1*s2*s3, -c2*s1, c1*s3+c3*s1*s2],
                         [c3*s1+c1*s2*s3, c1*c2, s1*s3-c1*c3*s2],
                         [-c2*s3, s2, c2*c3]])

    return matrix

def Rx(theta):
    return np.matrix([[ 1, 0, 0],
                      [ 0, np.cos(theta),-np.sin(theta)],
                      [ 0, np.sin(theta), np.cos(theta)]])
  
def Ry(theta):
    return np.matrix([[ np.cos(theta), 0, np.sin(theta)],
                      [ 0, 1, 0],
                      [-np.sin(theta), 0, np.cos(theta)]])
  
def Rz(theta):
    return np.matrix([[ np.cos(theta), -np.sin(theta), 0 ],
                      [ np.sin(theta), np.cos(theta) , 0 ],
                      [ 0, 0, 1 ]])

def euler_to_matrix(euler):
    return np.matmul(np.matmul(Rz(euler[2]), Ry(euler[1])), Rx(euler[0]))

def Transform(v, q):
    R = p.getMatrixFromQuaternion(q)  # 获取旋转矩阵
    # 拼凑齐次矩阵
    slices = [list(R[0: 3]), list(R[3: 6]), list(R[6: 9])]
    for i in range(3):
        slices[i].append(v[i])
    np_R = np.array(slices)
    bottom = np.array([0, 0, 0, 1])
    T = np.vstack((np_R, bottom))  # 在底部补齐
    # print(T)
    return T

# euler1 = [np.pi/2, 0, 0]
euler1 = [0, -1.3, 1]
t1 = [0,5,2]
q1 = p.getQuaternionFromEuler(euler1)
# matrix1 = euler_to_matrix(euler1)
# matrix2 = p.getMatrixFromQuaternion(q1)
transform1 = Transform(t1, q1)
# print(matrix1)
# print(matrix2)

euler2 = [0.3, -0.4, 0]
t2 = [1, 7, 3]
q2 = p.getQuaternionFromEuler(euler2)
transform2 = Transform(t2, q2)

T_test1 = transform1 @ transform2
T_test2 = transform2 @ transform1
print(T_test1)
print(T_test2)

p_transform = p.multiplyTransforms(t1, q1, t2, q2)
p_transform = Transform(*p_transform)
print(p_transform)


# euler2 = [0, 0, 0]
# t2 = [0,0,3]
# qut2 = p.getQuaternionFromEuler(euler2)
# matrix2 = euler_to_matrix(euler2)

# # euler1 x euler2
# result1 = p.multiplyTransforms(t2, qut2, t1, qut1)
# print(result1)

# transform1 = np.zeros((4,4))
# transform1[0:3, 0:3] = matrix1
# transform1[0:3, 3] = t1
# transform1[3,3] = 1

# transform2 = np.zeros((4,4))
# transform2[0:3, 0:3] = matrix2
# transform2[0:3, 3] = t2
# transform2[3,3] = 1
# result2 = np.matmul(transform1, transform2)
# print(result2)

