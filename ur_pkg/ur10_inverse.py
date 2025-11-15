#!/usr/bin/python
# -*- coding:utf-8 -*-
# 在使用C++代码遥操作采集数据过程中使用的是0.18为末端的长度
import numpy as np
from scipy.spatial.transform import Rotation as R, Rotation

# 官方示例参数
# d =[0, 0.1273,0,0,0.163941,0.1157,0.0922]
# a =[0,-0.612,-0.5723,0,0,0]
# 机械臂标定文件中的参数
L_robotiq = 0.06 # 0.06
L_gripper = 0.12 # 0.12
d = [0, 0.1279598369588654,0,0,0.1636727123724813,0.1156246017328714,0.09205355120243347 + 0.06 + 0.12]  # 示例值
a = [0,-0.6126666038983543,-0.5714813809834389,0,0,0]  # 示例值

alpha= [1.570313538404006, 0, 0, 1.568998582835579, -1.571395384373071, 0]
def quaternion_to_rotation_matrix(quat):
    x, y, z, w = quat
    rotation_matrix = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x**2 + z**2), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x**2 + y**2)]
    ])
    return rotation_matrix



def say_hello_world():
    print('Hello world!')
    
def quaternion2rot(quaternion):
    r = R.from_quat(quaternion)
    rot = r.as_matrix()  # 返回一个3x3的旋转矩阵
    return rot

# 返回 [x, y, z, w] 顺序的四元数
def rot2quaternion(rotation_matrix):
    r = R.from_matrix(rotation_matrix)
    quaternion = r.as_quat()
    return quaternion

def forward(theta_input):
    # DH参数
    L_robotiq = 0.06
    L_gripper = 0.12
    d = np.array([0, 0.1279598369588654, 0, 0, 0.1636727123724813,
                  0.1156246017328714, 0.09205355120243347 + L_robotiq + L_gripper])
    a = np.array([0, -0.6126666038983543, -0.5714813809834389, 0, 0, 0])
    alpha = np.array([1.570313538404006, 0, 0, 1.568998582835579,
                      -1.571395384373071, 0])

    # 构造每一节的齐次变换矩阵
    T = []
    for i in range(6):
        ct = np.cos(theta_input[i])
        st = np.sin(theta_input[i])
        ca = np.cos(alpha[i])
        sa = np.sin(alpha[i])

        T_i = np.array([
            [ct, -st * ca, st * sa, a[i] * ct],
            [st, ct * ca, -ct * sa, a[i] * st],
            [0, sa, ca, d[i + 1]],
            [0, 0, 0, 1]
        ])
        T.append(T_i)

    # 正向链式乘法
    T06 = T[0] @ T[1] @ T[2] @ T[3] @ T[4] @ T[5]

    # 提取末端位置与旋转矩阵
    pos_x_y_z = T06[:3, 3]
    rotation_matrix = T06[:3, :3]
    quat = R.from_matrix(rotation_matrix).as_quat()  # [x, y, z, w]

    return pos_x_y_z, quat  # 返回位置和四元数形式的姿态

def ur10_forward(theta_input):
    """
    正运动学计算
    :param theta_input: 关节角度列表 (单位：弧度)
    :param alpha: 链节的 alpha 参数列表 (单位：弧度)
    :param a: 链节的 a 参数列表
    :param d: 链节的 d 参数列表
    :return: 末端位置 pos_x_y_z 和旋转矩阵
    """
    T = []  # 存储每个关节的齐次变换矩阵
    for i in range(6):
        T_i = np.eye(4)  # 初始化 4x4 单位矩阵
        T_i[0, 0] = np.cos(theta_input[i])
        T_i[0, 1] = -np.sin(theta_input[i]) * np.cos(alpha[i])
        T_i[0, 2] = np.sin(theta_input[i]) * np.sin(alpha[i])
        T_i[0, 3] = a[i] * np.cos(theta_input[i])

        T_i[1, 0] = np.sin(theta_input[i])
        T_i[1, 1] = np.cos(theta_input[i]) * np.cos(alpha[i])
        T_i[1, 2] = -np.cos(theta_input[i]) * np.sin(alpha[i])
        T_i[1, 3] = a[i] * np.sin(theta_input[i])

        T_i[2, 0] = 0
        T_i[2, 1] = np.sin(alpha[i])
        T_i[2, 2] = np.cos(alpha[i])
        T_i[2, 3] = d[i + 1]  # 注意 d 的索引偏移

        T_i[3, 0:4] = [0, 0, 0, 1]
        T.append(T_i)

    # 累乘计算 T06
    T06 = T[0]
    for i in range(1, 6):
        T06 = np.dot(T06, T[i])
    # 提取末端位姿
    pos_x_y_z = T06[0:3, 3]  # 提取位置
    rotation_matrix = T06[0:3, 0:3]  # 提取旋转矩阵
    quaternion = rot2quaternion(rotation_matrix)  # 转为四元数 返回xyzw顺序
    # 输出调试信息
    # print(f"Position: {pos_x_y_z}")
    # print(f"Quaternion: {quaternion}")
    return pos_x_y_z, quaternion



def ur10_solution_filter_test(current, q_solutions):
    """
    过滤逆解，选择距离当前关节位置最近的解。
    :param current: 当前关节位置 (长度为6的列表或数组)
    :param q_solutions: 8x6 矩阵，表示八组逆解
    :return: 最优解 6x1 numpy数组
    """
    def calculate_distance(current, solution):
        """计算两组关节位置之间的欧几里得距离。"""
        return np.sqrt(np.sum((np.array(current) - np.array(solution)) ** 2))
    
    min_distance = 2 * np.pi * 6  # 初始化为一个很大的距离
    optimal_solution = np.zeros(6)  # 最优解的占位数组

    # 遍历8个逆解
    for i in range(8):
        is_solution_valid = True
        solution = np.zeros(6)
        
        for j in range(6):
            # 归一化到 [-2π, 2π]
            if q_solutions[i, j] < -2 * np.pi:
                q_solutions[i, j] += 2 * np.pi
            if q_solutions[i, j] > 2 * np.pi:
                q_solutions[i, j] -= 2 * np.pi
            solution[j] = q_solutions[i, j]

        # 如果解有效，计算距离
        if is_solution_valid:
            distance = calculate_distance(current, solution)
            # 如果当前解比之前的更优，则更新最优解
            if distance < min_distance:
                min_distance = distance
                optimal_solution = solution

    return optimal_solution



def ur10_inverse(target_pos, current, quat):
    """
    输入目标位置和四元数 返回8组逆解。
    """
    x, y, z = target_pos
    T06 = np.zeros((4, 4))  # 末端到基坐标系的变换矩阵
    q_solutions = np.zeros((8, 6))  # 用于存储8组逆解
    theta = np.zeros((9, 7))  # 用于存储计算中间值
    
    # 将四元数转换为旋转矩阵
    #rotation_matrix = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()
    rotation_matrix=quaternion2rot(quat)
    
    T06[:3, :3] = rotation_matrix
    T06[0, 3] = x
    T06[1, 3] = y
    T06[2, 3] = z
    T06[3, 3] = 1
    #print(T06)
    L_robotiq = 0.06 # 表示逆运动学中末端法兰到robotiq传感器的距离 还需要额外添加到夹爪的偏移量
    L_gripper = 0.12 # 0.10 # 表示机械臂末端夹爪的位置
    # 定义DH参数中的常量
    d = [0, 0.127960,0,0,0.1636727,0.1156246,0.0920536+L_robotiq+L_gripper]  # 示例值
    a = [0,-0.61267,-0.57148,0,0,0]  # 示例值
    #print(d[6])
    # 计算theta1的两个解
    A = d[6] * T06[1, 2] - T06[1, 3]
    B = d[6] * T06[0, 2] - T06[0, 3]
    C = d[4]
    R = A**2 + B**2
    #print(A,B,C)
    # theta1第一个解，赋值到1到4组
    theta[1][1] = np.arctan2(A, B) - np.arctan2(C, np.sqrt(A**2 + B**2 - C**2))
    #print(A**2 + B**2 - C**2)
    for i in range(1, 5):
        theta[i][1] = theta[1][1]
    # theta1第二个解，赋值到5到8组
    theta[5][1] = np.arctan2(A, B) - np.arctan2(C, -np.sqrt(A**2 + B**2 - C**2))
    for i in range(5, 9):
        theta[i][1] = theta[5][1]

    # theta5的四个解
    for i in range(1, 5):
        A = np.sin(theta[i][1]) * T06[0, 2] - np.cos(theta[i][1]) * T06[1, 2]
        theta[i][5] = np.arccos(A)
        theta[i + 4][5] = -np.arccos(A)

    # theta6的解
    for i in range(1, 9, 2):
        A = np.sin(theta[i][1]) * T06[0, 0] - np.cos(theta[i][1]) * T06[1, 0]
        B = np.sin(theta[i][1]) * T06[0, 1] - np.cos(theta[i][1]) * T06[1, 1]
        C = np.sin(theta[i][5])
        if np.abs(C) > 1e-5:
            theta[i][6] = np.arctan2(A, B) - np.arctan2(C, 0)
            theta[i + 1][6] = theta[i][6]
        else:
            theta[i][6] = theta[i + 1][6] = 0

    # theta3的解
    for i in range(1, 9, 2):
        C = np.cos(theta[i][1]) * T06[0, 0] + np.sin(theta[i][1]) * T06[1, 0]
        D = np.cos(theta[i][1]) * T06[0, 1] + np.sin(theta[i][1]) * T06[1, 1]
        E = np.cos(theta[i][1]) * T06[0, 2] + np.sin(theta[i][1]) * T06[1, 2]
        F = np.cos(theta[i][1]) * T06[0, 3] + np.sin(theta[i][1]) * T06[1, 3]
        G = np.cos(theta[i][6]) * T06[2, 1] + np.sin(theta[i][6]) * T06[2, 0]
        A = d[5] * (np.sin(theta[i][6]) * C + np.cos(theta[i][6]) * D) - d[6] * E + F
        B = T06[2, 3] - d[1] - T06[2, 2] * d[6] + d[5] * G
        if A**2 + B**2 <= (a[1] + a[2])**2:
            theta[i][3] = np.arccos((A**2 + B**2 - a[1]**2 - a[2]**2) / (2 * a[1] * a[2]))
            theta[i + 1][3] = -theta[i][3]
        else:
            theta[i][3] = theta[i + 1][3] = 0

    # theta2和theta4
    for i in range(1, 9):
        C = np.cos(theta[i][1]) * T06[0, 0] + np.sin(theta[i][1]) * T06[1, 0]
        D = np.cos(theta[i][1]) * T06[0, 1] + np.sin(theta[i][1]) * T06[1, 1]
        E = np.cos(theta[i][1]) * T06[0, 2] + np.sin(theta[i][1]) * T06[1, 2]
        F = np.cos(theta[i][1]) * T06[0, 3] + np.sin(theta[i][1]) * T06[1, 3]
        G = np.cos(theta[i][6]) * T06[2, 1] + np.sin(theta[i][6]) * T06[2, 0]
        A = d[5] * (np.sin(theta[i][6]) * C + np.cos(theta[i][6]) * D) - d[6] * E + F
        B = T06[2, 3] - d[1] - T06[2, 2] * d[6] + d[5] * G
        M = ((a[2] * np.cos(theta[i][3]) + a[1]) * B - a[2] * np.sin(theta[i][3]) * A) / (
            a[1]**2 + a[2]**2 + 2 * a[1] * a[2] * np.cos(theta[i][3])
        )
        N = (A + a[2] * np.sin(theta[i][3]) * M) / (a[2] * np.cos(theta[i][3]) + a[1])
        theta[i][2] = np.arctan2(M, N)
        theta[i][4] = np.arctan2((-np.sin(theta[i][6]) * C - np.cos(theta[i][6]) * D), G) - theta[i][2] - theta[i][3]
    #print(theta)
    # 将角度规范化到[-2π, 2π]并存储
    for i in range(1, 9):
        for j in range(1, 7):
            if theta[i][j] > 2*np.pi:
                theta[i][j]=theta[i][j] - 2*np.pi
            if theta[i][j] < -2*np.pi:
                theta[i][j]=theta[i][j] + 2*np.pi
            q_solutions[i - 1, j - 1] = theta[i][j]
    #print(q_solutions)
    # 过滤解并返回
    q_filter = ur10_solution_filter_test(current, q_solutions)
    return q_filter

def ur10_my_policy_inverse(T, current):
    """
    输入目标位置和四元数 返回8组逆解。
    """
    x, y, z = T[:3,3]
    T06 = np.zeros((4, 4))  # 末端到基坐标系的变换矩阵
    q_solutions = np.zeros((8, 6))  # 用于存储8组逆解
    theta = np.zeros((9, 7))  # 用于存储计算中间值
    
    # 将四元数转换为旋转矩阵
    # rotation_matrix = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()
    # rotation_matrix=quaternion2rot(quat)
    
    T06[:3, :3] = T[:3, :3]
    T06[0, 3] = x
    T06[1, 3] = y
    T06[2, 3] = z
    T06[3, 3] = 1
    #print(T06)
    L_robotiq = 0.06 # 表示逆运动学中末端法兰到robotiq传感器的距离 还需要额外添加到夹爪的偏移量
    L_gripper = 0.12 # 0.10 # 表示机械臂末端夹爪的位置 原来第一次demo是用的0.10
    # 定义DH参数中的常量
    d = [0, 0.127960,0,0,0.1636727,0.1156246,0.0920536+L_robotiq+L_gripper]  # 示例值
    a = [0,-0.61267,-0.57148,0,0,0]  # 示例值
    #print(d[6])
    # 计算theta1的两个解
    A = d[6] * T06[1, 2] - T06[1, 3]
    B = d[6] * T06[0, 2] - T06[0, 3]
    C = d[4]
    R = A**2 + B**2
    #print(A,B,C)
    # theta1第一个解，赋值到1到4组
    theta[1][1] = np.arctan2(A, B) - np.arctan2(C, np.sqrt(A**2 + B**2 - C**2))
    #print(A**2 + B**2 - C**2)
    for i in range(1, 5):
        theta[i][1] = theta[1][1]
    # theta1第二个解，赋值到5到8组
    theta[5][1] = np.arctan2(A, B) - np.arctan2(C, -np.sqrt(A**2 + B**2 - C**2))
    for i in range(5, 9):
        theta[i][1] = theta[5][1]

    # theta5的四个解
    for i in range(1, 5):
        A = np.sin(theta[i][1]) * T06[0, 2] - np.cos(theta[i][1]) * T06[1, 2]
        theta[i][5] = np.arccos(A)
        theta[i + 4][5] = -np.arccos(A)

    # theta6的解
    for i in range(1, 9, 2):
        A = np.sin(theta[i][1]) * T06[0, 0] - np.cos(theta[i][1]) * T06[1, 0]
        B = np.sin(theta[i][1]) * T06[0, 1] - np.cos(theta[i][1]) * T06[1, 1]
        C = np.sin(theta[i][5])
        if np.abs(C) > 1e-5:
            theta[i][6] = np.arctan2(A, B) - np.arctan2(C, 0)
            theta[i + 1][6] = theta[i][6]
        else:
            theta[i][6] = theta[i + 1][6] = 0

    # theta3的解
    for i in range(1, 9, 2):
        C = np.cos(theta[i][1]) * T06[0, 0] + np.sin(theta[i][1]) * T06[1, 0]
        D = np.cos(theta[i][1]) * T06[0, 1] + np.sin(theta[i][1]) * T06[1, 1]
        E = np.cos(theta[i][1]) * T06[0, 2] + np.sin(theta[i][1]) * T06[1, 2]
        F = np.cos(theta[i][1]) * T06[0, 3] + np.sin(theta[i][1]) * T06[1, 3]
        G = np.cos(theta[i][6]) * T06[2, 1] + np.sin(theta[i][6]) * T06[2, 0]
        A = d[5] * (np.sin(theta[i][6]) * C + np.cos(theta[i][6]) * D) - d[6] * E + F
        B = T06[2, 3] - d[1] - T06[2, 2] * d[6] + d[5] * G
        if A**2 + B**2 <= (a[1] + a[2])**2:
            theta[i][3] = np.arccos((A**2 + B**2 - a[1]**2 - a[2]**2) / (2 * a[1] * a[2]))
            theta[i + 1][3] = -theta[i][3]
        else:
            theta[i][3] = theta[i + 1][3] = 0

    # theta2和theta4
    for i in range(1, 9):
        C = np.cos(theta[i][1]) * T06[0, 0] + np.sin(theta[i][1]) * T06[1, 0]
        D = np.cos(theta[i][1]) * T06[0, 1] + np.sin(theta[i][1]) * T06[1, 1]
        E = np.cos(theta[i][1]) * T06[0, 2] + np.sin(theta[i][1]) * T06[1, 2]
        F = np.cos(theta[i][1]) * T06[0, 3] + np.sin(theta[i][1]) * T06[1, 3]
        G = np.cos(theta[i][6]) * T06[2, 1] + np.sin(theta[i][6]) * T06[2, 0]
        A = d[5] * (np.sin(theta[i][6]) * C + np.cos(theta[i][6]) * D) - d[6] * E + F
        B = T06[2, 3] - d[1] - T06[2, 2] * d[6] + d[5] * G
        M = ((a[2] * np.cos(theta[i][3]) + a[1]) * B - a[2] * np.sin(theta[i][3]) * A) / (
            a[1]**2 + a[2]**2 + 2 * a[1] * a[2] * np.cos(theta[i][3])
        )
        N = (A + a[2] * np.sin(theta[i][3]) * M) / (a[2] * np.cos(theta[i][3]) + a[1])
        theta[i][2] = np.arctan2(M, N)
        theta[i][4] = np.arctan2((-np.sin(theta[i][6]) * C - np.cos(theta[i][6]) * D), G) - theta[i][2] - theta[i][3]
    #print(theta)
    # 将角度规范化到[-2π, 2π]并存储
    for i in range(1, 9):
        for j in range(1, 7):
            if theta[i][j] > 2*np.pi:
                theta[i][j]=theta[i][j] - 2*np.pi
            if theta[i][j] < -2*np.pi:
                theta[i][j]=theta[i][j] + 2*np.pi
            q_solutions[i - 1, j - 1] = theta[i][j]
    #print(q_solutions)
    # 过滤解并返回
    q_filter = ur10_solution_filter_test(current, q_solutions)
    return q_filter


def ur10_pybullet_my_policy_inverse(target_pos, quat, current):
    """
    输入目标位置和四元数 返回8组逆解。
    """
    x, y, z = target_pos
    T06 = np.zeros((4, 4))  # 末端到基坐标系的变换矩阵
    q_solutions = np.zeros((8, 6))  # 用于存储8组逆解
    theta = np.zeros((9, 7))  # 用于存储计算中间值

    # 将四元数转换为旋转矩阵
    rotation = Rotation.from_quat(quat)
    rotation_matrix = rotation.as_matrix()

    T06[:3, :3] = rotation_matrix
    T06[0, 3] = x
    T06[1, 3] = y
    T06[2, 3] = z
    T06[3, 3] = 1
    # print(T06)
    L_robotiq = 0.06  # 表示逆运动学中末端法兰到robotiq传感器的距离 还需要额外添加到夹爪的偏移量
    L_gripper = 0.12  # 0.10 # 表示机械臂末端夹爪的位置 原来第一次demo是用的0.10
    # 定义DH参数中的常量
    d = [0, 0.1279598369588654,0,0,0.1636727123724813,0.1156246017328714,0.09205355120243347 + 0.06 + 0.12]  # 示例值
    a = [0,-0.6126666038983543,-0.5714813809834389,0,0,0]  # 示例值
    # print(d[6])
    # 计算theta1的两个解
    A = d[6] * T06[1, 2] - T06[1, 3]
    B = d[6] * T06[0, 2] - T06[0, 3]
    C = d[4]
    R = A ** 2 + B ** 2
    # print(A,B,C)
    # theta1第一个解，赋值到1到4组
    theta[1][1] = np.arctan2(A, B) - np.arctan2(C, np.sqrt(A ** 2 + B ** 2 - C ** 2))
    # print(A**2 + B**2 - C**2)
    for i in range(1, 5):
        theta[i][1] = theta[1][1]
    # theta1第二个解，赋值到5到8组
    theta[5][1] = np.arctan2(A, B) - np.arctan2(C, -np.sqrt(A ** 2 + B ** 2 - C ** 2))
    for i in range(5, 9):
        theta[i][1] = theta[5][1]

    # theta5的四个解
    for i in range(1, 5):
        A = np.sin(theta[i][1]) * T06[0, 2] - np.cos(theta[i][1]) * T06[1, 2]
        theta[i][5] = np.arccos(A)
        theta[i + 4][5] = -np.arccos(A)

    # theta6的解
    for i in range(1, 9, 2):
        A = np.sin(theta[i][1]) * T06[0, 0] - np.cos(theta[i][1]) * T06[1, 0]
        B = np.sin(theta[i][1]) * T06[0, 1] - np.cos(theta[i][1]) * T06[1, 1]
        C = np.sin(theta[i][5])
        if np.abs(C) > 1e-5:
            theta[i][6] = np.arctan2(A, B) - np.arctan2(C, 0)
            theta[i + 1][6] = theta[i][6]
        else:
            theta[i][6] = theta[i + 1][6] = 0

    # theta3的解
    for i in range(1, 9, 2):
        C = np.cos(theta[i][1]) * T06[0, 0] + np.sin(theta[i][1]) * T06[1, 0]
        D = np.cos(theta[i][1]) * T06[0, 1] + np.sin(theta[i][1]) * T06[1, 1]
        E = np.cos(theta[i][1]) * T06[0, 2] + np.sin(theta[i][1]) * T06[1, 2]
        F = np.cos(theta[i][1]) * T06[0, 3] + np.sin(theta[i][1]) * T06[1, 3]
        G = np.cos(theta[i][6]) * T06[2, 1] + np.sin(theta[i][6]) * T06[2, 0]
        A = d[5] * (np.sin(theta[i][6]) * C + np.cos(theta[i][6]) * D) - d[6] * E + F
        B = T06[2, 3] - d[1] - T06[2, 2] * d[6] + d[5] * G
        if A ** 2 + B ** 2 <= (a[1] + a[2]) ** 2:
            theta[i][3] = np.arccos((A ** 2 + B ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
            theta[i + 1][3] = -theta[i][3]
        else:
            theta[i][3] = theta[i + 1][3] = 0

    # theta2和theta4
    for i in range(1, 9):
        C = np.cos(theta[i][1]) * T06[0, 0] + np.sin(theta[i][1]) * T06[1, 0]
        D = np.cos(theta[i][1]) * T06[0, 1] + np.sin(theta[i][1]) * T06[1, 1]
        E = np.cos(theta[i][1]) * T06[0, 2] + np.sin(theta[i][1]) * T06[1, 2]
        F = np.cos(theta[i][1]) * T06[0, 3] + np.sin(theta[i][1]) * T06[1, 3]
        G = np.cos(theta[i][6]) * T06[2, 1] + np.sin(theta[i][6]) * T06[2, 0]
        A = d[5] * (np.sin(theta[i][6]) * C + np.cos(theta[i][6]) * D) - d[6] * E + F
        B = T06[2, 3] - d[1] - T06[2, 2] * d[6] + d[5] * G
        M = ((a[2] * np.cos(theta[i][3]) + a[1]) * B - a[2] * np.sin(theta[i][3]) * A) / (
                a[1] ** 2 + a[2] ** 2 + 2 * a[1] * a[2] * np.cos(theta[i][3])
        )
        N = (A + a[2] * np.sin(theta[i][3]) * M) / (a[2] * np.cos(theta[i][3]) + a[1])
        theta[i][2] = np.arctan2(M, N)
        theta[i][4] = np.arctan2((-np.sin(theta[i][6]) * C - np.cos(theta[i][6]) * D), G) - theta[i][2] - theta[i][3]
    # print(theta)

    # 将角度规范化到[-2π, 2π]并存储
    for i in range(1, 9):
        for j in range(1, 7):
            if theta[i][j] > 2 * np.pi:
                theta[i][j] = theta[i][j] - 2 * np.pi
            if theta[i][j] < -2 * np.pi:
                theta[i][j] = theta[i][j] + 2 * np.pi
            q_solutions[i - 1, j - 1] = theta[i][j]

    # print(q_solutions)
    # 过滤解并返回
    # q_filter = ur10_solution_filter_update_v2(current, q_solutions)
    # print("八组解",d)
    q_filter = ur10_solution_filter_test(current, q_solutions)

    return q_filter






# 数值解测试 测试通过 但没有关节角限制
def dh_transform(a, alpha, d, theta):
    """标准 DH 变换矩阵"""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,      sa,     ca,      d],
        [0,       0,      0,      1]
    ])

def compute_ur_jacobian_calibrated(joint_angles):
    """
    使用实测标定参数，计算 UR 机械臂末端的雅可比矩阵（6×6）

    参数：
        joint_angles (list or np.ndarray): 当前 6 个关节角

    返回：
        J: 6x6 几何雅可比矩阵
    """
    assert len(joint_angles) == 6, "需要 6 个关节角"

    # === 使用你给出的 DH 参数（已标定） ===
    L_robotiq = 0.06
    L_gripper = 0.12
    d = [0, 0.127960, 0, 0, 0.1636727, 0.1156246, 0.0920536 + L_robotiq + L_gripper]
    a = [0, -0.61267, -0.57148, 0, 0, 0]
    alpha = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]

    T = np.eye(4)
    zs = [np.array([0, 0, 1])]
    os = [np.array([0, 0, 0])]

    for i in range(6):
        A_i = dh_transform(a[i], alpha[i], d[i+1], joint_angles[i])
        T = T @ A_i
        zs.append(T[:3, 2])        # 当前 z_i 轴在 base 坐标系下
        os.append(T[:3, 3])        # 当前原点位置

    # 再加末端 Tool0 的偏移：d[6]
    T_tool = np.eye(4)
    T_tool[2, 3] = d[6]
    T = T @ T_tool
    o_n = T[:3, 3]  # 末端工具位置

    # === 几何雅可比 ===
    J = np.zeros((6, 6))
    for i in range(6):
        Jp = np.cross(zs[i], o_n - os[i])
        Jo = zs[i]
        J[:, i] = np.concatenate([Jp, Jo])

    return J

def normalize_angle(theta):
    """将角度归一化到 [-π, π]"""
    return (theta + np.pi) % (2 * np.pi) - np.pi

def numerical_ik(fk_func, jacobian_func, target_pos, target_quat,
                 init_joint_angles, max_iters=100, eps=1e-4, alpha=0.1):
    """
    数值法求解逆运动学，带 [-π, π] 归一化，避免多圈跳跃。

    参数：
        fk_func: 前向运动学函数
        jacobian_func: 雅可比矩阵函数
        target_pos: 目标位置 [x, y, z]
        target_quat: 目标姿态 [x, y, z, w]
        init_joint_angles: 初始猜解（6维）
        max_iters: 最大迭代次数
        eps: 收敛判据（误差阈值）
        alpha: 步长（比例系数）

    返回：
        joint_angles (np.ndarray): 收敛后的解，长度6
    """
    joint_angles = np.array(init_joint_angles, dtype=np.float64)

    for _ in range(max_iters):
        current_pos, current_quat = fk_func(joint_angles)

        # 位姿误差（位置 + 姿态）
        pos_err = target_pos - current_pos
        rot_err = (R.from_quat(target_quat) * R.from_quat(current_quat).inv()).as_rotvec()
        err = np.concatenate([pos_err, rot_err])

        if np.linalg.norm(err) < eps:
            break

        # 雅可比更新
        J = jacobian_func(joint_angles)
        dtheta = alpha * np.linalg.pinv(J) @ err
        joint_angles += dtheta

        # 归一化到 [-π, π]
        joint_angles = np.array([normalize_angle(θ) for θ in joint_angles])

    return joint_angles





