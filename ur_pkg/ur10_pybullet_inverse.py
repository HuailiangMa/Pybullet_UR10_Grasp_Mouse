#!/usr/bin/python
# -*- coding:utf-8 -*-
# 在使用C++代码遥操作采集数据过程中使用的是0.18为末端的长度
import numpy as np
from scipy.spatial.transform import Rotation as R, Rotation


def ur10_solution_filter(current, q_solutions):
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


def ur10_my_policy_inverse(T, current):
    """
    输入目标位置和四元数 返回8组逆解。
    """
    x, y, z = T[:3, 3]
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
    # print(T06)
    L_robotiq = 0.06  # 表示逆运动学中末端法兰到robotiq传感器的距离 还需要额外添加到夹爪的偏移量
    L_gripper = 0.12  # 0.10 # 表示机械臂末端夹爪的位置
    # 定义DH参数中的常量
    d = [0, 0.127960, 0, 0, 0.1636727, 0.1156246, 0.0920536 + L_robotiq + L_gripper]  # 示例值
    a = [0, -0.61267, -0.57148, 0, 0, 0]  # 示例值
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
    q_filter = ur10_solution_filter(current, q_solutions)
    return q_filter

def ur10_forward(theta_input):
    # 此函数用于求解机械臂的正向运动学
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

def ur10_forward_T(theta_input):
    # 此函数用于求解机械臂的正向运动学
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

    return T06  # 返回位置和四元数形式的姿态

def ur10_solution_filter_test(current, q_solutions):
    """
    对8组逆解进行筛选，剔除非法解，归一化角度差进行距离比较，选取与当前角度最接近的解。
    """

    def normalize_angle(theta):
        """将角度归一化到 (-π, π] 区间"""
        return (theta + np.pi) % (2 * np.pi) - np.pi

    def normalize_angles(q):
        return np.array([normalize_angle(a) for a in q])

    def compute_normalized_distance(q1, q2):
        """
        计算归一化角度差的L2距离，并加权 base 和 wrist3 的跳变惩罚
        """
        diff = normalize_angles(q1 - q2)
        norm_diff = diff / np.pi
        # === 特定关节加权 ===
        base_weight = 5.0     # 基座关节跳变惩罚（q[0]）
        wrist3_weight = 5.0   # 末端关节跳变惩罚（q[5]）
        norm_diff[0] *= base_weight
        norm_diff[5] *= wrist3_weight

        return np.linalg.norm(norm_diff)

    # Step 1: 过滤非法解
    valid_qs = []
    for q in q_solutions:
        if np.any(np.isnan(q)) or np.any(np.isinf(q)):
            continue
        if np.linalg.norm(q) < 1e-3:  # 全零解
            continue
        valid_qs.append(q)

    if len(valid_qs) == 0:
        print("[ERROR] 所有逆解非法（全0/NAN/inf），解算失败")
        return normalize_angles(current)

    # Step 2: 对 current 和解归一化
    current = normalize_angles(current)
    normalized_valid_qs = [normalize_angles(q) for q in valid_qs]

    # Step 3: 加入跳变惩罚的归一化距离度量
    min_score = float('inf')
    best_q = None
    for q in normalized_valid_qs:
        diff = normalize_angles(q - current)
        # jump_penalty = np.sum((np.abs(diff) > np.pi * 0.9).astype(float)) * 10.0
        jump_penalty = 0
        normalized_distance = compute_normalized_distance(q, current)
        # # === 肘部惩罚：若 q[2] < 0，说明肘向下，添加惩罚 ===
        # elbow_penalty = 5.0 if q[2] <-2 else 0.0  # 可调参数
        score = normalized_distance + jump_penalty  #  + elbow_penalty


        if score < min_score:
            min_score = score
            best_q = q

    return best_q


def ur10_inverse(target_pos, quat, current):
    # 此函数输入目标位置和四元数和当前关节角 输出一组最近的关节角解
    L_robotiq = 0.08
    L_gripper = 0.10
    d = np.array([0, 0.1279598369588654, 0, 0, 0.1636727123724813,
                  0.1156246017328714, 0.09205355120243347 + L_robotiq + L_gripper])
    a = np.array([0, -0.6126666038983543, -0.5714813809834389, 0, 0, 0])

    x, y, z = target_pos
    T06 = np.eye(4)
    R_mat = R.from_quat(quat).as_matrix()
    T06[:3, :3] = R_mat
    T06[0, 3] = x
    T06[1, 3] = y
    T06[2, 3] = z

    theta = np.zeros((9, 7))
    q_solutions = np.zeros((8, 6))

    A = d[6] * T06[1, 2] - T06[1, 3]
    B = d[6] * T06[0, 2] - T06[0, 3]
    C = d[4]
    theta[1][1] = np.arctan2(A, B) - np.arctan2(C, np.sqrt(A**2 + B**2 - C**2))
    for i in [2, 3, 4]:
        theta[i][1] = theta[1][1]
    theta[5][1] = np.arctan2(A, B) - np.arctan2(C, -np.sqrt(A**2 + B**2 - C**2))
    for i in [6, 7, 8]:
        theta[i][1] = theta[5][1]

    for i in range(1, 5):
        A = np.sin(theta[i][1]) * T06[0, 2] - np.cos(theta[i][1]) * T06[1, 2]
        theta[i][5] = np.arccos(A)
    for i in range(5, 9):
        A = np.sin(theta[i][1]) * T06[0, 2] - np.cos(theta[i][1]) * T06[1, 2]
        theta[i][5] = -np.arccos(A)

    for i in range(1, 9, 2):
        A = np.sin(theta[i][1]) * T06[0, 0] - np.cos(theta[i][1]) * T06[1, 0]
        B = np.sin(theta[i][1]) * T06[0, 1] - np.cos(theta[i][1]) * T06[1, 1]
        C = np.sin(theta[i][5])
        if abs(C) > 1e-5:
            theta[i][6] = np.arctan2(A, B) - np.arctan2(C, 0.0)
            theta[i + 1][6] = theta[i][6]
        else:
            theta[i][6] = theta[i + 1][6] = 0
            print('解算失误')

    for i in range(1, 9, 2):
        C = np.cos(theta[i][1]) * T06[0, 0] + np.sin(theta[i][1]) * T06[1, 0]
        D = np.cos(theta[i][1]) * T06[0, 1] + np.sin(theta[i][1]) * T06[1, 1]
        E = np.cos(theta[i][1]) * T06[0, 2] + np.sin(theta[i][1]) * T06[1, 2]
        F = np.cos(theta[i][1]) * T06[0, 3] + np.sin(theta[i][1]) * T06[1, 3]
        G = np.cos(theta[i][6]) * T06[2, 1] + np.sin(theta[i][6]) * T06[2, 0]
        A = d[5] * (np.sin(theta[i][6]) * C + np.cos(theta[i][6]) * D) - d[6] * E + F
        B = T06[2, 3] - d[1] - T06[2, 2] * d[6] + d[5] * G
        l2, l3 = a[1], a[2]
        if A**2 + B**2 <= (l2 + l3)**2:
            theta[i][3] = np.arccos((A**2 + B**2 - l2**2 - l3**2) / (2 * l2 * l3))
            theta[i + 1][3] = -theta[i][3]
        else:
            theta[i][3] = theta[i + 1][3] = 0

    for i in range(1, 9):
        C = np.cos(theta[i][1]) * T06[0, 0] + np.sin(theta[i][1]) * T06[1, 0]
        D = np.cos(theta[i][1]) * T06[0, 1] + np.sin(theta[i][1]) * T06[1, 1]
        E = np.cos(theta[i][1]) * T06[0, 2] + np.sin(theta[i][1]) * T06[1, 2]
        F = np.cos(theta[i][1]) * T06[0, 3] + np.sin(theta[i][1]) * T06[1, 3]
        G = np.cos(theta[i][6]) * T06[2, 1] + np.sin(theta[i][6]) * T06[2, 0]
        A = d[5] * (np.sin(theta[i][6]) * C + np.cos(theta[i][6]) * D) - d[6] * E + F
        B = T06[2, 3] - d[1] - T06[2, 2] * d[6] + d[5] * G
        l2, l3 = a[1], a[2]
        M = ((l3 * np.cos(theta[i][3]) + l2) * B - l3 * np.sin(theta[i][3]) * A) / \
            (l2**2 + l3**2 + 2 * l2 * l3 * np.cos(theta[i][3]))
        N = (A + l3 * np.sin(theta[i][3]) * M) / (l3 * np.cos(theta[i][3]) + l2)
        theta[i][2] = np.arctan2(M, N)
        theta[i][4] = np.arctan2((-np.sin(theta[i][6]) * C - np.cos(theta[i][6]) * D), G) - theta[i][2] - theta[i][3]

    for i in range(1, 9):
        for j in range(1, 7):
            if theta[i][j] > 2 * np.pi:
                theta[i][j] -= 2 * np.pi
            elif theta[i][j] < -2 * np.pi:
                theta[i][j] += 2 * np.pi
            q_solutions[i - 1][j - 1] = theta[i][j]

    return ur10_solution_filter_test(current, q_solutions),q_solutions

def ur10_inverse_matrix_test(T_tool, current):
    # 此函数输入旋转矩阵 输出机械臂的一组最近关节角逆解
    L_robotiq = 0.06
    L_gripper = 0.12
    d = np.array([0, 0.1279598369588654, 0, 0, 0.1636727123724813,
                  0.1156246017328714, 0.09205355120243347 + L_robotiq + L_gripper])
    a = np.array([0, -0.6126666038983543, -0.5714813809834389, 0, 0, 0])

    # x, y, z = target_pos
    # R_mat = R.from_quat(quat).as_matrix()

    T06 = np.eye(4)
    T06[:3, :3] = T_tool[:3, :3]
    T06[0, 3] = T_tool[0, 3]
    T06[1, 3] = T_tool[1, 3]
    T06[2, 3] = T_tool[2, 3]

    theta = np.zeros((9, 7))
    q_solutions = np.zeros((8, 6))

    A = d[6] * T06[1, 2] - T06[1, 3]
    B = d[6] * T06[0, 2] - T06[0, 3]
    C = d[4]
    temp = A ** 2 + B ** 2 - C ** 2
    delta = max(temp, 1e-8)
    # if temp < 0:
    #     print("[WARN] sqrt 负数，跳过该解：", A, B, C)
    #     return current  # 或 continue，或填默认解

    theta[1][1] = np.arctan2(A, B) - np.arctan2(C, np.sqrt(delta))
    for i in [2, 3, 4]:
        theta[i][1] = theta[1][1]
    theta[5][1] = np.arctan2(A, B) - np.arctan2(C, -np.sqrt(delta))
    for i in [6, 7, 8]:
        theta[i][1] = theta[5][1]

    for i in range(1, 5):
        A = np.sin(theta[i][1]) * T06[0, 2] - np.cos(theta[i][1]) * T06[1, 2]
        A = np.clip(A, -1.0, 1.0)
        theta[i][5] = np.arccos(A)
    for i in range(5, 9):
        A = np.sin(theta[i][1]) * T06[0, 2] - np.cos(theta[i][1]) * T06[1, 2]
        A = np.clip(A, -1.0, 1.0)
        theta[i][5] = -np.arccos(A)

    for i in range(1, 9, 2):
        A = np.sin(theta[i][1]) * T06[0, 0] - np.cos(theta[i][1]) * T06[1, 0]
        B = np.sin(theta[i][1]) * T06[0, 1] - np.cos(theta[i][1]) * T06[1, 1]
        C = np.sin(theta[i][5])
        if abs(C) > 1e-5:
            theta[i][6] = np.arctan2(A, B) - np.arctan2(C, 0.0)
            theta[i + 1][6] = theta[i][6]
        else:
            theta[i][6] = theta[i + 1][6] = 0
            print('theta[i][6] 解算失误')

    for i in range(1, 9, 2):
        C = np.cos(theta[i][1]) * T06[0, 0] + np.sin(theta[i][1]) * T06[1, 0]
        D = np.cos(theta[i][1]) * T06[0, 1] + np.sin(theta[i][1]) * T06[1, 1]
        E = np.cos(theta[i][1]) * T06[0, 2] + np.sin(theta[i][1]) * T06[1, 2]
        F = np.cos(theta[i][1]) * T06[0, 3] + np.sin(theta[i][1]) * T06[1, 3]
        G = np.cos(theta[i][6]) * T06[2, 1] + np.sin(theta[i][6]) * T06[2, 0]
        A = d[5] * (np.sin(theta[i][6]) * C + np.cos(theta[i][6]) * D) - d[6] * E + F
        B = T06[2, 3] - d[1] - T06[2, 2] * d[6] + d[5] * G
        l2, l3 = a[1], a[2]
        if A**2 + B**2 <= (l2 + l3)**2:
            cos_theta3 = (A ** 2 + B ** 2 - l2 ** 2 - l3 ** 2) / (2 * l2 * l3)
            cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
            theta[i][3] = np.arccos(cos_theta3)
            theta[i + 1][3] = -theta[i][3]
        else:
            theta[i][3] = theta[i + 1][3] = 0

    for i in range(1, 9):
        C = np.cos(theta[i][1]) * T06[0, 0] + np.sin(theta[i][1]) * T06[1, 0]
        D = np.cos(theta[i][1]) * T06[0, 1] + np.sin(theta[i][1]) * T06[1, 1]
        E = np.cos(theta[i][1]) * T06[0, 2] + np.sin(theta[i][1]) * T06[1, 2]
        F = np.cos(theta[i][1]) * T06[0, 3] + np.sin(theta[i][1]) * T06[1, 3]
        G = np.cos(theta[i][6]) * T06[2, 1] + np.sin(theta[i][6]) * T06[2, 0]
        A = d[5] * (np.sin(theta[i][6]) * C + np.cos(theta[i][6]) * D) - d[6] * E + F
        B = T06[2, 3] - d[1] - T06[2, 2] * d[6] + d[5] * G
        l2, l3 = a[1], a[2]
        M = ((l3 * np.cos(theta[i][3]) + l2) * B - l3 * np.sin(theta[i][3]) * A) / \
            (l2**2 + l3**2 + 2 * l2 * l3 * np.cos(theta[i][3]))
        N = (A + l3 * np.sin(theta[i][3]) * M) / (l3 * np.cos(theta[i][3]) + l2)
        theta[i][2] = np.arctan2(M, N)
        theta[i][4] = np.arctan2((-np.sin(theta[i][6]) * C - np.cos(theta[i][6]) * D), G) - theta[i][2] - theta[i][3]
    for i in range(1, 9):
        for j in range(1, 7):
            if theta[i][j] > 2 * np.pi:
                theta[i][j] -= 2 * np.pi
            elif theta[i][j] < -2 * np.pi:
                theta[i][j] += 2 * np.pi
            q_solutions[i - 1][j - 1] = theta[i][j]
    # for i in range(1, 9):
    #     for j in range(1, 7):
    #         angle = theta[i][j]
    #         # wrap angle to [-pi, pi]
    #         if angle > np.pi:
    #             angle -= 2 * np.pi
    #         elif angle < -np.pi:
    #             angle += 2 * np.pi
    #         q_solutions[i - 1][j - 1] = angle
    # print(q_solutions)
    return ur10_solution_filter_test(current, q_solutions),q_solutions


def ur10_inverse_matrix(T_tool, current):
    # 此函数输入旋转矩阵 输出机械臂的一组最近关节角逆解
    L_robotiq = 0.06
    L_gripper = 0.12
    d = np.array([0, 0.1279598369588654, 0, 0, 0.1636727123724813,
                  0.1156246017328714, 0.09205355120243347 + L_robotiq + L_gripper])
    a = np.array([0, -0.6126666038983543, -0.5714813809834389, 0, 0, 0])

    # x, y, z = target_pos
    # R_mat = R.from_quat(quat).as_matrix()

    T06 = np.eye(4)
    T06[:3, :3] = T_tool[:3, :3]
    T06[0, 3] = T_tool[0, 3]
    T06[1, 3] = T_tool[1, 3]
    T06[2, 3] = T_tool[2, 3]

    theta = np.zeros((9, 7))
    q_solutions = np.zeros((8, 6))

    A = d[6] * T06[1, 2] - T06[1, 3]
    B = d[6] * T06[0, 2] - T06[0, 3]
    C = d[4]
    temp = A ** 2 + B ** 2 - C ** 2
    delta = max(temp, 1e-8)
    # if temp < 0:
    #     print("[WARN] sqrt 负数，跳过该解：", A, B, C)
    #     return current  # 或 continue，或填默认解

    theta[1][1] = np.arctan2(A, B) - np.arctan2(C, np.sqrt(delta))
    for i in [2, 3, 4]:
        theta[i][1] = theta[1][1]
    theta[5][1] = np.arctan2(A, B) - np.arctan2(C, -np.sqrt(delta))
    for i in [6, 7, 8]:
        theta[i][1] = theta[5][1]

    for i in range(1, 5):
        A = np.sin(theta[i][1]) * T06[0, 2] - np.cos(theta[i][1]) * T06[1, 2]
        A = np.clip(A, -1.0, 1.0)
        theta[i][5] = np.arccos(A)
    for i in range(5, 9):
        A = np.sin(theta[i][1]) * T06[0, 2] - np.cos(theta[i][1]) * T06[1, 2]
        A = np.clip(A, -1.0, 1.0)
        theta[i][5] = -np.arccos(A)

    for i in range(1, 9, 2):
        A = np.sin(theta[i][1]) * T06[0, 0] - np.cos(theta[i][1]) * T06[1, 0]
        B = np.sin(theta[i][1]) * T06[0, 1] - np.cos(theta[i][1]) * T06[1, 1]
        C = np.sin(theta[i][5])
        if abs(C) > 1e-5:
            theta[i][6] = np.arctan2(A, B) - np.arctan2(C, 0.0)
            theta[i + 1][6] = theta[i][6]
        else:
            theta[i][6] = theta[i + 1][6] = 0
            print('theta[i][6] 解算失误')

    for i in range(1, 9, 2):
        C = np.cos(theta[i][1]) * T06[0, 0] + np.sin(theta[i][1]) * T06[1, 0]
        D = np.cos(theta[i][1]) * T06[0, 1] + np.sin(theta[i][1]) * T06[1, 1]
        E = np.cos(theta[i][1]) * T06[0, 2] + np.sin(theta[i][1]) * T06[1, 2]
        F = np.cos(theta[i][1]) * T06[0, 3] + np.sin(theta[i][1]) * T06[1, 3]
        G = np.cos(theta[i][6]) * T06[2, 1] + np.sin(theta[i][6]) * T06[2, 0]
        A = d[5] * (np.sin(theta[i][6]) * C + np.cos(theta[i][6]) * D) - d[6] * E + F
        B = T06[2, 3] - d[1] - T06[2, 2] * d[6] + d[5] * G
        l2, l3 = a[1], a[2]
        if A**2 + B**2 <= (l2 + l3)**2:
            cos_theta3 = (A ** 2 + B ** 2 - l2 ** 2 - l3 ** 2) / (2 * l2 * l3)
            cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
            theta[i][3] = np.arccos(cos_theta3)
            theta[i + 1][3] = -theta[i][3]
        else:
            theta[i][3] = theta[i + 1][3] = 0

    for i in range(1, 9):
        C = np.cos(theta[i][1]) * T06[0, 0] + np.sin(theta[i][1]) * T06[1, 0]
        D = np.cos(theta[i][1]) * T06[0, 1] + np.sin(theta[i][1]) * T06[1, 1]
        E = np.cos(theta[i][1]) * T06[0, 2] + np.sin(theta[i][1]) * T06[1, 2]
        F = np.cos(theta[i][1]) * T06[0, 3] + np.sin(theta[i][1]) * T06[1, 3]
        G = np.cos(theta[i][6]) * T06[2, 1] + np.sin(theta[i][6]) * T06[2, 0]
        A = d[5] * (np.sin(theta[i][6]) * C + np.cos(theta[i][6]) * D) - d[6] * E + F
        B = T06[2, 3] - d[1] - T06[2, 2] * d[6] + d[5] * G
        l2, l3 = a[1], a[2]
        M = ((l3 * np.cos(theta[i][3]) + l2) * B - l3 * np.sin(theta[i][3]) * A) / \
            (l2**2 + l3**2 + 2 * l2 * l3 * np.cos(theta[i][3]))
        N = (A + l3 * np.sin(theta[i][3]) * M) / (l3 * np.cos(theta[i][3]) + l2)
        theta[i][2] = np.arctan2(M, N)
        theta[i][4] = np.arctan2((-np.sin(theta[i][6]) * C - np.cos(theta[i][6]) * D), G) - theta[i][2] - theta[i][3]
    for i in range(1, 9):
        for j in range(1, 7):
            if theta[i][j] > 2 * np.pi:
                theta[i][j] -= 2 * np.pi
            elif theta[i][j] < -2 * np.pi:
                theta[i][j] += 2 * np.pi
            q_solutions[i - 1][j - 1] = theta[i][j]
    # for i in range(1, 9):
    #     for j in range(1, 7):
    #         angle = theta[i][j]
    #         # wrap angle to [-pi, pi]
    #         if angle > np.pi:
    #             angle -= 2 * np.pi
    #         elif angle < -np.pi:
    #             angle += 2 * np.pi
    #         q_solutions[i - 1][j - 1] = angle
    # print(q_solutions)
    return ur10_solution_filter_test(current, q_solutions)

def ur10_inverse_matrix_2cm(T_tool, current):
    # 此函数输入旋转矩阵 输出机械臂的一组最近关节角逆解
    L_robotiq = 0.06
    L_gripper = 0.10
    d = np.array([0, 0.1279598369588654, 0, 0, 0.1636727123724813,
                  0.1156246017328714, 0.09205355120243347 + L_robotiq + L_gripper])
    a = np.array([0, -0.6126666038983543, -0.5714813809834389, 0, 0, 0])

    # x, y, z = target_pos
    # R_mat = R.from_quat(quat).as_matrix()

    T06 = np.eye(4)
    T06[:3, :3] = T_tool[:3, :3]
    T06[0, 3] = T_tool[0, 3]
    T06[1, 3] = T_tool[1, 3]
    T06[2, 3] = T_tool[2, 3]

    theta = np.zeros((9, 7))
    q_solutions = np.zeros((8, 6))

    A = d[6] * T06[1, 2] - T06[1, 3]
    B = d[6] * T06[0, 2] - T06[0, 3]
    C = d[4]
    theta[1][1] = np.arctan2(A, B) - np.arctan2(C, np.sqrt(A**2 + B**2 - C**2))
    for i in [2, 3, 4]:
        theta[i][1] = theta[1][1]
    theta[5][1] = np.arctan2(A, B) - np.arctan2(C, -np.sqrt(A**2 + B**2 - C**2))
    for i in [6, 7, 8]:
        theta[i][1] = theta[5][1]

    for i in range(1, 5):
        A = np.sin(theta[i][1]) * T06[0, 2] - np.cos(theta[i][1]) * T06[1, 2]

        theta[i][5] = np.arccos(A)
    for i in range(5, 9):
        A = np.sin(theta[i][1]) * T06[0, 2] - np.cos(theta[i][1]) * T06[1, 2]
        theta[i][5] = -np.arccos(A)

    for i in range(1, 9, 2):
        A = np.sin(theta[i][1]) * T06[0, 0] - np.cos(theta[i][1]) * T06[1, 0]
        B = np.sin(theta[i][1]) * T06[0, 1] - np.cos(theta[i][1]) * T06[1, 1]
        C = np.sin(theta[i][5])
        if abs(C) > 1e-5:
            theta[i][6] = np.arctan2(A, B) - np.arctan2(C, 0.0)
            theta[i + 1][6] = theta[i][6]
        else:
            theta[i][6] = theta[i + 1][6] = 0
            print('解算失误')

    for i in range(1, 9, 2):
        C = np.cos(theta[i][1]) * T06[0, 0] + np.sin(theta[i][1]) * T06[1, 0]
        D = np.cos(theta[i][1]) * T06[0, 1] + np.sin(theta[i][1]) * T06[1, 1]
        E = np.cos(theta[i][1]) * T06[0, 2] + np.sin(theta[i][1]) * T06[1, 2]
        F = np.cos(theta[i][1]) * T06[0, 3] + np.sin(theta[i][1]) * T06[1, 3]
        G = np.cos(theta[i][6]) * T06[2, 1] + np.sin(theta[i][6]) * T06[2, 0]
        A = d[5] * (np.sin(theta[i][6]) * C + np.cos(theta[i][6]) * D) - d[6] * E + F
        B = T06[2, 3] - d[1] - T06[2, 2] * d[6] + d[5] * G
        l2, l3 = a[1], a[2]
        if A**2 + B**2 <= (l2 + l3)**2:
            theta[i][3] = np.arccos((A**2 + B**2 - l2**2 - l3**2) / (2 * l2 * l3))
            theta[i + 1][3] = -theta[i][3]
        else:
            theta[i][3] = theta[i + 1][3] = 0

    for i in range(1, 9):
        C = np.cos(theta[i][1]) * T06[0, 0] + np.sin(theta[i][1]) * T06[1, 0]
        D = np.cos(theta[i][1]) * T06[0, 1] + np.sin(theta[i][1]) * T06[1, 1]
        E = np.cos(theta[i][1]) * T06[0, 2] + np.sin(theta[i][1]) * T06[1, 2]
        F = np.cos(theta[i][1]) * T06[0, 3] + np.sin(theta[i][1]) * T06[1, 3]
        G = np.cos(theta[i][6]) * T06[2, 1] + np.sin(theta[i][6]) * T06[2, 0]
        A = d[5] * (np.sin(theta[i][6]) * C + np.cos(theta[i][6]) * D) - d[6] * E + F
        B = T06[2, 3] - d[1] - T06[2, 2] * d[6] + d[5] * G
        l2, l3 = a[1], a[2]
        M = ((l3 * np.cos(theta[i][3]) + l2) * B - l3 * np.sin(theta[i][3]) * A) / \
            (l2**2 + l3**2 + 2 * l2 * l3 * np.cos(theta[i][3]))
        N = (A + l3 * np.sin(theta[i][3]) * M) / (l3 * np.cos(theta[i][3]) + l2)
        theta[i][2] = np.arctan2(M, N)
        theta[i][4] = np.arctan2((-np.sin(theta[i][6]) * C - np.cos(theta[i][6]) * D), G) - theta[i][2] - theta[i][3]

    for i in range(1, 9):
        for j in range(1, 7):
            if theta[i][j] > 2 * np.pi:
                theta[i][j] -= 2 * np.pi
            elif theta[i][j] < -2 * np.pi:
                theta[i][j] += 2 * np.pi
            q_solutions[i - 1][j - 1] = theta[i][j]

    return ur10_solution_filter_test(current, q_solutions),q_solutions

def ur_numerical_ik(
    target_pos, target_quat, init_joint_angles,
    fk_func, max_iters=50, eps=1e-3,
    alpha=0.1, damping=1e-4, pos_weight=1.0, rot_weight=1.0
):
    def dh(a, alpha, d, theta):
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st * ca, st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0,      sa,     ca,      d],
            [0,       0,      0,      1]
        ])

    def compute_jacobian(joint_angles):
        L_robotiq, L_gripper = 0.06, 0.12
        d = [0, 0.127960, 0, 0, 0.1636727, 0.1156246, 0.0920536 + L_robotiq + L_gripper]
        a = [0, -0.61267, -0.57148, 0, 0, 0]
        alpha = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]

        T = np.eye(4)
        zs = np.empty((7, 3))
        os = np.empty((7, 3))
        zs[0] = [0, 0, 1]
        os[0] = [0, 0, 0]

        for i in range(6):
            A = dh(a[i], alpha[i], d[i + 1], joint_angles[i])
            T = T @ A
            zs[i + 1] = T[:3, 2]
            os[i + 1] = T[:3, 3]

        T_tool = np.eye(4)
        T_tool[2, 3] = d[6]
        T = T @ T_tool
        o_n = T[:3, 3]

        J = np.zeros((6, 6))
        for i in range(6):
            Jp = np.cross(zs[i], o_n - os[i])
            J[:, i] = np.hstack((Jp, zs[i]))
        return J

    def damped_pinv(J, lam=1e-4):
        JT = J.T
        return JT @ np.linalg.inv(J @ JT + lam * np.eye(J.shape[0]))
    def normalize_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    def normalize_angles(angles):
        return np.array([normalize_angle(a) for a in angles])
    joint_angles = np.array(init_joint_angles, dtype=np.float64)
    target_R = R.from_quat(target_quat)
    # === 记录角度偏移量（非归一化追踪） ===
    angle_offset = np.zeros(6)

    for _ in range(max_iters):
        current_pos, current_quat = fk_func(joint_angles)
        current_R = R.from_quat(current_quat)

        pos_err = target_pos - current_pos
        rot_err = (target_R * current_R.inv()).as_rotvec()
        err = np.concatenate([pos_weight * pos_err, rot_weight * rot_err])

        if np.linalg.norm(err) < eps:
            break

        J = compute_jacobian(joint_angles)
        J[:3, :] *= pos_weight
        J[3:, :] *= rot_weight
        dtheta = alpha * damped_pinv(J, damping) @ err
        joint_angles_new = joint_angles + dtheta

        for i in range(6):
            delta = joint_angles_new[i] - joint_angles[i]
            # 非归一化追踪：检查是否跳变
            if delta > np.pi:
                angle_offset[i] -= 2 * np.pi
            elif delta < -np.pi:
                angle_offset[i] += 2 * np.pi
            joint_angles[i] += delta

        # 加入偏移，保持全程连续
        joint_angles += angle_offset

    return joint_angles



def ur_numerical_ik_optimized(
    target_pos, target_quat, init_joint_angles,
    fk_func, max_iters=100, eps=5e-3,
    alpha=0.1, damping=1e-4, pos_weight=1.0, rot_weight=0.5,
    transform_matrix=None
):
    """
    优化整合版 UR 逆运动学数值解（含雅可比计算和角度归一化），支持解析解初始化
    参数：
        analytical_ik_func: 可选函数，输入目标位姿，返回候选关节解列表
    """
    def dh(a, alpha, d, theta):
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st * ca, st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0,      sa,     ca,      d],
            [0,       0,      0,      1]
        ])

    def compute_jacobian(joint_angles):
        L_robotiq, L_gripper = 0.025, 0.12
        d = [0, 0.127960, 0, 0, 0.1636727, 0.1156246, 0.0920536 + L_robotiq + L_gripper]
        a = [0, -0.61267, -0.57148, 0, 0, 0]
        alpha = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]
        T = np.eye(4)
        zs = np.empty((7, 3))
        os = np.empty((7, 3))
        zs[0] = [0, 0, 1]
        os[0] = [0, 0, 0]
        for i in range(6):
            A = dh(a[i], alpha[i], d[i + 1], joint_angles[i])
            T = T @ A
            zs[i + 1] = T[:3, 2]
            os[i + 1] = T[:3, 3]

        T_tool = np.eye(4)
        T_tool[2, 3] = d[6]
        T = T @ T_tool
        o_n = T[:3, 3]
        J = np.zeros((6, 6))
        for i in range(6):
            Jp = np.cross(zs[i], o_n - os[i])
            J[:, i] = np.hstack((Jp, zs[i]))
        return J

    def damped_pinv(J, lam=1e-4):
        JT = J.T
        return JT @ np.linalg.inv(J @ JT + lam * np.eye(J.shape[0]))

    def normalize_angles(angles):
        return (angles + np.pi) % (2 * np.pi) - np.pi


    if transform_matrix is not None:
        init_joint_angles = ur10_inverse_matrix(transform_matrix,init_joint_angles)

    joint_angles = init_joint_angles
    target_R = R.from_quat(target_quat)

    for _ in range(max_iters):
        current_pos, current_quat = fk_func(joint_angles)
        current_R = R.from_quat(current_quat)

        pos_err = target_pos - current_pos
        rot_err = (target_R * current_R.inv()).as_rotvec()
        err = np.concatenate([pos_weight * pos_err, rot_weight * rot_err])

        if np.linalg.norm(err) < eps:
            break

        J = compute_jacobian(joint_angles)
        J[:3, :] *= pos_weight
        J[3:, :] *= rot_weight
        dtheta = alpha * damped_pinv(J, damping) @ err
        joint_angles_new = joint_angles + dtheta

        # 保持角度连续
        for i in range(6):
            delta = normalize_angles(joint_angles_new[i] - joint_angles[i])
            joint_angles[i] += delta

        # 限制在 [-π, π]
        joint_angles = normalize_angles(joint_angles)

    return joint_angles






