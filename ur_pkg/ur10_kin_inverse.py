#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
from math import sin, cos
from scipy.spatial.transform import Rotation as R

# UR10 DH 参数（单位：米）
d1 = 0.1273
a2 = -0.612
a3 = -0.5723
d4 = 0.163941
d5 = 0.1157
d6 = 0.0922

ZERO_THRESH = 1e-8
PI = np.pi

# 替代 C 宏
def SIGN(x):
    return np.sign(x)

def forward_kinematics_ur10(q):
    q1, q2, q3, q4, q5, q6 = q

    s1, c1 = sin(q1), cos(q1)
    s2, c2 = sin(q2), cos(q2)
    s3, c3 = sin(q3), cos(q3)
    s4, c4 = sin(q4), cos(q4)
    s5, c5 = sin(q5), cos(q5)
    s6, c6 = sin(q6), cos(q6)

    q23 = q2 + q3
    q234 = q2 + q3 + q4
    s23, c23 = sin(q23), cos(q23)
    s234, c234 = sin(q234), cos(q234)

    T = np.zeros((4, 4))
    T[0, 0] = c234 * c1 * s5 - c5 * s1
    T[0, 1] = c6 * (s1 * s5 + c234 * c1 * c5) - s234 * c1 * s6
    T[0, 2] = -s6 * (s1 * s5 + c234 * c1 * c5) - s234 * c1 * c6
    T[0, 3] = d6 * c234 * c1 * s5 - a3 * c23 * c1 - a2 * c1 * c2 - d6 * c5 * s1 - d5 * s234 * c1 - d4 * s1

    T[1, 0] = c1 * c5 + c234 * s1 * s5
    T[1, 1] = -c6 * (c1 * s5 - c234 * c5 * s1) - s234 * s1 * s6
    T[1, 2] = s6 * (c1 * s5 - c234 * c5 * s1) - s234 * c6 * s1
    T[1, 3] = d6 * (c1 * c5 + c234 * s1 * s5) + d4 * c1 - a3 * c23 * s1 - a2 * c2 * s1 - d5 * s234 * s1

    T[2, 0] = -s234 * s5
    T[2, 1] = -c234 * s6 - s234 * c5 * c6
    T[2, 2] = s234 * c5 * s6 - c234 * c6
    T[2, 3] = d1 + a3 * s23 + a2 * s2 - d5 * (c23 * c4 - s23 * s4) - d6 * s5 * (c23 * s4 + s23 * c4)

    T[3, :] = [0, 0, 0, 1]

    position = T[:3, 3].tolist()

    # 使用 scipy 计算四元数（从旋转矩阵）
    rotation_matrix = T[:3, :3]
    quat = R.from_matrix(rotation_matrix).as_quat()  # [x, y, z, w] 格式
    quaternion = quat.tolist()

    return position, quaternion


def inverse(T, current, q6_des=0.0):
    T = np.array(T).reshape((4, 4))
    T02 = -T[0, 2]; T00 = T[0, 0]; T01 = T[0, 1]; T03 = -T[0, 3]
    T12 = -T[1, 2]; T10 = T[1, 0]; T11 = T[1, 1]; T13 = -T[1, 3]
    T22 = T[2, 2]; T20 = -T[2, 0]; T21 = -T[2, 1]; T23 = T[2, 3]

    q_sols = []
    q1 = []

    A = d6 * T12 - T13
    B = d6 * T02 - T03
    R = A**2 + B**2

    if abs(A) < ZERO_THRESH:
        div = -np.sign(d4) * np.sign(B) if abs(abs(d4) - abs(B)) < ZERO_THRESH else -d4 / B
        arcsin = np.arcsin(div)
        arcsin = 0.0 if abs(arcsin) < ZERO_THRESH else arcsin
        q1 = [(arcsin + 2*PI) if arcsin < 0 else arcsin, PI - arcsin]
    elif abs(B) < ZERO_THRESH:
        div = np.sign(d4) * np.sign(A) if abs(abs(d4) - abs(A)) < ZERO_THRESH else d4 / A
        arccos = np.arccos(div)
        q1 = [arccos, 2*PI - arccos]
    elif d4**2 > R:
        return []  # no solution
    else:
        arccos = np.arccos(d4 / np.sqrt(R))
        arctan = np.arctan2(-B, A)
        pos = arccos + arctan
        neg = -arccos + arctan
        q1 = [pos if pos >= 0 else 2*PI + pos, neg if neg >= 0 else 2*PI + neg]

    # wrist joint q5
    q5 = []
    for i in range(2):
        numer = T03 * np.sin(q1[i]) - T13 * np.cos(q1[i]) - d4
        div = np.sign(numer) * np.sign(d6) if abs(abs(numer) - abs(d6)) < ZERO_THRESH else numer / d6
        arccos = np.arccos(div)
        q5.append([arccos, 2*PI - arccos])

    # solve for remaining joints
    for i in range(2):
        for j in range(2):
            c1, s1 = np.cos(q1[i]), np.sin(q1[i])
            c5, s5 = np.cos(q5[i][j]), np.sin(q5[i][j])
            if abs(s5) < ZERO_THRESH:
                q6 = q6_des
            else:
                q6 = np.arctan2(-SIGN(s5)*(T01*s1 - T11*c1), SIGN(s5)*(T00*s1 - T10*c1))
                q6 = 0.0 if abs(q6) < ZERO_THRESH else (q6 + 2*PI if q6 < 0 else q6)

            c6, s6 = np.cos(q6), np.sin(q6)
            x04x = -s5*(T02*c1 + T12*s1) - c5*(s6*(T01*c1 + T11*s1) - c6*(T00*c1 + T10*s1))
            x04y = c5*(T20*c6 - T21*s6) - T22*s5
            p13x = d5*(s6*(T00*c1 + T10*s1) + c6*(T01*c1 + T11*s1)) - d6*(T02*c1 + T12*s1) + T03*c1 + T13*s1
            p13y = T23 - d1 - d6*T22 + d5*(T21*c6 + T20*s6)

            c3 = (p13x**2 + p13y**2 - a2**2 - a3**2) / (2*a2*a3)
            if abs(abs(c3) - 1.0) < ZERO_THRESH:
                c3 = np.sign(c3)
            elif abs(c3) > 1.0:
                continue  # no solution
            arccos = np.arccos(c3)
            q3 = [arccos, 2*PI - arccos]

            denom = a2**2 + a3**2 + 2*a2*a3*c3
            s3 = np.sin(arccos)
            A = a2 + a3*c3
            B = a3*s3
            q2 = [
                np.arctan2((A*p13y - B*p13x) / denom, (A*p13x + B*p13y) / denom),
                np.arctan2((A*p13y + B*p13x) / denom, (A*p13x - B*p13y) / denom)
            ]

            c23 = [np.cos(q2[0] + q3[0]), np.cos(q2[1] + q3[1])]
            s23 = [np.sin(q2[0] + q3[0]), np.sin(q2[1] + q3[1])]
            q4 = [
                np.arctan2(c23[0]*x04y - s23[0]*x04x, x04x*c23[0] + x04y*s23[0]),
                np.arctan2(c23[1]*x04y - s23[1]*x04x, x04x*c23[1] + x04y*s23[1])
            ]

            for k in range(2):
                q2[k] = 0.0 if abs(q2[k]) < ZERO_THRESH else (q2[k] + 2*PI if q2[k] < 0 else q2[k])
                q4[k] = 0.0 if abs(q4[k]) < ZERO_THRESH else (q4[k] + 2*PI if q4[k] < 0 else q4[k])
                q_sols.append([
                    q1[i], q2[k], q3[k], q4[k], q5[i][j], q6
                ])
            q_solutions = np.array(q_sols)
            optimal_solution = ur10_solution_filter_test(current,q_solutions)

    return optimal_solution

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
