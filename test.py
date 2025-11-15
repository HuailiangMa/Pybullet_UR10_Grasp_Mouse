import numpy as np
from ur10_pybullet_sim_env import UR10_Sim_Env
from ur10_pybullet_sim_env import UR10_Move_Controller
import os
from scipy.spatial.transform import Rotation as R, Slerp
import pybullet as pb
import time

# 相机标定结果
def get_T_base_camera():
    """
    定义并返回相机坐标系到基座坐标系的 4x4 变换矩阵 T_base_camera
    """
    # 替换为新的四元数和位置
    q_base_camera = [-0.9029638452119679, -0.37028542144547855,
                     0.09033867657688738, 0.19845383447869733]  # [x, y, z, w]
    t_base_camera = np.array([0.8748847230700205, -0.6808723436321604, 0.8521323625643371])  # [x, y, z]
    # 构造旋转矩阵和平移矩阵
    R_base_camera = R.from_quat(q_base_camera).as_matrix()
    T_base_camera = np.eye(4)
    T_base_camera[:3, :3] = R_base_camera
    T_base_camera[:3, 3] = t_base_camera
    return T_base_camera

def read_T_cam_obj_transformation_matrix(filename):
    """ 从TXT文件读取4x4变换矩阵 """
    with open(filename, 'r') as file:
        lines = file.readlines()
        matrix = np.array([list(map(float, line.strip().split())) for line in lines])
    return matrix

def get_T_base_obj_from_posefile(pose_path):
    """
    从 pose 文件读取 T_cam_obj，并转换为 T_base_obj。

    参数：
        pose_dir (str): 姿态文件所在目录
        pose_file (str): 姿态文件名

    返回：
        np.ndarray: 4x4 的 T_base_obj 变换矩阵
    """
    T_cam_block = read_T_cam_obj_transformation_matrix(pose_path)
    T_base_cam = get_T_base_camera()
    T_base_block = T_base_cam @ T_cam_block
    return T_base_block

def main():
    print("Hello world")
    env = UR10_Sim_Env()  # 加载pybullet环境类 包含机器人及夹爪、地面的生成
    client_id = env.client_id  # 获取client_id
    robot_info = env.robot_info  # 获取robot相关id
    robot_id = robot_info["robot_id"]
    joint_indices = robot_info["joint_indices"]
    table_position = np.array([0.58, -0.48, 0.14])
    table_size = np.array([0.4, 0.3, 0.14])
    table_id = env.load_table(table_position=table_position, table_size=table_size)  # 加载桌子
    table_z = 0.30

    # 初始化机械臂运动控制器类
    ur10_move_ctrl = UR10_Move_Controller(robot_info, client_id)
    ## --------该任务是一个简单的抓取-放置任务 先抓取鼠标 之后将鼠标放置在鼠标垫上---------- ##
    # 鼠标的obj文件路径
    mouse_mesh_path = 'assets/mouse_mesh/untitled.obj' # 鼠标的obj文件 纹理文件
    obj1_pose_filename = 'assets/mouse_pose.txt' # 鼠标的初始位姿
    ur10_trajectory1_filename = 'assets/ur10_grasp_mouse_trajectory1.npy' # 设定好的抓取路径
    ur10_gripper_cmd1 = 'assets/ur10_gripper_cmd1.npy' # 第一段抓取轨迹的夹爪控制指令
    # 鼠标垫的obj文件路径
    mouse_pad_mesh_path = 'assets/mouse_pad_mesh/untitled.obj' # 鼠标垫的obj文件 纹理文件
    obj2_pose_filename = 'assets/mouse_pad_pose.txt' # 鼠标垫的初始位姿
    ur10_trajectory2_filename = 'assets/ur10_place_mouse_trajectory2.npy'  # 设定好的放置路径
    ur10_gripper_cmd2 = 'assets/ur10_gripper_cmd2.npy' # 第二段放置轨迹的夹爪控制指令
    # 解包文件 生成物体
    T_base_obj1 = get_T_base_obj_from_posefile(obj1_pose_filename)
    T_base_obj2 = get_T_base_obj_from_posefile(obj2_pose_filename)
    # obj1_id = env.load_block(mouse_mesh_path, T_base_obj1, mass=0.1, lateral_friction=10,
    #                          baseInertialFramePosition=[0.0, 0.0, 0], scale=[0.8, 1.5, 1])  # 该物体的y轴实际上是z轴
    # obj2_id = env.load_block(mouse_pad_mesh_path, T_base_obj2, mass=0, lateral_friction=0.9,
    #                          baseInertialFramePosition=[0.0, 0.0, 0.0], table_z=0.28)  # 该物体的y轴实际上是z轴
    # 解包机械臂和夹爪控制指令
    T_base_ee_list1 = np.load(ur10_trajectory1_filename, allow_pickle=True)
    T_base_ee_list2 = np.load(ur10_trajectory2_filename, allow_pickle=True)
    gripper_cmd_list1 = np.load(ur10_gripper_cmd1, allow_pickle=True)
    gripper_cmd_list2 = np.load(ur10_gripper_cmd2, allow_pickle=True)
    table_position = np.array([0.58, -0.48, 0.14])
    table_size = np.array([0.4, 0.3, 0.13])
    obj1_center_base = (0.685, -0.382)  # 左半桌子
    obj2_center_base = (0.514, -0.563)  # 右半桌子
    init_flag = 0 # 状态切换标志位
    gripper_flag = 0 # 抓取标志位
    # # === 重置仿真环境 ===
    # ur10_move_ctrl.move_init_TopDownGrasp() # 调整到竖直抓取位姿 另一种是水平抓取位姿
    # env.gripper.open()
    while True:
        # if init_flag == 0:
        #     start_pose = T_base_ee_list1[0]
        #     move_flag = ur10_move_ctrl.move_cartesian_interpolation(start_pose, steps=30, speed=0.05)
        #     init_flag = 1
        # elif init_flag == 1:
        #     for i, T_base_ee in enumerate(T_base_ee_list1[1:]):
        #         move_flag = ur10_move_ctrl.move_ik(T_base_ee, speed=0.02)  # 带初值的数值解法实际并不好用 跟解析解的效果一模一样
        #         gripper_cmd = gripper_cmd_list1[i]
        #         ur10_move_ctrl.gripper_cmd = gripper_cmd_list1[i]
        #         gripper_flag = env.gripper.update_gripper(ur10_move_ctrl,gripper_cmd, gripper_flag)
        #     init_flag = 2
        # elif init_flag == 2:
        #     start_pose = T_base_ee_list2[0]
        #     move_flag = ur10_move_ctrl.move_cartesian_interpolation(start_pose, steps=30, speed=0.05)
        #     init_flag = 3
        # elif init_flag == 3:
        #     for i, T_base_ee in enumerate(T_base_ee_list2[1:]):
        #         move_flag = ur10_move_ctrl.move_ik(T_base_ee, speed=0.05)  # 带初值的数值解法实际并不好用 跟解析解的效果一模一样
        #         gripper_cmd = gripper_cmd_list2[i]
        #         ur10_move_ctrl.gripper_cmd = gripper_cmd_list2[i]
        #         gripper_flag = env.gripper.update_gripper(ur10_move_ctrl, gripper_cmd, gripper_flag)
        #     init_flag = 4
        # elif init_flag == 4:
        #     print(f"[DONE] 轨迹执行完成")
        #     pb.removeBody(obj1_id)
        #     pb.removeBody(obj2_id)
        #     time.sleep(0.005)  # 等待物体完全移除
        #     init_flag = 5
        pb.stepSimulation()
        time.sleep(1 / 240)

if __name__ == "__main__":
    main()
