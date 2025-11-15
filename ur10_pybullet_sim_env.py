# 此文件用于重构仿真加载类和夹爪抓取类
# 后续重构机械臂移动类和控制器类
# import csv
# import shutil

from scipy.spatial.transform import Rotation as R, Slerp
from ur_pkg import ur10_pybullet_inverse as ur_ik
import hashlib

import pybullet as pb
import pybullet_data as pd
import numpy as np
import time
import os

class UR10_Sim_Env:
    def __init__(self):
        self.client_id = pb.connect(pb.GUI)
        # self.client_id = pb.connect(pb.DIRECT)
        self._init_sim()
        self.robot_info = self._load_robot()
        self.setup_gripper()
        self.gripper = GripperController(robot_info=self.robot_info, client_id=self.client_id)
        self.brush_flag = 0

    def _init_sim(self):
        # 加载地面、机器人(附带夹爪)
        pb.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS, 0)
        pb.setGravity(0, 0, -9.81)
        pb.setAdditionalSearchPath(pd.getDataPath())
        self.plane_id = pb.loadURDF("plane.urdf")
        pb.changeVisualShape(
            self.plane_id,
            linkIndex=-1,
            rgbaColor=[0.7, 0.9, 1.0, 1.0] # 绿色，alpha=1为不透明
        )
        # 调整视角
        pb.resetDebugVisualizerCamera(
            cameraDistance=1.0,
            cameraYaw=45.199737548828125,
            cameraPitch=-38.19999694824219,
            cameraTargetPosition=[0.21293649077415466, -0.11691831052303314, 0.03891478106379509]
        )

    def _load_robot(self):
        def load_robot_and_get_indices(base_position=(0, 0, 0), base_orientation=(0, 0, 0)):
            """
            加载 UR10 机械臂并解析重要关节索引，可指定朝向。

            参数：
                base_position: 机械臂基座位置 (x, y, z)
                base_orientation: 基座朝向 (roll, pitch, yaw)，单位：弧度

            返回：
                字典，包含 robot_id 及各关节索引
            """
            urdf_path = "/home/robot/PycharmProjects/ACT_UR10/pybullet_ur10_project/assets/robot.urdf" # 这里替换为你自己的urdf文件位置
            quat = pb.getQuaternionFromEuler(base_orientation)
            robot_id = pb.loadURDF(
                urdf_path,
                basePosition=base_position,
                baseOrientation=quat,
                useFixedBase=True
            )
            print("==== 所有关节及其 child_link 名 ====")
            for i in range(pb.getNumJoints(robot_id)):
                joint_info = pb.getJointInfo(robot_id, i)
                print(f"Joint {i} -> link: {joint_info[12].decode()}")
            # 初始化索引
            ee_link_index = -1
            ee_tip_index = -1
            left_finger_joint_index = -1
            right_finger_joint_index = -1
            for i in range(pb.getNumJoints(robot_id)):
                joint_info = pb.getJointInfo(robot_id, i)
                name = joint_info[1].decode("utf-8")
                link_name = joint_info[12].decode("utf-8")
                if link_name == "wrist_3_link":
                    ee_link_index = i
                if name == "finger_joint":
                    left_finger_joint_index = i
                if name == "right_outer_knuckle_joint":
                    right_finger_joint_index = i
                if name == "dummy_center_fixed_joint":
                    ee_tip_index = i
            assert ee_link_index != -1, "未找到 wrist_3_link"
            assert left_finger_joint_index != -1, "未找到 finger_joint"
            assert right_finger_joint_index != -1, "未找到 right_outer_knuckle_joint"
            assert ee_tip_index != -1, "未找到 dummy_center_fixed_joint"
            print(f"末端: {ee_link_index}, 夹爪: {left_finger_joint_index}, {right_finger_joint_index}")
            joint_indices = [1, 2, 3, 4, 5, 6]
            return {
                "robot_id": robot_id,
                "joint_indices": joint_indices,
                "ee_link_index": ee_link_index,
                "ee_tip_index": ee_tip_index,
                "left_finger_joint_index": left_finger_joint_index,
                "right_finger_joint_index": right_finger_joint_index
            }
        robot_info = load_robot_and_get_indices(
            base_position=(0, 0, 0),
            base_orientation=(0, 0, np.pi)
        )
        return robot_info

    def setup_gripper(self):
        setup_gripper_mimic_constraints_v3(
            self.robot_info["robot_id"],
            self.client_id,
            self.robot_info["left_finger_joint_index"]
        )

    def load_table(self, table_position=[0.5, -0.5, 0.15], table_size = [0.25, 0.25, 0.11]):
        return self.load_demo_box(table_position=table_position, table_size=table_size)

    def load_demo_box(self,table_position,table_size):
        # === 1. 创建桌子 ===
        def create_table_box(
                position=[0.5, -0.5, 0.15],
                size=[0.4, 0.4, 0.3],
                rgba=[1.0, 1.0, 1.0, 1.0],
                mass=0,
                orientation_euler=(0, 0, 0)
        ):
            quat = pb.getQuaternionFromEuler(orientation_euler)
            visual_shape = pb.createVisualShape(
                shapeType=pb.GEOM_BOX,
                halfExtents=size,
                rgbaColor=rgba
            )
            collision_shape = pb.createCollisionShape(
                shapeType=pb.GEOM_BOX,
                halfExtents=size
            )
            table_id = pb.createMultiBody(
                baseMass=mass,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=position,
                baseOrientation=quat
            )
            # ✅ 附加刚度和阻尼
            # Step 4: 设置摩擦力参数
            pb.changeDynamics(
                bodyUniqueId=table_id,
                linkIndex=-1,
                lateralFriction=0.5,
                rollingFriction=0.003,  # 非0
                spinningFriction=0.003,
                contactStiffness=5000,
                contactDamping=1  # 调大之后物体越倾向于旋转和乱动
            )
            return table_id
        table_id = create_table_box(
            table_position,
            table_size,
            orientation_euler=(0, 0, np.pi / 4)  # 旋转 45°
        )
        return table_id

    # 新版加载物体程序 需要已知重建后的纹理文件和重建后的几何文件
    def load_cup(self,fixed_obj_path, convex_obj_path, transform_matrix,mass=0.3, lateral_friction = 0.8, scale=[1, 1, 1],
                 baseInertialFramePosition=[0, 0, 0],table_z=0.30,color=[0.7, 0.85, 1.0, 1.0]):
        return load_object_fixed_and_convex(fixed_obj_path, convex_obj_path, transform_matrix,mass=mass, scale=scale,lateral_friction=lateral_friction,
                                            rolling_friction=0.001, spinning_friction=0.001,baseInertialFramePosition=baseInertialFramePosition, table_z=table_z,color=color)
    # 旧版加载物体程序 需要已知物体的obj格式文件
    def load_block(self, path, transform_matrix, mass=0.3, lateral_friction = 0.5, scale=[1, 1, 1],baseInertialFramePosition=[0.0, 0.0, 0.0],table_z=0.30):
        return load_object_vhacd_mesh(path, transform_matrix, mass=mass, lateral_friction = lateral_friction, scale=scale,baseInertialFramePosition=baseInertialFramePosition,table_z=table_z)




class GripperController:
    def __init__(self, robot_info: dict, client_id: int, joint_name: str = "finger_joint"):
        self.robot_id = robot_info["robot_id"]
        self.client_id = client_id
        self.joint_name = joint_name
        self.joint_index = self._find_joint_index()
        self.grasp_flag = False

    def _find_joint_index(self):
        for i in range(pb.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
            info = pb.getJointInfo(self.robot_id, i, physicsClientId=self.client_id)
            if info[1].decode("utf-8") == self.joint_name:
                return i
        raise ValueError(f"[ERROR] Joint '{self.joint_name}' not found.")

    def close(self, target_angle=0.43, force=1000, timeout=3.0): # 以前默认的闭合角度为0.51
        """
        快速控制夹爪闭合，使用高速度/高增益控制器，仍保留初始清除阶段。
        """
        # 1. 清除 VELOCITY 控制
        pb.setJointMotorControl2(
            bodyIndex=self.robot_id,
            jointIndex=self.joint_index,
            controlMode=pb.VELOCITY_CONTROL,
            targetVelocity=0.0,
            force=0.0,
            physicsClientId=self.client_id
        )
        for _ in range(2):  # 稳定帧
            pb.stepSimulation(physicsClientId=self.client_id)

        # 2. 设置 POSITION 控制
        pb.setJointMotorControl2(
            bodyIndex=self.robot_id,
            jointIndex=self.joint_index,
            controlMode=pb.POSITION_CONTROL,
            targetPosition=target_angle,
            force=force,
            positionGain=1.0,
            maxVelocity=1.5,
            physicsClientId=self.client_id
        )

        # 3. 等待收敛或超时
        t0 = time.time()
        while time.time() - t0 < timeout:
            joint_pos = pb.getJointState(self.robot_id, self.joint_index, physicsClientId=self.client_id)[0]

            if abs(joint_pos - target_angle) < 0.002:
                return True
            pb.stepSimulation(physicsClientId=self.client_id)
            time.sleep(1 / 2000)

        print("[WARNING] close_gripper_position 超时未完全闭合")
        return False

    def open(self, target_angle=0.14, force=30, timeout=10.0):
        """
        快速稳定地打开夹爪到指定角度，适用于仿真中张开速度加快。
        """
        # 1. 清除可能存在的 VELOCITY 控制（避免抖动）
        pb.setJointMotorControl2(
            bodyIndex=self.robot_id,
            jointIndex=self.joint_index,
            controlMode=pb.VELOCITY_CONTROL,
            targetVelocity=0.0,
            force=0.0,
            physicsClientId=self.client_id
        )
        for _ in range(2):
            pb.stepSimulation(physicsClientId=self.client_id)

        # 2. 设置位置控制目标
        pb.setJointMotorControl2(
            bodyIndex=self.robot_id,
            jointIndex=self.joint_index,
            controlMode=pb.POSITION_CONTROL,
            targetPosition=target_angle,
            force=force,
            positionGain=1.0,
            maxVelocity=2.0,  # 提高张开速度
            physicsClientId=self.client_id
        )

        # 3. 等待达到目标位置或超时
        t0 = time.time()
        while time.time() - t0 < timeout:
            joint_pos = pb.getJointState(self.robot_id, self.joint_index, physicsClientId=self.client_id)[0]
            if abs(joint_pos - target_angle) < 0.003:
                # print(f"[INFO] 夹爪已成功张开至 {joint_pos:.4f}")
                return True
            pb.stepSimulation(physicsClientId=self.client_id)
            time.sleep(1 / 2000)

        print("[WARNING] 超时未完全张开夹爪")
        return False

    def _move(self, target_angle, force, timeout, action):
        pb.setJointMotorControl2(
            bodyIndex=self.robot_id,
            jointIndex=self.joint_index,
            controlMode=pb.VELOCITY_CONTROL,
            targetVelocity=0.0,
            force=0.0,
            physicsClientId=self.client_id
        )
        for _ in range(2):
            pb.stepSimulation(physicsClientId=self.client_id)

        pb.setJointMotorControl2(
            bodyIndex=self.robot_id,
            jointIndex=self.joint_index,
            controlMode=pb.POSITION_CONTROL,
            targetPosition=target_angle,
            force=force,
            positionGain=1.0,
            maxVelocity=2.0 if action == "open" else 1.5,
            physicsClientId=self.client_id
        )

        t0 = time.time()
        while time.time() - t0 < timeout:
            joint_pos = pb.getJointState(self.robot_id, self.joint_index, physicsClientId=self.client_id)[0]
            if abs(joint_pos - target_angle) < 0.003:
                print(f"[INFO] Gripper {action} done: {joint_pos:.4f}")
                return True
            pb.stepSimulation(physicsClientId=self.client_id)
            time.sleep(1 / 2000)

        print(f"[WARNING] Gripper {action} timeout")
        return False

    def check_grasp_success(self, block_id, pad_link_indices=[13, 18], depth_thresh=1, contact_thresh=10,
                            extra_push=0.05, force=50):
        result, contact_count = get_pad_contact_depths(
            self.robot_id, block_id, pad_link_indices, client_id=self.client_id
        )
        print(result)
        if result[13] is not None or result[18] is not None:
            # print(1)
            # print(result[12],result[17],contact_count)
            if result[13] > depth_thresh or result[18] > depth_thresh:
                #print(2)
                #print("[SUCCESS] Grasp detected via contact depth > threshold")
                self.grasp_flag = True
                return True
            else:
                return False
        else:
            return False

    def check_placement_success(self, ur10_move_ctrl):
        if ur10_move_ctrl.task_success_flag:
            all_in_contact = True
        else:
            all_in_contact = False
            return all_in_contact
        return all_in_contact

    def update_gripper(self, ur10_move_ctrl, gripper_cmd, gripper_flag,target_angle=0.60):
        """
        根据 gripper 控制指令更新夹爪状态。

        参数：
            env: 仿真环境，需包含 env.gripper
            gripper_cmd_list: 控制指令列表（float 数组）
            index: 当前轨迹点索引
            gripper_flag: 当前夹爪状态标志（0=开，1=闭）

        返回：
            更新后的 gripper_flag
        """
        ur10_move_ctrl.gripper_cmd = gripper_cmd
        if gripper_cmd > 0.5 and gripper_flag == 0:
            self.close(target_angle)
            return 1
        elif gripper_cmd < 0.5 and gripper_flag == 1:
            # pb.changeDynamics(block_id, -1, lateralFriction=0.3, rollingFriction=0.01, spinningFriction=0.01) # 该指令可以动态调整摩擦力
            self.open()
            return 0
        return gripper_flag


# UR10控制器类 用于状态更新
class UR10_Move_Controller:
    def __init__(self, robot_info: dict, client_id: int):
        self.robot_id = robot_info["robot_id"]
        self.joint_indices = robot_info["joint_indices"]
        self.ee_link_index = robot_info["ee_link_index"]
        self.wrist3_link_index = robot_info["ee_link_index"]
        self.client_id = client_id
        self.init_joint_angles = [-0.09524376, -1.1650527,  -2.21238966, -2.88003804, -0.9131, 0.08560491]
        self.state_record_flag = False # 开启记录标志位 用于记录机械臂移动过程中的末端位姿和关节角和夹爪指令
        self.relative_state_record_flag = False # 开启相对位姿变换记录标志位 记录机械臂移动过程中相对位姿和夹爪指令
        self.state_history = [] # 记录信息的列表
        self.gripper_cmd = 0.0 # 用于记录机械臂的夹爪控制指令
        self.gripper_obs = 0.0
        self.obj_obs_pose = np.zeros(7) # 用于记录七维物体位姿
        self.ur_obs_pose = np.zeros(7) # 用于记录机械臂末端的七维位姿
        self.ur_cmd = np.zeros(7) # 用于记录机械臂末端的控制指令
        self.current_joint_angles = np.zeros(6) # 用于记录当前关节角
        self.command_joint_angles = np.zeros(6) # 用于记录当前关节角控制指令
        self.save_sate_filepath = "./dataset_logs"
        self.save_counter = 0

        # 另一种保存数据的方式
        self.T_obj_ee_obs = np.eye(4)
        self.T_obj_ee_cmd = np.eye(4)
        self.T_base_obj_obs = np.eye(4)
        self.T_obj_base = np.eye(4)
        self.T_obj_base_obs = np.eye(4)

        self.obj1_id = None
        self.obj2_id = None
        self.T_base_obj1 = np.eye(4)
        self.T_base_obj2 = np.eye(4)
        self.T_base_obj3 = np.eye(4)
        self.T_base_obj4 = np.eye(4)

        self.collision_obj2_flag = False
        self.obj2_collision_flag = False
        self.task_success_flag = False

    def get_joint_angles(self):
        return [pb.getJointState(self.robot_id, i, physicsClientId=self.client_id)[0] for i in self.joint_indices]

    def move_matrix(self, T_world_tool0, speed=0.01, timeout=3):
        current_angles = self.get_joint_angles()
        start_time = time.time()
        joint_angles = ur_ik.ur10_inverse_matrix(T_world_tool0, current_angles)
        # === 结束计时 ===
        elapsed_time = time.time() - start_time
        print(f"[IK] 解算用时: {elapsed_time:.4f} 秒")
        print("[IK] 结果关节角:", joint_angles)
        return self.move_joints(joint_angles, speed=speed, timeout=timeout)

    def move_quat(self,target_position,target_quat,speed=0.02, timeout=3):
        current_angles = self.get_joint_angles()
        joint_angles, qsols = ur_ik.ur10_inverse(target_position, target_quat, current_angles)
        print(joint_angles)
        joint_angles = [2.09, -1.97, 2.21, -0.25641185442079717, 1.2748560905456543, 0.1050417348742485]
        return self.move_joints(joint_angles, speed=speed, timeout=timeout),qsols

    def move_init(self,speed=0.01, timeout=3):
        # joint_angles = [2.09, -1.97, 2.21, -0.25641185442079717, 1.2748560905456543, 0.1050417348742485]
        joint_angles = [2.2587430477142334, -1.85327655473818, 2.182936191558838, -0.3368290106402796, 1.4757771492004395, 0.0005033374764025211]
        return self.move_joints(joint_angles, speed=speed, timeout=timeout)

    def move_init_HorizontalGrasp(self,speed=0.01, timeout=3):
        # joint_angles = [2.09, -1.97, 2.21, -0.25641185442079717, 1.2748560905456543, 0.1050417348742485]
        # joint_angles = [2.2587430477142334, -1.85327655473818, 2.182936191558838, -0.3368290106402796, 1.4757771492004395, 0.0005033374764025211]
        joint_angles = [2.25974989, -2.15307743, 2.08053637, 0.06548142, 1.47576523, 0.00045540]
        return self.move_joints(joint_angles, speed=speed, timeout=timeout)

    def move_init_TopDownGrasp(self,speed=0.01, timeout=3):
        # joint_angles = [2.09, -1.97, 2.21, -0.25641185442079717, 1.2748560905456543, 0.1050417348742485]
        joint_angles = [2.34667492, -1.51858265, 1.48441792, 1.58592689, 1.57081020, -0.00441009]
        return self.move_joints(joint_angles, speed=speed, timeout=timeout)

    def move_my_policy_inverse(self,  T_world_tool0, speed=0.01, timeout=3):
        current_angles = self.get_joint_angles()
        joint_angles = ur_ik.ur10_my_policy_inverse(T_world_tool0, current_angles)
        return self.move_joints(joint_angles, speed=speed, timeout=timeout)
    def get_T_base_ee(self):
        self.current_joint_angles = np.array(self.get_joint_angles())
        T_base_ee = ur_ik.ur10_forward_T(self.current_joint_angles)
        return T_base_ee
    def get_pos_quat(self):
        self.current_joint_angles = np.array(self.get_joint_angles())
        pos,quat = ur_ik.ur10_forward(self.current_joint_angles)
        return pos,quat

    def get_obj1_pose(self):
        """
        获取物体的世界位姿（位置 + 四元数）
        Returns:
            tuple:
                - pos (np.ndarray): 3D 位置 (x, y, z)
                - quat (np.ndarray): 四元数姿态 (x, y, z, w)
        """
        pos, quat = pb.getBasePositionAndOrientation(self.obj1_id, physicsClientId=self.client_id)
        return np.array(pos), np.array(quat)

    def get_T_base_obj1(self):
        pos = self.T_base_obj1[:3,3]
        rot = self.T_base_obj1[:2, :3].reshape(6)
        return pos, rot
    def get_T_base_obj2(self):
        pos = self.T_base_obj2[:3,3]
        rot = self.T_base_obj2[:2, :3].reshape(6)
        return pos, rot
    def get_T_base_obj3(self):
        pos = self.T_base_obj3[:3,3]
        rot = self.T_base_obj3[:2, :3].reshape(6)
        return pos, rot
    def get_T_base_obj4(self):
        pos = self.T_base_obj4[:3,3]
        rot = self.T_base_obj4[:2, :3].reshape(6)
        return pos, rot

    def get_obj2_pose(self):
        """
        获取物体的世界位姿（位置 + 四元数）
        Returns:
            tuple:
                - pos (np.ndarray): 3D 位置 (x, y, z)
                - quat (np.ndarray): 四元数姿态 (x, y, z, w)
        """
        pos, quat = pb.getBasePositionAndOrientation(self.obj2_id, physicsClientId=self.client_id)
        return np.array(pos), np.array(quat)

    def move_cartesian_interpolation(self, target_T, steps=30, speed=0.05):
        current_angles = self.get_joint_angles()
        current_pos, current_quat = ur_ik.ur10_forward(current_angles)
        target_pos = target_T[:3, 3]
        target_quat = R.from_matrix(target_T[:3, :3]).as_quat()
        # 修正四元数方向一致性，防止插值翻转
        if np.dot(current_quat, target_quat) < 0:
            target_quat = -target_quat
        last_joint_angles = current_angles
        for i in range(1, steps + 1):
            alpha = i / steps
            interp_pos = (1 - alpha) * current_pos + alpha * target_pos
            interp_quat = (1 - alpha) * current_quat + alpha * target_quat
            interp_quat /= np.linalg.norm(interp_quat)
            joint_angles = ur_ik.ur_numerical_ik(
                fk_func=ur_ik.ur10_forward,
                target_pos=interp_pos,
                target_quat=interp_quat,
                init_joint_angles=last_joint_angles
            )
            if joint_angles is None:
                print(f"[WARN] IK failed at step {i}/{steps}")
                return False

            self.move_joints(joint_angles, speed=speed)
            pb.stepSimulation()
            time.sleep(1 / 240)
            last_joint_angles = joint_angles
        return True

    def move_joint_interpolation(self, target_T, steps=50, speed=0.05):
        """
        在关节空间中插值移动到目标末端位姿。
        仅计算一次逆运动学，并在关节角之间线性插值。
        """
        current_joints = np.array(self.get_joint_angles())
        # 计算目标关节角（只求一次IK）
        target_pos = target_T[:3, 3]
        target_quat = R.from_matrix(target_T[:3, :3]).as_quat()
        target_joints = ur_ik.ur_numerical_ik(
            target_pos, target_quat, current_joints, fk_func=ur_ik.ur10_forward
        )
        if target_joints is None:
            print("[ERROR] IK failed for target pose.")
            return False
        target_joints = np.array(target_joints)
        # 插值执行（可扩展为s型轨迹或带加速度约束的轨迹）
        for t in range(1, steps + 1):
            alpha = t / steps
            interp_joints = (1 - alpha) * current_joints + alpha * target_joints
            self.move_joints(interp_joints, speed=speed)
            pb.stepSimulation()
            time.sleep(1 / 240)
        return True

    def move_ik(self, T_world_tool0, speed=0.02, timeout=3):
        current_angles = self.get_joint_angles()
        # current_angles[3] = -2.5
        R_mat = T_world_tool0[:3, :3]
        rot = R.from_matrix(R_mat)
        target_quat = rot.as_quat()  # 四元数格式为 (x, y, z, w)
        # 提取位置
        target_pos = T_world_tool0[:3, 3]
        start_time = time.time()
        joint_angles = ur_ik.ur_numerical_ik(
                fk_func=ur_ik.ur10_forward,
                target_pos=target_pos,
                target_quat=target_quat,
                init_joint_angles=current_angles
                # elbow_preference_axis = elbow_preference_axis,
                # elbow_weight=1.0  # 可调整为 0.5~2.0，看效果
            )
        # === 结束计时 ===
        elapsed_time = time.time() - start_time
        return self.move_joints(joint_angles, speed=speed, timeout=timeout)

    def move_ik_analytic_init(self, T_world_tool0, speed=0.02, timeout=3):
        current_angles = self.get_joint_angles()
        # current_angles[3] = -2.5
        R_mat = T_world_tool0[:3, :3]
        rot = R.from_matrix(R_mat)
        target_quat = rot.as_quat()  # 四元数格式为 (x, y, z, w)
        # 提取位置
        target_pos = T_world_tool0[:3, 3]
        start_time = time.time()
        joint_angles = ur_ik.ur_numerical_ik_optimized(
                fk_func=ur_ik.ur10_forward,
                target_pos=target_pos,
                target_quat=target_quat,
                transform_matrix=T_world_tool0,
                init_joint_angles=current_angles
                # elbow_preference_axis = elbow_preference_axis,
                # elbow_weight=1.0  # 可调整为 0.5~2.0，看效果
            )
        # === 结束计时 ===
        elapsed_time = time.time() - start_time
        print(f"[IK] 解算用时: {elapsed_time:.4f} 秒")
        print("[IK] 结果关节角:", joint_angles)
        return self.move_joints(joint_angles, speed=speed, timeout=timeout)

    def move_joints(self, target_joints, speed=0.1, timeout=3, tol=0.001):
        target_joints = np.array(target_joints)
        t0 = time.time()
        while (time.time() - t0) < timeout:
            current_joints = np.array(self.get_joint_angles())
            diff = target_joints - current_joints
            abs_diff = np.abs(diff)

            if np.all(abs_diff < tol):
                for _ in range(10):
                    pb.stepSimulation(physicsClientId=self.client_id)
                return True

            if np.all(abs_diff < 0.005):
                pb.setJointMotorControlArray(
                    bodyIndex=self.robot_id,
                    jointIndices=self.joint_indices,
                    controlMode=pb.POSITION_CONTROL,
                    targetPositions=target_joints,
                    positionGains=[1.5] * len(self.joint_indices),
                    forces=[1500] * len(self.joint_indices),
                    physicsClientId=self.client_id
                )
                for _ in range(10):
                    pb.stepSimulation(physicsClientId=self.client_id)
                    time.sleep(1 / 1800)
                return True

            factor = min(1.0, np.max(abs_diff) / (tol * 10))
            step = diff * speed * factor
            next_joints = current_joints + step

            pb.setJointMotorControlArray(
                bodyIndex=self.robot_id,
                jointIndices=self.joint_indices,
                controlMode=pb.POSITION_CONTROL,
                targetPositions=next_joints,
                positionGains=[1.2] * len(self.joint_indices),
                forces=[1000] * len(self.joint_indices),
                physicsClientId=self.client_id
            )
            pb.stepSimulation(physicsClientId=self.client_id)
            time.sleep(1 / 1800)

        print("[WARNING] move_joints timeout. Resetting to initial pose...")
        if hasattr(self, "init_joint_angles"):
            self.move_joints_once(self.init_joint_angles, speed=0.02)

        return False

    def move_joints_once(self, target_joints, speed=0.1):
        """
        仅发送一次目标关节角命令，无反馈控制。
        """
        target_joints = np.array(target_joints)
        pb.setJointMotorControlArray(
            bodyIndex=self.robot_id,
            jointIndices=self.joint_indices,
            controlMode=pb.POSITION_CONTROL,
            targetPositions=target_joints,
            positionGains=[1.2] * len(self.joint_indices),
            forces=[1000] * len(self.joint_indices),
            physicsClientId=self.client_id
        )
        # 可选地模拟几步让动作生效
        for _ in range(10):
            pb.stepSimulation(physicsClientId=self.client_id)
            time.sleep(1 / 240)  # 你也可以写更高的仿真频率，如 1/1800
        return True

    def solve_ik(self, T, current_pose):
        R_mat = T[:3, :3]
        rot = R.from_matrix(R_mat)
        quat = rot.as_quat()  # 四元数格式为 (x, y, z, w)
        # 提取位置
        pos = T[:3, 3]

        joints = pb.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.ee_link_index,
            targetPosition=pos,
            targetOrientation=quat,
            lowerLimits=[-6.283, -6.283, -3.141, -6.283, -6.283, -6.283],
            upperLimits=[6.283, 6.283, 3.141, 6.283, 6.283, 6.283],
            jointRanges=[12.566, 12.566, 6.282, 12.566, 12.566, 12.566],
            restPoses=current_pose,
            maxNumIterations=100,
            residualThreshold=1e-5,
        )
        joints = np.array(joints, dtype=np.float32)
        for i in range(len(joints)):
            if joints[i] > 2 * np.pi:
                joints[i] -= 2 * np.pi
            elif joints[i] < -2 * np.pi:
                joints[i] += 2 * np.pi
        return joints

    def move_q_solutions(self, T_world_tool0, speed=0.01, timeout=3):
        current_angles = self.get_joint_angles()
        current_angles = self.get_joint_angles()
        current_angles[3] = -100
        joint_angles, q_sols = ur_ik.ur10_inverse_matrix_test(T_world_tool0, current_angles)
        return q_sols

    def check_motion_collision(self, table_id, obj_id):
        """
        检查机械臂是否与桌面碰撞，如果碰撞则清理传入的 obj_ids 列表
        :param table_id: 桌子 ID
        :param obj_ids: 需要清理的物体 ID 列表
        :return: True 表示发生碰撞并已清理，False 表示无碰撞
        """
        collision_flag = False
        if self.check_collision(table_id):
            print("[COLLISION] 机械臂与桌面发生碰撞，轨迹判为失败")
            self.state_history.clear()
            try:
                pb.removeBody(obj_id)
            except Exception as e:
                print(f"[WARN] 移除物体 {obj_id} 失败: {e}")
            collision_flag = True
        if self.check_collision(obj_id) and self.task_success_flag == False:
            self.task_success_flag = True
        return collision_flag

    def check_motion_collision_one_obj(self, table_id, block_id):
        collision_flag = False
        if self.check_collision(table_id):
            print("[COLLISION] 机械臂与桌面发生碰撞2，轨迹判为失败")
            pb.removeBody(block_id)
            collision_flag = True
            if self.state_record_flag:
                self.state_history.clear()
                print('检测到碰撞 清除状态记录')
            return collision_flag
        return collision_flag
#
# def get_pad_contact_depths(robot_id, block_id, pad_link_indices, client_id=0):
#     """
#     获取夹爪两个 pad 与物体的第一个接触点的接触深度。
#     同时返回与两个 pad 相关的接触点数量总和。
#
#     参数：
#         robot_id (int): 夹爪的 body ID
#         block_id (int): 目标物体的 body ID
#         pad_link_indices (list[int]): 两个 pad 的 link 索引
#         client_id (int): PyBullet 客户端 ID
#
#     返回：
#         tuple(dict, int):
#             - result: {pad_link_index: contact_depth or None}
#             - contact_count: 与两个 pad 相关的接触点总数
#     """
#     import pybullet as pb
#     result = {pad_link_indices[0]: None, pad_link_indices[1]: None}
#     recorded = set()
#     contact_count = 0
#     contacts = pb.getContactPoints(bodyA=robot_id, bodyB=block_id, physicsClientId=client_id)
#     for contact in contacts:
#         linkA = contact[2]
#         linkB = contact[3]
#         depth = contact[9]
#
#         for link in [linkA, linkB]:
#             if link in pad_link_indices:
#                 contact_count += 1
#                 if link not in recorded:
#                     result[link] = depth
#                     recorded.add(link)
#
#         if len(recorded) == 2:
#             # 已记录两个 pad 的第一个接触点，但要继续统计 contact_count
#             continue
#
#     return result, contact_count
#
#
#
import sys
import contextlib
import os

@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as fnull:
        # 低层文件描述符
        fd_stdout = sys.stdout.fileno()
        fd_stderr = sys.stderr.fileno()

        # 保存原始文件描述符
        saved_stdout = os.dup(fd_stdout)
        saved_stderr = os.dup(fd_stderr)

        try:
            os.dup2(fnull.fileno(), fd_stdout)
            os.dup2(fnull.fileno(), fd_stderr)
            yield
        finally:
            os.dup2(saved_stdout, fd_stdout)
            os.dup2(saved_stderr, fd_stderr)
            os.close(saved_stdout)
            os.close(saved_stderr)
#
def load_object_vhacd_mesh(
    path, transform_matrix, mass=0.5, scale=[1, 1, 1],
    lateral_friction=0.7, rolling_friction=0.001, spinning_friction=0.001, baseInertialFramePosition = [0,0,0],table_z=0.24
):
    """
    优化版：只对每个 mesh 路径运行一次 VHACD，之后复用。
    """
    if mass == 0:
        transform_matrix[2,3]=table_z
    transform_matrix = np.array(transform_matrix).reshape((4, 4))
    position = transform_matrix[:3, 3].tolist()
    rotation_matrix = transform_matrix[:3, :3]
    orientation_quat = R.from_matrix(rotation_matrix).as_quat()  # xyzw

    # === 基于路径构建 VHACD 缓存文件名 ===
    path_abs = os.path.abspath(path)
    hash_str = hashlib.md5(path_abs.encode()).hexdigest()[:8]
    output_dir = os.path.join(os.getcwd(), "vhacd_cache")
    os.makedirs(output_dir, exist_ok=True)

    vhacd_output = os.path.join(output_dir, f"{hash_str}_vhacd.obj")
    log_file = os.path.join(output_dir, f"{hash_str}_vhacd_log.txt")

    # === 若未生成则执行 VHACD ===
    if not os.path.exists(vhacd_output):
        with suppress_stdout_stderr():
            pb.vhacd(path, vhacd_output, log_file, concavity=1, pca=True)

    # === 创建视觉形状和碰撞形状 ===
    visual_shape_id = pb.createVisualShape(
        shapeType=pb.GEOM_MESH,
        fileName=path,
        meshScale=scale,
        # rgbaColor=[0.7, 0.6, 0.5, 1.0]  # 更浅的木色，亮度高一些 [0.6, 0.5, 0.4, 1.0]
    )
    collision_shape_id = pb.createCollisionShape(
        shapeType=pb.GEOM_MESH,
        fileName=vhacd_output,
        meshScale=scale
    )

    # === 创建物体 ===
    body_id = pb.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=position,
        baseOrientation=orientation_quat.tolist(),
        baseInertialFramePosition = baseInertialFramePosition
    )
    # body_id = pb.createMultiBody(
    #     baseMass=mass,
    #     baseCollisionShapeIndex=collision_shape_id,
    #     baseVisualShapeIndex=visual_shape_id,
    #     basePosition=position,
    #     baseOrientation=orientation_quat.tolist(),
    #     baseInertialFramePosition=baseInertialFramePosition
    # )
    #
    # # === 禁用碰撞（可选，确保不参与任何碰撞）===
    # pb.setCollisionFilterGroupMask(body_id, -1, 0, 0)

    # === 设置摩擦参数 ===
    pb.changeDynamics(
        bodyUniqueId=body_id,
        linkIndex=-1,
        lateralFriction=lateral_friction,
        rollingFriction=rolling_friction,
        spinningFriction=spinning_friction,
        contactStiffness=10000,
        contactDamping=1,
    )
    if mass == 0:
        # 可选：禁用物体响应，保留检测
        pb.changeDynamics(body_id, -1, contactProcessingThreshold=0)
        # 可选：开启组过滤（默认 1,1）
        pb.setCollisionFilterGroupMask(body_id, -1, 1, 1)


    return body_id
#


# def load_mesh_object_quat(path, position, orientation_quat, mass=0.5, scale=[1, 1, 1]):
#     visual_shape_id = pb.createVisualShape(
#         shapeType=pb.GEOM_MESH,
#         fileName=path,
#         meshScale=scale
#     )
#     collision_shape_id = pb.createCollisionShape(
#         shapeType=pb.GEOM_MESH,
#         fileName=path,
#         meshScale=scale
#     )
#     body_id = pb.createMultiBody(
#         baseMass=mass,
#         baseCollisionShapeIndex=collision_shape_id,
#         baseVisualShapeIndex=visual_shape_id,
#         basePosition=position,
#         baseOrientation=orientation_quat
#     )
#     return body_id
#
# def load_object_fixed_and_convex(
#     fixed_obj_path, convex_obj_path, transform_matrix0,
#     mass=0.5, scale=[1, 1, 1],
#     lateral_friction=0.9, rolling_friction=0.001, spinning_friction=0.001,
#     baseInertialFramePosition=[0, 0, 0], table_z=0.30,color=[0.7, 0.85, 1.0, 1.0]
# ):
#     """
#     加载已处理模型：
#       fixed_obj_path  - 清洗后的 OBJ（visual，用原材质/纹理）
#       convex_obj_path - 凸包 OBJ（collision，物理碰撞用）
#     """
#
#     # === 如果是静态物体（质量=0），把位置抬到桌面上 ===
#     transform_matrix = transform_matrix0.copy()
#     if mass == 0:
#         transform_matrix[2, 3] = table_z
#
#     transform_matrix = np.array(transform_matrix).reshape((4, 4))
#     # transform_matrix[2, 3] = table_z
#     position = transform_matrix[:3, 3].tolist()
#     rotation_matrix = transform_matrix[:3, :3]
#     orientation_quat = R.from_matrix(rotation_matrix).as_quat()  # xyzw
#
#     # === 创建视觉形状（保留原 MTL 纹理） ===
#     visual_shape_id = pb.createVisualShape(
#         shapeType=pb.GEOM_MESH,
#         fileName=fixed_obj_path,
#         meshScale=scale,
#         # rgbaColor=[0.9, 0.9, 0.9, 1.0]# 不传 rgbaColor，这样 PyBullet 会读取 OBJ 对应的 MTL 材质和纹理
#         rgbaColor=color
#     )
#
#
#     # === 创建碰撞形状（凸包版本） ===
#     collision_shape_id = pb.createCollisionShape(
#         shapeType=pb.GEOM_MESH,
#         fileName=convex_obj_path,
#         meshScale=scale
#     )
#
#     # === 创建物体 ===
#     body_id = pb.createMultiBody(
#         baseMass=mass,
#         baseCollisionShapeIndex=collision_shape_id,
#         baseVisualShapeIndex=visual_shape_id,
#         basePosition=position,
#         baseOrientation=orientation_quat.tolist(),
#         baseInertialFramePosition=baseInertialFramePosition
#     )
#
#     # 覆盖颜色（这一步会忽略掉 mtl 纹理）
#     pb.changeVisualShape(body_id, -1, rgbaColor=color)
#
#     # === 设置摩擦 & 接触属性 ===
#     pb.changeDynamics(
#         bodyUniqueId=body_id,
#         linkIndex=-1,
#         lateralFriction=lateral_friction,
#         rollingFriction=rolling_friction,
#         spinningFriction=spinning_friction,
#         contactStiffness=1000000,
#         contactDamping=1
#     )
#
#     if mass == 0:
#         # 静态物体可选：禁用动力学响应但保留检测
#         pb.changeDynamics(body_id, -1, contactProcessingThreshold=0)
#         pb.setCollisionFilterGroupMask(body_id, -1, 1, 1)
#
#     return body_id
#

#
#

#
#
def setup_gripper_mimic_constraints_v3(gripper_id, client_id, finger_joint_index):
    """
    给 robotiq gripper 添加 mimic gear 约束，只控制一个 finger_joint，其余关节跟随。
    适配 UR10 使用。
    """
    def set_gripper_friction(gripper_id, client_id, friction=5.0):
        for i in range(pb.getNumJoints(gripper_id, physicsClientId=client_id)):
            joint_info = pb.getJointInfo(gripper_id, i, physicsClientId=client_id)
            joint_name = joint_info[1].decode("utf-8")
            # print(joint_name)
            if "inner_finger_pad" in joint_name or "finger_tip" in joint_name:
                pb.changeDynamics(
                    gripper_id,
                    i,
                    lateralFriction=0,
                    rollingFriction=0.1,
                    spinningFriction=0.1,
                    physicsClientId=client_id,
                )
                # print(f"[INFO] 设置摩擦力: {joint_name} linkIndex={i}")

    # 1. 获取所有 joint 名称映射
    name_to_index = {
        pb.getJointInfo(gripper_id, i, physicsClientId=client_id)[1].decode("utf-8"): i
        for i in range(pb.getNumJoints(gripper_id, physicsClientId=client_id))
    }

    # 2. 定义 mimic joints
    mimic_pairs = {
        "left_inner_finger_joint": 1,
        "left_inner_knuckle_joint": -1,
        "right_outer_knuckle_joint": -1,
        "right_inner_finger_joint": 1,
        "right_inner_knuckle_joint": -1,
    }

    # 3. 设置 mimic gear constraint
    for name, gear in mimic_pairs.items():
        if name not in name_to_index:
            print(f"[warn] 未找到 mimic joint: {name}")
            continue
        mimic_index = name_to_index[name]
        c = pb.createConstraint(
            parentBodyUniqueId=gripper_id,
            parentLinkIndex=finger_joint_index,
            childBodyUniqueId=gripper_id,
            childLinkIndex=mimic_index,
            jointType=pb.JOINT_GEAR,
            jointAxis=[0, 0, 1],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        pb.changeConstraint(c, gearRatio=gear, erp=0.9, maxForce=5000)
    # 3.5 设置指尖 pad 的平行运动约束（可选增强）
    pad_joint_names = [
        "left_inner_finger_pad_joint",
        "right_inner_finger_pad_joint"
    ]

    if all(name in name_to_index for name in pad_joint_names):
        left_pad_index = name_to_index[pad_joint_names[0]]
        right_pad_index = name_to_index[pad_joint_names[1]]

        # 添加 gear 约束使得左右指尖平行移动（同步夹紧）
        c = pb.createConstraint(
            parentBodyUniqueId=gripper_id,
            parentLinkIndex=left_pad_index,
            childBodyUniqueId=gripper_id,
            childLinkIndex=right_pad_index,
            jointType=pb.JOINT_GEAR,
            jointAxis=[1, 1, 1],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
            physicsClientId=client_id,
        )
        pb.changeConstraint(c, gearRatio=-1, erp=0.9, maxForce=2000)

    # 4. 额外连接左右夹爪：主控 -> right_outer_knuckle_joint
    if "right_outer_knuckle_joint" in name_to_index:
        c = pb.createConstraint(
            parentBodyUniqueId=gripper_id,
            parentLinkIndex=finger_joint_index,
            childBodyUniqueId=gripper_id,
            childLinkIndex=name_to_index["right_outer_knuckle_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[0, 1, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
            physicsClientId=client_id,
        )
        pb.changeConstraint(c, gearRatio=-1, erp=0.9, maxForce=3000)

    # 5. 设置摩擦力
    set_gripper_friction(gripper_id, client_id)
    pb.changeDynamics(
        bodyUniqueId=gripper_id,
        linkIndex=18,  # 示例，指尖 link index
        lateralFriction=1.0,
        rollingFriction=0.003,
        spinningFriction=0.003,
        contactStiffness=10000,
        contactDamping=1
    )
    set_gripper_friction(gripper_id, client_id)
    pb.changeDynamics(
        bodyUniqueId=gripper_id,
        linkIndex=13,  # 示例，指尖 link index
        lateralFriction=1.0,
        rollingFriction=0.003,
        spinningFriction=0.003,
        contactStiffness=10000,
        contactDamping=1
    )





