import pybullet as pb
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

def get_link_pose(body, link):
    result = pb.getLinkState(body, link)
    return result[4], result[5]

def get_joint_angles(robot_id, joint_indices):
    """
    获取机器人指定关节的当前角度。

    参数：
        robot_id (int): 机器人在 PyBullet 中的 ID
        joint_indices (list[int]): 要读取的关节索引列表

    返回：
        np.ndarray: 当前每个关节的角度（弧度）
    """
    return np.array([
        pb.getJointState(robot_id, i)[0] for i in joint_indices
    ])

def solve_ik(robot_id, ee_link_index, pose, rest_pose):
    joints = pb.calculateInverseKinematics(
        bodyUniqueId=robot_id,
        endEffectorLinkIndex=ee_link_index,
        targetPosition=pose[0],
        targetOrientation=pose[1],
        lowerLimits=[-6.283, -6.283, -3.141, -6.283, -6.283, -6.283],
        upperLimits=[6.283, 6.283, 3.141, 6.283, 6.283, 6.283],
        jointRanges=[12.566, 12.566, 6.282, 12.566, 12.566, 12.566],
        restPoses=rest_pose.tolist(),
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

def move_joints(robot_id, joint_indices, target_joints, speed=0.01, timeout=3, tol=0.001, client_id=0):
    """Move robot to target joint configuration, switch to position control near target."""
    import time
    target_joints = np.array(target_joints)
    t0 = time.time()
    stable_count = 0

    while (time.time() - t0) < timeout:
        current_joints = np.array([
            pb.getJointState(robot_id, i, physicsClientId=client_id)[0]
            for i in joint_indices
        ])
        diff = target_joints - current_joints
        abs_diff = np.abs(diff)

        # 精度判断：完全收敛
        if np.all(abs_diff < tol):
            for _ in range(10):
                pb.stepSimulation(physicsClientId=client_id)
            return True

        # 若误差小于0.1，直接切换为目标位置控制并退出
        if np.all(abs_diff < 0.005):
            pb.setJointMotorControlArray(
                bodyIndex=robot_id,
                jointIndices=joint_indices,
                controlMode=pb.POSITION_CONTROL,
                targetPositions=target_joints,
                positionGains=[1.5] * len(joint_indices),
                forces=[1500] * len(joint_indices),
                physicsClientId=client_id
            )
            for _ in range(10):
                pb.stepSimulation(physicsClientId=client_id)
                time.sleep(1 / 1800)
            return True

        # 否则按比例推进一步
        factor = min(1.0, np.max(abs_diff) / (tol * 10))
        step = diff * speed * factor
        next_joints = current_joints + step

        pb.setJointMotorControlArray(
            bodyIndex=robot_id,
            jointIndices=joint_indices,
            controlMode=pb.POSITION_CONTROL,
            targetPositions=next_joints,
            positionGains=[1.2] * len(joint_indices),
            forces=[1000] * len(joint_indices),
            physicsClientId=client_id
        )

        pb.stepSimulation(physicsClientId=client_id)
        time.sleep(1 / 1800)

    print("Warning: move_joints timeout.")
    return False

def compute_joint_angles_from_tool_pose(target_pos, target_quat, robot_id, ee_link_index):
    rest_pose = np.array([0, -1.5, 1.5, -1.5, -1.57, 0], dtype=np.float32)
    T_wrist_tool0 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0.18],
        [0, 0, 0, 1],
    ])
    T_world_tool0 = np.eye(4)
    T_world_tool0[:3, :3] = R.from_quat(target_quat).as_matrix()
    T_world_tool0[:3, 3] = target_pos
    T_world_wrist = T_world_tool0 @ np.linalg.inv(T_wrist_tool0)
    wrist_pos = T_world_wrist[:3, 3]
    wrist_quat = R.from_matrix(T_world_wrist[:3, :3]).as_quat()
    joint_angles = solve_ik(robot_id, ee_link_index, (wrist_pos, wrist_quat), rest_pose)
    return joint_angles

def compute_joint_angles_from_tool_pose_matrix(T_world_tool0, robot_id, ee_link_index):
    rest_pose = np.array([0, -1.5, 1.5, -1.5, -1.57, 0], dtype=np.float32)
    T_wrist_tool0 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0.18],
        [0, 0, 0, 1],
    ])
    T_world_wrist = T_world_tool0 @ np.linalg.inv(T_wrist_tool0)
    wrist_pos = T_world_wrist[:3, 3]
    wrist_quat = R.from_matrix(T_world_wrist[:3, :3]).as_quat()
    joint_angles = solve_ik(robot_id, ee_link_index, (wrist_pos, wrist_quat), rest_pose)
    return joint_angles

