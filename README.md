# 🦾 Pybullet UR10 Grasp Mouse
**基于 PyBullet 的 UR10 机械臂抓取鼠标示例项目**

本项目提供一个完整的 PyBullet 仿真环境，展示 UR10 机械臂如何抓取鼠标并放置到鼠标垫上。  
结构清晰、注释完善，适合 **PyBullet 初学者 / 机械臂仿真学习 / 强化学习场景构建**。

详细讲解在CSDN博客中:https://blog.csdn.net/qq_53186375/article/details/154876227?fromshare=blogdetail&sharetype=blogdetail&sharerId=154876227&sharerefer=PC&sharesource=qq_53186375&sharefrom=from_link

---

## 📦 文件结构说明
```text
Pybullet_UR10_Grasp_Mouse/
├── test.py
├── ur10_pybullet_sim_env.py
├── assets/
│   ├── gripper/           # 加载 urdf 使用
│   ├── mouse_mesh/        # 鼠标 mesh 文件 
│   ├── mouse_pad_mesh/    # 鼠标垫 mesh 文件
│   ├── ur10/              # 机械臂 mesh 文件
│   ├── mouse_pad_pose.txt 
│   ├── mouse_pose.txt
│   ├── robot.urdf         # UR10机械臂带夹爪的 urdf 文件
│   ├── ur10_grasp_mouse_trajectory1.npy
│   ├── ur10_gripper_cmd1.npy
│   ├── ur10_gripper_cmd2.npy
│   └── ur10_place_mouse_trajectory2.npy
└── ur_pkg/
    ├── __init__.py
    ├── ur10_inverse.py
    ├── ur10_kin_inverse.py
    ├── ur10_move.py
    └── ur10_pybullet_inverse.py
```

### 📌 **主要文件说明**
---

#### **test.py**
- 项目的入口文件  
- 包含 `main()` 函数和辅助工具  
- 负责执行 **完整抓取任务流程**
---

#### **ur10_pybullet_sim_env.py**
PyBullet 环境配置类，包含：
- UR10 机械臂加载  
- 夹爪初始化与控制  
- 机械臂运动控制器  
- 鼠标 / 鼠标垫等物体加载  
---

#### **assets/**
包含任务需要的全部资源：

- UR10 + 夹爪 URDF  
- 鼠标、鼠标垫 OBJ 网格模型  
- 纹理文件  
- 预定义运动轨迹  
- 夹爪控制命令  
- 物体初始位姿配置  

---

#### **ur_pkg/**
包含机器人运动学相关模块：

- 数值逆运动学（Newton / Jacobian-based）  
- 正运动学（FK）  
- 工具函数  

---

# 🚀 使用说明（中文）

确保已安装 PyBullet 后，直接运行：

```bash
python3 test.py
``` 

# 🦾 PyBullet UR10 Grasp Mouse (English)

A clean, beginner-friendly PyBullet simulation project demonstrating how a **UR10 robot arm grasps a computer mouse and places it onto a mousepad**.  
This repository is designed for learners who want to understand:

- How to load and control a UR10 robot in PyBullet  
- How to load custom objects (mouse, mousepad, etc.)  
- How to control a gripper  
- How to build a simple manipulation task environment  

This project contains clear code structure, detailed comments, and modular design that makes it easy to extend for your own research, robotics experiments, or reinforcement-learning environments.

---

## 📁 Project Structure

```text
Pybullet_UR10_Grasp_Mouse/
├── test.py
├── ur10_pybullet_sim_env.py
├── assets/
│   ├── robot.urdf
│   ├── mouse.obj
│   ├── mousepad.obj
│   ├── textures/
│   ├── trajectories/
│   ├── gripper_cmd/
│   └── initial_pose/
└── ur_pkg/
    ├── forward_kinematics.py
    ├── inverse_kinematics.py
    └── utils/
``` 

yaml
复制代码


### 📌 File Descriptions

#### **test.py**
- The entry point of the project  
- Contains the `main()` function  
- Runs the entire grasping task  
- Includes helper utility functions  

#### **ur10_pybullet_sim_env.py**
Encapsulates the full PyBullet environment setup, including:

- Loading the UR10 robot arm  
- Loading and initializing the gripper  
- Motion controllers  
- Loading objects (mouse, mousepad, etc.)  
- Utility functions for reset / simulation steps  

#### **assets/**
Contains all the necessary resources for simulation:

- UR10 robot + gripper URDF  
- Task objects (mouse.obj, mousepad.obj)  
- Textures  
- Pre-defined robot trajectories  
- Gripper control command files  
- Initial pose configuration files  

#### **ur_pkg/**
Custom UR10 kinematics implementation:

- **Forward kinematics**  
- **Numerical inverse kinematics**  
- Helper utilities  

---

# 🚀 How to Use

This project requires only PyBullet.  
Once installed, simply run:

```bash
python3 test.py


