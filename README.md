# 机器人动力学参数辨识库

## 1. 简介
### 1.1 模型定义  
* SDH/MDH
* 摩擦力模型，当前仅支持库伦摩擦，粘滞摩擦，及定量偏置
* 可添加弹簧
### 1.2 运动学建模
* 正常计算，没啥说的
### 1.3 动力学建模
- 基于拉格朗日法的动力学方程
- MCG方程
- 动力学方程线性化
- 基于QR分解的最小参数集
- C/C++代码生成
### 1.4 激励轨迹设计
- 自定义级数的傅里叶激励轨迹设计
- 基于教学算法的激励轨迹设计（待完成）
- 李萨如激励轨迹设计（适用于静态辨识）（待完成）
### 1.5 采样数据处理
- 可自定义数据预处理
- 零相位差低通滤波器 (Zero-phase low-pass filter)
### 1.6 参数辨识
- 普通最小二乘法 (Ordinary Least Square, OLS)
- 基于权重的最小二乘法 (Weighted Least Square, WLS)
- 凸优化算法 (Convex Optimization)
- 辨识结果保存为CSV文件

## 2. 环境及依赖
* Anaconda
* Python 3.12
* Python库
    * NumPy, SymPy, Matplotlib, cloudpickle, Pandas, SciPy, CvxOpt, PyOpt, pathlib

## 3. 参考文献
[A convex optimization-based dynamic model identification package for the da Vinci Research Kit](https://ieeexplore.ieee.org/document/8758871)

~~~
@ARTICLE{8758871, 
author={Y. {Wang} and R. {Gondokaryono} and A. {Munawar} and G. S. {Fischer}}, 
journal={IEEE Robotics and Automation Letters}, 
title={A Convex Optimization-Based Dynamic Model Identification Package for the da Vinci Research Kit}, 
year={2019}, 
volume={4}, 
number={4}, 
pages={3657-3664}, 
keywords={Manipulator dynamics;Kinematics;Dynamics;Open source software;Couplings;Surgical Robotics: Laparoscopy;Dynamics;Calibration and Identification}, 
doi={10.1109/LRA.2019.2927947}, 
ISSN={2377-3766}, 
month={Oct},}
~~~

## 4. 参考开源库
- 本仓库fork了[dvrk_dynamics_identification](https://github.com/WPI-AIM/dvrk_dynamics_identification),在此基础上，新建了develop分支，develop分支的开发参考了大量源仓库的master分支代码。
- 另外参考的开源代码有[SymPyBotics](https://github.com/cdsousa/SymPyBotics)


