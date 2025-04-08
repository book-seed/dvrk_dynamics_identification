import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import entry
import sympy
from collections import namedtuple
from utils import utils
from numpy import deg2rad

# ------------------------- 模型定义 ------------------------------------------------------------------------------------

model_name = 'mtm'          # 模型名称
model_folder = '/data/' + model_name + '/model/'        # 模型保存路径

q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10 = utils.new_sym('q:11')
_pi = sympy.pi

# define paralelogram coordinate relation
# qd -> coordinate for dvrk_ros package
# qmd -> coordinate for the modeling joints
# q -> coordinate for motors
qd2 = q2
qd3 = -q2 + q3
qd4 = 0.6697*q2 - 0.6697*q3 + q4

qmd1 = q1
qmd2 = qd2
qmd30 = qd3
qmd31 = qd3 + qd2
qmd32 = -qd3
qmd33 = q3
qmd4 = qd4
qmd5 = q5
qmd6 = q6
qmd7 = q7

# q31 = q3 + q2
# q32 = -q3

l_b2p = 215.4 * 0.001
l_arm = 279.4 * 0.001
l_b2f = 100 * 0.001
l_fa = 364.5 * 0.001
h =105.6 * 0.001

# define link number
L_b = 0
L_1 = 1
L_2 = 2
L_30 = 3
L_31 = 4
L_32 = 5
L_4 = 6
L_5 = 7
L_6 = 8
L_7 = 9
M_4 = 10

# define spring delta L
dlN = None

q = qmd5
r_s = 0.0075
h_s = 0.1035
l_r = 0.0613
q_o = 23.0/180.0*_pi
l = sympy.sqrt(r_s**2 + h_s**2 - 2*r_s*h_s*sympy.cos(_pi + q_o - q))
d_l = l - l_r
r_f = r_s * h_s * sympy.sin(_pi + q_o - q) / l
dl5 = r_f * d_l # * 27.86

# DH
dh = [(L_b,  -1,   [L_1, M_4],  0,     0,      0,       0,           False, False, False, dlN),
      (L_1,  L_b,  [L_2, L_31], 0,     0,      -l_b2p,  qmd1,        True,  False, True, dlN),
      (L_2,  L_1,  [L_30],      0,     -_pi/2, 0,       qmd2+_pi/2,  True,  False, True, dlN),
      (L_30, L_2,  [L_4],       l_arm, 0,      0,       qmd30+_pi/2, True,  False, True, dlN),
      (L_31, L_1,  [L_32],      0,     -_pi/2, 0,       qmd31+_pi,   True,  False, True, dlN),
      (L_32, L_31, [],          l_b2f, 0,      0,       qmd32-_pi/2, True,  False, True, dlN),
      (L_4,  L_30, [L_5],       l_fa,  -_pi/2, 0.151,   qmd4,        True,  False, True, dlN),
      (L_5,  L_4,  [L_6],       0,     _pi/2,  0,       qmd5,        True,  False, True, dl5),
      (L_6,  L_5,  [L_7],       0,     -_pi/2, 0,       qmd6+_pi/2,  True,  False, True, dlN),
      (L_7,  L_6,  [],          0,     -_pi/2, 0,       qmd7+_pi,    True,  False, True, dlN),
      (M_4,  L_b,  [],          0,     0,      0,       q4,          False, True,  True, dlN)]

# Friction
friction_type = ['coulomb', 'viscous', 'offset']

Robot = namedtuple('Robot', ['name_', 'model_folder_', 'dh_', 'dh_convention_', 'friction_type_'])
robot = Robot(name_=model_name, model_folder_=model_folder, dh_=dh, dh_convention_='mdh', friction_type_=friction_type)

# ------------------------- 激励轨迹定义 ---------------------------------------------------------------------------------

trajectory_name = 'three_order_fourier_traj'

optimal_traj_folder = '/data/' + model_name + '/optimal_traj/'

base_freq = 0.1

fourier_order = 6

cartesian_constraints = []

joint_constraints = [(qmd1,  deg2rad(-57),  deg2rad(29),  deg2rad(-160), deg2rad(160), deg2rad(-1000), deg2rad(1000)),
                     (qmd2,  deg2rad(-10),  deg2rad(60),  deg2rad(-180), deg2rad(180), deg2rad(-1000), deg2rad(1000)),
                     (qmd30, deg2rad(-30),  deg2rad(30),  deg2rad(-180), deg2rad(180), deg2rad(-1000), deg2rad(1000)),
                     (qmd4,  deg2rad(-40),  deg2rad(195), deg2rad(-360), deg2rad(360), deg2rad(-1000), deg2rad(1000)),
                     (qmd5,  deg2rad(-87),  deg2rad(180), deg2rad(-360), deg2rad(360), deg2rad(-1000), deg2rad(1000)),
                     (qmd6,  deg2rad(-40),  deg2rad(38),  deg2rad(-360), deg2rad(360), deg2rad(-1000), deg2rad(1000)),
                     (qmd7,  deg2rad(-460), deg2rad(450), deg2rad(-720), deg2rad(720), deg2rad(-1000), deg2rad(1000)),
                     (qmd31, deg2rad(-9),   deg2rad(39),  deg2rad(-360), deg2rad(360), deg2rad(-1000), deg2rad(1000))]

Excitation_Traj = namedtuple('Excitation_Traj',
                             ['traj_name_', 'traj_folder_', 'base_freq_', 'fourier_order_',
                              'joint_constraints_', 'cartesian_constraints_'])
trajectory = Excitation_Traj(traj_name_=trajectory_name, traj_folder_=optimal_traj_folder,
                             base_freq_=base_freq, fourier_order_=fourier_order,
                             joint_constraints_=joint_constraints, cartesian_constraints_=cartesian_constraints)

# ------------------------- 选项配置 ------------------------------------------------------------------------------------

config = [('load_kinematic_from_file', True),
          ('load_dynamic_from_file', True),
          ('load_robot_model_from_file', True)]

# ------------------------- 运行 ----------------------------------------------------------------------------------------

entry.run(robot, trajectory, config)
