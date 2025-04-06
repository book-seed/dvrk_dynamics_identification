import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import utils, robot_model_data
from model import robot_def
from kinematics import geometry
from dynamics import dynamics

import sympy

# 模型名称
model_name = 'mtm'

# 模型保存路径
model_folder = 'data/' + model_name + '/model/'

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


config = [('load_kinematic_from_file', True),
          ('load_dynamic_from_file', False)]



# run
print("\nstep1: Robot Define -----------------------------------------------------")
robot_def = robot_def.RobotDef(name=model_name, params=dh, dh_convention='mdh', friction_type=friction_type)

print("\nstep2: Create Kinematic Chain -------------------------------------------")
geom = geometry.Geometry(robot_def, config)

print("\nstep3: Create Dynamic Chain ---------------------------------------------")
dyn = dynamics.Dynamics(robot_def, geom, config)

print("\nstep4: Save Robot Model -------------------------------------------------")
robot_model = robot_model_data.RobotModel(dyn)
utils.save_data(model_folder, model_name, robot_model)
