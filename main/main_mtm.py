import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import entry
import sympy
from collections import namedtuple
from utils import utils
from numpy import deg2rad

########################################################################################################################
#---------------------------------------------------- 模型定义 ---------------------------------------------------------#
########################################################################################################################
model_name = 'mtm'
q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10 = utils.new_sym('q:11')
pi = sympy.pi

qmd1 = q1
qmd2 = q2
qmd30 = -q2 + q3
qmd31 = q3
qmd32 = q2 - q3
qmd33 = q3
qmd4 = 0.6697*q2 - 0.6697*q3 + q4
qmd5 = q5
qmd6 = q6
qmd7 = q7

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
q_o = 23.0/180.0*pi
l = sympy.sqrt(r_s**2 + h_s**2 - 2*r_s*h_s*sympy.cos(pi + q_o - q))
d_l = l - l_r
r_f = r_s * h_s * sympy.sin(pi + q_o - q) / l
dl5 = r_f * d_l

# Joint number | prev link | succ links | a | alpha | d | theta | link inertia | motor inertia | friction | spring
dh = [( L_b,   -1,   [L_1, M_4],      0,      0,       0,           0, False,  False, False, dlN),
      ( L_1,  L_b,  [L_2, L_31],      0,      0,  -l_b2p,        qmd1,  True,  False,  True, dlN),
      ( L_2,  L_1,       [L_30],      0,  -pi/2,       0,   qmd2+pi/2,  True,  False,  True, dlN),
      (L_30,  L_2,        [L_4],  l_arm,      0,       0,  qmd30+pi/2,  True,  False,  True, dlN),
      (L_31,  L_1,       [L_32],      0,  -pi/2,       0,    qmd31+pi,  True,  False,  True, dlN),
      (L_32, L_31,           [],  l_b2f,      0,       0,  qmd32-pi/2,  True,  False,  True, dlN),
      ( L_4, L_30,        [L_5],   l_fa,  -pi/2,   0.151,        qmd4,  True,  False,  True, dlN),
      ( L_5,  L_4,        [L_6],       0,  pi/2,       0,        qmd5,  True,  False,  True, dl5),
      ( L_6,  L_5,        [L_7],       0, -pi/2,       0,   qmd6+pi/2,  True,  False,  True, dlN),
      ( L_7,  L_6,           [],       0, -pi/2,       0,     qmd7+pi,  True,  False,  True, dlN),
      ( M_4,  L_b,           [],       0,     0,       0,          q4, False,   True,  True, dlN)]

friction_type = ['coulomb', 'viscous', 'offset']
dh_method = 'mdh'

Robot = namedtuple('Robot', ['name_', 'dh_', 'dh_convention_', 'friction_type_'])
robot = Robot(name_=model_name, dh_=dh, dh_convention_=dh_method, friction_type_=friction_type)

########################################################################################################################
#---------------------------------------------------- 激励轨迹 ---------------------------------------------------------#
########################################################################################################################
trajectory_name = 'three_order_fourier_traj'
optimal_traj_folder = '/data/' + model_name + '/optimal_traj/'
base_freq = 0.1
fourier_order = 6
control_freq = 200
cartesian_constraints = []
joint_constraints = [(qmd1,  deg2rad(-57),  deg2rad(29),  deg2rad(-160), deg2rad(160), deg2rad(-1600), deg2rad(1600)),
                     (qmd2,  deg2rad(-10),  deg2rad(60),  deg2rad(-180), deg2rad(180), deg2rad(-1800), deg2rad(1800)),
                     (qmd30, deg2rad(-30),  deg2rad(30),  deg2rad(-180), deg2rad(180), deg2rad(-1800), deg2rad(1800)),
                     (qmd4,  deg2rad(-40),  deg2rad(195), deg2rad(-360), deg2rad(360), deg2rad(-3600), deg2rad(3600)),
                     (qmd5,  deg2rad(-87),  deg2rad(180), deg2rad(-360), deg2rad(360), deg2rad(-3600), deg2rad(3600)),
                     (qmd6,  deg2rad(-40),  deg2rad(38),  deg2rad(-360), deg2rad(360), deg2rad(-3600), deg2rad(3600)),
                     (qmd7,  deg2rad(-460), deg2rad(450), deg2rad(-720), deg2rad(720), deg2rad(-7200), deg2rad(7200)),
                     (qmd31, deg2rad(-9),   deg2rad(39),  deg2rad(-360), deg2rad(360), deg2rad(-3600), deg2rad(3600))]

Excitation_Traj = namedtuple('Excitation_Traj',
                             ['traj_name_', 'traj_folder_', 'base_freq_', 'fourier_order_',
                              'control_freq_', 'joint_constraints_', 'cartesian_constraints_'])
trajectory = Excitation_Traj(traj_name_=trajectory_name,
                             traj_folder_=optimal_traj_folder,
                             base_freq_=base_freq,
                             fourier_order_=fourier_order,
                             control_freq_=control_freq,
                             joint_constraints_=joint_constraints,
                             cartesian_constraints_=cartesian_constraints)

########################################################################################################################
#---------------------------------------------------- 采样数据处理 ------------------------------------------------------#
########################################################################################################################

sample_traj_name = 'two_results'
sample_freq = 200                                                       # 数据采样频率(周期性采样，设置和激励轨迹中的控制频率相同)
cutoff_freq = 5 * trajectory.base_freq_ * trajectory.fourier_order_     # 低通滤波器截止频率
cut_num = 200                                                           # 数据掐头去尾(去掉前后{cut_num}个数据)
filter_order = 6

# 定义回调函数，从文件中读取数据后，根据模型要求，预先对数据进行处理
def data_pre_process_callback(pre_q, pre_dq, pre_tau):
    return pre_q, pre_dq, pre_tau

Sample_Data_Process = namedtuple('Sample_Data_Process',[
    'sample_traj_name_',
    'sample_freq_',
    'cutoff_freq_',
    'filter_order_',
    'cut_num_',
    'callback_'])
sample_data = Sample_Data_Process(
    sample_traj_name_ = sample_traj_name,
    sample_freq_=sample_freq,
    cutoff_freq_=cutoff_freq,
    filter_order_=filter_order,
    cut_num_=cut_num,
    callback_=data_pre_process_callback)

########################################################################################################################
#------------------------------------------------- 辨识策略 ------------------------------------------------------------#
########################################################################################################################

# 是否需要重新生成Wb (如果机器人模型未变，采样数据未变，求解器未变，Wb存在的情况下，无需重新生成Wb)
gen_regressor = False
# 求解器定义，可选的求解器有
# --OLS (Ordinary Least Square)
# --WLS (Wight Least Square)
# --CVX (Convex Optimization)
solver = 'CVX'
# 定义回调函数，根据需要处理辨识前后的数据
def iden_res_callback(filt_q, filt_dq, filt_tau, iden_tau):
    return filt_q, filt_dq, filt_tau, iden_tau

Iden = namedtuple('Iden', ['gen_regressor_', 'solver_', 'callback_'])
iden = Iden(gen_regressor_=gen_regressor, solver_=solver, callback_=iden_res_callback)

########################################################################################################################
#------------------------------------------------- 选项配置 ------------------------------------------------------------#
########################################################################################################################
Config = namedtuple('Config', ['load_kinematic_from_file_', 'load_dynamic_from_file_',
                               'create_robot_model_', 'design_excitation_traj_', 'sample_data_process_',
                               'dynamics_identification_'])

config = Config(load_kinematic_from_file_=False,
                load_dynamic_from_file_=False,
                create_robot_model_=True,
                design_excitation_traj_=True,
                sample_data_process_=True,
                dynamics_identification_=True)

# ------------------------- 运行 ----------------------------------------------------------------------------------------

entry.run(robot, trajectory, sample_data, iden, config)
