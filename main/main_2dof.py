import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import entry
import sympy
from collections import namedtuple
from utils import utils
from numpy import deg2rad

# ------------------------- 模型定义 ------------------------------------------------------------------------------------

model_name = '2dof'          # 模型名称
model_folder = '/data/' + model_name + '/model/'        # 模型保存路径

q0, q1 = utils.new_sym('q:2')
pi = sympy.pi

# define link number
L_b = 0
L_1 = 1
L_2 = 2

# DH
# Joint number | prev link | succ links | a | alpha | d | theta | link inertia | motor inertia | friction | spring
dh = [(L_b,  -1, L_1,  0,     0, 0,       0, False, False, False, None),
      (L_1, L_b, L_2,  0, -pi/2, 0, q0+pi/2,  True, False, False, None),
      (L_2, L_1, [ ],  0,  pi/2, 0, q1-pi/2,  True, False, False, None)]


# Friction
friction_type = ['coulomb', 'viscous']

Robot = namedtuple('Robot', ['name_', 'model_folder_', 'dh_', 'dh_convention_', 'friction_type_'])
robot = Robot(name_=model_name, model_folder_=model_folder, dh_=dh, dh_convention_='sdh', friction_type_=friction_type)

# ------------------------- 激励轨迹定义 ---------------------------------------------------------------------------------

trajectory_name = 'three_order_fourier_traj'
base_freq = 0.1
fourier_order = 6
cartesian_constraints = []
joint_constraints = []

Excitation_Traj = namedtuple('Excitation_Traj',
                             ['traj_name_', 'base_freq_', 'fourier_order_',
                              'joint_constraints_', 'cartesian_constraints_'])
trajectory = Excitation_Traj(traj_name_=trajectory_name,
                             base_freq_=base_freq,
                             fourier_order_=fourier_order,
                             joint_constraints_=joint_constraints,
                             cartesian_constraints_=cartesian_constraints)

############################################################################
#----------------------------- 采样数据处理 ---------------------------------#
############################################################################

sample_traj_name = 'two_results'
sample_freq = 200   # 数据采样频率
cutoff_freq = 5 * trajectory.base_freq_ * trajectory.fourier_order_  # 低通滤波器截止频率
cut_num = 200       # 数据掐头去尾
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

############################################################################
#------------------------------ 辨识策略 -----------------------------------#
############################################################################

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

############################################################################
#------------------------------ 选项配置 -----------------------------------#
############################################################################
Config = namedtuple('Config', ['load_kinematic_from_file_', 'load_dynamic_from_file_',
                               'create_robot_model_', 'design_excitation_traj_', 'sample_data_process_',
                               'dynamics_identification_', 'code_generation_'])

config = Config(load_kinematic_from_file_=False,
                load_dynamic_from_file_=False,
                create_robot_model_=False,
                code_generation_=False,
                design_excitation_traj_=True,
                sample_data_process_=False,
                dynamics_identification_=False)

# ------------------------- 运行 ----------------------------------------------------------------------------------------

entry.run(robot, trajectory, sample_data, iden, config)
