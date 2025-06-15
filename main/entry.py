from pathlib import Path
from kinematics import geometry
from dynamics import dynamics
from utils import utils
from model import robot_def, robot_model_data
from trajectory_optimization import traj_optimizer
from identification import data_processing, identification
from codegen import robotcode



def run(robot, trajectory, data, iden, config):

    # 是否需要计算或加载机器人模型
    request_robot_model_ = (config.create_robot_model_ or
                            config.code_generation_ or
                            config.design_excitation_traj_ or
                            config.dynamics_identification_)
    
    # 数据保存路径
    model_folder_ = Path.cwd().parent / 'data' / robot.name_ / 'model'
    sample_traj_folder_ = Path.cwd().parent / 'data' / robot.name_ / 'sample_trajectory'
    identification_folder_ = Path.cwd().parent / 'data' / robot.name_ / 'identification'
    codegen_folder_ = Path.cwd().parent / 'data' / robot.name_ / 'codegen'
    excitation_traj_folder_ = Path.cwd().parent / 'data' / robot.name_ / 'excitation_trajectory'

    if config.create_robot_model_ is True :

        print("\nstep1: Robot Define -----------------------------------------------------")
        robot_define = robot_def.RobotDef(
            name=robot.name_, params=robot.dh_, dh_convention=robot.dh_convention_,
            friction_type=robot.friction_type_)
        print("\nstep2: Create Kinematic Chain -------------------------------------------")
        geom = geometry.Geometry(robot_define, model_folder_, config.load_kinematic_from_file_)
        print("\nstep3: Create Dynamic Chain ---------------------------------------------")
        dyn = dynamics.Dynamics(robot_define, geom, model_folder_, config.load_dynamic_from_file_)
        print("\nstep4: Save Robot Model -------------------------------------------------")
        robot_model = robot_model_data.RobotModel(dyn)
        utils.save_data(model_folder_, robot.name_, robot_model)

    else:
        print("\nstep1: Robot Define Skipping ...")
        print("\nstep2: Create Kinematic Chain Skipping ...")
        print("\nstep3: Create Dynamic Chain Skipping ...")
        print("\nstep4: Save Robot Model Skipping ...")

    robot_model = None
    if request_robot_model_ is True:
        print("\nstep5: Load Robot Model -------------------------------------------------")
        robot_model = utils.load_data(model_folder_, robot.name_)
    else:
        print("\nstep5: Load Robot Model Skipping ...")

    if config.code_generation_ is True:
        print("\nstep6: Code Generation --------------------------------------------------")
        robotcode.CodeGen(robot_model, codegen_folder_)

    if config.design_excitation_traj_ is True:
        print("\nstep7: Excitation Trajectory Optimization -------------------------------")
        optimal_traj = traj_optimizer.TrajOptimizer(robot_model, trajectory.fourier_order_, trajectory.base_freq_,
                                                    joint_constraints=trajectory.joint_constraints_,
                                                    cartesian_constraints=trajectory.cartesian_constraints_)
        utils.save_data(excitation_traj_folder_, trajectory.traj_name, optimal_traj)
    else:
        print("\nstep7: Excitation Trajectory Optimization Skipping ...")

    if config.sample_data_process_ is True:
        print("\nstep8: Sample Data Process  ---------------------------------------------")
        data_processing.DataProcessor(data, sample_traj_folder_)
    else:
        print("\nstep8: Sample Data Process Skipping ...")

    if config.dynamics_identification_ is True:
        print("\nstep9: Dynamics_Parameters Identification -------------------------------")
        # 辨识前数据预处理(耗时操作)
        regressor_matrix_file_name = f"regressor_matrix"
        if iden.gen_regressor_ is True:
            sample_traj_processed = sample_traj_folder_ / f"{data.sample_traj_name_}_processed.csv"
            pre = identification.IdenPreDeal(iden, robot_model, sample_traj_processed)
            utils.save_data(identification_folder_, regressor_matrix_file_name, pre)
        regressor = utils.load_data(identification_folder_, regressor_matrix_file_name)
        identification.Identification(iden, regressor, robot_model, identification_folder_)



       

