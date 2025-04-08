from kinematics import geometry
from dynamics import dynamics
from utils import utils
from model import robot_def, robot_model_data
from trajectory_optimization import traj_optimizer

def run(robot, trajectory, config):
    trajopt_condition = 'load_robot_model_from_file'
    if next((value for key, value in config if key == trajopt_condition), None) is False :

        print("\nstep1: Robot Define -----------------------------------------------------")
        robot_define = robot_def.RobotDef(
            name=robot.name_, params=robot.dh_, dh_convention=robot.dh_convention_,
            friction_type=robot.friction_type_)

        print("\nstep2: Create Kinematic Chain -------------------------------------------")
        geom = geometry.Geometry(robot_define, config)

        print("\nstep3: Create Dynamic Chain ---------------------------------------------")
        dyn = dynamics.Dynamics(robot_define, geom, config)

        print("\nstep4: Save Robot Model -------------------------------------------------")
        robot_model = robot_model_data.RobotModel(dyn)
        utils.save_data(robot.model_folder_, robot.name_, robot_model)

    else:
        print("\nstep1: Load Robot Model Skipping ...")
        print("\nstep2: Create Kinematic Chain Skipping ...")
        print("\nstep3: Create Dynamic Chain Skipping ...")
        print("\nstep4: Save Robot Model Skipping ...")

    print("\nstep5: Load Robot Model -------------------------------------------------")
    robot_model = utils.load_data(robot.model_folder_, robot.name_)

    print("\nstep6: Excitation Trajectory Optimization -------------------------------")
    optimal_traj = traj_optimizer.TrajOptimizer(robot_model, trajectory.fourier_order_, trajectory.base_freq_,
                                                joint_constraints=trajectory.joint_constraints_,
                                                cartesian_constraints=trajectory.cartesian_constraints_)


