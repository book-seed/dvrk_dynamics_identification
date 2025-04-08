from dynamics.dynamics import Dynamics


class RobotModel:
    def __init__(self, dyn):
        self.dof = dyn.rbt_def.dof
        self.coordinates = dyn.rbt_def.coordinates
        self.d_coordinates = dyn.rbt_def.d_coordinates
        self.dd_coordinates = dyn.rbt_def.dd_coordinates
        self.coordinates_joint_type = dyn.rbt_def.coordinates_joint_type
        self.joint_type = dyn.rbt_def.joint_type
        
        # 最小惯性参数集个数
        self.base_num = dyn.base_num
        
        # 最小惯性参数集
        self.base_param = dyn.base_param
        
        # 全惯性参数集（基于质心坐标系）
        self.std_param = dyn.rbt_def.std_params
        
        # 全惯性参数集（基于连杆坐标系）
        self.bary_param = dyn.rbt_def.bary_params
        
        self.p_n_func = dyn.geom.p_n_func

        self.H = dyn.H
        self.M = dyn.M
        self.C = dyn.C
        self.G = dyn.G
        self.H_func = dyn.H_func
        self.H_b = dyn.H_b
        self.H_b_func = dyn.H_b_func

        self.frame_num = dyn.rbt_def.frame_num
        self.use_inertia = dyn.rbt_def.use_inertia
        self.use_friction = dyn.rbt_def.use_friction
        self.friction_type = dyn.rbt_def.friction_type
        self.use_Ia = dyn.rbt_def.use_Ia
        self.spring_num = dyn.rbt_def.spring_num
        self.spring_dl = dyn.rbt_def.spring_dl

        self.T_0n = dyn.geom.T_0n
        self.T_0nc = dyn.geom.T_0nc
        self.p_n = dyn.geom.p_n
        self.p_c = dyn.geom.p_c
        self.R = dyn.geom.R
        self.v_cw = dyn.geom.v_cw
        self.w_b = dyn.geom.w_b
