import time
import numpy as np
from trajectory_optimization.fourier_traj import FourierTraj
import csv
import sympy
from scipy.optimize import minimize

q0_scale = np.pi
fourier_scale = 10 * np.pi

# joint constraints
# [(joint_var, q_low, q_upper, dq_low, dq_upper, ddq_low, ddq_upper), ..., (...)]

# cartesian constraints
# [(joint_num, x_low, x_high, y_low, y_high, z_low, z_high), ..., (...)]

class TrajOptimizer:
    def __init__(self, model, trajectory, q0_min=-q0_scale, q0_max=q0_scale,
                 ab_min=-fourier_scale, ab_max=fourier_scale, verbose=False):
        self.reg_norm_mat = None
        self.f_result = None
        self.x_result = None
        self._order = trajectory.fourier_order_
        self._base_freq = trajectory.base_freq_
        self._control_freq = trajectory.control_freq_
        self._model = model
        self._joint_constraints = trajectory.joint_constraints_
        self._joint_const_num = len(self._joint_constraints)
        print('joint constraint number: {}'.format(self._joint_const_num))
        self._cartesian_constraints = trajectory.cartesian_constraints_
        self._cartesian_const_num = len(self._cartesian_constraints)
        print('cartesian constraint number: {}'.format(self._cartesian_const_num))
        self._const_num = self._joint_const_num * 6 + self._cartesian_const_num * 3
        print('constraint number: {}'.format(self._const_num))

        self._q0_min = q0_min
        self._q0_max = q0_max
        self._ab_min = ab_min
        self._ab_max = ab_max

        self.sample_num = self._control_freq / self._base_freq

        self.fourier_traj = FourierTraj(self._model.dof, self._order, self._base_freq,
                                        self._control_freq, self.sample_num)
        self._prepare_opt()

        self.frame_pos = np.zeros((self.sample_num, 3))
        self.const_frame_ind = np.array([])

        for c_c in self._cartesian_constraints:
            frame_num, bool_max, c_x, c_y, c_z = c_c

            if frame_num not in self.const_frame_ind:
                self.const_frame_ind = np.append(self.const_frame_ind, frame_num)

        self.frame_traj = np.zeros((len(self.const_frame_ind), self.sample_num, 3))

        print('frames_constrained: {}'.format(self.const_frame_ind))

        self._obj_cnt = 0

        start_time = time.time()
        self._optimize()
        print("excitation trajectory optimization finished. Cost Time: {} seconds".format(round(time.time() - start_time, 6)))

    def _prepare_opt(self):
        self.H = np.zeros((self._model.dof * self.sample_num, self._model.base_num))
        self.H_norm = np.zeros((self._model.dof * self.sample_num, self._model.base_num))

    def _obj_func(self, x):
        # objective
        q, dq, ddq = self.fourier_traj.fourier_base_x2q(x)

        for n in range(self.sample_num):
            vars_input = q[n, :].tolist() + dq[n, :].tolist() + ddq[n, :].tolist()
            self.H[n * self._model.dof:(n + 1) * self._model.dof, :] = self._model.H_b_func(*vars_input)

        self.H /= np.subtract(self.H.max(axis=0), self.H.min(axis=0))

        f = np.linalg.cond(self.H)

        if self._obj_cnt % 50 == 0:
            print("{}. condition number: {}".format(self._obj_cnt, f))
        self._obj_cnt += 1

        return f

    def _constraint_func(self, x):
        q, dq, ddq = self.fourier_traj.fourier_base_x2q(x)
        g = []

        # Joint constraints (with composite joint angle considered)
        q_ss = [c[0] for c in self._joint_constraints]
        A, _ = sympy.linear_eq_to_matrix(q_ss, self._model.coordinates)
        A_T = np.matrix(A).astype(np.float64).transpose()
        q_c = np.matmul(q, A_T)
        dq_c = np.matmul(dq, A_T)
        ddq_c = np.matmul(ddq, A_T)

        for j, j_c in enumerate(self._joint_constraints):
            _, q_l, q_u, dq_l, dq_u, ddq_l, ddq_u = j_c
            for qt, dqt, ddqt in zip(q_c[:, j], dq_c[:, j], ddq_c[:, j]):
                g.extend([qt - q_u, q_l - qt, dqt - dq_u, dq_l - dqt, ddqt - ddq_u, ddq_l - ddqt])

        # Cartesian Constraints
        for c_c in self._cartesian_constraints:
            frame_num, bool_max, c_x, c_y, c_z = c_c
            for num in range(q.shape[0]):
                vars_input = q[num, :].tolist()
                p_num = self._model.p_n_func[frame_num](*vars_input)
                self.frame_pos[num, 0] = p_num[0, 0]
                self.frame_pos[num, 1] = p_num[1, 0]
                self.frame_pos[num, 2] = p_num[2, 0]
                if bool_max == 'max':
                    g.extend([p_num[0, 0] - c_x, p_num[1, 0] - c_y, p_num[2, 0] - c_z])
                elif bool_max == 'min':
                    g.extend([-p_num[0, 0] + c_x, -p_num[1, 0] + c_y, -p_num[2, 0] + c_z])

        g_array = np.array(g)
        g_array = g_array.flatten()  
        return g_array

    def _add_vars2bounds(self):

        bounds = []
        for num in range(self._model.dof):
            # q0
            bounds.append((self._q0_min, self._q0_max))
            # a sin
            for o in range(self._order):
                bounds.append((self._ab_min, self._ab_max))
            # b cos
            for o in range(self._order):
                bounds.append((self._ab_min, self._ab_max))

        return bounds

    def _initial_guess(self):

        def rand_local(l, u, scale):
            return (np.random.random() * (u - l) / 2 + (u + l) / 2) * scale
        x = []
        for num in range(self._model.dof):
            # q0
            x.append(rand_local(self._q0_min, self._q0_max, 0.1))
            # a sin
            for o in range(self._order):
                x.append(rand_local(self._ab_min, self._ab_max, 0.1))
            # b cos
            for o in range(self._order):
                x.append(rand_local(self._ab_min, self._ab_max, 0.1))
        return x

    def _optimize(self):
        self._prepare_opt()
        bounds = self._add_vars2bounds()
        cons = ({'type': 'ineq', 'fun': lambda x: -self._constraint_func(x)})

        x0 = np.array(self._initial_guess())
        result = minimize(fun=self._obj_func,
                          x0=x0,
                          method='SLSQP',
                          bounds=bounds,
                          constraints=cons,
                          tol=1e-5,
                          options={'maxiter': 2000, 'disp': True})

        self.f_result = result.fun
        self.x_result = result.x

        print('condition number: {}'.format(self.f_result))
        print('x: {}'.format(self.x_result))

    def calc_normalize_mat(self):
        q, dq, ddq = self.fourier_traj.fourier_base_x2q(self.x_result)

        for n in range(self.sample_num):
            vars_input = q[n, :].tolist() + dq[n, :].tolist() + ddq[n, :].tolist()
            self.H[n * self._model.dof:(n + 1) * self._model.dof, :] = self._model.H_b_func(*vars_input)
            self.reg_norm_mat = self.H.max(axis=0) - self.H.min(axis=0)

        return self.reg_norm_mat

    def calc_frame_traj(self):
        """
        计算约束坐标系在轨迹上的空间位置
        通过优化得到的傅里叶系数，计算每个约束坐标系在轨迹采样点上的三维坐标位置，
        结果存储在self.frame_traj中
        输出维度: [约束坐标系数量 × 采样点数 × 3(xyz)]
        """
        # 获取轨迹上每个采样点的位置，速度，加速度信息
        q, dq, ddq = self.fourier_traj.fourier_base_x2q(self.x_result)
        # 遍历所有需要约束的坐标系
        for i in range(len(self.const_frame_ind)):
            # 遍历轨迹上的每个采样点
            for num in range(q.shape[0]):
                vars_input = q[num, :].tolist() # 当前采样点每个dof的位置
                p_num = self._model.p_n_func[int(self.const_frame_ind[i])](*vars_input)
                self.frame_traj[i, num, :] = p_num[:, 0]

    def make_traj_csv(self, folder, name):
        q, dq, ddq = self.fourier_traj.fourier_base_x2q(self.x_result)

        with open(folder + 'x_result' + '.csv', 'w', newline='') as x_result_file:
            x_result_wr = csv.writer(x_result_file, quoting=csv.QUOTE_NONE)
            x_result_wr.writerow(self.x_result)

        with open(folder + 'trajectory_points' + '.csv', 'w', newline='') as my_file:
            wr = csv.writer(my_file, quoting=csv.QUOTE_NONE)
            for i in range(np.size(q, 0) - 10):
                wr.writerow(np.concatenate(q[i], dq[i], ddq[i]))


     