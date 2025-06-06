import numpy as np
from pathlib import Path
import pandas as pd
from collections import namedtuple
import time
import matplotlib.pyplot as plt
import sympy as sp
import cvxpy
from cvxpy.expressions.cvxtypes import problem
from pexpect.screen import constrain

TrajData = namedtuple('TrajData', [
    't', 'q_raw', 'q_filter','dq_raw', 'dq_filter',
    'ddq_raw', 'ddq_filter','tau_raw', 'tau_filter',
])


class IdenPreDeal:
    def __init__(self, iden, robot_model, traj_file):
        self.Wb = None
        # 参与辨识的力矩数据，即经过处理后的采集的力矩(向量)
        self.tau_s = None
        # 存储 q,dq,ddq等观测数据
        self.traj_data = None

        start_time = time.time()
        self._load_processed_sampled_data(traj_file)
        self._gen_regressor(iden, robot_model)
        print("regressor matrix created. Cost Time: {} seconds".format(
            round(time.time() - start_time, 6)))

    def _load_processed_sampled_data(self, traj_file):
        f = np.array(pd.read_csv(traj_file, sep=','))
        row, col = f.shape
        dof = int((col - 1) / 8)

        self.traj_data = TrajData(
            t = f[:, 0],
            q_raw = f[:, 1 + 0 * dof: 1 + 1 * dof],
            q_filter = f[:, 1 + 1 * dof: 1 + 2 * dof],
            dq_raw = f[:, 1 + 2 * dof: 1 + 3 * dof],
            dq_filter = f[:, 1 + 3 * dof: 1 + 4 * dof],
            ddq_raw = f[:, 1 + 4 * dof: 1 + 5 * dof],
            ddq_filter = f[:, 1 + 5 * dof: 1 + 6 * dof],
            tau_raw = f[:, 1 + 6 * dof: 1 + 7 * dof],
            tau_filter = f[:, 1 + 7 * dof: 1 + 8 * dof])

    def _gen_regressor(self, iden, robot_model):
        sample_num, dof = self.traj_data.q_filter.shape
        self.Wb = np.zeros((sample_num * dof, robot_model.base_num))
        self.tau_s = np.zeros(sample_num * dof)

        for i in range(sample_num):
            vars_input = (self.traj_data.q_filter[i, :].tolist()
                          + self.traj_data.dq_filter[i, :].tolist()
                          + self.traj_data.ddq_filter[i, :].tolist())
            self.Wb[i * dof:(i + 1) * dof, :] = robot_model.H_b_func(*vars_input)

            for d in range(dof):
                self.tau_s[i * dof + d] = self.traj_data.tau_filter[i, d]


class Identification:

    def __init__(self, iden, regressor, robot_model, save_folder):

        # 预测的力矩数据，即通过最小惯性参数计算得到的力矩(向量)
        self.predict_tau_s = None
        # 预测的力矩数据，即通过最小惯性参数计算得到的力矩(矩阵)
        self.predict_tau = None
        # 求解器
        self.solve_mth = iden.solver_
        # 最小参数集对应的辨识结果
        self.xb = None

        print("identification solver is {}".format(self.solve_mth))
        self._solver(regressor, robot_model)

        print("identification result:")
        # 将机器人的基础参数和辨识结果水平拼接成一个矩阵
        identification_result = sp.Matrix.hstack(sp.Matrix(robot_model.base_param), sp.Matrix(self.xb))
        # 逐行打印拼接后的矩阵元素
        for row in range(identification_result.shape[0]):
            sp.pprint(identification_result.row(row))
        print()

        # 保存辨识结果（最小参数集）
        file = Path(save_folder) / f"xb_{self.solve_mth}.csv"
        df = pd.DataFrame(self.xb)
        df.to_csv(file, header=False, index=False)  # 保留列名，不保留行索引

        self._evaluation(robot_model, regressor)
        self._plot_measured_and_predict_tau(robot_model, regressor, save_folder)


    def _solver(self, regressor, robot_model):

        def solve_wls():
            weight = (np.max(regressor.traj_data.tau_filter, axis=0)
                      - np.min(regressor.traj_data.tau_filter, axis=0))
            # repeat the weight to generate a large vector for all the data
            weights = 1.0 / np.tile(weight, int(regressor.Wb.shape[0]/weight.shape[0]))
            Wb_wls = np.multiply(regressor.Wb, np.asmatrix(weights).transpose())
            tau_s_wls = np.multiply(regressor.tau_s, weights)
            self.xb = np.linalg.lstsq(Wb_wls, tau_s_wls)[0]

        def solve_ols():
            self.xb = np.linalg.lstsq(regressor.Wb, regressor.tau_s, rcond=None)[0]

        def solve_cvx():
            weight = (np.max(regressor.traj_data.tau_filter, axis=0)
                      - np.min(regressor.traj_data.tau_filter, axis=0))
            weights = 1.0 / np.tile(weight, int(regressor.Wb.shape[0]/weight.shape[0]))
            W_cvx = np.multiply(regressor.Wb, np.asmatrix(weights).transpose())
            tau_s_cvx = np.multiply(regressor.tau_s, weights)
            xb = cvxpy.Variable(robot_model.base_num)
            objective = cvxpy.Minimize(cvxpy.norm2(W_cvx @ xb - tau_s_cvx) + 0.002 * cvxpy.norm1(xb))
            constrains = []
            prob = cvxpy.Problem(objective, constrains)
            result = prob.solve()
            print(f"identification result using convex optimization: {result}")
            self.xb = xb.value

        if self.solve_mth == 'OLS':
            solve_ols()
        elif self.solve_mth == 'WLS':
            solve_wls()
        elif self.solve_mth == 'CVX':
            solve_cvx()
        else:
            raise ValueError('solver does not exist')

    def _evaluation(self, robot_model, regressor):
        sample_num, dof = regressor.traj_data.q_filter.shape
        self.predict_tau_s = regressor.Wb.dot(self.xb)
        self.predict_tau = np.zeros(regressor.traj_data.tau_filter.shape)
        for i in range(dof):
            self.predict_tau[:,i] = self.predict_tau_s[i::dof]

        # 默认 Frobenius 范数，即矩阵元素的平方和的平方根
        # 无偏估计(统计学)：
        # 计算自由度调整项的参考依据主要来源于统计学中无偏估计的理论，目的是避免因模型
        # 过拟合而低估误差，让误差方差的估计值更能反映模型的真实泛化能力。
        var_regression_error = (np.linalg.norm(self.predict_tau_s - regressor.tau_s) /
                                (regressor.tau_s.size - robot_model.base_num))
        print(f"variance of regression error:")
        print(var_regression_error)

        std_dev_xb = np.sqrt(np.diag(var_regression_error *
                                     np.linalg.inv(regressor.Wb.transpose().dot(regressor.Wb))))
        print(f"standard deviation of xb:")
        print(std_dev_xb)

        pct_std_dev_xb = std_dev_xb / np.abs(self.xb)
        print("percentage of standard deviation of xb: ")
        print(pct_std_dev_xb)

    def _plot_measured_and_predict_tau(self, robot_model, regressor, save_folder):
        sample_num, dof = regressor.traj_data.tau_filter.shape
        t = regressor.traj_data.t - regressor.traj_data.t[0]

        fig = plt.figure()

        font_size_def = 5.0

        for i in range(dof):
            plt_tau = fig.add_subplot(dof, 1, i + 1)
            plt_tau.margins(x=0.002, y=0.005)
            plt_tau.plot(t, regressor.traj_data.tau_filter[:, i], 'r', label="Measured", linewidth=1)
            plt_tau.plot(t, self.predict_tau[:, i], 'b', label="Predicted", linewidth=1)
            plt_tau.plot(t, self.predict_tau[:, i] - regressor.traj_data.tau_filter[:, i], 'k--', label="Error", linewidth=1)

            if i == dof - 1:
                plt_tau.set_xlabel(r'$t$ (s)', fontsize=font_size_def)
            if robot_model.coordinates_joint_type[i] == 'R':
                plt_tau.set_ylabel(r'$\tau^m_{}$ (Nm)'.format(robot_model.coordinates[i].name[1:]),
                                   fontsize=font_size_def)
            else:
                plt_tau.set_ylabel(r'$f^m_{}$ (N)'.format(robot_model.coordinates[i].name[1:],
                                                          fontsize=font_size_def))
            if i == 0:
                plt_tau.legend(bbox_to_anchor=(0.0, 1.60, 1.0, .102), loc='upper center', ncol=3,
                               mode="expand", borderaxespad=0., fontsize=font_size_def)
            plt_tau.tick_params(labelsize=font_size_def)

        plt.tight_layout()
        file = Path(save_folder) / f"measured_and_predicted_tau_{self.solve_mth}.png"

        plt.savefig(file, dpi=500)
        # plt.show()







