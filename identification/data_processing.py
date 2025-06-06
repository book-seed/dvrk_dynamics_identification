# This file is originally adopted from https://github.com/cdsousa/wam7_dyn_ident and modified by Yan Wang
from pathlib import Path
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from utils import utils
from utils import diff
import sympy

# 采集的数据的保存格式为：
# q0, dq0 tau0, q1, dq1, tau1, ..., qn, dqn, taun
# 处理后的数据的保存为csv,格式：(每行有 1 + 8 * dof)个元素
# t, q0_raw, ..., qn_raw, q0_filter, ..., qn_filter，
# dq0_raw, ..., dqn_raw, dq0_filter, ..., dqn_filter，
# ddq0_raw, ..., ddqn_raw, ddq0_filter, ..., ddqn_filter，
# tau0_raw, ..., taun_raw, tau0_filter, ..., taun_filter

class DataProcessor:
    def __init__(self, data, sample_traj_folder):

        self._measured_data_file = Path(sample_traj_folder / data.sample_traj_name_).with_suffix(".csv")
        self._sample_freq = data.sample_freq_
        self._cutoff_freq = data.cutoff_freq_
        self._cut_num = data.cut_num_
        self._filter_order = data.filter_order_
        self._callback = data.callback_

        self._load_trajectory_data()
        self._diff_and_filt_data()
        self._plot_and_save_measured_data(data)

    def _load_trajectory_data(self):
        f = np.array(pd.read_csv(self._measured_data_file, sep=',', header=None))
        row, col = f.shape
        sample_num = row
        self.dof = int(col/3)

        # 此处设置的采样频率要和实际数据的采样频率一致
        self.t_raw = np.array(range(sample_num), dtype=float) / self._sample_freq
        self.q_raw = np.zeros((row, self.dof))
        self.dq_raw = np.zeros((row, self.dof))
        self.ddq_raw = np.zeros((row, self.dof))
        self.tau_raw = np.zeros((row, self.dof))

        for d in range(self.dof):
            self.q_raw[:, d] = f[:, d]
            self.dq_raw[:, d] = f[:, self.dof + d]
            self.tau_raw[:, d] = f[:, 2 * self.dof + d]

        self.q_raw, self.dq_raw, self.tau_raw = \
            self._callback(self.q_raw, self.dq_raw, self.tau_raw)

    def _diff_and_filt_data(self):

        q_tmp = np.zeros_like(self.q_raw)
        dq_tmp = np.zeros_like(self.dq_raw)
        ddq_tmp = np.zeros_like(self.dq_raw)
        tau_tmp = np.zeros_like(self.tau_raw)

        butter_coef = scipy.signal.butter(self._filter_order, self._cutoff_freq / (self._sample_freq / 2))

        for i in range(self.dof):
            self.ddq_raw[:, i] = diff.central_diff(self.dq_raw[:, i], 1.0 / self._sample_freq, order='4th_order_precision')

            q_tmp[:, i] = scipy.signal.filtfilt(butter_coef[0], butter_coef[1], self.q_raw[:, i])
            dq_tmp[:, i] = scipy.signal.filtfilt(butter_coef[0], butter_coef[1], self.dq_raw[:, i])
            ddq_tmp[:, i] = scipy.signal.filtfilt(butter_coef[0], butter_coef[1], self.ddq_raw[:, i])
            tau_tmp[:, i] = scipy.signal.filtfilt(butter_coef[0], butter_coef[1], self.tau_raw[:, i])

        self.t_cut = self.t_raw[self._cut_num:-self._cut_num]

        self.q_filt_cut = q_tmp[self._cut_num:-self._cut_num, :]
        self.dq_filt_cut = dq_tmp[self._cut_num:-self._cut_num, :]
        self.ddq_filt_cut = ddq_tmp[self._cut_num:-self._cut_num, :]
        self.tau_filt_cut = tau_tmp[self._cut_num:-self._cut_num, :]

        self.q_raw_cut = self.q_raw[self._cut_num:-self._cut_num, :]
        self.dq_raw_cut = self.dq_raw[self._cut_num:-self._cut_num, :]
        self.ddq_raw_cut = self.ddq_raw[self._cut_num:-self._cut_num, :]
        self.tau_raw_cut = self.tau_raw[self._cut_num:-self._cut_num, :]

    def _plot_and_save_measured_data(self, data):

        t = self.t_cut
        q_raw = self.q_raw_cut
        q_filter = self.q_filt_cut
        dq_raw = self.dq_raw_cut
        dq_filter = self.dq_filt_cut
        ddq_raw = self.ddq_raw_cut
        ddq_filter = self.ddq_filt_cut
        tau_raw = self.tau_raw_cut
        tau_filter = self.tau_filt_cut
        
        dof = q_raw.shape[1]

        # 合并所有数组
        save_data = np.column_stack([t, q_raw, q_filter, dq_raw, dq_filter, ddq_raw, ddq_filter, tau_raw, tau_filter])
        # 创建列名列表
        columns = ['time']
        prefixes = ['q_raw', 'q_filter', 'dq_raw', 'dq_filter', 'ddq_raw', 'ddq_filter', 'tau_raw', 'tau_filter']
        for prefix in prefixes:
            for i in range(dof):  # 假设每个变量有3个维度
                columns.append(f'{prefix}_{i}')
            
        # 创建DataFrame并保存为CSV
        file = Path(self._measured_data_file).parent / f"{data.sample_traj_name_}_processed.csv"
        df = pd.DataFrame(save_data, columns=columns)
        df.to_csv(file, index=False)  # 保留列名，不保留行索引  

        # 验证保存的文件
        print(f"数据已保存到{file}，包含 {len(columns)} 列和 {len(df)} 行")
        
        # 绘制每个变量的时间序列图
        fig, axes = plt.subplots(dof, 4, figsize=(10,15))
        fig.suptitle('trajectory data before and after processing', fontsize=18)

        top_labels = ['q', r'$\dot{q}$', r'$\ddot{q}$', r'$\tau$']
        left_labels = [f'J{i+1}' for i in range(dof)]

        for j, label in enumerate(top_labels):
            axes[0, j].set_title(label, fontsize=14, pad=20)

        for i, label in enumerate(left_labels):
            axes[i, 0].set_ylabel(label, fontsize=14, rotation=0, labelpad=20)

        # dq_raw, ddq_raw 的噪声比较大，可能会导致dq_filter, ddq_filter被过度压缩
        for i in range(dof):
            axes[i, 0].plot(t, q_raw[:, i])
            axes[i, 0].plot(t, q_filter[:, i])
            # axes[i, 1].plot(t, dq_raw[:, i])
            axes[i, 1].plot(t, dq_filter[:, i])
            # axes[i, 2].plot(t, ddq_raw[:, i])
            axes[i, 2].plot(t, ddq_filter[:, i])
            axes[i, 3].plot(t, tau_raw[:, i])
            axes[i, 3].plot(t, tau_filter[:, i])

        plt.subplots_adjust(hspace=0.6, wspace=0.4)
        file = Path(self._measured_data_file).parent / f"{data.sample_traj_name_}_processed.png"
        plt.savefig(file, dpi=300)
        # plt.show()








def plot_meas_2pred_tau(t, tau_m, tau_p1, tau_p2, joint_type, coordinates):
    sample_num, dof = tau_m.shape
    t = t - t[0]

    fig = plt.figure()
    # matplotlib.pyplot.rcParams['pdf.fonttype'] = 42
    # matplotlib.pyplot.rcParams['ps.fonttype'] = 42

    # font_size_def = 9.0
    font_size_def = 14.0

    for i in range(dof):
        plt_tau = fig.add_subplot(dof, 1, i + 1)
        plt_tau.margins(x=0.002, y=0.02)

        plt_tau.plot(t, tau_p1[:, i], color=(0,0,1), label="Predicted-F", linewidth=1)
        plt_tau.plot(t, tau_p1[:, i] - tau_m[:, i], color='c', ls='-.', label="Error-F", linewidth=0.8)
        plt_tau.plot(t, tau_p2[:, i], color=(1,0,0), label="Predicted-Y", linewidth=1)
        plt_tau.plot(t, tau_p2[:, i] - tau_m[:, i], color='y', ls='-.', label="Error-Y", linewidth=0.8)
        plt_tau.plot(t, tau_m[:, i], color='k', ls='--', label="Measured", linewidth=1.5)
        # plt_tau.plot(t, tau_p[:, i] - tau_m[:, i], 'k--', label="Error", linewidth=1)
        # zeros = np.zeros(tau_p[:, i].shape)
        # plt_tau.plot(t, zeros, color='0.5', linewidth=0.75)
        if i == dof-1:
            plt_tau.set_xlabel(r'$t$ (s)', fontsize=font_size_def)
        if joint_type[i] == 'R':
            plt_tau.set_ylabel(r'$\tau^m_{}$ (Nm)'.format(coordinates[i].name[1:]), fontsize=font_size_def)
        else:
            plt_tau.set_ylabel(r'$f^m_{}$ (N)'.format(coordinates[i].name[1:], fontsize=font_size_def))
        # plt_tau.legend(['Measured', "Predicted"])
        if i == 0:
            plt_tau.legend(bbox_to_anchor=(0.0, 1.60, 1.0, 0.502), loc='upper center', ncol=3,
                           mode="expand", borderaxespad=0., fontsize=font_size_def)
        plt_tau.tick_params(labelsize=font_size_def)
    plt.tight_layout()
    plt.show()


def barycentric2standard_params(x, rbt_def, Rs=None):
    i = 0
    i_link = 1
    x_out = []
    while i_link < rbt_def.frame_num:
        if rbt_def.use_inertia[i_link]:
            m = x[i + 9]
            rx, ry, rz = x[i + 6] / m, x[i + 7] / m, x[i + 8] / m
            r = [rx, ry, rz]
            L_mat = inertia_vec2tensor(x[i: i + 6])
            I_vec = inertia_tensor2vec(Lmr2I(L_mat, m, r))

            if Rs:
                R = Rs[i_link - 1]
                RI = np.matmul(R, np.matrix(inertia_vec2tensor(I_vec)).astype(np.float64))
                I_vec = inertia_tensor2vec(np.matmul(RI, R.transpose()))
                I_vec = np.array(I_vec).astype(np.float64).tolist()
                r = np.matmul(R, np.array(r).transpose()).tolist()[0]

            x_out += I_vec + r + [m]

            i += 10
        if rbt_def.use_friction[i_link]:
            if 'Coulomb' in rbt_def.friction_type:
                x_out += [x[i]]
                i += 1

            if 'viscous' in rbt_def.friction_type:
                x_out += [x[i]]
                i += 1

            if 'offset' in rbt_def.friction_type:
                x_out += [x[i]]
                i += 1

        if rbt_def.use_Ia[i_link]:
            x_out += [x[i]]
            i += 1

        if rbt_def.spring_dl[i_link] != None:
            x_out += [x[i]]
            i += 1

        i_link += 1

    return x_out


def params_array2table(param_value, rbt_def, std_or_bary):
    i = 0 # param_value count
    i_link = 1
    table = sympy.zeros(rbt_def.frame_num, 15+1)
    std_param_str = ['link', 'I_xx', 'I_xy', 'I_xz', 'I_yy', 'I_yz', 'I_zz', 'r_x', 'r_y', 'r_z', 'm', 'F_c', 'F_v', 'F_o', 'I_m',
                 'K']
    bary_param_str = ['link', 'L_xx', 'L_xy', 'L_xz', 'L_yy', 'L_yz', 'L_zz', 'l_x', 'l_y', 'l_z', 'm', 'F_c', 'F_v', 'F_o', 'I_m',
                 'K']
    param_str = std_param_str
    if std_or_bary == 'bary':
        param_str = bary_param_str

    n_sym = 0
    for k in range(len(param_str)):
        table[0, k] = param_str[k]

    while i_link < rbt_def.frame_num:
        table[i_link, 0] = i_link
        for j in range(10):
            if rbt_def.use_inertia[i_link]:
                table[i_link, j + 1] = param_value[i]
                i += 1
            else:
                table[i_link, j + 1] = n_sym

        if rbt_def.use_friction[i_link]:
            if 'Coulomb' in rbt_def.friction_type:
                table[i_link, 11] = param_value[i]
                i += 1
            else:
                table[i_link, 11] = n_sym

            if 'viscous' in rbt_def.friction_type:
                table[i_link, 12] = param_value[i]
                i += 1
            else:
                table[i_link, 12] = n_sym

            if 'offset' in rbt_def.friction_type:
                table[i_link, 13] = param_value[i]
                i += 1
            else:
                table[i_link, 13] = n_sym
        else:
            table[i_link, 11] = n_sym
            table[i_link, 12] = n_sym
            table[i_link, 13] = n_sym

        if rbt_def.use_Ia[i_link]:
            table[i_link, 14] += param_value[i]
            i += 1
        else:
            table[i_link, 14] = n_sym

        if rbt_def.spring_dl[i_link] != None:
            table[i_link, 15] += param_value[i]
            i += 1
        else:
            table[i_link, 15] = n_sym

        i_link += 1

    return table


def write_parameters2json(table, folder, name):
    json_data = {}
    for i in range(table.shape[0] - 1):
        link_num = i + 1
        link_data = {
            'I': [float(table[link_num, 1]),
                  float(table[link_num, 2]),
                  float(table[link_num, 3]),
                  float(table[link_num, 4]),
                  float(table[link_num, 5]),
                  float(table[link_num, 6])],
            'r': [float(table[link_num, 7]),
                  float(table[link_num, 8]),
                  float(table[link_num, 9])],
            'm': float(table[link_num, 10]),
            'Fc': float(table[link_num, 11]),
            'Fv': float(table[link_num, 12]),
            'Fo': float(table[link_num, 13]),
            'Im': float(table[link_num, 14]),
            'K': float(table[link_num, 15])
        }
        json_data[str(link_num)] = link_data

    import io, json, os, errno

    file_name = folder + name + '.json'

    if not os.path.exists(os.path.dirname(file_name)):
        try:
            os.makedirs(os.path.dirname(file_name))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with io.open(file_name, 'wb') as f:
        f.write(json.dumps(json_data, sort_keys=True, indent=4))

    print("Parameters have been written into [{}] successfully!".format(file_name))


def trans_inertia(I, r, R):
    I1 = np.matmul(np.matmul(R.transpose(), I), R)
    r1 = np.matmul(R.transpose(), r)
    return I1, r1