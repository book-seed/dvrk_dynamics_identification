import numpy as np
import matplotlib.pyplot as plt

linestyles = [('solid',               (0, ())),
              ('solid',      (0, ())),
              ('solid', (0, ())),

              ('solid', (0, ())),
              ('solid', (0, ())),

              ('solid', (0, ())),
              ('solid', (0, ())),
              ('solid', (0, ())),

              ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
              ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
              ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
              ('loosely dotted',      (0, (1, 10))),
              ('dotted',              (0, (1, 5)))]


class TrajPlotter:
    def __init__(self, fourier_traj, frame_traj = [], const_frame_num = []):
        self._fourier_traj = fourier_traj
        self._frame_traj = frame_traj
        self._const_frame_ind = const_frame_num

    def plot_desired_traj(self, fourier_x):

        # 设置默认字体大小
        font_size_def = 15.0

        # 获取时间序列
        x = self._fourier_traj.t

        q, dq, ddq = self._fourier_traj.fourier_base_x2q(fourier_x)

        # 创建图形窗口
        fig = plt.figure(1)
        plt_q = fig.add_subplot(311)
        plt_q.margins(x=0.002, y=0.12)
        plt_q.set_title("Optimal Excitation Trajectory")

        # position
        for d in range(self._fourier_traj.dof):
            _, linestyle = linestyles[d]
            co_num = str(d + 1)
            plt_q.plot(x, q[:, d], label=(r"$q^m_" + co_num +"$"), linestyle=linestyle)

        # 自定义图例的显示位置和样式
        # bbox_to_anchor 定义图例的边界框位置和大小
        # loc 表示图例的锚点位置为上中心
        # ncol 表示图例分几列显示，这里根据自由度数量设置列数
        # mode="expand" 让图例水平扩展填充边界框
        # borderaxespad 是图例与坐标轴的间距
        # fontsize 是图例文字的大小
        plt_q.legend(bbox_to_anchor=(0., 1.22, 1., .102), loc='upper center', ncol=self._fourier_traj.dof,
                     mode="expand", borderaxespad=0., fontsize=font_size_def)

        # 设置子图的 x 轴标签，使用 LaTeX 格式表示时间，设置字体大小
        plt_q.set_xlabel(r'$t$ (s)', fontsize=font_size_def)

        # 设置子图的 y 轴标签，使用 LaTeX 格式表示位置，设置字体大小
        plt_q.set_ylabel(r'$q^m$ (rad or m)', fontsize=font_size_def)

        # 设置子图坐标轴刻度标签的字体大小
        plt_q.tick_params(labelsize=font_size_def)

        # velocity
        plt_dq = fig.add_subplot(312)
        plt_dq.margins(x=0.002, y=0.12)
        for d in range(self._fourier_traj.dof):
            _, linestyle = linestyles[d]
            plt_dq.plot(x, dq[:, d], label=(r"$\dot{q}^m_"+str(d+1)+"$"), linestyle=linestyle)

        plt_dq.set_xlabel(r'$t$ (s)', fontsize=font_size_def)
        plt_dq.set_ylabel(r'$\dot{q}^m$ (rad/s or m/s)', fontsize=font_size_def)
        plt_dq.tick_params(labelsize=font_size_def)

        # acceleration
        plt_ddq = fig.add_subplot(313)
        plt_ddq.margins(x=0.002, y=0.12)
        for d in range(self._fourier_traj.dof):
            _, linestyle = linestyles[d]
            plt_ddq.plot(x, ddq[:, d], label=(r"$\ddot{q}^m_"+str(d+1)+"$"), linestyle=linestyle)

        plt_ddq.set_xlabel(r'$t$ (s)', fontsize=font_size_def)
        plt_ddq.set_ylabel(r'$\ddot{q}^m$ (rad/s$^2$ or m/s$^2$)', fontsize=font_size_def)
        plt_ddq.tick_params(labelsize=font_size_def)

        plt.tight_layout()
        plt.show()

    def plot_measured_traj(self):
        pass

    def plot_frame_traj(self):

        x = self._fourier_traj.t
        map1 = ['x', 'y', 'z']

        const_size = len(self._const_frame_ind)

        subplotnum = const_size*100 + 11

        fig = plt.figure(2)
        for i in range(const_size):
            plt_q = fig.add_subplot(subplotnum + i)

            for d in range(3):
                _, linestyle = linestyles[d]
                plt_q.plot(x, self._frame_traj[i, :, d], label=(str(map1[d])), linestyle=linestyle)

            plt_q.legend()
            plt_q.set_ylabel(r'$q$ (m)')
            plt_q.set_title('Frame ' + str(int(self._const_frame_ind[i]))+ ' Trajectory')
        plt.show()