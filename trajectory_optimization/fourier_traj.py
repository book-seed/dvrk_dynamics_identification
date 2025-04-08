import numpy as np
import math

verbose = False

if verbose:
    def vprint(*args):
        for arg in args:
            print(arg, end=' ')
        print()
else:
    vprint = lambda *a: None      # do-nothing function


class FourierTraj:
    def __init__(self, dof, order, base_freq, sample_num_per_period=10, frequency='nan', stable_time=0, final_time=10):
        self.dof = dof
        self.order = order
        self.base_freq = base_freq
        self.sample_num_per_period = sample_num_per_period
        self.stable_time = stable_time
        self.frequency = frequency      # 采样频率 ？

        # if no specified frequency and final_time, generate a one-period trajectory.
        if math.isnan(float(frequency)):
            self.sample_num = round(self.order * self.sample_num_per_period + 1)
            self.period = 1.0 / self.base_freq
            self.frequency = self.sample_num/(final_time + stable_time)
        else:
            self.sample_num = round(float(frequency) * (final_time + stable_time))
            self.period = final_time

        self.q = np.zeros((self.sample_num, self.dof))
        self.dq = np.zeros((self.sample_num, self.dof))
        self.ddq = np.zeros((self.sample_num, self.dof))

        self._gen_q_base()

    def _gen_q_base(self):
        self.t = np.linspace(0, self.period, self.sample_num)

        self.fourier_q_base = np.zeros((self.sample_num, 2 * self.order + 1))
        self.fourier_dq_base = np.zeros((self.sample_num, 2 * self.order + 1))
        self.fourier_ddq_base = np.zeros((self.sample_num, 2 * self.order + 1))

        for n in range(self.sample_num):
            self.fourier_q_base[n, 0] = 1

            if not self.stable_time == 0:
                ramp_up = float(n)/float(self.stable_time * self.frequency)     # 在 stable_time 内，ramp_up为0~1之间的一个值
                if ramp_up > 1:
                    ramp_up = 1
            else:
                ramp_up = 1

            for o in range(self.order):

                # 等价于论文中 omega_f * l * t, 其中 omega_f = 2 * pi * f_f    f_f:基频
                phase = 2 * np.pi * (o + 1) * self.t[n] * self.base_freq

                # 等价于论文中 omega_f * l, 其中 omega_f = 2 * pi * f_f    f_f:基频
                c = 2 * np.pi * (o + 1) * self.base_freq

                # a_lk / (omega_f * l) * sin(omega_f * l * t)
                self.fourier_q_base[n, o + 1] = ramp_up * np.sin(phase) / c

                # -b_lk / (omega_f * l) * cos(omega_f * l * t)
                self.fourier_q_base[n, self.order + o + 1] = -ramp_up * np.cos(phase) / c

                # 位置轨迹的一阶导数
                self.fourier_dq_base[n, o + 1] =  ramp_up * np.cos(phase)
                self.fourier_dq_base[n, self.order + o + 1] = ramp_up * np.sin(phase)

                # 位置轨迹的二阶导数
                self.fourier_ddq_base[n, o + 1] = -ramp_up *c * np.sin(phase)
                self.fourier_ddq_base[n, self.order + o + 1] = ramp_up * c * np.cos(phase)

        vprint('fourier_q_base:')
        vprint(self.fourier_q_base)
        vprint('fourier_dq_base:')
        vprint(self.fourier_dq_base)
        vprint('fourier_ddq_base:')
        vprint(self.fourier_ddq_base)

    def fourier_base_x2q(self, x):

        # x 表征激励轨迹的系数，对于单个dof的5级傅里叶级数, [q_ok, a1k, a2k, a3k, a4k ,a5k, b1k, b2k, b3k, b4k, b5k]

        for d in range(self.dof):
            start = d * (2 * self.order + 1)
            end = (d + 1) * (2 * self.order + 1)
            self.q[:, d] = np.matmul(self.fourier_q_base, x[start:end])
            self.dq[:, d] = np.matmul(self.fourier_dq_base, x[start:end])
            self.ddq[:, d] = np.matmul(self.fourier_ddq_base, x[start:end])
        return self.q, self.dq, self.ddq
