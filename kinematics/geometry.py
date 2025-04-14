import sympy
import numpy as np
from kinematics.frame_drawer import FrameDrawer
from utils import utils
import time
import multiprocessing
import dill
from multiprocessing.reduction import ForkingPickler

ForkingPickler.dumps = dill.dumps

verbose = False

if verbose:
    def vprint(*args):
        for arg in args:
            print(arg, end=' ')
        print()
else:
    vprint = lambda *a: None      # do-nothing function


class Geometry:
    def __init__(self, rbt_def, load_data_from_file):
        
        self.rbt_def = rbt_def
        self.model_folder = '/data/' + self.rbt_def.name + '/model/'

        self.T_0n = list(sympy.simplify(range(self.rbt_def.frame_num)))
        self.p_n = list(sympy.simplify(range(self.rbt_def.frame_num)))
        self.T_0nc = list(sympy.simplify(range(self.rbt_def.frame_num)))
        self.p_c = list(sympy.simplify(range(self.rbt_def.frame_num)))
        self.R = list(sympy.simplify(range(self.rbt_def.frame_num)))
        self.v_cw = list(sympy.simplify(range(self.rbt_def.frame_num)))
        self.w_b = list(sympy.simplify(range(self.rbt_def.frame_num)))

        condition = 'load_kinematic_from_file'
        if load_data_from_file is True :
            print("load kinematic data from file starting.")
            start_time = time.time()
            # self._load_data_sp()
            self._load_data()
            print("load kinematic data from file finished. Cost Time: {} seconds".format(round(time.time() - start_time, 6)))
            
            # 验证多进程结果与单进程结果的一致性
            # err_v = [sympy.simplify(a - b) for a, b in zip(self.v_cw, self.v_cw_sp)]
            # err_w = [sympy.simplify(a - b) for a, b in zip(self.w_b, self.w_b_sp)]
            # print(f'err_v: {err_v}')
            # print(f'err_w: {err_w}')
        else:
            start_time = time.time()
            self._multi_process_cal_geom()
            print("_multi_process_cal_geom all geometries finished. Cost Time: {} seconds".format(round(time.time() - start_time, 6)))
            self._save_data()
            
            # start_time = time.time()
            # self._cal_geom()
            # print("_cal_geom all geometries finished. Cost Time: {} seconds".format(round(time.time() - start_time, 6)))
            # self._save_data_sp()
            
        self._calc_functions()
        # self.draw_geom()

    @staticmethod
    def _calc_v_cw(p_c, rbt_def, num):
        t = sympy.symbols('t')
        v_cw = sympy.diff(p_c.subs(rbt_def.subs_q2qt), t)
        v_cw = v_cw.subs(rbt_def.subs_dqt2dq + rbt_def.subs_qt2q)
        v_cw = sympy.simplify(v_cw) 
        print(f'frame {num}: calc v_cw finished')
        return v_cw
        
    @staticmethod
    def _calc_w_b(R, rbt_def, num):
        R_t = R.subs(rbt_def.subs_q2qt)
        dR_t = sympy.diff(R_t)
        dR = dR_t.subs(rbt_def.subs_dqt2dq + rbt_def.subs_qt2q)
        w_b = sympy.simplify(utils.so32vec(R.transpose() * dR))
        print(f'frame {num}: calc w_b finished')
        return w_b
    
    def _multi_process_cal_geom(self):
        self.T_0n = list(sympy.simplify(range(self.rbt_def.frame_num)))
        self.p_n = list(sympy.simplify(range(self.rbt_def.frame_num)))
        self.T_0nc = list(sympy.simplify(range(self.rbt_def.frame_num)))
        self.p_c = list(sympy.simplify(range(self.rbt_def.frame_num)))
        self.R = list(sympy.simplify(range(self.rbt_def.frame_num)))
        self.v_cw = list(sympy.simplify(range(self.rbt_def.frame_num)))
        self.w_b = list(sympy.simplify(range(self.rbt_def.frame_num)))

        for num in self.rbt_def.link_nums:
            if num == 0:
                self.T_0n[num] = self.rbt_def.dh_T[num]
                continue
            self.T_0n[num] = self.T_0n[self.rbt_def.prev_link_num[num]] * self.rbt_def.dh_T[num]
            self.R[num] = self.T_0n[num][0:3, 0:3]
            self.p_n[num] = self.T_0n[num][0:3, 3]
            self.T_0nc[num] = sympy.sympify(self.T_0n[num] * utils.translation_transmat(self.rbt_def.r_by_ml[num]))
            self.p_c[num] = self.T_0nc[num][0:3, 3]
            
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        task_1_results = [pool.apply_async(self._calc_v_cw, args=(self.p_c[num], self.rbt_def, num)) for num in self.rbt_def.link_nums if num != 0]
        task_2_results = [pool.apply_async(self._calc_w_b, args=(self.R[num], self.rbt_def, num)) for num in self.rbt_def.link_nums if num != 0]
        pool.close()
        pool.join()

        self.v_cw[1:] = [res.get() for res in task_1_results]
        self.w_b[1:] = [res.get() for res in task_2_results]
        

    def _cal_geom(self):
        self.T_0n = list(sympy.simplify(range(self.rbt_def.frame_num)))
        self.p_n = list(sympy.simplify(range(self.rbt_def.frame_num)))
        self.T_0nc = list(sympy.simplify(range(self.rbt_def.frame_num)))
        self.p_c = list(sympy.simplify(range(self.rbt_def.frame_num)))
        self.R = list(sympy.simplify(range(self.rbt_def.frame_num)))
        self.v_cw = list(sympy.simplify(range(self.rbt_def.frame_num)))
        self.w_b = list(sympy.simplify(range(self.rbt_def.frame_num)))

        t = sympy.symbols('t')

        for num in self.rbt_def.link_nums:
            print('Frame: {}'.format(num))
            start_time = time.time()
            if num == 0:
                self.T_0n[num] = self.rbt_def.dh_T[num]
                continue
            self.T_0n[num] = self.T_0n[self.rbt_def.prev_link_num[num]] * self.rbt_def.dh_T[num]
            self.R[num] = self.T_0n[num][0:3, 0:3]
            self.p_n[num] = self.T_0n[num][0:3, 3]
            self.T_0nc[num] = sympy.sympify(self.T_0n[num] * utils.translation_transmat(self.rbt_def.r_by_ml[num]))
            self.p_c[num] = self.T_0nc[num][0:3, 3]
            
            v_cw = sympy.diff(self.p_c[num].subs(self.rbt_def.subs_q2qt), t)
            v_cw = v_cw.subs(self.rbt_def.subs_dqt2dq + self.rbt_def.subs_qt2q)
            self.v_cw[num] = sympy.simplify(v_cw)

            R_t = self.R[num].subs(self.rbt_def.subs_q2qt)
            dR_t = sympy.diff(R_t)
            dR = dR_t.subs(self.rbt_def.subs_dqt2dq + self.rbt_def.subs_qt2q)
            self.w_b[num] = sympy.simplify(utils.so32vec(self.R[num].transpose() * dR))
   

    def _calc_functions(self):
        self.p_n_func = ["" for x in range(self.rbt_def.frame_num)]
        #self.p_n_func = np.zeros(self.rbt_def.dof)
        input_vars = tuple(self.rbt_def.coordinates)

        for num in range(self.rbt_def.frame_num):
            self.p_n_func[num] = sympy.lambdify(input_vars, self.p_n[num])


    def draw_geom(self, angle=0):
        frame_drawer = FrameDrawer((-0.6, 0.6), (-0.4, 0.4), (-0.7, 0.7))

        if angle == 0:
            subs_q2zero = [(q, angle) for q in self.rbt_def.coordinates]
        else :
            subs_q2zero = []
            x = self.rbt_def.coordinates
            for i in range(len(x)):
                subs_q2zero.append((x[i], angle[i]))

        for num in self.rbt_def.link_nums:
            T = np.matrix(self.T_0n[num].subs(subs_q2zero))
            frame_drawer.draw_frame(T, num)
            #print(T[0:3, 3])
            if num != 0:
                T_prev = np.matrix(self.T_0n[self.rbt_def.prev_link_num[num]].subs(subs_q2zero))
                frame_drawer.drawSegment(T_prev, T)

        frame_drawer.show()

    def _save_data(self):
        data = [('T_0n', self.T_0n),
                ('p_n', self.p_n),
                ('T_0nc', self.T_0nc),
                ('p_c', self.p_c),
                ('R', self.R),
                ('v_cw', self.v_cw),
                ('w_b', self.w_b)]

        utils.save_data(self.model_folder, 'kinematics', data)
        
    def _save_data_sp(self):
        data = [('T_0n', self.T_0n),
                ('p_n', self.p_n),
                ('T_0nc', self.T_0nc),
                ('p_c', self.p_c),
                ('R', self.R),
                ('v_cw', self.v_cw),
                ('w_b', self.w_b)]

        utils.save_data(self.model_folder, 'kinematics_sp', data)

    def _load_data(self):
        data = utils.load_data(self.model_folder, 'kinematics')
        for key, value in data:
            match key:
                case 'T_0n':
                    self.T_0n = value
                case 'p_n':
                    self.p_n = value
                case 'T_0nc':
                    self.T_0nc = value
                case 'p_c':
                    self.p_c = value
                case 'R':
                    self.R = value
                case 'v_cw':
                    self.v_cw = value
                case 'w_b':
                    self.w_b = value
                    
    def _load_data_sp(self):
        data = utils.load_data(self.model_folder, 'kinematics_sp')
        for key, value in data:
            match key:
                case 'T_0n':
                    self.T_0n_sp = value
                case 'p_n':
                    self.p_n_sp = value
                case 'T_0nc':
                    self.T_0nc_sp = value
                case 'p_c':
                    self.p_c_sp = value
                case 'R':
                    self.R_sp = value
                case 'v_cw':
                    self.v_cw_sp = value
                case 'w_b':
                    self.w_b_sp = value

