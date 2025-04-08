import sympy
import numpy as np
from dynamics.dyn_param_dep import find_dyn_parm_deps
import copy
import time
from utils import utils
import multiprocessing
import dill
from multiprocessing.reduction import ForkingPickler


# 让 ForkingPickler 使用 dill 进行序列化
ForkingPickler.dumps = dill.dumps


class Dynamics:
    def __init__(self, rbt_def, geom, config, g=[0, 0, -9.81]):

        self.rbt_def = rbt_def
        self.model_folder = '/data/' + self.rbt_def.name + '/model/'
        self.geom = geom
        self._g = np.matrix(g)

        condition = 'load_dynamic_from_file'
        if next((value for key, value in config if key == condition), None) is True:
            print("load dynamic data from file starting.")
            start_time = time.time()
            self._load_data()
            # self._load_data_sp()
            print("load dynamic data from file finished. Cost Time: {} seconds".format(round(time.time() - start_time, 6)))
            
            # 验证多进程结果与单进程结果的一致性
            # err_tau = [sympy.simplify(a - b) for a, b in zip(self.tau, self.tau_sp)]
            # err_H   = [sympy.simplify(a - b) for a, b in zip(self.H, self.H_sp)]
            # err_M   = [sympy.simplify(a - b) for a, b in zip(self.M, self.M_sp)]
            # err_C   = [sympy.simplify(a - b) for a, b in zip(self.C, self.C_sp)]
            # err_G   = [sympy.simplify(a - b) for a, b in zip(self.G, self.G_sp)]
            # print(f'err_tau: {err_tau}')
            # print(f'err_H: {err_H}')
            # print(f'err_M: {err_M}')
            # print(f'err_C: {err_C}')
            # print(f'err_G: {err_G}')
        else:
            start_time = time.time()
            self.tau = self._multi_process_calc_dyn()
            print("_multi_process_calc_dyn calc dynamic finished. Cost Time: {} seconds".format(round(time.time() - start_time, 6)))
            start_time = time.time()
            self._multi_process_calc_H_MCG()
            print("_multi_process_calc_HMCG Cost Time: {} seconds".format(round(time.time() - start_time, 6)))
            self._save_data()
            
            # 单进程
            # start_time = time.time()
            # self._calc_dyn()
            # print("_calc_dyn calc dynamic finished. Cost Time: {} seconds".format(round(time.time() - start_time, 6)))
            # start_time = time.time()
            # self._calc_H_MCG()
            # print("_calc_HMCG Cost Time: {} seconds".format(round(time.time() - start_time, 6)))
            # self._save_data_sp()

        self._calc_H_func()
        self._calc_base_param()


        print("finished creating robot dynamics")

    def _calc_dyn(self):
        print("calculating lagrangian...")
        # Calculate kinetic energy and potential energy
        p_e = 0
        k_e = 0
        start_time = time.time()
        for num in self.rbt_def.link_nums[1:]:
            print("calculating the link kinetic energy of {}/{}".format(num, self.rbt_def.link_nums[-1]))
            p_e += -self.rbt_def.m[num] * self.geom.p_c[num].dot(self._g)  # m*g*h

            k_e_n = 0
            if self.rbt_def.use_inertia[num]:
                # 1/2 * m * v^2 + 1/2 * I * w^2
                k_e_n = self.rbt_def.m[num] * self.geom.v_cw[num].dot(self.geom.v_cw[num])/2 +\
                       (self.geom.w_b[num].transpose() * self.rbt_def.I_by_Llm[num] * self.geom.w_b[num])[0, 0]/2

                # k_e_n = sympy.simplify(k_e_n) # this is replaced by the following code to reduce time cost
                k_e_n = sympy.factor(sympy.expand(k_e_n) - sympy.expand(k_e_n * self.rbt_def.m[num]).subs(self.rbt_def.m[num], 0)/self.rbt_def.m[num])

            k_e += k_e_n

        # Lagrangian
        L = k_e - p_e
        print("_calc_dyn_L Cost Time: {} seconds".format(round(time.time() - start_time, 6)))   

        start_time = time.time()
        tau = []
        print("calculating joint torques...")
        for q, dq in zip(self.rbt_def.coordinates, self.rbt_def.d_coordinates):
            print("tau of {}".format(q))
            dk_ddq = sympy.diff(k_e, dq)
            dk_ddq_t = dk_ddq.subs(self.rbt_def.subs_q2qt + self.rbt_def.subs_dq2dqt)
            dk_ddq_dtt = sympy.diff(dk_ddq_t, sympy.Symbol('t'))
            dk_ddq_dt = dk_ddq_dtt.subs(self.rbt_def.subs_ddqt2ddq + self.rbt_def.subs_dqt2dq + self.rbt_def.subs_qt2q)
            dL_dq = sympy.diff(L, q)
            tau.append(sympy.expand(dk_ddq_dt - dL_dq))
        print("_calc_dyn_tau Cost Time: {} seconds".format(round(time.time() - start_time, 6)))   

        print("adding frictions and springs...")
        tau = copy.deepcopy(tau)

        for i in range(self.rbt_def.frame_num):
            dq = self.rbt_def.dq_for_frame[i]

            if self.rbt_def.use_friction[i]:
                tau_f = sympy.sign(dq) * self.rbt_def.Fc[i] + dq * self.rbt_def.Fv[i] + self.rbt_def.Fo[i]
                for j in range(len(self.rbt_def.d_coordinates)):
                    dq_da = sympy.diff(dq, self.rbt_def.d_coordinates[j])
                    tau[j] += dq_da * tau_f

            if self.rbt_def.spring_dl[i] is not None:
                tau_s = self.rbt_def.spring_dl[i] * self.rbt_def.K[i]
                for j in range(len(self.rbt_def.d_coordinates)):
                    dq_da = sympy.diff(dq, self.rbt_def.d_coordinates[j])
                    tau[j] -= dq_da * tau_s

        print("Add motor inertia...")

        for i in range(self.rbt_def.frame_num):
            if self.rbt_def.use_Ia[i]:
                tau_Ia = self.rbt_def.ddq_for_frame[i] * self.rbt_def.Ia[i]
                tau_index = self.rbt_def.dd_coordinates.index(self.rbt_def.ddq_for_frame[i].free_symbols.pop())
                tau[tau_index] += tau_Ia

        self.tau = tau
    
    # @staticmethod  
    # def _static_calc_Ln(num, rbt_def, geom, _g):
    #     print("calculating the link kinetic energy of {}/{}".format(num, rbt_def.link_nums[-1]))
    #     p_e_n = -rbt_def.m[num] * geom.p_c[num].dot(_g)  
    #     k_e_n = 0
    #     if rbt_def.use_inertia[num]:
    #         k_e_n = rbt_def.m[num] * geom.v_cw[num].dot(geom.v_cw[num])/2 +\
    #                    (geom.w_b[num].transpose() * rbt_def.I_by_Llm[num] * geom.w_b[num])[0, 0]/2
    #         k_e_n = sympy.factor(sympy.expand(k_e_n) - sympy.expand(k_e_n * rbt_def.m[num]).subs(rbt_def.m[num], 0)/rbt_def.m[num])
        
    #     L_n = k_e_n - p_e_n
    #     return k_e_n, L_n
    
    @staticmethod  
    def _static_calc_Ln(m, use_inertia, I_by_Llm, p_c, v_cw, w_b, g):
        p_e_n = -m * p_c.dot(g)  
        k_e_n = 0
        if use_inertia:
            k_e_n = m * v_cw.dot(v_cw)/2 + (w_b.transpose() * I_by_Llm * w_b)[0, 0]/2
            k_e_n = sympy.factor(sympy.expand(k_e_n) - sympy.expand(k_e_n * m).subs(m, 0)/m)
        
        L_n = k_e_n - p_e_n
        return k_e_n, L_n
    
    @staticmethod
    def _static_calc_tau_for_pair(q, dq, rbt_def, k_e, L):
        dk_ddq = sympy.diff(k_e, dq)
        dk_ddq_t = dk_ddq.subs(rbt_def.subs_q2qt + rbt_def.subs_dq2dqt)
        dk_ddq_dtt = sympy.diff(dk_ddq_t, sympy.Symbol('t'))
        dk_ddq_dt = dk_ddq_dtt.subs(rbt_def.subs_ddqt2ddq + rbt_def.subs_dqt2dq + rbt_def.subs_qt2q)
        dL_dq = sympy.diff(L, q)
        return sympy.expand(dk_ddq_dt - dL_dq)
      
    def _multi_process_calc_dyn(self):
        
        print("calculating lagrangian...")
        start_time = time.time()
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        tasks = []
        for num in self.rbt_def.link_nums[1:]:
            # task = pool.apply_async(self._static_calc_Ln, args=(num, self.rbt_def, self.geom, self._g))
            task = pool.apply_async(self._static_calc_Ln, args=(self.rbt_def.m[num], self.rbt_def.use_inertia[num], self.rbt_def.I_by_Llm[num], 
                                                                self.geom.p_c[num], self.geom.v_cw[num], self.geom.w_b[num], self._g))
            tasks.append(task)
        pool.close()
        pool.join()
        results = [task.get() for task in tasks]
        k_e = sum([result[0] for result in results])
        L = sum([result[1] for result in results])
        print("_multi_process_calc_dyn_Le Cost Time: {} seconds".format(round(time.time() - start_time, 6)))    

        print("calculating joint torques...")  
        start_time = time.time()    
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        tasks = []
        for q, dq in zip(self.rbt_def.coordinates, self.rbt_def.d_coordinates):
            task = pool.apply_async(self._static_calc_tau_for_pair, args=(q, dq, self.rbt_def, k_e, L))
            tasks.append(task)
        pool.close()
        pool.join()
        tau = [task.get() for task in tasks]
        print("_multi_process_calc_dyn_tau Cost Time: {} seconds".format(round(time.time() - start_time, 6)))   
 
        print("adding frictions and springs...")
        tau = copy.deepcopy(tau)

        for i in range(self.rbt_def.frame_num):
            dq = self.rbt_def.dq_for_frame[i]

            if self.rbt_def.use_friction[i]:
                tau_f = sympy.sign(dq) * self.rbt_def.Fc[i] + dq * self.rbt_def.Fv[i] + self.rbt_def.Fo[i]
                for j in range(len(self.rbt_def.d_coordinates)):
                    dq_da = sympy.diff(dq, self.rbt_def.d_coordinates[j])
                    tau[j] += dq_da * tau_f

            if self.rbt_def.spring_dl[i] is not None:
                tau_s = self.rbt_def.spring_dl[i] * self.rbt_def.K[i]
                for j in range(len(self.rbt_def.d_coordinates)):
                    dq_da = sympy.diff(dq, self.rbt_def.d_coordinates[j])
                    tau[j] -= dq_da * tau_s

        print("Add motor inertia...")

        for i in range(self.rbt_def.frame_num):
            if self.rbt_def.use_Ia[i]:
                tau_Ia = self.rbt_def.ddq_for_frame[i] * self.rbt_def.Ia[i]
                tau_index = self.rbt_def.dd_coordinates.index(self.rbt_def.ddq_for_frame[i].free_symbols.pop())
                tau[tau_index] += tau_Ia

        return tau

    def _calc_H(self):
        print("calculating H ...")
        self.H, b = sympy.linear_eq_to_matrix(self.tau, self.rbt_def.bary_params)

    @staticmethod
    def _static_calc_H(tau, rbt_def):
        print("calculating H ...")
        A, b = sympy.linear_eq_to_matrix(tau, rbt_def.bary_params)
        return A  
    
    def _calc_H_func(self):
        print("calculating H_func ...")
        input_vars = tuple(self.rbt_def.coordinates +self.rbt_def.d_coordinates + self.rbt_def.dd_coordinates)
        self.H_func = sympy.lambdify(input_vars, self.H)
          
    def _calc_M(self):
        print("calculating M ...")
        self.M, b = sympy.linear_eq_to_matrix(self.tau, self.rbt_def.dd_coordinates)
    
    @staticmethod
    def _static_calc_M(tau, rbt_def):
        print("calculating M ...")
        A, b = sympy.linear_eq_to_matrix(tau, rbt_def.dd_coordinates)
        return A
    
    def _calc_G(self):
        print("calculating G ...")
        subs_qdq2zero = [(dq, 0) for dq in self.rbt_def.d_coordinates]
        subs_qdq2zero += [(ddq, 0) for ddq in self.rbt_def.dd_coordinates]
        self.G = sympy.Matrix(self.tau).subs(subs_qdq2zero)

    @staticmethod
    def _static_calc_G(tau, rbt_def):
        print("calculating G ...")
        subs_qdq2zero = [(dq, 0) for dq in rbt_def.d_coordinates]
        subs_qdq2zero += [(ddq, 0) for ddq in rbt_def.dd_coordinates]
        A = sympy.Matrix(tau).subs(subs_qdq2zero)
        return A
    
    def _calc_C(self):
        print("calculating C ...")
        subs_ddq2zero = [(ddq, 0) for ddq in self.rbt_def.dd_coordinates]
        self.C = sympy.Matrix(self.tau).subs(subs_ddq2zero) - self.G

    @staticmethod
    def _static_calc_C(tau, rbt_def):
        print("calculating C ...")
        subs_ddq2zero = [(ddq, 0) for ddq in rbt_def.dd_coordinates]
        CG = sympy.Matrix(tau).subs(subs_ddq2zero)
        return CG

    def _calc_H_MCG(self):
        print("Calculating H, M, C and G...")
        self._calc_H()
        self._calc_M()
        self._calc_G()
        self._calc_C()
        
    def _multi_process_calc_H_MCG(self):
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            tasks = [(self._static_calc_H, (self.tau, self.rbt_def)),
                     (self._static_calc_M, (self.tau, self.rbt_def)),
                     (self._static_calc_C, (self.tau, self.rbt_def)),
                     (self._static_calc_G, (self.tau, self.rbt_def))]

            results = []
            for task, args in tasks:
                result = pool.apply_async(task, args)
                results.append(result)

            pool.close()
            pool.join()

            task_results = []
            for i, result in enumerate(results):
                task_results.append((tasks[i][0].__name__, result.get()))
                
            res_dict = dict(task_results)
            if '_static_calc_H' in res_dict:
                self.H = res_dict['_static_calc_H']
            if '_static_calc_M' in res_dict:
                self.M = res_dict['_static_calc_M']
            if '_static_calc_G' in res_dict:
                self.G = res_dict['_static_calc_G']
            if '_static_calc_C' in res_dict:
                self.C = res_dict['_static_calc_C'] - self.G        
        
    def _calc_base_param(self):
        print("calculating base parameter...")
        r, P_X, P = find_dyn_parm_deps(len(self.rbt_def.coordinates), len(self.rbt_def.bary_params), self.H_func)
        self.base_num = r
        print('base parameter number: {}'.format(self.base_num))
        self.base_param = P_X.dot(np.matrix(self.rbt_def.bary_params).transpose())

        P_b = P[:r].tolist()

        self.H_b = self.H[:, P_b]

        input_vars = tuple(self.rbt_def.coordinates + self.rbt_def.d_coordinates + self.rbt_def.dd_coordinates)

        print("Creating H_b function...")
        self.H_b_func = sympy.lambdify(input_vars, self.H_b)

    def _save_data(self):
        data = [('tau', self.tau),
                ('H', self.H),
                ('M', self.M),
                ('C', self.C),
                ('G', self.G)]

        utils.save_data(self.model_folder, 'dynamics', data)

    def _load_data(self):
        data = utils.load_data(self.model_folder, 'dynamics')
        for key, value in data:
            match key:
                case 'tau':
                    self.tau = value
                case 'H':
                    self.H = value
                case 'M':
                    self.M = value
                case 'C':
                    self.C = value
                case 'G':
                    self.G = value
                    
    def _save_data_sp(self):
        data = [('tau', self.tau),
                ('H', self.H),
                ('M', self.M),
                ('C', self.C),
                ('G', self.G)]

        utils.save_data(self.model_folder, 'dynamics_sp', data)

    def _load_data_sp(self):
        data = utils.load_data(self.model_folder, 'dynamics_sp')
        for key, value in data:
            match key:
                case 'tau':
                    self.tau_sp = value
                case 'H':
                    self.H_sp = value
                case 'M':
                    self.M_sp = value
                case 'C':
                    self.C_sp = value
                case 'G':
                    self.G_sp = value
