import numpy as np
import sympy
import os
from IPython.display import HTML, display
import tabulate
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors

class Identification:

    def __init__(self, iden, robot_model, traj_data):
        self._save_path = os.path.dirname(os.getcwd()) + iden.iden_res_file_ + '.pdf'
        self._param_num = robot_model.base_num
        self._H = robot_model.H_b_func
        self._base_param = robot_model.base_param
        self._Wb = None
        self._tau_s = None
        self._solve_mth = iden.solver_
        self._xb = None

        self._gen_regressor(traj_data)
        self._solver()
        self._save_and_evaluation()


    def _gen_regressor(self, data):
        sample_num, dof = data.q_filt_cut.shape
        self._Wb = np.zeros((sample_num * dof, self._param_num))
        self._tau_s = np.zeros(sample_num * dof)

        for i in range(sample_num):
            vars_input = data.q_filt_cut[i, :].tolist() + data.dq_filt_cut[i, :].tolist() + data.ddq_filt_cut[i, :].tolist()
            self._Wb[i * dof:(i + 1) * dof, :] = self._H(*vars_input)

            for d in range(dof):
                self._tau_s[i * dof + d] = data.tau_filt_cut[i, d]


    def _solver(self):
        if self._solve_mth == 'OLS':
            self._xb = np.linalg.lstsq(self._Wb, self._tau_s, rcond=None)[0]
        else:
            raise ValueError('solver does not exist')


    def _save_and_evaluation(self):

        table = [["Base Parameter", "Value"]]

        for i in range(self._param_num):
            param_str = str(sympy.Matrix(self._base_param)[i])
            max_disp_len = 50
            line = [param_str if len(param_str) <= max_disp_len
                    else param_str[:max_disp_len] + '...', self._xb[i]]
            table.append(line)

        html_table = tabulate.tabulate(table, tablefmt='html')
        doc = SimpleDocTemplate(self._save_path, pagesize=letter)
        elements = []

        # 将HTML表格转换为reportlab可处理的表格结构
        data = [row.split('<td>')[1:] for row in html_table.split('<tr>') if row.strip()]
        data = [[cell.rstrip('</td>') for cell in row] for row in data if row]
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
        doc.build(elements)





