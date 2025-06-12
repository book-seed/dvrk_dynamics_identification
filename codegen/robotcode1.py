import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from subexprs_yh import SubExprs
from ccodegen1 import robot_code_to_func
import sympy


def generate_M_code(robot_model):
    print('generating M matrix code')
    M_se = SubExprs()

    M_code = M_se.get(robot_model.M)

    M_func_def = robot_code_to_func('c', M_code, 'M', 'get_dyn_M_func', robot_model)
    print(M_func_def)

def generate_C_code(robot_model):
    print('generating C matrix code')
    C_se = SubExprs()
    C_func_def = robot_code_to_func('c', C_se.get(robot_model.C), 'C', 'get_dyn_C_func', robot_model)
    print(C_func_def)

def generate_tau_code(robot_model):
    print('generating tau code')
    C_se = SubExprs()
    c_code = C_se.get(sympy.Matrix(robot_model.tau))
    C_func_def = robot_code_to_func('c', c_code, 'C', 'get_dyn_tau_func', robot_model)
    print(C_func_def)