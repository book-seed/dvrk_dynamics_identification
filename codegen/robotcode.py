import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ccodegen import generate_c_code



class CodeGen:
    def __init__(self, model, folder):
        symbol_vars_ = model.coordinates + model.d_coordinates + model.dd_coordinates
        param_vars = [f'p{i}' for i in range(model.base_num)]

        # H
        res = generate_c_code(model.H, symbol_vars_,
                              output_file=folder / f"{model.name}_H_function.cpp",
                              func_name="get_H_function")
        print(res)

        # H_b
        res = generate_c_code(model.H_b, symbol_vars_,
                              output_file=folder / f"{model.name}_Hb_function.cpp",
                              func_name="get_Hb_function")
        print(res)

        # tau
        res = generate_c_code(model.tau, symbol_vars_, param_vars=param_vars,
                              output_file=folder / f"{model.name}_tau_function.cpp",
                              func_name="get_tau_function")
        print(res)

        # G
        res = generate_c_code(model.G, symbol_vars_, param_vars=param_vars,
                              output_file=folder / f"{model.name}_G_function.cpp",
                              func_name="get_G_function")
        print(res)

        # C
        res = generate_c_code(model.C, symbol_vars_, param_vars=param_vars,
                              output_file=folder / f"{model.name}_C_function.cpp",
                              func_name="get_C_function")
        print(res)

        # M
        res = generate_c_code(model.M, symbol_vars_, param_vars=param_vars,
                              output_file=folder / f"{model.name}_M_function.cpp",
                              func_name="get_M_function")
        print(res)