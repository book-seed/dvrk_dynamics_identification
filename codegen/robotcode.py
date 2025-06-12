import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ccodegen import generate_c_code



class CodeGen:
    def __init__(self, model):
        symbol_vars_ = model.coordinates + model.d_coordinates + model.dd_coordinates

        # H_b
        generate_c_code(symbol_vars_, model.H_b, f"dyn_Hb_function.cpp", "get_Hb_function")
        print("C代码已生成到 Hb_function.cpp")