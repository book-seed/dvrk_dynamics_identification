from sympy import symbols, sin, cos, ccode, cse

"""
symbol_vars 为 q,dq,ddq组成的列表
"""

def generate_optimized_code(symbol_vars, expr_matrix):
    """
    生成优化后的C代码,使用SymPy的CSE消除公共子表达式

    参数:
        symbol_vars: 符号变量列表
        expr_matrix: 包含表达式的SymPy矩阵

    返回:
        tuple: (subexprs_str, results_str)
               subexprs_str: 优化后的子表达式字符串
               results_str : 最终结果表达式字符串
    """
    # 将矩阵转换为列表
    expr_list = expr_matrix.flat()

    # 使用 CSE 优化表达式
    subexprs_, reduced_exprs = cse(expr_list)

    # 构建子表达式字符串
    subexprs_str = "    //公共子表达式\n"
    for var, expr in subexprs_:
        subexprs_str += f"double {var} = {ccode(expr)};\n"

    # 构建结果表达式字符串
    results_str = "\n    //结果表达式\n"
    for i, expr in enumerate(reduced_exprs):
        results_str += f"double result{i + 1} = {ccode(expr)};\n"

    return subexprs_str, results_str


# ... 已有代码 ...

def generate_c_code(symbol_vars, expr_matrix, output_file=None, func_name="compute_expressions"):
    """
    生成完整的C++代码文件，包含优化后的表达式

    参数:
        symbol_vars: 符号变量列表
        expr_matrix: 包含表达式的SymPy矩阵
        output_file: 输出文件名，如果为None则返回代码字符串
        func_name: 生成的C++函数名，默认为 "compute_expressions"
    """
    subexprs_str, results_str = generate_optimized_code(symbol_vars, expr_matrix)

    # 假设 symbol_vars 按 q, dq, ddq 顺序排列，且数量相同
    num_vars = len(symbol_vars) // 3

    # 构建完整的C++代码
    c_code = f"""/*
 * 自动生成的代码 - 使用SymPy CSE优化
 * 表达式计算函数
 */

#include <cmath>

void {func_name}(const double* q, const double* dq, const double* ddq, double* results) 
{{
"""
    # 为每个符号变量添加解引用操作，包含 q1, dq1, ddq1
    for i in range(num_vars):
        c_code += f"    const double q{i} = q[{i}];\n"
        c_code += f"    const double dq{i} = dq[{i}];\n"
        c_code += f"    const double ddq{i} = ddq[{i}];\n"
    c_code += "\n"
    c_code += subexprs_str.replace("\n", "\n    ")
    c_code += "\n"
    # 直接将结果赋值给 results 数组
    results_lines = results_str.strip().split('\n')[1:]  # 跳过注释行
    for i, line in enumerate(results_lines):
        expr = line.split('=')[1].strip().rstrip(';')
        c_code += f"    results[{i}] = {expr};\n"
    c_code += "}"

    if output_file:
        with open(output_file, 'w') as f:
            f.write(c_code)
        return f"C++代码已写入到 {output_file}"
    else:
        return c_code