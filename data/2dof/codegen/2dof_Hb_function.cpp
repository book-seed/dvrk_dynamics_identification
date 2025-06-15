/*
 * 自动生成的代码 - 使用SymPy CSE优化
 * 表达式计算函数
 */

#include <cmath>

void get_Hb_function(const double* q, const double* dq, const double* ddq, double* results) {

    //符号变量
    const double q0 = q[0];
    const double dq0 = dq[0];
    const double ddq0 = ddq[0];
    const double q1 = q[1];
    const double dq1 = dq[1];
    const double ddq1 = ddq[1];

    //公共子表达式
    double x0 = sin(q1);
    double x1 = cos(q1);
    double x2 = ddq0*x1;
    double x3 = pow(x0, 2);
    double x4 = 2*dq0*dq1;
    double x5 = pow(x1, 2);
    double x6 = pow(dq1, 2);
    double x7 = x0*x1;
    double x8 = pow(dq0, 2);
    
    results[0] = 0;
    results[1] = 0;
    results[2] = 2*x0*x2 - x3*x4 + x4*x5;
    results[3] = ddq1*x1 - x0*x6;
    results[4] = ddq1*x0 + x1*x6;
    results[5] = ddq0*x3 + x4*x7;
    results[6] = 0;
    results[7] = ddq0;
    results[8] = -9.8100000000000005*x0;
    results[9] = 9.8100000000000005*x1;
    results[10] = x3*x8 - x5*x8;
    results[11] = x2;
    results[12] = ddq0*x0;
    results[13] = -x7*x8;
    results[14] = ddq1;
    results[15] = 0;
}