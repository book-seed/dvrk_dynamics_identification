/*
 * 自动生成的代码 - 使用SymPy CSE优化
 * 表达式计算函数
 */

#include <cmath>

void get_H_function(const double* q, const double* dq, const double* ddq, double* results) {

    //符号变量
    const double q0 = q[0];
    const double dq0 = dq[0];
    const double ddq0 = ddq[0];
    const double q1 = q[1];
    const double dq1 = dq[1];
    const double ddq1 = ddq[1];

    //公共子表达式
    double x0 = cos(q1);
    double x1 = pow(x0, 2);
    double x2 = sin(q1);
    double x3 = x0*x2;
    double x4 = 2*dq0*dq1;
    double x5 = x3*x4;
    double x6 = pow(dq1, 2);
    double x7 = ddq0*x0;
    double x8 = pow(x2, 2);
    double x9 = pow(dq0, 2);
    double x10 = x3*x9;
    
    results[0] = 0;
    results[1] = 0;
    results[2] = 0;
    results[3] = ddq0;
    results[4] = 0;
    results[5] = 0;
    results[6] = 0;
    results[7] = 0;
    results[8] = 0;
    results[9] = 0;
    results[10] = ddq0*x1 - x5;
    results[11] = ddq1*x0 - x2*x6;
    results[12] = x1*x4 + 2*x2*x7 - x4*x8;
    results[13] = 0;
    results[14] = ddq1*x2 + x0*x6;
    results[15] = ddq0*x8 + x5;
    results[16] = 0;
    results[17] = 0;
    results[18] = 0;
    results[19] = 0;
    results[20] = 0;
    results[21] = 0;
    results[22] = 0;
    results[23] = 0;
    results[24] = 0;
    results[25] = 0;
    results[26] = 0;
    results[27] = 0;
    results[28] = 0;
    results[29] = 0;
    results[30] = x10;
    results[31] = x7;
    results[32] = -x1*x9 + x8*x9;
    results[33] = ddq1;
    results[34] = ddq0*x2;
    results[35] = -x10;
    results[36] = -9.8100000000000005*x2;
    results[37] = 0;
    results[38] = 9.8100000000000005*x0;
    results[39] = 0;
}