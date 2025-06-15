/*
 * 自动生成的代码 - 使用SymPy CSE优化
 * 表达式计算函数
 */

#include <cmath>

void get_tau_function(const double* q, const double* dq, const double* ddq, const double* p, double* results) {

    //符号变量
    const double q0 = q[0];
    const double dq0 = dq[0];
    const double ddq0 = ddq[0];
    const double q1 = q[1];
    const double dq1 = dq[1];
    const double ddq1 = ddq[1];

    //参数变量
    const double p0 = params[0];
    const double p1 = params[1];
    const double p2 = params[2];
    const double p3 = params[3];
    const double p4 = params[4];
    const double p5 = params[5];
    const double p6 = params[6];
    const double p7 = params[7];

    //公共子表达式
    double x0 = sin(q1);
    double x1 = cos(q1);
    double x2 = pow(dq1, 2);
    double x3 = 2*q1;
    double x4 = sin(x3);
    double x5 = cos(x3);
    double x6 = 2*dq0*dq1;
    double x7 = ddq0*x5;
    double x8 = x4*x6;
    double x9 = (1.0/2.0)*p5;
    double x10 = (1.0/2.0)*p7;
    double x11 = pow(dq0, 2);
    double x12 = x11*x4;
    
    results[0] = p2*(ddq0*x4 + x5*x6) + p3*(ddq1*x1 - x0*x2) + p4*(ddq1*x0 + x1*x2) + x10*(ddq0 - x7 + x8) + x9*(ddq0 + x7 - x8);
    results[1] = ddq0*p3*x1 + ddq0*p4*x0 + ddq1*p6 - 9.8100000000000005*p0*x0 + 9.8100000000000005*p1*x1 - p2*x11*x5 - x10*x12 + x12*x9;
}