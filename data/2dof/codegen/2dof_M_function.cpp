/*
 * 自动生成的代码 - 使用SymPy CSE优化
 * 表达式计算函数
 */

#include <cmath>

void get_M_function(const double* q, const double* dq, const double* ddq, const double* p, double* results) {

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
    double x0 = 2*q1;
    double x1 = p3*cos(q1) + p4*sin(q1);
    
    results[0] = p2*sin(x0) + (1.0/2.0)*p5*(1 - cos(x0)) + p7;
    results[1] = x1;
    results[2] = x1;
    results[3] = p6;
}