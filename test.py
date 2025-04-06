import sympy as sp
from utils import utils

q0, q1, q2, q3= utils.new_sym('q:4')
p0, p1, p2, p3= utils.new_sym('p:4')
s0, s1, s2, s3= utils.new_sym('s:4')
z0, z1, z2, z3= utils.new_sym('z:4')

A1 = [q0, q1, q2, q3]
A2 = [p0, p1, p2, p3]
B1 = [s0, s1, s2, s3]
B2 = [z0, z1, z2, z3]

def main():
    
    err1 = []
    err2 = []

    for i in range(0,4) :
        err1.append(sp.simplify(A1[i] - A2[i]))
        err2.append(sp.simplify(B1[i] - B2[i]))
                
    print(f'err1 = {err1}')
    print(f'err2 = {err2}')


if __name__ == "__main__":
    main()

    