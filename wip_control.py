import numpy as np
from sympy import *
import control as ct

# NOTE:
# if link has more than 2 bodies, assume dqq[j]=0 for j > 2
# This measn bias force Cj and inertial force H[:,:2] * q[:2] is
# canceled with joint force for j >2.
def wip_gain(model, context, a=symbols('a'), av=0, Q=np.diag([1,1,1]), R=1):
    phi,dphi,ddphi = symbols('phi dphi ddphi')
    th, dth, ddth  = symbols('th dth ddth')

    [q1, q2] = model.q()
    [dq1, dq2] = model.dq()
    [ddq1, ddq2] = model.ddq()

    u = model.jointlinks[1].tau
    H0 = model.H[:2,:2]
    C0 = Matrix(model.counter_joint_force()[:2]) # normal force is ignorable
    lhs = ((H0) * Matrix([[ddq1],[ddq2]]) + C0 - Matrix([[0], [u]])).subs(context)

    # q1 = -phi + th - a
    # q2 = phi
    lhs1 = expand(simplify(lhs.subs([ (q1,-phi+th-a)
                    , (q2, phi)
                    , (dq1, -dphi+dth)
                    , (dq2, dphi)
                    , (ddq1, -ddphi+ddth)
                    , (ddq2, ddphi)
                    ])),trig=True)
    lhs2 = Matrix([[lhs1[0]], [lhs1[0] - lhs1[1]]])

    # liniarize
    lhs2 = lhs2.subs([ (cos(th), 1)
                     , (sin(th), th)
                     , (dth**2, 0)
                     , (a, av)
                     ])

    H2 = lhs2.jacobian(Matrix([ddphi, ddth]))
    C2 = (lhs2 - H2 * Matrix([[ddphi], [ddth]]))

    H3 = H2.col_insert(2, Matrix([[0], [0]])).row_insert(2, Matrix([[0, 0, 1]]))
    rhs3 = H3.inv() * (-C2).row_insert(2, Matrix([[dth]]))

    x = Matrix([dphi, dth, th])
    A = (rhs3.jacobian(x))
    B = ((rhs3 - A * x).jacobian(Matrix([u])))
    #D = (rhs3 - A * x - B * u)

    print("A", A)
    print("B", B)
    # H x' = Ax + Bu
    K, _, E = ct.lqr(A, B, Q, R)
    print("gain", K)
    print("pole", E)

    return K

def wip_wheel_torq(K, ref_v, q, dq, a=0):
    # phi:q2
    # th:q2+q1 + a
    # NOTE: dth:q2+q1
    phi, th = q[1], q[0] + q[1] + a
    dphi, dth = dq[1], dq[0] + dq[1]
    ret = K @ np.array([ref_v - dphi, 0 - dth, 0 - th])
    return ret[0]

