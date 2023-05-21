from sympy import *
import numpy as np

# fx: sym of contact force x
# r: sym of wheel radius
def gen_friction_impulse(model_a, fx, r, context):
    [q1, q2, q3, q4] = model_a.q()
    [dq1, dq2, dq3, dq4] = model_a.dq()
    [ddq1, ddq2, ddq3, ddq4] = model_a.ddq()
    fext = [zeros(3,1) for i in range(4)]
    #fext[2][0] = - fx * (q2 - r) # global coordinate torq!! but q2 == 0
    fext[2][0] = - fx * ( - r) # global coordinate torq!! but q2 == 0
    fext[2][1] = fx
    syms = model_a.q() + model_a.dq()
    Cfric = diff(model_a.inverse_dynamics([0 for i in range(model_a.NB)], fext, impulse=True).subs(context), fx)
    Cfric_f = lambdify(syms, Cfric)
    H = model_a.H.subs(context)
    H_f = lambdify(syms, H)
    rv = context[r]
    # ddq[i] = a[i] * fx (impulse)
    # -r * (dq3 + ddq3) =  dq1 + ddq1
    # <=> (-ddq1 - r ddq3) = r * dq3 + dq1
    # <=>  fx = (r * dq3 + dq1) / (-a1 - r a3)

    def friction_impulse(qv, dqv):
        Hv = H_f(*qv, *dqv)
        Cv = Cfric_f(*qv, *dqv)
        ddqv = np.linalg.solve(Hv, -Cv).reshape(-1)
        dq3v = dqv[2]
        dq1v = dqv[0]
        ddqv1 = ddqv[0]
        ddqv3 = ddqv[2]
        fxv = (rv * dq3v + dq1v) / (-ddqv1 - rv * ddqv3)
        new_dqv = dqv + ddqv * fxv
        return new_dqv
    return friction_impulse

# k: elastic
# b: friction
# r: wheel radius value
def normal(k, b, r, qv, dqv):
    y = qv[1] - r
    dy = dqv[1]
    if y >= 0:
        return 0
    return - k * y - b * dy


