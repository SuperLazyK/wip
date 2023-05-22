import numpy as np

def euler_step(ddq_f, q, dq, fext, dt, v_u):
    ddq = ddq_f(q, dq, fext, v_u)
    new_dq = dq + ddq * dt
    new_q = q + new_dq * dt
    return new_q, new_dq

