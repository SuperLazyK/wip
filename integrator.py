import numpy as np

def euler_step(ddq_f, q, dq, fext, dt, v_u):
    ddq = ddq_f(q, dq, fext, v_u)
    new_dq = dq + ddq * dt
    new_q = q + new_dq * dt
    return new_q, new_dq

def rk4_step(ddq_f, q, dq, dt, v_u):

    k1_q = q
    k1_dq  = dq
    k1_ddq = ddq_f(q , k1_dq, v_u)

    k2_q = q + dt/2 * k1_dq
    k2_dq  = dq + dt/2 * k1_ddq
    k2_ddq = ddq_f(k2_q, k2_dq, v_u)

    k3_q = q + dt/2 * k2_dq
    k3_dq = dq + dt/2 * k2_ddq
    k3_ddq = ddq_f(k3_q, k3_dq, v_u)

    k4_q = q + dt * k3_dq
    k4_dq = dq + dt/2 * k3_ddq
    k4_ddq = ddq_f(k4_q, k4_dq, v_u)

    ret_dq = (k1_dq + 2*k2_dq + 2*k3_dq + k4_dq)/6
    ret_ddq = (k1_ddq + 2*k2_ddq + 2*k3_ddq + k4_ddq)/6

    return q + dt*ret_dq, dq + dt*ret_ddq


