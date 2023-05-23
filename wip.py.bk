from sym_linktree import *
import numpy as np
from sym_jointlink import *
import sys
import time
from sym_arith import *
from sympy import *
import control as ct
from integrator import *
from wip_control import *
from wip_env import *

# ref
# https://www.ascento.ethz.ch/wp-content/uploads/2019/05/AscentoPaperICRA2019.pdf
# https://arxiv.org/pdf/2005.11431.pdf
# https://www2.akita-nct.ac.jp/libra/report/46/46038.pdf

r,l,Iw,Ib,mw,mb = symbols('r l Iw Ib mw mb')
fx,fy = symbols('fx fy') # ext force to bottom of wheel
u = symbols('u') # motor torq
x0 = symbols('x0') # x-offset
a0 = symbols('a0') # angle-offset
g = symbols('g')

context = { l: 0.5, r: 0.05,
        mw: 1, mb: 9,
        Iw: 1./800, Ib: 2.25,
        g: 9.81
        }

# NOTE normal force is ignored due to constraint
def sym_models_ground():
    # initial wheel angle should be vertical
    jl1 = WheelJointLink("qw", mw, r, RackPinionJoint(r, x0), XT=Xpln(pi/2, 0, 0), Icog=Iw)
    I = transInertia(mcI(mb, [l, 0], Ib), Xpln(a0, 0, 0))
    jl2 = StickJointLink("qs", mb, l, RevoluteJoint(), I=I, tau=u)
    #jl1 = WheelJointLink("qw", mw, r, RackPinionJoint(r, x0), XT=Xpln(pi/2, 0, 0), Icog=Iw)
    #jl2 = StickJointLink("qs", mb, l, RevoluteJoint(), cx=l, I=transInertia(mcI(mb, [l,0], Ib), Xpln(a0, 0,0)), tau=u)
    plant_model = LinkTreeModel([jl1, jl2], g)
    return plant_model

# NOTE: axis order is important
# initial wheel angle should be vertical
def sym_models_air():
    jl1 = StickJointLink("x", 0, 0, PrismaticJoint(), XT=Xpln(pi/2, 0, 0)) # fict
    jl2 = StickJointLink("y", 0, 0, PrismaticJoint(), XT=Xpln(0, 0, 0))    # fict
    jl3 = WheelJointLink("qw", mw, r, RevoluteJoint(), XT=Xpln(0, 0, 0), Icog=Iw)
    I = transInertia(mcI(mb, [l, 0], Ib), Xpln(a0, 0, 0))
    jl4 = StickJointLink("qs", mb, l, RevoluteJoint(), I=I, tau=u)
    plant_model = LinkTreeModel([jl1, jl2, jl3, jl4], g)
    return plant_model

def test():
    a0v = np.pi/8
    model_g = sym_models_ground()
    ddqf_g = model_g.gen_ddq_f([u, x0], context | {a0:a0v})
    draw_g_cmds = model_g.gen_draw_cmds([x0], context | {a0:a0v})

    model_a = sym_models_air()
    ddqf_a = model_a.gen_ddq_f([u, fx, fy], context | {a0:a0v})
    draw_a_cmds = model_a.gen_draw_cmds([], context | {a0:a0v})

    reuse = True
    if reuse: # K is independent of a0
        K = np.array([[-1, 41.26540066, 125.12381105]])
    else:
        A, B = linealize(model_g, a0)
        K = wip_gain(context, A, B)

    friction_impulse = gen_friction_impulse(model_a, fx, r, context | {a0:a0v})

    import graphic
    viewer = graphic.Viewer(scale=200, offset=[0, 0.2])
    dt = 0.001
    v_ref = 0

    def event_handler(key, shifted):
        nonlocal v_ref
        if key == 'q':
            sys.exit()
        elif key == 'l':
            v_ref = 5
        elif key == 'h':
            v_ref = -5
        elif key == 'j':
            v_ref = 0

    t = 0
    in_air = True
    x0_v = 0

    in_air = True
    #in_air = False

    if in_air:
        q_v = np.array([-1, 1, 0, 0], dtype=np.float64)
        dq_v = np.array([1, 0, 0, 0], dtype=np.float64)
    else:
        q_v = np.array([-2.60627075e-15,  2.61932652e-15])
        dq_v= np.array([ 13.17562667, -13.11639202])

    while True:
        t = t + dt
        if in_air:
            b = 6000.
            k = b * b / 4*(context[mb] + context[mw]) # zeta == 1
            fyv = normal(k, b, context[r], q_v, dq_v)
            q_v, dq_v = euler_step(ddqf_a, q_v, dq_v, dt, [0, 0, fyv])
            cmds = draw_a_cmds(q_v, dq_v, [])
            #print(t, "air", q_v, dq_v)
            if fyv > 0:
                print("contact!!")
                dq_v = friction_impulse(q_v, dq_v)
                x0_v = q_v[0] - context[r] * q_v[2]
                q_v = np.array([q_v[2] , q_v[3]])
                dq_v = dq_v[2:]
                in_air = False
                #print(t, "air->gnd", x0_v + context[r] * q_v[0], -(context[r] * dq_v[0]))
                #print(t, "air->gnd", q_v, dq_v)
        else:
            v_u = wip_wheel_torq(K, v_ref, q_v, dq_v, a0v)
            q_v, dq_v = euler_step(ddqf_g, q_v, dq_v, dt, [v_u, x0_v])
            cmds = draw_g_cmds(q_v, dq_v, [x0_v])
            #print(t, "gnd", dq_v)
        viewer.handle_event(event_handler)
        viewer.clear()
        #viewer.text([f"t: {t:.03f} :q {q[0]:.03f} {q[1]:.03f}"])
        viewer.draw(cmds)
        viewer.draw_horizon(0)
        viewer.flush(dt)

if __name__ == '__main__':
    test()
