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

# ref
# https://www.ascento.ethz.ch/wp-content/uploads/2019/05/AscentoPaperICRA2019.pdf
# https://arxiv.org/pdf/2005.11431.pdf
# https://www2.akita-nct.ac.jp/libra/report/46/46038.pdf

r,ll,lh,Iw,Il,Ih,mw,ml,mh = symbols('r ll lh Iw Il Ih mw ml mh') # wheel leg hip
fx,fy = symbols('fx fy') # ext force to bottom of wheel
uw, uk = symbols('uw uk') # motor torq (wheel, knee)
x0 = symbols('x0') # offset
k = symbols('k') # torsion spring elastic coeff
g = symbols('g')

context = { mw: 1, Iw: 1./200, r: 0.1,
            ml: 1, Il: 0.09/12, ll: 0.3,
            mh: 8, Ih: 8*0.09/12, lh: 0.3,
            k:200,
            g: 9.81
        }

# NOTE normal force is ignored due to constraint
def sym_models_ground():
    # initial wheel angle should be vertical
    jl1 = WheelJointLink("qw", mw, r, RackPinionJoint(r, x0), XT=Xpln(pi/2, 0, 0), Icog=Iw)
    jl2 = StickJointLink("ql", ml, ll, RevoluteJoint(), XT=Xpln(-pi/2, ll, 0), cx=ll, Icog=Il, tau=uw)
    jl3 = StickSpringJointLink("qh", mh, lh, k, RevoluteJoint(), XT=Xpln(0, lh, 0), cx=lh, Icog=Ih, tau=uk)
    plant_model = LinkTreeModel([jl1, jl2, jl3], g)
    return plant_model


def virtual_wip_model_ground(model):
    vI = simplify(model.Ic[1])
    vm, cx, cy, _ = I2mc(vI)
    vl = simplify(sqrt(cx**2+cy**2))
    vtheta = atan2(cy, cx)
    X = Xpln(-vtheta, 0, 0)
    jl1 = WheelJointLink("vqw", mw, r, RackPinionJoint(r, x0), XT=Xpln(pi/2, 0, 0), Icog=Iw)
    jl2 = StickJointLink("vqs", vm, vl, RevoluteJoint(), cx=vl, I=transInertia(vI, X), tau=uw)
    vwip_model = LinkTreeModel([jl1, jl2], g)
    return vwip_model, vtheta


def test():
    model_g = sym_models_ground()
    max_torq_w = 3.5 # Nm
    max_torq_k = 40 # Nm
    max_knee_angle = np.deg2rad(-150)
    min_knee_angle = np.deg2rad(-45)
    qh = model_g.q()[2]
    vmodel_g, a0 = virtual_wip_model_ground(model_g)
    # [lh*mh*(-(dql+dqw)**2*ll*cos(qh) + g*cos(qh + ql + qw))]])
    cancel_bias_force = lambdify(model_g.q() + model_g.dq(), simplify(model_g.counter_joint_force()[2,0]).subs(context))
    a0f = lambdify([qh], a0.subs(context))
    reuse = True
    if reuse:
        K = np.array([[-1., 11.20462078, 50.02801177]])
    else:
        K = wip_gain(vmodel_g, context | {qh:0})

    ddqf_g = model_g.gen_ddq_f([uw, uk, x0], context)
    draw_g_cmds = model_g.gen_draw_cmds([x0], context)

    q_v = np.array([0, 0, 0], dtype=np.float64)
    dq_v = np.array([3, 0, 0], dtype=np.float64)
    #q_v = np.array([0, 0], dtype=np.float64)
    #dq_v = np.array([0, 0], dtype=np.float64)

    import graphic
    viewer = graphic.Viewer(scale=200, offset=[0, 0.2])

    t = 0
    dt = 0.001
    in_air = False

    v_ref = 0 # horizontal velocity
    p_ref = 0 # knee angle

    v_uw = 0
    v_uk = 0
    x0_v = 0
    printM(model_g.H)

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

    while True:
        t = t + dt
        if in_air:
            pass
        else:
            Kp = 1000
            Kd = 100
            v_uk = Kp*(p_ref - q_v[2]) - Kd * dq_v[2] + cancel_bias_force(*q_v, *dq_v)
            v_uw = wip_wheel_torq(K, v_ref, q_v, dq_v, a0f(0))
            v_uk = np.clip(v_uk, -max_torq_k, max_torq_k)
            v_uw = np.clip(v_uw, -max_torq_w, max_torq_w)
            q_v, dq_v = euler_step(ddqf_g, q_v, dq_v, dt, [v_uw, v_uk, x0_v])
            cmds = draw_g_cmds(q_v, dq_v, [x0_v])
        viewer.handle_event(event_handler)
        viewer.clear()
        viewer.text([ f"t: {t:.03f}"
                    , f"q: {q_v[0]:.03f} {q_v[1]:.03f}  {q_v[2]:.03f}"
                    , f"dq: {dq_v[0]:.03f} {dq_v[1]:.03f}  {dq_v[2]:.03f}"
                    , f"tau: N/A {v_uw:.01f} {v_uk:.01f})"
                    ])
        viewer.draw(cmds)
        viewer.draw_horizon(0)
        viewer.flush(dt)

if __name__ == '__main__':
    test()
