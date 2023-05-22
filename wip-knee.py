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

class WIPG(LinkTreeModel):

    IDX_W=0
    IDX_L=1
    IDX_H=2

    max_knee_angle = np.deg2rad(-150)
    min_knee_angle = np.deg2rad(-45)

    def __init__(self):
        # initial wheel angle should be vertical
        jl0 = StickJointLink("y", 0, 0, PrismaticJoint(), XT=Xpln(-pi/2, 0, 0))
        jl1 = WheelJointLink("qw", mw, r, RackPinionJoint(r, x0), XT=Xpln(pi/2, 0, 0), Icog=Iw)
        jl2 = StickJointLink("ql", ml, ll, RevoluteJoint(), XT=Xpln(-pi/2, ll, 0), cx=ll, Icog=Il, tau=uw)
        #jl3 = StickJointLink("qh", mh, lh, RevoluteJoint(), XT=Xpln(0, lh, 0), cx=lh, Icog=Ih, tau=uk)
        jl3 = StickSpringJointLink("qh", mh, lh, k, 0, RevoluteJoint(), XT=Xpln(0, lh, 0), cx=lh, Icog=Ih, tau=uk)
        #plant_model = LinkTreeModel([jl0, jl1, jl2, jl3], g, X0=Xpln(pi/2, 0, 0))
        super().__init__([jl1, jl2, jl3], g, X0=Xpln(0, 0, 0))

        # initial status
        #self.q_v = self.qv(qh=np.deg2rad(0))
        self.q_v = self.qv(qh=np.deg2rad(45))
        self.dq_v = self.qv(qw=3)
        self.v_fext =np.zeros((self.NB, 3))
        self.v_ref = 0 # horizontal velocity
        self.p_ref = self.qh_v() # knee angle
        self.x0_v = 0
        self.v_uk = 0
        self.v_uw = 0

        qh = self.qh()
        A, B, a0 = self.virtual_wip_model_ground()
        self.Af = lambdify([qh], A.subs(context))
        self.Bf = lambdify([qh], B.subs(context))
        self.a0f = lambdify([qh], a0.subs(context))
        self.cancel_force_knee = self.cancel_bias_force_knee()
        self.update_gain()

        self.ddqf_g = self.gen_ddq_f(self.sim_input(), context)
        self.draw_g_cmds = self.gen_draw_cmds(self.draw_input(), context)

    def update_gain(self):
        self.K, _, _ = ct.lqr(self.Af(self.p_ref), self.Bf(self.p_ref), np.diag([1,1,1]), 1)

    def sim_input(self):
        return [uw, uk, x0]

    def draw_input(self):
        return [x0]

    def qv(self, qw=0, ql=0, qh=0):
        return np.array([qw, ql, qh], dtype=np.float64)

    def qw_v(self):
        return self.q_v[self.IDX_W]

    def ql_v(self):
        return self.q_v[self.IDX_L]

    def qh_v(self):
        return self.q_v[self.IDX_H]

    def dqw_v(self):
        return self.dq_v[self.IDX_W]

    def dql_v(self):
        return self.dq_v[self.IDX_L]

    def dqh_v(self):
        return self.dq_v[self.IDX_H]

    def qw(self):
        return self.q()[self.IDX_W]

    def ql(self):
        return self.q()[self.IDX_L]

    def qh(self):
        return self.q()[self.IDX_H]

    def ref_v(self):
        return np.array([self.v_ref, self.p_ref])

    def tau_v(self):
        return np.array([self.v_uw, self.v_uk])

    def virtual_wip_model_ground(self):
        vI = simplify(self.Ic[self.IDX_L]) # stick inertia
        vmb, cx, cy, vIb = I2mc(vI)
        vl = simplify(sqrt(cx**2+cy**2))
        vtheta = atan2(cy, cx)
        A, B = wip_lin_system(g, r, vl, mw, vmb, Iw, vIb)
        return A, B, vtheta

    def cancel_bias_force_knee(self):
        return lambdify(self.syms(), simplify(self.counter_joint_force()[self.IDX_H,0]).subs(context))

    def draw(self):
        return self.draw_g_cmds(self.q_v, self.dq_v, [self.x0_v])

    def step(self, dt):
        Kp = 1000
        Kd = 100
        max_torq_w = 3.5 # Nm
        max_torq_k = 40 # Nm
        self.v_uk = Kp*(self.p_ref - self.qh_v()) - Kd * self.dqh_v() + self.cancel_force_knee(*self.q_v, *self.dq_v, *self.v_fext.reshape(-1))
        self.v_uw = wip_wheel_torq(self.K, self.v_ref, self.q_v, self.dq_v, self.a0f(self.p_ref))
        #self.v_uk = np.clip(self.v_uk, -max_torq_k, max_torq_k)
        #self.v_uw = np.clip(self.v_uw, -max_torq_w, max_torq_w)
        self.q_v, self.dq_v = euler_step(self.ddqf_g, self.q_v, self.dq_v, self.v_fext, dt, [self.v_uw, self.v_uk, self.x0_v])

def test():
    model_g = WIPG()

    import graphic
    viewer = graphic.Viewer(scale=200, offset=[0, 0.2])

    t = 0
    dt = 0.001
    in_air = False

    pause = True
    def event_handler(key, shifted):
        nonlocal pause
        if key == 'q':
            sys.exit()
        elif key == 's':
            pause = pause ^ True
        elif key == 'l':
            model_g.v_ref = 5
        elif key == 'h':
            model_g.v_ref = -5
        elif key == 'j':
            model_g.v_ref = 0
        elif key == 'p':
            model_g.p_ref = np.deg2rad(45)
            model_g.K = np.array([[-1., 13.53793325, 55.79660305]])
        elif key == 'n':
            model_g.p_ref = np.deg2rad(-45)
            model_g.K = np.array([[-1., 7.98246651, 39.40143798]])
        elif key == 'k':
            model_g.p_ref = 0
            model_g.K = np.array([[-1., 11.20462078, 50.02801177]])

    while True:
        if in_air:
            pass
        else:
            cmds = model_g.draw()

        viewer.handle_event(event_handler)
        viewer.clear()
        viewer.text([ f"t: {t:.03f}"
                    , graphic.arr2txt(model_g.q_v, " q")
                    , graphic.arr2txt(model_g.dq_v, "dq")
                    , graphic.arr2txt(model_g.ref_v(), "ref")
                    , graphic.arr2txt(model_g.tau_v(), "tau")
                    ])
        viewer.draw(cmds)
        viewer.draw_horizon(0)
        viewer.flush(dt)

        if pause:
            continue

        t = t + dt

        if in_air:
            pass
        else:
            model_g.step(dt)

if __name__ == '__main__':
    test()
