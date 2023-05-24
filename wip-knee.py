import numpy as np
import sys
import time
from sym_arith import *
from sympy import *
from sym_linktree import *
import control as ct
from integrator import *
from wip_control import *

# ref
# https://www.ascento.ethz.ch/wp-content/uploads/2019/05/AscentoPaperICRA2019.pdf
# https://arxiv.org/pdf/2005.11431.pdf
# https://www2.akita-nct.ac.jp/libra/report/46/46038.pdf

r,ll,lh,Iw,Il,Ih,mw,ml,mh = symbols('r ll lh Iw Il Ih mw ml mh') # wheel leg hip
uw, uk = symbols('uw uk') # motor torq (wheel, knee)
fyw = symbols('fyw') # offset
x0 = symbols('x0') # offset
k = symbols('k') # torsion spring elastic coeff
g = symbols('g')

IDX_W=0
IDX_L=1
IDX_H=2

context = { mw: 1, Iw: 1./200, r: 0.1,
            ml: 1, Il: 0.09/12, ll: 0.3,
            mh: 8, Ih: 8*0.09/12, lh: 0.3,
            k:0,
            g:9.81
        }

class WIPG(LinkTreeModel):

    def __init__(self, simp=True):
        # initial wheel angle should be vertical
        jl1 = WheelJointLink("qw", mw, r, RackPinionJoint(r, x0), XT=Xpln(pi/2, 0, 0), Icog=Iw)
        jl2 = StickJointLink("ql", ml, ll, RevoluteJoint(), XT=Xpln(-pi/2, ll, 0), cx=ll, Icog=Il, tau=uw)
        jl3 = StickJointLink("qh", mh, lh, RevoluteJoint(), XT=Xpln(0, lh, 0), cx=lh, Icog=Ih, tau=uk)
        super().__init__([jl1, jl2, jl3], g, X0=Xpln(0, 0, 0))
        self.gen_function(context)
        self.reset()

    def reset(self):
        self.q_v = np.array([0, 0, np.deg2rad(45)])
        self.dq_v = np.array([0, 0, 0])
        self.v_ref = 0 # horizontal velocity
        self.qh_ref = 0 # knee
        self.x0_v = 0
        self.v_uk = 0
        self.v_uw = 0
        #self.check_vwip_param()

    def check_vwip_param(self):
        qh = self.q()[IDX_H]
        vI = simplify(self.Ic[IDX_L]) # stick inertia
        vmb, cx, cy, vIb = I2mc(vI)
        vl = simplify(sqrt(cx**2+cy**2))
        a0 = atan2(cy, cx)
        A, B = wip_lin_system(g, r, vl, mw, vmb, Iw, vIb)
        Af = lambdify([qh], A.subs(context))
        Bf = lambdify([qh], B.subs(context))
        a0f = lambdify([qh], a0.subs(context))

        for qh_ref in [0, np.deg2rad(-45), np.deg2rad(45)]:
            K, _, _ = ct.lqr(Af(qh_ref), Bf(qh_ref), np.diag([1,1,1]), 1)
            print(qh_ref, K, a0f(qh_ref))

    def sim_input(self):
        return [uw, uk, x0]

    def v_sim_input(self):
        return np.array([self.v_uw, self.v_uk, self.x0_v])

    def draw_input(self):
        return [x0]

    def v_draw_input(self):
        return [self.x0_v]

    def update_fext(self):
        pass

    def update_sim_input(self):
        Kp = 100
        Kd = Kp * 0.1

        v_uk = Kp*(self.qh_ref - self.q_v[IDX_H]) - Kd * self.dq_v[IDX_H] + self.cancel_force[IDX_H]()

        if self.qh_ref  == 0:
            K = np.array([[-1., 11.20462078, 50.02801177]])
            a0_v = -0.7266423406817263
        elif self.qh_ref == np.deg2rad(-45):
            K = np.array([[-1., 7.98246651, 39.40143798]])
            a0_v = -1.0370279769387782
        elif self.qh_ref == np.deg2rad(45):
            K = np.array([[-1., 13.53793325, 55.79660305]])
            a0_v = -0.36833839806552643
        v_uw = wip_wheel_torq(K, self.v_ref, self.q_v, self.dq_v, a0_v)

        #max_torq_w = 3.5 # Nm
        #max_torq_k = 40 # Nm
        self.v_uk = v_uk
        self.v_uw = v_uw


def test():
    model_g = WIPG()

    def event_handler(key, shifted):
        if key == 'l':
            model_g.v_ref = 5
        elif key == 'h':
            model_g.v_ref = -5
        elif key == 'j':
            model_g.v_ref = 0
        elif key == 'p':
            model_g.qh_ref = np.deg2rad(45)
        elif key == 'n':
            model_g.qh_ref = np.deg2rad(-45)
        elif key == 'k':
            model_g.qh_ref = 0

    view(model_g, event_handler)

if __name__ == '__main__':
    test()
