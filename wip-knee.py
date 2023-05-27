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
fwy = symbols('fwy') # offset
x0 = symbols('x0') # offset
k = symbols('k') # torsion spring elastic coeff
g = symbols('g')

context = { mw: 1, Iw: 1./200, r: 0.1,
            ml: 1, Il: 0.09/12, ll: 0.3,
            mh: 8, Ih: 8*0.09/12, lh: 0.3,
            k:0,
            g:9.81
        }
b = 400.
k = b * b / 4*(context[mh] + context[ml] + context[mw]) # zeta == 1

class WIPG(LinkTreeModel):

    def __init__(self):
        # initial wheel angle should be vertical
        self.IDX_Y=0
        self.IDX_W=1
        self.IDX_L=2
        self.IDX_H=3
        jl0 = StickJointLink("y", 0, 0, PrismaticJoint(), XT=Xpln(-pi/2, 0, 0), Icog=0)
        jl1 = WheelJointLink("qw", mw, r, RackPinionJoint(r, x0), XT=Xpln(pi/2, 0, 0), Icog=Iw)
        jl2 = StickJointLink("ql", ml, ll, RevoluteJoint(), XT=Xpln(-pi/2, ll, 0), cx=ll, Icog=Il, tau=uw)
        jl3 = StickJointLink("qh", mh, lh, RevoluteJoint(), XT=Xpln(0, lh, 0), cx=lh, Icog=Ih, tau=uk)
        super().__init__([jl0, jl1, jl2, jl3], g, X0=Xpln(pi/2, 0, 0))
        _, _, x, _ = Xtoscxy(jl1.X_r_to)
        jl1.fa = x * fwy
        jl1.fy = fwy
        self.gen_function(context)
        self.reset()

    def reset(self):
        self.q_v[self.IDX_L]=np.pi/4
        self.v_ref = 0 # horizontal velocity
        self.qh_ref = 0 # knee
        self.x0_v = 0
        self.v_uk = 0
        self.v_uw = 0
        self.v_fwy = 0
        #self.check_vwip_param()

    def check_vwip_param(self):
        qh = self.q()[self.IDX_H]
        vI = simplify(self.Ic[self.IDX_L]) # stick inertia
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
        return [fwy, uw, uk, x0]

    def v_sim_input(self):
        return np.array([self.v_fwy, self.v_uw, self.v_uk, self.x0_v])

    def draw_input(self):
        return [x0]

    def v_draw_input(self):
        return [self.x0_v]

    def update_fext(self):
        if self.q_v[self.IDX_Y] < 0:
            self.v_fwy =-k * self.q_v[self.IDX_Y] -b * self.dq_v[self.IDX_Y]
        else:
            self.v_fwy = 0

    def update_sim_input(self):
        Kp = 100
        Kd = Kp * 0.1

        v_uk = Kp*(self.qh_ref - self.q_v[self.IDX_H]) - Kd * self.dq_v[self.IDX_H] + self.cancel_force[self.IDX_H]()

        if self.qh_ref  == 0:
            K = np.array([[-1., 11.20462078, 50.02801177]])
            a0_v = -0.7266423406817263
        elif self.qh_ref == np.deg2rad(-45):
            K = np.array([[-1., 7.98246651, 39.40143798]])
            a0_v = -1.0370279769387782
        elif self.qh_ref == np.deg2rad(45):
            K = np.array([[-1., 13.53793325, 55.79660305]])
            a0_v = -0.36833839806552643

        v_uw = wip_wheel_torq(K, self.v_ref, self.q_v[self.IDX_W:], self.dq_v[self.IDX_W:], a0_v)

        max_torq_w = 3.5 # Nm
        max_torq_k = 40 # Nm
        #self.v_uw = np.clip(-max_torq_w, max_torq_w, v_uw)
        #self.v_uk = np.clip(-max_torq_k, max_torq_k, v_uk)
        self.v_uk = v_uk
        self.v_uw = v_uw
        #self.v_uk = 0
        #self.v_uw = 0

    def hook_pre_step(self):
        print("q", self.q_v)
        print("dq", self.dq_v)
        print("uw", self.v_uw)

    def hook_post_step(self):
        print("ddq", self.ddq_v)

    def set_vel_ref(self, v):
        self.v_ref = v

    def set_knee_ref(self, v):
        self.qh_ref = v

class WIPA(LinkTreeModel):

    def __init__(self):
        # initial wheel angle should be vertical
        self.IDX_X=0
        self.IDX_Y=1
        self.IDX_W=2
        self.IDX_L=3
        self.IDX_H=4
        jl0x = StickJointLink("x", 0, 0, PrismaticJoint(), XT=Xpln(pi/2, 0, 0), Icog=0)
        jl0y = StickJointLink("y", 0, 0, PrismaticJoint(), XT=Xpln(-pi/2, 0, 0), Icog=0)
        jl1 = WheelJointLink("qw", mw, r, RevoluteJoint(), XT=Xpln(pi/2, 0, 0), Icog=Iw)
        jl2 = StickJointLink("ql", ml, ll, RevoluteJoint(), XT=Xpln(-pi/2, ll, 0), cx=ll, Icog=Il, tau=uw)
        jl3 = StickJointLink("qh", mh, lh, RevoluteJoint(), XT=Xpln(0, lh, 0), cx=lh, Icog=Ih, tau=uk)
        super().__init__([jl0x, jl0y, jl1, jl2, jl3], g, X0=Xpln(0, 0, 0))
        self.gen_function(context)
        self.reset()

    def reset(self):
        self.q_v[self.IDX_L]=np.pi/4
        self.v_ref = 0 # horizontal velocity
        self.qh_ref = 0 # knee
        self.x0_v = 0
        self.v_uk = 0
        self.v_uw = 0

    def sim_input(self):
        return [uw, uk]

    def v_sim_input(self):
        return np.array([self.v_uw, self.v_uk])

    def update_sim_input(self):
        Kp = 100
        Kd = Kp * 0.1
        v_uk = Kp*(self.qh_ref - self.q_v[self.IDX_H]) - Kd * self.dq_v[self.IDX_H] + self.cancel_force[self.IDX_H]()
        max_torq_k = 40 # Nm
        self.v_uk = np.clip(-max_torq_k, max_torq_k, v_uk)
        #self.v_uk = 0
        self.v_uw = 0

    def hook_post_step(self):
        pass

#class WIP():
#    def __init__(self):
#        self.model_g = WIPG()
#        self.model_a = WIPA()
#        self.ground = True
#
#    def step(self):
#        if self.ground:
#        pass
#
#    def draw(self):
#        if self.ground:
#        pass
#
#    def set_vel_ref(self, v):
#        self.model_g.v_ref = v
#
#    def set_knee_ref(self, v):
#        self.model_g.qh_ref = v
#        self.model_a.qh_ref = v


def test():
    model = WIPG()

    def event_handler(key, shifted):
        if key == 'l':
            model.set_vel_ref(20)
        elif key == 'h':
            model.set_vel_ref(-20)
        elif key == 'j':
            model.set_vel_ref(0)
        elif key == 'p':
            model.set_knee_ref(np.deg2rad(45))
        elif key == 'n':
            model.set_knee_ref(np.deg2rad(-45))
        elif key == 'k':
            model.set_knee_ref(np.deg2rad(0))

    view(model, event_handler, dt=0.001)

if __name__ == '__main__':
    test()
