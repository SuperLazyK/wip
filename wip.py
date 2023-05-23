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
uw = symbols('uw') # motor torq
x0 = symbols('x0') # x-offset
g = symbols('g')

context = { l: 0.5, r: 0.05,
        mw: 1, mb: 9,
        Iw: 1./800, Ib: 2.25,
        g: 9.81
        }

vy = False
if vy:
    IDX_W = 1
    IDX_Y = 0
else:
    IDX_W = 0

class WIPG(LinkTreeModel):

    def __init__(self, simp=True):
        # initial wheel angle should be vertical
        if vy:
            jl0 = StickJointLink("y", 0, 0, PrismaticJoint(), XT=Xpln(-pi/2, 0, 0), Icog=0)
            jl1 = WheelJointLink("qw", mw, r, RackPinionJoint(r, x0), XT=Xpln(pi/2, 0, 0), Icog=Iw)
            jl2 = StickJointLink("qs", mb, l, RevoluteJoint(), I=transInertia(mcI(mb, [l, 0], Ib), Xpln(0, 0, 0)), tau=uw)
            super().__init__([jl0, jl1, jl2], g, X0=Xpln(pi/2, 0, 0))
        else:
            jl1 = WheelJointLink("qw", mw, r, RackPinionJoint(r, x0), XT=Xpln(pi/2, 0, 0), Icog=Iw)
            jl2 = StickJointLink("qs", mb, l, RevoluteJoint(), I=transInertia(mcI(mb, [l, 0], Ib), Xpln(0, 0, 0)), tau=uw)
            super().__init__([jl1, jl2], g, X0=Xpln(0, 0, 0))
        self.gen_function(context)
        self.reset()

    def reset(self):
        self.v_ref = 0 # horizontal velocity
        self.x0_v = 0
        self.v_uw = 0

    def sim_input(self):
        return [uw, x0]

    def v_sim_input(self):
        return np.array([self.v_uw, self.x0_v])

    def draw_input(self):
        return [x0]

    def v_draw_input(self):
        return [self.x0_v]

    def feedback(self):
        K = np.array([[-1, 41.26540066, 125.12381105]])
        v_uw = wip_wheel_torq(K, self.v_ref, self.q_v[IDX_W:], self.dq_v[IDX_W:], 0)
        self.v_uw = v_uw
        b = 400.
        k = b * b / 4*(context[mb] + context[mw]) # zeta == 1
        if vy:
            if self.q_v[IDX_Y] < 0:
                self.fext_v[IDX_W][2] =-k * self.q_v[IDX_Y] -b * self.dq_v[IDX_Y]
            else:
                self.fext_v[IDX_W][2] = 0

def test():
    model_g = WIPG()

    def event_handler(key, shifted):
        if key == 'l':
            model_g.v_ref = 5
        elif key == 'h':
            model_g.v_ref = -5
        elif key == 'j':
            model_g.v_ref = 0

    view(model_g, event_handler)

if __name__ == '__main__':
    test()
