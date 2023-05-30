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
fwx = symbols('fwx') # for impulse
fwy = symbols('fwy') # normal
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

np.set_printoptions(precision=3, suppress=True)

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
        #super().__init__([jl0, jl1], g, X0=Xpln(pi/2, 0, 0))
        _, _, x, _ = Xtoscxy(jl1.X_r_to)
        jl1.fa = x * fwy
        jl1.fy = fwy
        self.gen_function(context)
        self.reset_state()

    def reset_state(self):
        super().reset_state()
        self.q_v[self.IDX_L]=np.pi/4
        self.v_ref = 0 # horizontal velocity
        self.qh_ref = 0 # knee
        self.x0_v = 0
        self.v_uk = 0
        self.v_uw = 0
        self.v_fwy = 0
        self.t = 0
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

        max_torq_w = 4 # Nm
        max_torq_k = 50 # Nm
        self.v_uw = np.clip(v_uw, -max_torq_w, max_torq_w)
        self.v_uk = np.clip(v_uk, -max_torq_k, max_torq_k)

    def draw_text(self):
        return [ f"q : {self.q_v}"
               , f"dq : {self.dq_v}"
               , f"knee: {self.v_uk:.3f}"
               , f"wheel: {self.v_uw:.3f}"
               ]

    def set_vel_ref(self, v):
        self.v_ref = v

    def set_knee_ref(self, v):
        self.qh_ref = v

    def on_ground(self):
        return self.q_v[self.IDX_Y] <= 0.00001

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
        #super().__init__([jl0x, jl0y, jl1], g, X0=Xpln(0, 0, 0))
        super().__init__([jl0x, jl0y, jl1, jl2, jl3], g, X0=Xpln(0, 0, 0))
        self.gen_function(context)
        self.reset_state()

    def reset_state(self):
        super().reset_state()
        self.q_v[self.IDX_L]=np.pi/4
        self.q_v[self.IDX_Y]=0.4
        self.dq_v[self.IDX_X]=1
        #self.dq_v[self.IDX_W]=10
        #self.dq_v[self.IDX_L]=10
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
        self.v_uk = np.clip(v_uk, -max_torq_k, max_torq_k)
        #self.v_uk = 0
        self.v_uw = 0

    def hook_post_step(self):
        pass

    def set_vel_ref(self, v):
        self.v_ref = v

    def set_knee_ref(self, v):
        self.qh_ref = v

    def on_ground(self):
        return self.q_v[self.IDX_Y] < context[r]

    def draw_text(self):
        return [ f"q : {self.q_v}"
               , f"dq : {self.dq_v}"
               , f"knee: {self.v_uk:.3f}"
               , f"wheel: {self.v_uw:.3f}"
               ]

class WIP():

    def __init__(self):
        self.model_g = WIPG()
        self.model_a = WIPA()
        self.gen_friction_impulse(self.model_a, context)
        self.reset_state()

    def reset_state(self):
        self.use_ground = False
        self.model_a.reset_state()
        self.model_g.reset_state()
        self.t = 0

    def step(self, dt):
        if self.use_ground:
            self.model_g.step(dt)
            if not self.model_g.on_ground():
                self.jump()
        else:
            self.model_a.step(dt)
            if self.model_a.on_ground():
                self.land()
        self.t = self.t+dt

    def draw(self):
        if self.use_ground:
            return self.model_g.draw()
        else:
            return self.model_a.draw()

    def draw_text(self):
        if self.use_ground:
            return ["mode : grd"] + self.model_g.draw_text()
        else:
            return ["mode : air"] + self.model_a.draw_text()

    def jump(self):
        print("jump")
        #print(self.model_g.dq_v)
        model_a = self.model_a
        model_g = self.model_g
        model_a.q_v[model_a.IDX_X] = self.model_g.x0_v-model_g.q_v[model_g.IDX_W] * context[r]
        model_a.q_v[model_a.IDX_Y] = model_g.q_v[model_g.IDX_Y] + context[r]
        model_a.q_v[model_a.IDX_W] = model_g.q_v[model_g.IDX_W]
        model_a.q_v[model_a.IDX_L] = model_g.q_v[model_g.IDX_L]
        model_a.q_v[model_a.IDX_H] = model_g.q_v[model_g.IDX_H]
        model_a.dq_v[model_a.IDX_X] = -model_g.dq_v[model_g.IDX_W] * context[r]
        model_a.dq_v[model_a.IDX_Y] = model_g.dq_v[model_g.IDX_Y]
        model_a.dq_v[model_a.IDX_W] = model_g.dq_v[model_g.IDX_W]
        model_a.dq_v[model_a.IDX_L] = model_g.dq_v[model_g.IDX_L]
        model_a.dq_v[model_a.IDX_H] = model_g.dq_v[model_g.IDX_H]
        self.use_ground = False
        #print(self.model_a.dq_v)

    def land(self):
        print("land")
        #print(self.model_a.dq_v)
        self.model_a.dq_v = self.impulse()
        #print(self.model_a.dq_v)
        self.model_g.q_v = self.model_a.q_v[self.model_a.IDX_Y:]
        self.model_g.q_v[self.model_g.IDX_Y] = self.model_g.q_v[self.model_g.IDX_Y] - context[r]
        self.model_g.dq_v = self.model_a.dq_v[self.model_a.IDX_Y:]
        self.model_g.x0_v = self.model_a.q_v[self.model_a.IDX_X] + context[r] * self.model_a.q_v[self.model_a.IDX_W]
        self.use_ground = True
        #print(self.model_g.dq_v)
        #print(self.model_g.q_v.tolist())

    def gen_friction_impulse(self, model, context):
        fext = [zeros(3,1) for i in range(model.NB)]
        fext[model.IDX_W][0] = - fwx * ( - r) # global coordinate torq!! but q2 == 0
        fext[model.IDX_W][1] = fwx
        q = model.q()
        dq = model.dq()
        Cfric = model.inverse_dynamics([0 for i in range(model.NB)], fext, impulse=True).subs(context)
        Hinv = MatrixSymbol("Hinv", model.NB, model.NB)
        delta = Hinv * (-Cfric)
        sol = solve(-r * (dq[model.IDX_W] + delta[model.IDX_W, 0]) - (dq[model.IDX_X] + delta[model.IDX_X, 0]), fwx)[0]
        delta_sol = delta.subs(context | {fwx:sol})
        new_dqv = Matrix(dq) + delta_sol
        f = lambdify([Hinv] + q + dq, new_dqv)

        def friction_impulse():
            Hv = np.linalg.inv(model.H_f(*model.all_vals()))
            return f(Hv, *model.q_v, *model.dq_v).reshape(-1)
        self.impulse = friction_impulse


    def set_vel_ref(self, v):
        self.model_g.set_vel_ref(v)

    def set_knee_ref(self, v):
        self.model_g.set_knee_ref(v)
        self.model_a.set_knee_ref(v)


def test():
    model = WIP()

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
        elif key == 'r':
            model.reset_state()

    view(model, event_handler, dt=0.001)

if __name__ == '__main__':
    test()

