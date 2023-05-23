from sympy import *
from sym_linktree import *

# ref
# https://www.ascento.ethz.ch/wp-content/uploads/2019/05/AscentoPaperICRA2019.pdf
# https://arxiv.org/pdf/2005.11431.pdf
# https://www2.akita-nct.ac.jp/libra/report/46/46038.pdf

r,ll,lh,Iw,Il,Ih,mw,ml,mh = symbols('r ll lh Iw Il Ih mw ml mh') # wheel leg hip
uw, uk = symbols('uw uk') # motor torq (wheel, knee)
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

    def __init__(self):
        # initial wheel angle should be vertical
        jl1 = WheelJointLink("qw", mw, r, RackPinionJoint(r, x0), XT=Xpln(pi/2, 0, 0), Icog=Iw)
        jl2 = StickJointLink("ql", ml, ll, RevoluteJoint(), XT=Xpln(-pi/2, ll, 0), cx=ll, Icog=Il, tau=uw)
        jl3 = StickJointLink("qh", mh, lh, RevoluteJoint(), XT=Xpln(0, lh, 0), cx=lh, Icog=Ih, tau=uk)
        super().__init__([jl1, jl2, jl3], g, X0=Xpln(0, 0, 0))
        self.gen_function(context)
        self.cancel_force_knee = lambdify(self.syms(), simplify(self.counter_joint_force()[self.IDX_H,0]).subs(context))
        self.reset()

    def reset(self):
        self.q_v = np.array([0, 0, np.deg2rad(45)])
        self.dq_v = np.array([0, 0, 0])
        self.v_ref = 0 # horizontal velocity
        self.p_ref = 0 # knee
        self.x0_v = 0
        self.v_uk = 0
        self.v_uw = 0
        self.K  = xx
        self.a0 = xx

    def check_vwip_param():
        qh = self.q()[IDX_H]
        vI = simplify(self.Ic[self.IDX_L]) # stick inertia
        vmb, cx, cy, vIb = I2mc(vI)
        vl = simplify(sqrt(cx**2+cy**2))
        a0 = atan2(cy, cx)
        A, B = wip_lin_system(g, r, vl, mw, vmb, Iw, vIb)
        Af = lambdify([qh], A.subs(context))
        Bf = lambdify([qh], B.subs(context))
        a0f = lambdify([qh], a0.subs(context))
        K, _, _ = ct.lqr(self.Af(self.p_ref), self.Bf(self.p_ref), np.diag([1,1,1]), 1)

    def sim_input(self):
        return [uw, uk, x0]

    def v_sim_input(self):
        return np.array([self.v_uw, self.v_uk, self.x0_v])

    def draw_input(self):
        return [x0]

    def feedback(self):
        Kp = 100
        Kd = Kp * 0.1
        max_torq_w = 3.5 # Nm
        max_torq_k = 40 # Nm
        self.v_uk = Kp*(self.p_ref - self.qh_v()) - Kd * self.dqh_v() + self.cancel_force_knee(*self.q_v, *self.dq_v, *self.fext_v.reshape(-1))
        self.v_uw = wip_wheel_torq(self.K, self.v_ref, self.q_v, self.dq_v, self.a0f(self.p_ref))

