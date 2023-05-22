from sympy import *
from sym_jointlink import *
import scipy
import numpy as np
#from sympy import init_printing
#init_printing() 


# NOTE: the original index 1,2,... and 0 means base body(basically not moved)
# NOTE: body velocity in the body coordinate does not mean 0 (5.14)
# symbolic calculation should be executed only once
# NOTE
#  Ic[i] is i-th body corrdinate inertia matrix of accumlated bodies after i
#  but it does NOT depend on its joint angle.
# X_r_to[i] transfer matrix to i-th body local coordinate from global(root) coordinate
# NOTE:
# i-th body local coordinate is transfered with its joint to the parent!!
# => X_r_to[0] may not be same as X0
class LinkTreeModel:

    def __init__(self, jointlinks, g, X0=eye(3), parent_idx=None):
        self.jointlinks = jointlinks
        self.NB = len(jointlinks)
        self.dim = 3 # num of dimension of spatial/planar vector
        self.X_parent_to = [zeros(self.dim, self.dim) for i in range(self.NB)]
        self.g = g
        self.accb = zeros(self.dim, 1)
        self.velb = zeros(self.dim, 1)
        self.X0 = X0

        if parent_idx is None:
            # list
            self.parent_idx = list(range(-1, self.NB-1))
        else:
            for i in parent_idx:
                assert -1 <= i and i < self.NB
            self.parent_idx = parent_idx

        self.Ic = [zeros(self.dim, self.dim) for i in range(self.NB)]

        # symbolic calculation should be executed only once
        self.update_vel_X()
        self.composite_inertia()
        self.cog = self.calc_cog()


    def update_vel_X(self):
        for i in range(self.NB):
            I  = self.jointlinks[i].I
            XJ = self.jointlinks[i].XJ()
            j = self.parent(i)
            if j != -1: # parent is not root
                XT = self.jointlinks[j].XT
            else:
                XT = self.X0
            X_j_to_i = XJ * XT
            self.X_parent_to[i] = X_j_to_i

            vJ = self.jointlinks[i].vJ()
            if j != -1: # parent is not root
                self.jointlinks[i].X_r_to = X_j_to_i * self.jointlinks[j].X_r_to
                self.jointlinks[i].vel = X_j_to_i * self.jointlinks[j].vel + vJ
            else:
                self.jointlinks[i].X_r_to = X_j_to_i
                self.jointlinks[i].vel = X_j_to_i * self.velb + vJ

    def q(self):
        return [jl.q for jl in self.jointlinks]

    def dq(self):
        return [jl.dq for jl in self.jointlinks]

    def ddq(self):
        return [jl.ddq for jl in self.jointlinks]

    def fext(self):
        return sum([[jl.fa, jl.fx, jl.fy] for jl in self.jointlinks],[])

    def syms(self, others=[]):
        return self.q() +  self.dq() + self.fext() + others

    def parent(self, i):
        return self.parent_idx[i]

    # recursive newton-euler on each body coordinate
    # NOTE: fext : ext-force to i-th body in root coordinate
    def inverse_dynamics(self, ddq, fext, impulse=False):
        q = self.q()
        dq = self.dq()
        NB = self.NB
        dim = self.dim

        assert NB == len(ddq)
        assert NB == len(fext)
        assert (dim, 1) == fext[0].shape

        acc = [zeros(dim, 1) for i in range(NB)]
        tau = [0 for i in range(NB)]
        f = [zeros(dim, 1) for i in range(NB)]

        S = [self.jointlinks[i].S() for i in range(NB)]

        for i in range(NB):
            I  = self.jointlinks[i].I
            vJ = self.jointlinks[i].vJ()
            cJ = self.jointlinks[i].cJ()
            j = self.parent(i)
            X_j_to_i = self.X_parent_to[i]

            if j != -1: # parent is not root
                accj = acc[j]
            else:
                accj = self.accb

            vel = self.jointlinks[i].vel
            acc[i] = X_j_to_i * accj  + S[i] * ddq[i] + cJ + crm(vel) * vJ

            f_g = self.jointlinks[i].gravity_force(self.g)
            # Euler's body coordinate dynamics equation
            # Inertia and angular velocity always changes in reference cordinate!!
            # NOTE w, I, M(tora) are descirbed in body coordinate
            # M = I dw/dt + w x (Iw)
            # f[i] is interaction force to i "from its parent j"
            # passive joint force is eliminated
            if impulse:
                f[i] = - dualm(self.jointlinks[i].X_r_to) * fext[i]
            else:
                f[i] = I * acc[i] + crf(vel) * I * vel - dualm(self.jointlinks[i].X_r_to) * fext[i] - f_g

        for i in range(NB-1, -1, -1):
            # parent force projeced to S(non-constraint-dimension)
            if impulse:
                tau[i] = S[i].dot(f[i])
            else:
                tau[i] = S[i].dot(f[i]) - self.jointlinks[i].passive_joint_force()
            j = self.parent(i)
            if j != -1: # parent is root
                X_j_to_i = self.X_parent_to[i]
                f[j] = f[j] + X_j_to_i.T * f[i]

        return Matrix(tau)


    def composite_inertia(self):

        X_parent_to = self.X_parent_to
        NB = self.NB
        dim = self.dim
        H = zeros(NB, NB)
        S = [self.jointlinks[i].S() for i in range(NB)]

        Ic = [zeros(dim, dim) for i in range(NB)]
        for i in range(NB):
            Ic[i] = self.jointlinks[i].I

        for i in range(NB-1, -1, -1):
            j = self.parent(i)
            if j != -1: # parent is root
                X_j_to_i = X_parent_to[i]
                Ic[j] = Ic[j] + X_j_to_i.T * Ic[i] * X_j_to_i
            F = Ic[i] * S[i]
            H[i,i] = S[i].T * F

            j = i
            while self.parent(j) != -1:
                F = X_parent_to[j].T * F
                j = self.parent(j)
                H[i,j] = F.T * S[j]
                H[j,i] = H[i,j]

        self.Ic = Ic
        # local to root
        #for i in range(NB):
        #    self.Ic[i] = self.jointlinks[i].X_r_to.T * Ic[i] * self.jointlinks[i].X_r_to

        self.H = H

    def kinetic_energy(self):
        # other approach: Sum of dq H dq/2
        # NOTE: vJ is relative. So Sum of vJ I vJ/2 is NOT kinetic energy
        return sum([jl.kinetic_energy() for jl in self.jointlinks])

    def potential_energy(self):
        return sum([jl.potential_energy(self.g) for jl in self.jointlinks])

    # joint force to keep the attitude
    def counter_joint_force(self):
        fext = [Matrix([jl.fa, jl.fx, jl.fy]) for jl in self.jointlinks]
        C = self.inverse_dynamics([0 for i in range(self.NB)], fext)
        return C

    # foward dynqmics
    def gen_ddq_f(self, input_sym_list=[], ctx={}):
        tau = Matrix([jl.active_joint_force() for jl in self.jointlinks])
        # force to cancel for no joint acc
        C = self.counter_joint_force()
        fext = [[jl.fa, jl.fx, jl.fy] for jl in self.jointlinks]
        syms = self.syms(input_sym_list)
        Hevalf = lambdify(syms, self.H.subs(ctx))
        rhs = lambdify(syms, (tau-C).subs(ctx))
        def ddq_f(qv, dqv, fextv, uv):
            b = rhs(*qv, *dqv, *fextv.reshape(-1), *uv).reshape(-1).astype(np.float64)
            A = Hevalf(*qv, *dqv, *fextv.reshape(-1), *uv)
            return np.linalg.solve(A, b)
        return ddq_f

    def calc_cog(self, ith=0): # global coordinate
        _, cx, cy, _ = I2mc(transInertia(self.Ic[ith], self.jointlinks[ith].X_r_to))
        return Matrix([cx, cy])

    def gen_draw_cmds(self, input_sym_list, ctx):
        q_sym_list = self.q()
        dq_sym_list = self.dq()
        draw_cmd_fns = [jl.gen_draw_cmds(q_sym_list + dq_sym_list + input_sym_list, ctx) for jl in self.jointlinks]
        eval_cog_pos = lambdify(q_sym_list + input_sym_list, self.cog.subs(ctx))
        def draw_cmds(qv, dqv, v):
            p = eval_cog_pos(*qv, *v)
            return sum([f(np.concatenate([qv, dqv, v])) for f in draw_cmd_fns], plot_point_cmd(p[0,0], p[1,0], 0.01, color="blue", name="cog"))
        return draw_cmds

    def joint_info(self, simp=True):
        for jl in self.jointlinks:
            print(jl.name, fromX(jl.X_r_to, simp))

def test1():
    g = symbols('g')
    l1, m1 = symbols('l1 m1')
    l2, m2 = symbols('l2 m2')
    m1 = 1
    m2 = 1
    l1 = 1
    l2 = 1
    jl1 = StickJointLink("q1", m1, l1, RevoluteJoint(), XT=Xpln(pi/2, l1, 0)) # fict
    q1 = jl1.q
    jl2 = StickJointLink("q2", m2, l2, RevoluteJoint(), XT=Xpln(0, 0, 0))    # fict
    model = LinkTreeModel([jl1, jl2], g)
    I0 = simplify(model.Ic[0])
    I1 = simplify(model.Ic[1])
    printM(I0)
    print(I2mc(I0))
    printM(I1)
    print(I2mc(I1))
    print(jl2.I)
    print("cog x", model.cog[0,0])
    print("cog y", model.cog[1,0])

    cog = (model.calc_cog(0).subs({q:0 for q in model.q()[:1]}))
    th = atan2(cog[1,0], cog[0,0]).subs({q1:0})
    print(th)

    th = atan2(model.cog[1,0], model.cog[0,0]).subs({q1:0})
    print(th)
    printM(simplify(transInertia(I, Xpln(-th, 0, 0))), "XIX")

def test1():
    g = symbols("g")
    m1, l1 = symbols("m1 l1")
    Ic1 = stickI(m1, l1)
    jl1 = StickJointLink("q1", m1, l1, RevoluteJoint(), cx=l1, XT=Xpln(pi/2, l1, 0), Icog=Ic1)
    model = LinkTreeModel([jl1], g)
    print(simplify(model.calc_cog()))

def test2():
    g = symbols("g")
    m1, l1 = symbols("m1 l1")
    m2, l2 = symbols("m2 l2")
    Ic1 = stickI(m1, l1)
    Ic2 = stickI(m2, l2)
    jl1 = StickJointLink("q1", m1, l1, RevoluteJoint(), cx=l1, XT=Xpln(pi/2, l1, 0), Icog=Ic1)
    jl2 = StickJointLink("q2", m2, l2, RevoluteJoint(), cx=l2, XT=Xpln(0, 0, 0), Icog=Ic2)
    q1 = jl1.q
    q2 = jl2.q
    model = LinkTreeModel([jl1, jl2], g)
    print("tree")
    printM(simplify(model.Ic[0]))

    I1 = mcI(m1, [l1, 0], Ic1)
    I2 = mcI(m2, [l2, 0], Ic2)
    X = Xpln(q2, l1, 0)
    I2X = simplify(transInertia(I2, X))
    print("sum")
    printM(I1 + I2X)
    _, cx, cy, _ = I2mc(I1 + I2X)
    th = atan2(cy, cx)
    Xc = Xpln(-th, 0, 0)
    Ic = simplify(transInertia(I1 + I2X, Xc))
    print("rot")
    printM(Ic)

def test3():
    g = symbols("g")
    a = symbols("a")
    m0, l0 = symbols("m0 l0")
    m1, l1 = symbols("m1 l1")
    m2, l2 = symbols("m2 l2")
    Ic0 = circleI(m0, l0)
    Ic1 = stickI(m1, l1)
    Ic2 = stickI(m2, l2)
    jl0 = WheelJointLink("q1", m0, l0, RevoluteJoint(), XT=Xpln(pi/2+a, 0, 0), Icog=Ic0)
    jl1 = StickJointLink("q1", m1, l1, RevoluteJoint(), cx=l1, XT=Xpln(pi/2, l1, 0), Icog=Ic1)
    jl2 = StickJointLink("q2", m2, l2, RevoluteJoint(), cx=l2, XT=Xpln(0, 0, 0), Icog=Ic2)
    q0 = jl0.q
    q1 = jl1.q
    q2 = jl2.q
    model = LinkTreeModel([jl0, jl1, jl2], g)
    print("tree")
    I1 = simplify(model.Ic[1])
    _, cx, cy, _ = I2mc(I1)
    th = atan2(cy, cx)
    Xc = Xpln(-th, 0, 0)
    Ic = simplify(transInertia(I1, Xc))
    printM(Ic)


def test():
    g = symbols("g")
    m, r, l = symbols("m, r, l")
    js = [ StickJointLink("y", 0, 0, PrismaticJoint(), XT=Xpln(-pi/2, 0, 0))
         , StickJointLink("x", 0, 0, PrismaticJoint(), XT=Xpln(0, 0, 0)) 
         , WheelJointLink("qw", m, r, RackPinionJoint(r), XT=Xpln(pi/2, 0, 0))
         , StickJointLink("ql", m, l, RevoluteJoint(), cx=l)
         ]
    js = [ StickJointLink("x", 0, 0, PrismaticJoint(), XT=Xpln(pi/2, 0, 0))
         , StickJointLink("y", 0, 0, PrismaticJoint(), XT=Xpln(-pi/2, 0, 0)) 
         , WheelJointLink("qw", m, r, RackPinionJoint(r), XT=Xpln(pi/2, 0, 0))
         , StickJointLink("ql", m, l, RevoluteJoint(), cx=l)
         ]
    #js = [WheelJointLink("qw", m, r, RackPinionJoint(r))]
    model = LinkTreeModel(js, g, X0=Xpln(0, 0, 0))
    for i in range(len(js)):
        print(i)
        th, x, y = fromX(model.jointlinks[i].X_r_to)
        print((simplify(th), simplify(x),simplify(y)))
    draw_cmds = model.gen_draw_cmds([], {r:0.1, l:0.5})
    cmds = draw_cmds([0 for i in range(len(js))], [0 for i in range(len(js))], [])
    import graphic
    viewer = graphic.Viewer(scale=200, offset=[0, 0.2])

    def event_handler(key, shifted):
        if key == 'q':
            sys.exit()


    while True:
        viewer.handle_event(event_handler)
        viewer.clear()
        #viewer.text([f"t: {t:.03f} :q {q[0]:.03f} {q[1]:.03f}"])
        viewer.draw(cmds)
        viewer.draw_horizon(0)
        viewer.flush(0.01)
    sys.exit(0)

if __name__ == '__main__':
    test()

