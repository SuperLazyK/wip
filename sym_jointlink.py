from sympy import *
from geometry import *
from sym_arith import *

#-------------------------
# Joint/Link
#-------------------------
def try_subs(x, ctx):
    return x.subs(ctx) if hasattr(x, 'subs') else x

class RevoluteJoint:
    def __init__(self):
        pass

    def XJ(self, q):
        return Xpln(q, 0, 0)

    def S(self, q):
        return Matrix([1, 0, 0])

    def vJ(self, q, dq):
        return (self.S(q) * dq).reshape(3,1)

    def cJ(self, q, dq):
        # ring(S) == 0
        return zeros(3, 1)


class PrismaticJoint:
    def __init__(self):
        pass

    def XJ(self, q):
        return Xpln(0, q, 0)

    def S(self, q):
        return Matrix([0, 1, 0])

    def vJ(self, q, dq): # see Ex 4.6
        return (self.S(q) * dq).reshape(3,1)

    def cJ(self, q, dq):
        # ring(S) == 0
        return zeros(3, 1)

# NOTE: ex 4.6 is rack is set over the pinion
class RackPinionJoint:
    def __init__(self, r, offset=0):
        self.r = r
        self.offset = offset

    def XJ(self, q):
        return Xpln(q, self.offset - self.r*q, self.r)
        #return Matrix([[1, 0, 0], [q*r*sin(q), cos(q), sin(q)], [q*r*cos(q), -sin(q), cos(q)]])

    def S(self, q):
        return Matrix([1, -self.r*cos(q), self.r*sin(q)])

    def vJ(self, q, dq):
        return (self.S(q) * dq).reshape(3,1)

    def cJ(self, q, dq):
        return Matrix([0, self.r*sin(q)*dq**2, self.r*cos(q)*dq**2])


# planar inertia on body coordinate: M -> F
#
# c : cog position on the body coordinate Fi
# 
# note:
#  - center coordinate inertia tensor: diag(Ic, m, m)
#  - X_body_to_center : Xpln(0, c)
#  - X_center_to_dual : inv(Xpln(0, c))
#  see (2.62~2.63)
#  Ibody =  dual(inv(Xplan(0,c))) diag(Ic, m, m) Xplan(0,c) vbody
#  note dual(inv(X)) = X^T
def mcI(m, c, Ic):
    cx = c[0]
    cy = c[1]
    return Matrix([
        [Ic + m * (cx**2 + cy**2), -m * cy, m * cx],
        [-m * cy, m, 0],
        [m * cx, 0, m]
        ])

def I2mc(I):
    m = I[1,1]
    cx = I[0,2]/m
    cy = -I[0,1]/m
    Ic = I[0,0] - m * (cx**2 + cy**2)
    return m, cx, cy, Ic

# center of mass inertia
def Ic(m, I):
    return Matrix([
        [I, 0, 0],
        [0, m, 0],
        [0, 0, m]
        ])

def stickI(m, l):
    return m * l**2 / 12

def circleI(m, r):
    return r**2 * m / 2

DIM=3

# kinematics tree element
# body with a joint
# index -1 means root
# NOTE: body coordinate's z axis is same as joint axis. origin is NOT CoG.
class JointLink():
    # note:
    # Link and joint attached to the previous link
    # current body coordinate' origin is at the previous joint position
    # XT: base frame to attachment point to the child
    # the next joint is attached at XT on the current body coordinate
    # dim == 1 because 2-dim joint can decomposed into 2 joints
    def __init__(self, name, I, m, XT, q, dq, ddq, joint):
        self.name = name
        self.joint = joint
        self.X_r_to = zeros(3, 3)
        self.vel = zeros(3, 1) # velocity on body coordinate
        self.I = I # Inertia tensor on body coordinate
        self.m = m
        self.XT = XT
        if q is None:
            self.q = symbols(name)
            self.dq = symbols("d"+name)
            self.ddq = symbols("dd"+name)
        else:
            self.q = q
            self.dq = dq
            self.ddq = ddq

    # XJ: attachment point from the parent to base frame of the body
    def XJ(self):
        return self.joint.XJ(self.q)

    def vJ(self):
        return self.joint.vJ(self.q, self.dq)

    def cJ(self):
        return self.joint.cJ(self.q, self.dq)

    def S(self):
        return self.joint.S(self.q)

    def gen_draw_cmds(self, sym_list, ctx):
        return []

    def joint_force(self):
        return 0

    def kinetic_energy(self):
        return (1/2 * self.vel.T * self.I * self.vel)[0,0]

    # g force on body coordinate
    def gravity_force(self, g):
        return self.I * self.X_r_to * Matrix([0, 0, -g])

    def Xcog(self):
        _, cx, cy, _ = I2mc(self.I)
        Xc = Xpln(0, cx, cy)
        return Xc * self.X_r_to

    def potential_energy(self, g):
        _, _, _, y = Xtoscxy(self.Xcog())
        return y * g * self.m


class StickJointLink(JointLink):
    def __init__(self, name, m, l, joint, q=None, dq=None, ddq=None, cx=None, Icog=None, I=None, XT=None, tau=0):
        if I is None:
            if Icog is None:
                Icog = stickI(m,l)
            if cx is None:
                I = mcI(m, [l/2, 0], Icog)
            else:
                I = mcI(m, [cx, 0], Icog)
        if XT is None:
            XT = Xpln(0, l, 0)
        super().__init__(name, I, m, XT, q, dq, ddq, joint)
        self.tau = tau
        self.l = l

    def gen_draw_cmds(self, sym_list, ctx):
        X_r_to_f = lambdify(sym_list, self.X_r_to.subs(ctx))
        l_f = lambdify(sym_list, try_subs(self.l, ctx))
        def draw_cmd(v):
            X = X_r_to_f(*v)
            l = l_f(*v)
            return draw_lineseg_cmd(X, l) + draw_circle_cmd(X, 0.01)
        return draw_cmd

    def joint_force(self):
        return self.tau

class StickSpringJointLink(StickJointLink):
    def __init__(self, name, m, l, k, joint, q=None, dq=None, ddq=None, cx=None, Icog=None, I=None, XT=None, tau=0):
        super().__init__(name, m, l, joint, q, dq, ddq, cx, Icog, I, XT, tau)
        self.k = k

    def joint_force(self):
        return self.tau - self.k * self.q

class WheelJointLink(JointLink):
    def __init__(self, name, m, r, joint, q=None, dq=None, ddq=None, Icog=None, XT=None, tau=0):
        if Icog is None:
            Icog = circleI(m,r)
        I = mcI(m, [0, 0], Icog)
        if XT is None:
            XT = Xpln(0, 0, 0)
        super().__init__(name, I, m, XT, q, dq, ddq, joint)
        self.tau = tau
        self.r = r

    def joint_force(self):
        return self.tau

    def gen_draw_cmds(self, sym_list, ctx):
        X_r_to_f = lambdify(sym_list, self.X_r_to.subs(ctx))
        r_f = lambdify(sym_list, try_subs(self.r, ctx))
        def draw_cmd(v):
            X = X_r_to_f(*v)
            r = r_f(*v)
            return draw_lineseg_cmd(X, r) + draw_circle_cmd(X, r)
        return draw_cmd

def test1():
    m, l, q = symbols("m l q")
    Ic = stickI(m, l)
    I = mcI(m, [l, 0], Ic)
    X = Xpln(q, 0, 0)
    print(simplify(transInertia(I, X)-mcI(m, [l*cos(q), l*sin(q)], Ic)))

if __name__ == '__main__':
    test1()

