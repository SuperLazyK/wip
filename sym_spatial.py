from sympy import *

# NOTE: dR(th)/dt = tilda(omega) R
# <=> detilda(dR(th)/dt R.T)
def tilda(v):
    return Matrix([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
        ])

def detilda(m):
    return Matrix([m[2,1], m[0,2], m[1,0]]).T

# The Pluker transform from A to B coordinates
# Let A and B be Cartesian frames with origins at O and P , respectively
# let r be the coordinate vector expressing OP in A coordinates
# let E be the rotation matrix to transforms 3D vectors **from A to B**
def plx(E,r):
    return Matrix(BlockMatrix([[E, zeros(3,3)], [-E * tilda(r), E]]))

def rz(th):
    s = sin(th)
    c = cos(th)
    return Matrix([
        [c, s, 0],
        [-s, c, 0],
        [0, 0, 1]
        ])

def mv(omega, v):
    return Matrix(BlockMatrix([[omega], [v]]))

def toPlanarMat(m):
    return m[2:5,2:5]

def toPlanarVec(v):
    return v[2:5]


def test():
    x, y, th = symbols('x y th')
    E = rz(th)
    p = Matrix([[x, y, 0]]).T
    pprint(toPlanarMat(plx(E,p)))



def rac_and_pinion():
    t = symbols('t')
    q1 = Function('q1')(t)
    dq1 = diff(q1,t)
    r = symbols('r')
    E = rz(q1)
    # ex 4.6
    p = Matrix([-r * q1, 0, 0])
    #p = Matrix([r * q1, 0, 0])
    dq1 = diff(q1,t)
    Xp_s = plx(E,p)
    print("Xp_s", Xp_s)
    dp = diff(p,t)
    omega = detilda(simplify(diff(E,t).T * E)).T
    print("omega", omega)
    vJ_p = mv(omega, tilda(p) * omega + dp)
    print("vJ_p", vJ_p)
    vJ = Xp_s * vJ_p
    print("vJ", toPlanarVec(vJ))
    S = vJ.jacobian(Matrix([dq1]))
    print("S", toPlanarVec(S))
    cJ = S.jacobian(Matrix([q1])) * Matrix([[dq1]]) * dq1
    print(simplify(cJ))



rac_and_pinion()

