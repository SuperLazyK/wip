from sympy import *
import sys

#init_printing(use_unicode=True)

#-------------------------
# Arith
#-------------------------

def printM(M, name="M"):
    r,c = shape(M)
    for i in range(r):
        for j in range(c):
            print(f"{name}[{i}:{j}] {M[i,j]}")

# scalar product: *
# matrix product: *
# matrix-vector product: *

# right-hand coordinate
# torq: clock-wise is positive!!

# General Transform : 2.24~2.27
def inversem(X):
    c = X[1,1]
    s = X[1,2]
    x = s*X[1,0] + c*X[2,0]
    y = -c*X[1,0] + s*X[2,0]
    return Matrix([
        [1, 0, 0],
        [y, c, -s],
        [-x, s, c]
        ])

def dualm(X):
    return inversem(X).T

# planar coordinate transform matrix : M -> M
# see the figure Table A.1
# base coordinate -> offset c,s rotated th rad toward counter-clockwise
# r : x, y
# for pos, not vel,acc
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !! This is not position converter !!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# position converter is (2.28)
# see 2.24 and just clip submatrix from Spatial matrix 
# X * r means convert r in dst(X)-coordinate into X*r in original-coordinate
def Xpln(th, x, y):
    c = cos(th)
    s = sin(th)
    return Matrix([
        [1, 0, 0],
        [s * x - c * y, c, s],
        [c * x + s * y, -s, c]
        ])

# planar coordinate transform matrix : M -> M
# offset c,s rotated th rad toward counter-clockwise -> base
def Xplnr(th, x, y):
    c = cos(th)
    s = sin(th)
    return Matrix([
        [1, 0, 0],
        [x, c, -s],
        [y, s, c]
        ])



def fromX(X, simp=False):
    th = atan2(X[1,2], X[1,1])
    x = X[1,0] * X[1,2] + X[2,0] * X[2,2]
    y = -X[1,0] * X[2,2] + X[2,0] * X[1,2] 
    return (simplify(th), simplify(x), simplify(y))

# NOTE:
# vector does not express "position"
# position can be expressed by velocity (unit time)??
# but origin-offset is not expressed.
# position can be represented by X

def vec(th, x, y):
    return Matrix([
        [th],
        [x],
        [y],
        ])

def Xtoscxy(X):
    #th = np.arctan2(X[1][2], X[1][1])
    x = X[1,0] * X[1,2] + X[2,0] * X[2,2]
    y = -X[1,0] * X[2,2] + X[2,0] * X[1,2] 
    return ([X[1,2], X[1,1], x, y])

# planar cross for motion
def crm(v):
    omega = v[0]
    vx = v[1]
    vy = v[2]
    return Matrix([
        [0, 0, 0],
        [vy, 0, -omega],
        [-vx, omega, 0]
        ])


# planar cross for force
def crf(v):
    omega = v[0]
    vx = v[1]
    vy = v[2]
    return Matrix([
        [0, -vy, vx],
        [0, 0, -omega],
        [0, omega, 0]
        ])

# X: A -> B
# I: inertia at A
# return: inertia at B
def transInertia(I, X):
    return X.T * I * X


if __name__ == '__main__':
    q1, x1, y1 = symbols('q1 x1 y1')
    fx, fy, xf, yf = symbols('fx fy x2 y2')
    tau = symbols('tau')
    Xr_to = Xpln(0, x1, y1)
    X_to_r = inversem(Xr_to)
    # torq: counter-clock-wise is positive!!
    torq = - fx * y1 + fy * x1

    # tau is global so offseted to - fx * y1 + fy * x1.
    # local coodinate torq is de-offseted
    print(simplify(dualm(Xr_to) * vec(torq, fx, fy))[0])

