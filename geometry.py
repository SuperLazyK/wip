from arith import *
import numpy as np

#-------------------------
# Geometory
#-------------------------

def draw_lineseg_cmd(X, l, color=None):
    s, c, x, y = Xtoscxy(X)
    dir = np.array([c,s])
    org = np.array([x,y])
    return [{"type": "lineseg", "color":color, "start":org, "end":org+dir*l}]

def draw_circle_cmd(X, r, color=None):
    s, c, x, y = Xtoscxy(X)
    return [ {"type": "circle", "color":color, "origin":(x,y), "r":r}]

def plot_point_cmd(x, y, r, color=None):
    return [ {"type": "circle", "color":color, "origin":(x,y), "r":r}]

