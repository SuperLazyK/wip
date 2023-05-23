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
from wipg import WIPG

def test():
    model_g = WIPG()

    import graphic
    viewer = graphic.Viewer(scale=200, offset=[0, 0.2])

    t = 0
    dt = 0.001
    in_air = False

    pause = True
    def event_handler(key, shifted):
        nonlocal pause
        if key == 'q':
            sys.exit()
        elif key == 's':
            pause = pause ^ True
        elif key == 'l':
            model_g.v_ref = 5
        elif key == 'h':
            model_g.v_ref = -5
        elif key == 'j':
            model_g.v_ref = 0
        #elif key == 'p':
        #    model_g.p_ref = np.deg2rad(45)
        #    model_g.update_gain()
        #elif key == 'n':
        #    model_g.p_ref = np.deg2rad(-45)
        #    model_g.update_gain()
        #elif key == 'k':
        #    model_g.p_ref = 0
        #    model_g.update_gain()

    while True:
        if in_air:
            pass
        else:
            cmds = model_g.draw()

        viewer.handle_event(event_handler)
        viewer.clear()
        viewer.text([ f"t: {t:.03f}"
                    , graphic.arr2txt(model_g.q_v, " q")
                    , graphic.arr2txt(model_g.dq_v, "dq")
                    ])
        viewer.draw(cmds)
        viewer.draw_horizon(0)
        viewer.flush(dt)

        if pause:
            continue

        t = t + dt

        if in_air:
            pass
        else:
            model_g.step(dt)

if __name__ == '__main__':
    test()
