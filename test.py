    def test(self):

        q_sym = [q1, q2]
        dq_sym = [dq1, dq2]
        u_sym =[u]
        ddq_f = self.model.gen_ddq_f(q_sym, dq_sym, u_sym, self.context, fext=[Matrix([0, 0, (mw+mb)*g]), zeros(3,1)])
        def nonlin_rhs(t, x, u, params):
            v_q = phith2q(x[2:4])
            v_dq = phith2q(x[:2])
            ret = q2phith(ddq_f(v_q, v_dq, [u[0]])) # normal force : fn does not have effect to dynamics
            return [ret[0], ret[1], x[0], x[1]]

        nonlin_cplant = ct.NonlinearIOSystem( nonlin_rhs, lambda t, x, u, params: [x[0], x[1],x[3]], inputs='u', outputs=('dphi', 'dth', 'th'), states=4, name='P')
        lin_cplant = ct.ss(self.A, self.B, np.identity(3), 0, name='LP', inputs=('u'), outputs=('dphi', 'dth', 'th'))

        #lin_csys = ct.ss(self.A - self.B @ K, self.B @ K, np.identity(3), 0)
        lin_csys = ct.feedback(ct.series(self.K, lin_cplant), np.identity(3), sign= -1)
        nonlin_csys = ct.feedback(ct.series(self.K, nonlin_cplant), np.identity(3), sign= -1)
        ts = np.linspace(0, 5, 100)
        #ts, xs = ct.step_response(lin_csys, input=0, T=T)
        #ts, xs = ct.input_output_response(lin_csys, ts, U=[1, 0, 0])
        ts, xs = ct.input_output_response(nonlin_csys, ts, U=[1, 0, 0])

        import matplotlib.pyplot as plt
        plt.subplot(111)
        plt.title("Identity weights")
        plt.plot(ts.T, xs[0].T, '-', label="phi'")
        plt.plot(ts.T, xs[1].T, '--', label="th'")
        plt.plot(ts.T, xs[2].T, '--', label="th")
        plt.legend()
        plt.show()
        sys.exit(0)


def test():
    model, jl1, jl2 = sym_models()

    q = np.array([0, -0.5], dtype=np.float64)
    dq = np.array([0, 1], dtype=np.float64)
    controller = WIPController(model, context)

    #controller.test()
    q_sym = [q1, q2]
    dq_sym = [dq1, dq2]
    u_sym =[u]
    ddq_f = model.gen_ddq_f(q_sym, dq_sym, u_sym, context)
    draw_cmds = model.gen_draw_cmds(q_sym, dq_sym, context)

    import graphic
    viewer = graphic.Viewer(scale=400, offset=[0, 0.4])
    dt = 0.001
    t = 0
    v_ref = 0

    def event_handler(key, shifted):
        nonlocal v_ref
        if key == 'q':
            sys.exit()
        elif key == 'k':
            v_ref = 5
        elif key == 'j':
            v_ref = -5
        elif key == 'l':
            v_ref = 0

    while True:
        t = t + dt
        v_u = controller.tau(v_ref, q, dq)
        ddq = ddq_f(q, dq, v_u)

        dq[:2] = dq[:2] + ddq * dt
        q = q + dq * dt

        #scipy.integrate.RK45(ddq, t, q, dq)
        k = model.kinetic_energy()
        cmds = draw_cmds(q, dq)
        viewer.handle_event(event_handler)
        viewer.clear()
        viewer.text([f"t: {t:.03f} :q {q[0]:.03f} {q[1]:.03f}"])
        viewer.draw(cmds)
        viewer.draw_horizon(0)
        viewer.flush(dt)

def test2():
    model, jls = sym_model_air()
    q = np.array([-1, 1, 0, 0], dtype=np.float64)
    dq = np.array([1, 0, 0, 0], dtype=np.float64)
    #dq = np.array([0.45, 0, 1, 0], dtype=np.float64)
    global context
    #context[g] = 0

    q_sym = [q1, q2, q3, q4]
    dq_sym = [dq1, dq2, dq3, dq4]
    fx,fy = symbols('fx fy')
    u_sym =[u, fx, fy]

    ei = EnvironmentInteruction(model, context)

    #ddq_f = model.gen_ddq_f(q_sym, dq_sym, u_sym, context)
    ddq_f = model.gen_ddq_f(q_sym, dq_sym, u_sym, context, fext=[zeros(3,1), zeros(3,1), Matrix([- fx * (q2 - r) + fy * q1, fx, fy]), zeros(3,1)])
    draw_cmds = model.gen_draw_cmds(q_sym, dq_sym, context)

    import graphic
    viewer = graphic.Viewer(scale=100, offset=[0, 0.2])
    dt = 0.001
    t = 0
    def event_handler(key, shifted):
        nonlocal dq
        if key == 'q':
            sys.exit()
        elif key == 'h':
            dq = ei.friction_impulse(q, dq)
        elif key == 'l':
            dq = np.array([1, 0, 0, 0], dtype=np.float64)

    in_air = True

    while True:
        t = t + dt
        fyv = ei.normal(q, dq)
        q, dq = euler_step(ddq_f, q, dq, dt, [0, 0, fyv])
        if in_air and fyv > 0:
            #print("conflict")
            dq = ei.friction_impulse(q, dq)
            in_air = False
        #print(dq)
        k = model.kinetic_energy()
        cmds = draw_cmds(q, dq)
        viewer.handle_event(event_handler)
        viewer.clear()
        viewer.text([f"t: {t:.03f} :q {q[0]:.03f} {q[1]:.03f} {q[2]:.03f} {q[3]:.03f}"])
        viewer.draw(cmds)
        viewer.draw_horizon(0)
        viewer.flush(dt)

if __name__ == '__main__':
    #test()
    test2()

