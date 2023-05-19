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

