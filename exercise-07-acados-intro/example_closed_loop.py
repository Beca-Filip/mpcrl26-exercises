import numpy as np
from acados_template import AcadosOcpSolver, AcadosSimSolver, AcadosSim
from plot_utils import plot_trajectories
from setup_ocp import setup_ocp
import matplotlib.pyplot as plt

def main():
    Fmax = 80

    Tf = 1.
    N_horizon = 20
    dt = Tf/N_horizon

    ocp = setup_ocp(Fmax, N_horizon, Tf)
    ocp_solver = AcadosOcpSolver(ocp)

    # create simulator
    sim = AcadosSim.from_ocp(ocp)
    sim.solver_options.T = dt
    sim.parameter_values = np.array([1.])
    integrator = AcadosSimSolver(sim)

    # initialize simulation
    Nsim = 80
    nx = ocp_solver.acados_ocp.dims.nx
    nu = ocp_solver.acados_ocp.dims.nu

    simX = np.zeros((Nsim+1, nx))
    simU = np.zeros((Nsim, nu))

    simX[0,:] = ocp.constraints.x0

    timings = np.zeros((Nsim,))
    reward = 0.0
    reward_function = lambda state: 1 if np.all(np.abs(state) < np.array([0.1, 0.02, 0.1, 0.02])) else 0

    # closed loop
    for i in range(Nsim):

        # solve ocp and get next control input
        simU[i,:] = ocp_solver.solve_for_x0(x0_bar = simX[i, :], fail_on_nonzero_status=False)

        timings[i] = ocp_solver.get_stats('time_tot')

        # environment rules.
        simU[i, :] = np.clip(simU[i, :], -Fmax, Fmax)
        reward += reward_function(simX[i, :])

        # simulate system
        simX[i+1, :] = integrator.simulate(x=simX[i, :], u=simU[i,:])

    # evaluate timings: scale to milliseconds
    timings *= 1e3
    print(f'Computation time in ms: min {np.min(timings):.3f} median {np.median(timings):.3f} max {np.max(timings):.3f}')
    print(f'reward: {reward}')

    # plot results
    model = ocp_solver.acados_ocp.model
    plot_trajectories(
        x_traj_list=[simX],
        u_traj_list=[simU],
        time_traj_list=[np.linspace(0, Nsim*dt, Nsim+1)],
        time_label=model.t_label,
        labels_list=['closed loop result'],
        x_labels=model.x_labels,
        u_labels=model.u_labels,
        idxbu=ocp.constraints.idxbu,
        lbu=ocp.constraints.lbu,
        ubu=ocp.constraints.ubu,
        show_plot=False,
        fig_filename='closed_loop_results.png',
        latexify=False,
        background_transparent=False,
    )

if __name__ == '__main__':
    main()
