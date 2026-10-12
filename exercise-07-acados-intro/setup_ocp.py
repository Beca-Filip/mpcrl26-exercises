import casadi as ca
import numpy as np
from acados_template import AcadosOcp, AcadosModel


def export_pendulum_model() -> AcadosModel:
    model_name = 'pendulum_ode'

    # constants
    g = 9.81 # gravity constant [m/s^2]
    l = 0.8 # length of the rod [m]
    m_cart = ca.SX.sym('m') #1. # mass of the cart [kg]
    m = 0.1 # mass of the ball [kg]

    # set up states & controls
    x1 = ca.SX.sym('x1')
    theta = ca.SX.sym('theta')
    v1 = ca.SX.sym('v1')
    dtheta = ca.SX.sym('dtheta')

    x = ca.vertcat(x1, theta, v1, dtheta)

    F = ca.SX.sym('F')
    u = ca.vertcat(F)

    # xdot
    x1_dot = ca.SX.sym('x1_dot')
    theta_dot = ca.SX.sym('theta_dot')
    v1_dot = ca.SX.sym('v1_dot')
    dtheta_dot = ca.SX.sym('dtheta_dot')

    xdot = ca.vertcat(x1_dot, theta_dot, v1_dot, dtheta_dot)

    # dynamics
    cos_theta = ca.cos(theta)
    sin_theta = ca.sin(theta)
    denominator = m_cart + m - m*cos_theta*cos_theta
    f_expl_expr = ca.vertcat(v1,
                     dtheta,
                     (-m*l*sin_theta*dtheta*dtheta + m*g*cos_theta*sin_theta+F)/denominator,
                     (-m*l*cos_theta*sin_theta*dtheta*dtheta + F*cos_theta+(m_cart+m)*g*sin_theta)/(l*denominator)
                     )

    model = AcadosModel()
    model.x = x
    model.xdot = xdot
    model.u = u
    model.p = m_cart
    model.name = model_name
    model.f_expl_expr = f_expl_expr

    # store meta information
    model.x_labels = ['$x$ [m]', r'$\theta$ [rad]', '$v$ [m]', r'$\dot{\theta}$ [rad/s]']
    model.u_labels = ['$F$']
    model.t_label = '$t$ [s]'
    
    return model


def setup_ocp(Fmax: float, N_horizon: int, Tf: float,
              w_x: float = 1., w_theta: float = 1., w_v: float = 1e-4, w_dtheta: float = 1e-4,
              w_u: float = 1e-4) -> AcadosOcp:
    """Create an ocp object to formulate the OCP for the cartpole system.
    
    Args:
        Fmax: Maximum force that can be applied to the cart.
        N_horizon: Number of steps in the prediction horizon.
        Tf: Total time horizon of the prediction horizon.
        w_x: Weight for the x state.
        w_theta: Weight for the theta state.
        w_v: Weight for the v state.
        w_dtheta: Weight for the dtheta state.
        w_u: Weight for the u input.
    """
    # TODO Exercise: Fill in the blanks in this method
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()
    # set model
    ocp.model = export_pendulum_model()
    ocp.parameter_values = np.array([1.])

    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()

    # ======================= set cost =======================
    # NOTE: NONLINEAR_LS pseudo: 0.5 * (y_expr - yref)^T W (y_expr - yref)
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    # weight matrices
    Wx = np.diag([4., 1., 1e-3, 1e-3])
    Wu = np.diag([1e-4])
    W = np.block([
        [Wx, np.zeros((nx, nu))],   
        [np.zeros((nu, nx)), Wu]
    ])
    # Wx = np.diag([w_x, w_theta, w_v, w_dtheta])
    # Wu = np.diag([w_u])
    # W = np.block([
    #     [Wx, np.zeros((nx, nu))],   
    #     [np.zeros((nu, nx)), Wu]
    # ])
    ocp.cost.W = W # TODO: set something # state and input weights
    ocp.cost.W_e = ocp.cost.W[:nx, :nx]

    # expressions for ys
    x = ocp.model.x
    u = ocp.model.u
    # NOTE (use x and u from the two lines above here)
    # ocp.model.cost_y_expr = 0.5 * ca.vertcat(x, u).T @ ocp.cost.W @ ca.vertcat(x, u) 
    # ocp.model.cost_y_expr_e = 0.5 * ca.vertcat(x).T @ ocp.cost.W_e @ ca.vertcat(x)
    ocp.model.cost_y_expr = ca.vertcat(x, u)
    ocp.model.cost_y_expr_e = ca.vertcat(x)

    # references
    # raise NotImplementedError()
    y_ref = np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])
    ocp.cost.yref = y_ref # desired state and input
    ocp.cost.yref_e = ocp.cost.yref[:nx]

    # ============== set constraints ================
    # Actuator constraints
    # NOTE: Use Fmax
    ocp.constraints.lbu = np.array([-Fmax])
    ocp.constraints.ubu = np.array([Fmax])
    ocp.constraints.idxbu = np.array([0])

    # Will be changed in the closed loop
    ocp.constraints.x0 = np.array([0.0, np.pi, 0.0, 0.0])

    # ============== set ocp options ================
    # prediction horizon
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = Tf

    # additional ocp options
    # ocp.solver_options.nlp_solver_type = 'SQP'
    # ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    # ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.integrator_type = 'ERK'
    # ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.qp_tol = 1e-8
    # ocp.solver_options.qp_tol = 1e-5
    # ocp.solver_options.tol = 1e-6
    ocp.solver_options.tol = 1e-4
    # ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    ocp.solver_options.globalization = 'FIXED_STEP'
    ocp.name = 'pendulum_ocp'

    return ocp