import warnings

import cvxpy as cp
import numpy as np
from kinematic_constraint import Constraint, calc_dofs_basis
from kinematic_constraint.geometry import Vec3
from matplotlib import pyplot as plt
from matplotlib.axes import Axes


def calc_wire_tensions_two_body(
    wire_connections: list[Constraint],
    platform_com: Vec3,
    platform_weight: Vec3,
    set_tensions: list[tuple[int, float]] | None = None,
    relative_stiffness: list[float] | None = None,
) -> list[float]:
    """Calculate the tension forces in the wires of a 2-body tensegrity structure.

    This function assumes the [tensegrity](https://en.wikipedia.org/wiki/Tensegrity)
    structure consists of two rigid bodies, a platform and a base, which are connected
    by thin wires. The wires can only carry tension forces. The platform is suspended
    in space by the wires. This function calculates the tension forces in the wires.

    Args:
        wire_connections: For each wire supporting the platform, the point at which it connects
            to the platform [m], and the direction from the remote end to the connection point
            [dimensionless].
        platform_com: [m] The location of the platform's center of mass.
        platform_weight: [N] The weight vector of the platform.
        set_tensions: Optionally, set the tension force in zero or more wires.
            For each item, the first element is the index in the wire lists, and
            the second element is the fixed tension force in newtons.
        relative_stiffness: [dimensionless] the relative stiffness of each wire. For statically
            indeterminate arrangements, the relative stiffnesses are needed to calculate accurate
            tension forces. If the wires are all made of the same material and have the same diameter,
            their stiffnesses are inversely proportional to their lengths.

    The vector arguments may be in any coordinate system, but they must all be given in
    the same coordinate system.

    Returns:
        [N] the tension force in each wire.
    """
    n_wires = len(wire_connections)

    # Check if the structure is over- or under-constrained.
    dofs = calc_dofs_basis(wire_connections)
    if len(dofs) > 0:
        warnings.warn(
            f"The structure is under-constrained and has {len(dofs)} degrees of freedom for small displacements:\n"
            + "\n".join(str(dof) for dof in dofs)
        )
    if n_wires > 6 and relative_stiffness is None:
        raise ValueError(
            "The structure is over-constrained / statically indeterminate, and the tensions"
            + " in the wires depend on their relative stiffnesses."
            + "  The `relative_stiffness` argument must be provided."
        )

    if set_tensions is None:
        set_tensions = []
    for i, fs in set_tensions:
        if i < 0 or i >= n_wires:
            raise ValueError(f"Invalid wire index {i} in set_tensions.")
        if fs <= 0:
            raise ValueError(f"Invalid set tension of {fs} N for wire {i}.")

    if relative_stiffness is None:
        relative_stiffness = n_wires * [1.0]
    if len(relative_stiffness) != n_wires:
        raise ValueError(
            "The length of `relative_stiffness` and `wire_connection_points` must be the same."
        )
    relative_stiffness_ = np.array(relative_stiffness)
    if not np.all(np.array(relative_stiffness_) > 0.0):
        raise ValueError("Relative stiffness must be positive.")

    # Set up a quadratic program
    #     minimize    0.5 * f^T P f
    #     subject to  A f = b
    # where the unknown f is the tension force in each wire
    #
    # The number of equality constraints is 3 + 3 + len(set_tensions),
    # for 3 directions of force balance, 3 directions of torque balance,
    # and each set tension force.
    #
    # The quadratic objective function is the total elastic energy in the wires.
    # The structure will naturally assume an energy-minimizing configuration,
    # so minimizing the energy gives the physically correct forces.
    #
    # From Hooke's law, the tension force and elastic energy in each wire are:
    #     force = k * delta_length
    #     energy = 0.5 * k * delta_length^2 = 0.5 * force^2 / k
    # where k is the stiffness of the wire.
    f = cp.Variable(n_wires, name="f")  # [N]
    P = np.diag(1.0 / relative_stiffness_)  # inverse of dimensionless relative stiffness

    d = np.array([-w.direction for w in wire_connections])  # [dimensionless]
    com = np.array(platform_com)  # [m]
    r = np.array(
        [np.array(w.point) - com for w in wire_connections]
    )  # [m] connection points relative to center of mass
    arm = np.array(
        [np.cross(ri, di) for ri, di in zip(r, d)]
    )  # [m] Moment arm of each wire about the com

    net_force_0 = d[:, 0] @ f + platform_weight[0]  # [N]
    net_force_1 = d[:, 1] @ f + platform_weight[1]  # [N]
    net_force_2 = d[:, 2] @ f + platform_weight[2]  # [N]
    net_torque_0 = arm[:, 0] @ f  # [N m]
    net_torque_1 = arm[:, 1] @ f  # [N m]
    net_torque_2 = arm[:, 2] @ f  # [N m]

    constraints = [
        f >= 0.0,  # wires can only carry tension
        net_force_0 == 0.0,
        net_force_1 == 0.0,
        net_force_2 == 0.0,
        net_torque_0 == 0.0,
        net_torque_1 == 0.0,
        net_torque_2 == 0.0,
    ]
    for i, fs in set_tensions:
        constraints.append(f[i] == fs)

    problem = cp.Problem(cp.Minimize(cp.quad_form(f, P)), constraints)
    problem.solve(solver="CLARABEL", verbose=True)
    if problem.status in ("infeasible", "unbounded"):
        raise ValueError(f"Solve failed.\n{str(problem)}")

    return f.value.tolist()


def _draw_vector_three_view(
    top_xy: Axes,
    front_xz: Axes,
    right_yz: Axes,
    start: Vec3,
    length: Vec3,
    **kwargs,
):
    hw = 0.03
    top_xy.arrow(
        start[0],
        start[1],
        length[0],
        length[1],
        head_width=hw,
        **kwargs,
    )
    front_xz.arrow(
        start[0],
        start[2],
        length[0],
        length[2],
        head_width=hw,
        **kwargs,
    )
    right_yz.arrow(
        start[1],
        start[2],
        length[1],
        length[2],
        head_width=hw,
        **kwargs,
    )


def draw_three_view(
    wire_connections: list[Constraint],
    platform_com: Vec3,
    platform_weight: Vec3,
    tensions: list[float],
    meter_per_newton: float = 1.0,
):
    """Visualize the wire forces from `calc_wire_tensions_two_body`.
    See `calc_wire_tensions_two_body` for a description of the arguments.

    `meters_per_newton` converts the force vectors to lengths on the plot.
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(15, 15))
    top_xy, leg = axes[0]
    front_xz, right_yz = axes[1]
    max_tension = max(tensions)
    for i, w in enumerate(wire_connections):
        color = f"C{i}"
        _draw_vector_three_view(
            top_xy,
            front_xz,
            right_yz,
            w.point,
            -meter_per_newton * tensions[i] * w.direction,
            color=color,
            label=f"wire {i}",
        )
        # Draw "ghost" of short arrows to indicate their direction.
        if tensions[i] < 0.5 * max_tension:
            _draw_vector_three_view(
                top_xy,
                front_xz,
                right_yz,
                w.point,
                -meter_per_newton * 0.5 * max_tension * w.direction,
                color=color,
                alpha=0.25,
            )
        leg.scatter(0.0, 0.0, color=color, label=f"wire {i}")
    _draw_vector_three_view(
        top_xy,
        front_xz,
        right_yz,
        platform_com,
        meter_per_newton * np.array(platform_weight),
        color="black",
        label="weight",
    )
    leg.scatter(0.0, 0.0, color="black", label="weight")
    leg.legend(loc="upper right")

    top_xy.set_title("Top")
    top_xy.set_xlabel("$x$")
    top_xy.set_ylabel("$y$")
    front_xz.set_title("Front")
    front_xz.set_xlabel("$x$")
    front_xz.set_ylabel("$z$")
    right_yz.set_title("Right")
    right_yz.set_xlabel("$y$")
    right_yz.set_ylabel("$z$")

    for ax in (top_xy, front_xz, right_yz):
        ax.set_aspect("equal")

    fig.tight_layout()
