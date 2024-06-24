import warnings

import cvxpy as cp
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy.linalg import null_space
from dataclasses import dataclass


def _unit(x: tuple[float, float, float] | NDArray) -> NDArray:
    x_ = np.array(x)
    return x_ / np.linalg.norm(x_)


def calc_wire_tensions_two_body(
    wire_connection_points: list[tuple[float, float, float]],
    wire_directions: list[tuple[float, float, float]],
    platform_com: tuple[float, float, float],
    platform_weight: tuple[float, float, float],
    set_tensions: list[tuple[int, float]] | None = None,
    relative_stiffness: list[float] | None = None,
) -> list[float]:
    """Calculate the tension forces in the wires of a 2-body tensegrity structure.

    This function assumes the [tensegrity](https://en.wikipedia.org/wiki/Tensegrity)
    structure consists of two rigid bodies, a platform and a base, which are connected
    by thin wires. The wires can only carry tension forces. The platform is suspended
    in space by the wires. This function calculates the tension forces in the wires.

    Args:
        wire_connection_points: [m] A list of 3-vectors giving the points at which the
            wires connect to the platform.
        wire_directions: [dimensionless] A list of 3-vectors giving the direction along which the
            wire emanates from each connection point.
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
    n_wires = len(wire_connection_points)
    if len(wire_directions) != n_wires:
        raise ValueError(
            "The length of `wire_directions` and `wire_connection_points` must be the same."
        )
    if n_wires < 6:
        warnings.warn("With fewer than 6 wires, the structure is under-constrained.")
    if n_wires > 6 and relative_stiffness is None:
        raise ValueError(
            "With over 6 wires, the structure is statically indeterminate, and the tensions"
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

    d = np.array([_unit(wire_direction) for wire_direction in wire_directions])  # [dimensionless]
    com = np.array(platform_com)  # [m]
    r = np.array([np.array(x) - com for x in wire_connection_points])  # [m] center of mass
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


def get_translation_linear_operator(
    directions: list[tuple[float, float, float]],
) -> NDArray:
    """Get a linear operator which maps translations of the body -> changes in the constraint lengths.

    The sign convention for length changes is that motion in the constraint direction is a positive
    length change.

    The null space of this operator represents the translational degrees of freedom of the body,
    if any exist.

    See `calc_translation_dofs` for a description of the arguments.
    """
    return np.array([_unit(d) for d in directions])


def get_rotation_linear_operator(
    connection_points: list[tuple[float, float, float]],
    directions: list[tuple[float, float, float]],
) -> NDArray:
    """Get a linear operator which maps rotations of the body -> changes in the constraint lengths.
    This linearization is only valid for very small rotations.

    The input space for this operator is column vectors like
    ```
    [[r[0]],
     [r[1]],
     [r[2]],
     [rp[0]],
     [rp[1]],
     [rp[2]]]
    ```
    where $r$ is the axis of rotation, $p$ is the point about which the body is rotated,
    and $rp = r \\cross p$.

    The sign convention for length changes is that motion in the constraint direction is a positive
    length change.

    The null space of this operator represents the rotational degrees of freedom of the body,
    if any exist.

    See `calc_dofs` for a description of the arguments.
    """
    if len(directions) != len(connection_points):
        raise ValueError("`directions` must have the same length as `connection_points`")

    # Let
    #   $r$ be a rotation vector,
    #   $c$ be a connection point vector,
    #   $d$ be a unit direction vector,
    #   $dl$ be the scalar length change of the corresponding constraint.
    # We seek a vector $a$ such that $a \dot r = dl$. $a$ will be the row of the linear
    # operator for the corresponding constraint.
    #
    # Let $m$ be the direction of motion of $c$ for an infinitesimal rotation about $r$:
    #   $m = r \cross c$
    #
    # The constraint length change is the motion along the constraint direction:
    #   $dl = m \dot d$
    #
    # Combining these two equations:
    #   $(r \cross c) \dot d = dl$
    #
    # Use the vector triple product rule:
    #   $(c \cross d) \dot r = dl$
    #
    # Thus our linear operator row $a = (c \cross d)$.
    # TODO document
    return np.array(
        [
            np.concatenate((np.cross(c, _unit(d)), -1 * _unit(d)))
            for c, d in zip(connection_points, directions)
        ]
    )


@dataclass
class Rotation:
    axis: NDArray  # shape (3,)
    center: NDArray  # shape (3,)


def extract_rotation(r_and_rp: NDArray) -> Rotation:
    assert r_and_rp.shape == (6,)
    r = r_and_rp[:3]
    rp = r_and_rp[3:]
    p = np.cross(r, rp) / (r @ r)
    return Rotation(axis=_unit(r), center=p)


@dataclass
class DoF:
    translation: NDArray | None  # shape (3,)
    rotation: Rotation | None


def calc_dofs(
    connection_points: list[tuple[float, float, float]],
    directions: list[tuple[float, float, float]],
) -> list[DoF]:
    """Calculate the translational and rotational degrees of freedom of a constrained rigid body.

    This function models a rigid body supported by point-contact "constraint lines",
    as defined in Blanding [1].

    Args:
        directions: Directions of the constraints.

    Returns:
        basis: Dimensionless matrix of shape (6 x n_dof).
            The columns of this matrix form an orthonormal basis of the system's
            degrees of freedom; `basis.shape[1]` is the number of degrees of freedom.
            Within each column, the first three elements are a translation axis and
            the second three elements are a


    References:
        [1] D. Blanding, Exact Constraint: Machine Design Using Kinematic Processing.
            New York: American Society of Mechanical Engineers, 1999.
    """
    n_constraints = len(connection_points)
    if n_constraints == 0:
        return []

    linop_trans = get_translation_linear_operator(directions)
    linop_rot = get_rotation_linear_operator(connection_points, directions)

    basis = null_space(np.hstack((linop_trans, linop_rot)))
    print(basis)
    assert basis.shape[0] == 9

    # TODO the basis can contain extraneous columns where
    # the rotation axis is zero, but the rotation center and translation are equal.
    # Further, I suspect the rotation center and translation couple together somehow.
    # I need to correct for this when converting the null space basis into degrees of freedom.

    # Each constraint removes at most 1 degree of freedom.
    n_dof = basis.shape[1]
    assert n_dof >= 3 - n_constraints

    dofs = []
    for i in range(n_dof):
        col = basis[:, i]
        translation = None if np.linalg.norm(col[:3]) < 1e-6 else _unit(col[:3])
        rotation = None if np.linalg.norm(col[3:]) < 1e-6 else extract_rotation(col[3:])
        dofs.append(DoF(translation=translation, rotation=rotation))
    return dofs


def _draw_vector_three_view(
    top_xy: Axes,
    front_xz: Axes,
    right_yz: Axes,
    start: tuple[float, float, float] | NDArray,
    length: tuple[float, float, float] | NDArray,
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
    wire_connection_points: list[tuple[float, float, float]],
    wire_directions: list[tuple[float, float, float]],
    platform_com: tuple[float, float, float],
    platform_weight: tuple[float, float, float],
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
    for i in range(len(wire_connection_points)):
        color = f"C{i}"
        _draw_vector_three_view(
            top_xy,
            front_xz,
            right_yz,
            wire_connection_points[i],
            meter_per_newton * tensions[i] * _unit(wire_directions[i]),
            color=color,
            label=f"wire {i}",
        )
        # Draw "ghost" of short arrows to indicate their direction.
        if tensions[i] < 0.5 * max_tension:
            _draw_vector_three_view(
                top_xy,
                front_xz,
                right_yz,
                wire_connection_points[i],
                meter_per_newton * 0.5 * max_tension * _unit(wire_directions[i]),
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
