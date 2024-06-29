import copy
from dataclasses import dataclass
from typing import Self, Sequence
import warnings

import cvxpy as cp
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy.linalg import null_space
from mpl_toolkits.mplot3d import Axes3D


Vec3 = tuple[float | int, float | int, float | int] | NDArray


def _unit(x: Vec3) -> NDArray:
    x_ = np.array(x, dtype=np.float64)
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


def check_len_3(x: Vec3, name: str = "vector"):
    if isinstance(x, np.ndarray) and x.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {x.shape}")
    elif len(x) != 3:
        raise ValueError(f"{name} must have length 3, got {len(x)}")


class Line3:
    def __init__(self, point: Vec3, direction: Vec3) -> None:
        """A line in 3d space, represented by a point and a direction."""
        self.point = point
        self.direction = direction

    @property
    def point(self) -> NDArray:
        """A point the line passes through, with shape `(3,)` and units of length."""
        return self._point

    @point.setter
    def point(self, value: Vec3):
        check_len_3(value, "point")
        self._point = np.array(value, dtype=np.float64)

    @property
    def direction(self) -> NDArray:
        """The direction of the line, with shape `(3,)` and dimensionless unit length."""
        return self._direction

    @direction.setter
    def direction(self, value: Vec3):
        check_len_3(value, "direction")
        self._direction = _unit(np.array(value, dtype=np.float64))

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(point=({self.point[0]}, {self.point[1]}, {self.point[2]}), direction=({self.direction[0]}, {self.direction[1]}, {self.direction[2]}))"

    def coincident(self, other: Self, ptol: float = 1e-12, dtol: float = 1e-12):
        if np.linalg.norm(np.cross(self.direction, other.direction)) > dtol:
            return False
        return self.shortest_distance_to_point(other.point) <= ptol

    def shortest_distance_to_point(self, p: Vec3) -> float:
        # https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
        p = np.array(p, dtype=np.float64)
        # The norm of `line.direction`` is 1, so we don't need to divide by it.
        return float(np.linalg.norm(np.cross(self.direction, self.point - p)))


Constraint = Line3
"""A constraint line which connects to a rigid body at a point, and prevents the connection
point from moving along the direction of the constraint.

`point` is where the constraint connects to the object.

`direction` is a unit vector from the remote end of the constraint to the connection point.

This definition of a constraint is from Section 1.3 of Blanding [1].

References:
    [1] D. Blanding, Exact Constraint: Machine Design Using Kinematic Processing.
        New York: American Society of Mechanical Engineers, 1999.
"""


def get_translation_linear_operator(constraints: list[Constraint]) -> NDArray:
    """Get a linear operator which maps translations of the body -> changes in the constraint lengths.

    The sign convention for length changes is that motion in the constraint direction is a positive
    length change.

    The null space of this operator represents the translational degrees of freedom of the body,
    if any exist.
    """
    return np.array([_unit(cst.direction) for cst in constraints])


def get_rotation_linear_operator(constraints: list[Constraint]) -> NDArray:
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
    # TODO document for rotation about point p. The above describes rotation about the origin.
    return np.array(
        [
            np.concatenate((np.cross(cst.point, _unit(cst.direction)), -1 * _unit(cst.direction)))
            for cst in constraints
        ]
    )


Rotation = Line3
"""An axis about which a body can rotate.

`point` is a point through which the line of rotation passes.

`direction` is a unit vector along the axis of rotation.
"""


class DoF:
    def __init__(self, translation: Vec3 | None, rotation: Rotation | None):
        if translation is not None:
            check_len_3(translation, "translation")
            self._translation = _unit(np.array(translation, dtype=np.float64))
        else:
            self._translation = None
        self._rotation = rotation

    @property
    def translation(self) -> NDArray | None:
        return self._translation

    @property
    def rotation(self) -> Rotation | None:
        return self._rotation


def calc_rotation_point(constraints: list[Constraint], axis: Vec3) -> NDArray:
    """Given a set of constraints and a rotation axis,
    solve for the point about which the rotation will not change the length
    of any constraint.
    """
    A = np.array([np.cross(_unit(cst.direction), axis) for cst in constraints])
    b = np.array([np.cross(cst.point, _unit(cst.direction)) @ axis for cst in constraints])
    p, resid, rank, s = np.linalg.lstsq(A, b)
    print(f"{p=}, {resid=}, {rank=}, {s=}")
    if resid.size > 0 and np.max(resid) > 1e-6:
        assert False  # TODO handle case where there is no good rotation point.
    return p


def shortest_dist_between_lines(a: Line3, b: Line3) -> float:
    """Calculate the shortest distance between two 3d lines."""
    d1xd2 = np.cross(a.direction, b.direction)
    d1xd2_norm = np.linalg.norm(d1xd2)
    if d1xd2_norm < 1e-9:
        # Lines are almost parallel, avoid dividing by zero.
        # N.B. the norm of the directions is 1.
        return float(np.linalg.norm(np.cross(a.direction, (b.point - a.point))))
    return float(np.linalg.norm(d1xd2 @ (b.point - a.point)) / d1xd2_norm)


def closest_points_on_lines(a: Line3, b: Line3) -> tuple[NDArray, NDArray]:
    """Calculate the points where two 3d lines are closest to each other.

    Returns two arrays:
    * The point on line `a` closest to line `b`.
    * The point on line `b` closest to line `a`.

    Both are of shape `(3,)` and in the same length units as the `point`s of the lines.

    If the lines are parallel, returns an arbitrary pair of points on the lines.
    """
    d1xd2 = np.cross(a.direction, b.direction)
    d1xd2_mag2 = d1xd2 @ d1xd2
    if d1xd2_mag2 < 1e-12:
        ta = 0.0
        # Formula from https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
        # The magnitude of b.direction is 1, so we don't need to divide by it.
        tb = -(b.point - a.point) @ b.direction
    else:
        # Formula from https://math.stackexchange.com/a/4764188
        ta = np.cross((b.point - a.point), b.direction) @ d1xd2 / d1xd2_mag2
        tb = np.cross((b.point - a.point), a.direction) @ d1xd2 / d1xd2_mag2
    closest_a = a.point + ta * a.direction
    closest_b = b.point + tb * b.direction
    return closest_a, closest_b


@dataclass
class _PointAndIndexSet:
    point: NDArray
    indexes: set[int]


def use_common_point_for_intersecting_lines(lines: list[Line3], tol: float = 1e-12):
    """For any sub-set of lines which intersect, set their `point`s to be the intersection point.

    Modifies the argument in-place.
    """
    # (intersection point, indexes of lines which intersect that point)
    intersections: list[_PointAndIndexSet] = []

    # Find all points where two lines intersect.
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            pa, pb = closest_points_on_lines(lines[i], lines[j])
            if np.linalg.norm(pa - pb) < tol:
                # The lines intersect to within the tolerance.
                intersections.append(_PointAndIndexSet((pa + pb) / 2, set([i, j])))

    if len(intersections) == 0:
        return

    # Simplify the list of intersections by grouping close points together.
    while len(intersections) > 1:
        # Find the pair of intersections which are closest to each other.
        closest_dist = np.inf
        closest_pair = None
        for i in range(len(intersections)):
            for j in range(i + 1, len(intersections)):
                dist = np.linalg.norm(intersections[i].point - intersections[j].point)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_pair = (i, j)
        if closest_dist > tol:
            break
        assert closest_pair is not None
        # Merge the closest pair of intersections.
        a = intersections.pop(closest_pair[0])
        b = intersections.pop(closest_pair[1] - 1)  # we changed the indexes with the previous pop
        intersections.append(_PointAndIndexSet((a.point + b.point) / 2, a.indexes.union(b.indexes)))

    for intersection in intersections:
        for i in intersection.indexes:
            lines[i].point = intersection.point


def basis_contains_vector(basis: Sequence[Vec3], vector: Vec3) -> bool:
    A = np.stack([*basis, vector])
    return np.linalg.matrix_rank(A) <= len(basis)


def orthogonal_subspace(basis: Sequence[Vec3], vector: Vec3) -> NDArray:
    A = np.stack([vector, *basis], axis=-1)
    Q, R = np.linalg.qr(A)
    return Q[:, 1:]


def simplify_dofs(dofs: list[DoF]) -> list[DoF]:
    """Convert one set of degrees of freedom into an equivalent set,
    which a human may find more intuitive.
    """
    new_dofs = copy.deepcopy(dofs)
    use_common_point_for_intersecting_lines(
        [dof.rotation for dof in new_dofs if dof.rotation is not None]
    )
    return new_dofs


def calc_dofs(constraints: list[Constraint], simplify: bool = True) -> list[DoF]:
    """Calculate the translational and rotational degrees of freedom of a constrained rigid body.

    This function models a rigid body supported by point-contact "constraint lines",
    as defined in Blanding [1].

    References:
        [1] D. Blanding, Exact Constraint: Machine Design Using Kinematic Processing.
            New York: American Society of Mechanical Engineers, 1999.
    """
    n_constraints = len(constraints)
    if n_constraints == 0:
        return [
            DoF(translation=(1, 0, 0), rotation=None),
            DoF(translation=(0, 1, 0), rotation=None),
            DoF(translation=(0, 0, 1), rotation=None),
            DoF(translation=None, rotation=Rotation(point=(0, 0, 0), direction=(1, 0, 0))),
            DoF(translation=None, rotation=Rotation(point=(0, 0, 0), direction=(0, 1, 0))),
            DoF(translation=None, rotation=Rotation(point=(0, 0, 0), direction=(0, 0, 1))),
        ]

    if len(constraints) == 1 or all(
        constraints[0].coincident(constraints[i]) for i in range(1, len(constraints))
    ):
        print("special")
        # Either there is only one constraint, or all the constraints are coincident.
        # In this case, there are 3 rotation DoFs and 2 translation DoFs.
        # The rotation DoFs are ambiguous, as they could be any three orthogonal lines which
        # all intersect the one constraint line.
        # Just choose an intuitive set as a special case.
        translation_basis = orthogonal_subspace(
            [(1, 0, 0), (0, 1, 0), (0, 0, 1)], constraints[0].direction
        )
        return [
            DoF(translation_basis[:, 0], None),
            DoF(translation_basis[:, 1], None),
            DoF(
                translation=None, rotation=Rotation(point=constraints[0].point, direction=(1, 0, 0))
            ),
            DoF(
                translation=None, rotation=Rotation(point=constraints[0].point, direction=(0, 1, 0))
            ),
            DoF(
                translation=None, rotation=Rotation(point=constraints[0].point, direction=(0, 0, 1))
            ),
        ]

    linop_rt = get_rotation_linear_operator(constraints)

    basis = null_space(linop_rt)
    print(basis)
    assert basis.shape[0] == 6

    # Each constraint removes at most 1 degree of freedom.
    n_dof = basis.shape[1]
    assert n_dof >= 3 - n_constraints

    dofs = []
    for i in range(n_dof):
        col = basis[:, i]

        translation = None
        rotation = None
        if np.linalg.norm(col[:3]) < 1e-6:
            translation = col[3:]
        else:
            direction = _unit(col[:3])
            point = calc_rotation_point(constraints, direction)
            rotation = Rotation(point=point, direction=direction)
        dofs.append(DoF(translation=translation, rotation=rotation))

    if simplify:
        dofs = simplify_dofs(dofs)
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


CONSTRAINT_LINEWIDTH = 2.0
CONSTRAINT_EXTENDED_LINEWIDTH = 0.5
CONSTRAINT_MARKERSIZE = 10.0
DOF_LINEWIDTH = 1.0
DOF_EXTENDED_LINEWIDTH = 0.5
DOF_MARKERSIZE = 10.0
DOF_TRANSLATION_LENGTH = 0.5


def draw_constraint_three_view(
    top_xy: Axes,
    front_xz: Axes,
    right_yz: Axes,
    constraint: Constraint,
):
    for ax, i, j in ((top_xy, 0, 1), (front_xz, 0, 2), (right_yz, 1, 2)):
        if not (abs(constraint.direction[i]) < 1e-9 and abs(constraint.direction[j]) < 1e-9):
            ax.axline(
                (constraint.point[i], constraint.point[j]),
                (
                    constraint.point[i] + constraint.direction[i],
                    constraint.point[j] + constraint.direction[j],
                ),
                linestyle="--",
                linewidth=CONSTRAINT_EXTENDED_LINEWIDTH,
                color="gray",
            )
        ax.plot(
            [
                constraint.point[i],
                constraint.point[i] - constraint.direction[i],
            ],
            [
                constraint.point[j],
                constraint.point[j] - constraint.direction[j],
            ],
            linewidth=CONSTRAINT_LINEWIDTH,
            color="black",
        )
        ax.plot(
            [constraint.point[i]],
            [constraint.point[j]],
            markersize=CONSTRAINT_MARKERSIZE,
            marker=".",
            color="black",
        )


def draw_constraint_3d(ax: Axes3D, constraint: Constraint):
    ax.plot(
        [constraint.point[0], constraint.point[0] - constraint.direction[0]],
        [constraint.point[1], constraint.point[1] - constraint.direction[1]],
        [constraint.point[2], constraint.point[2] - constraint.direction[2]],
        linewidth=CONSTRAINT_LINEWIDTH,
        color="black",
    )
    ax.plot(
        [constraint.point[0]],
        [constraint.point[1]],
        [constraint.point[2]],
        markersize=CONSTRAINT_MARKERSIZE,
        marker=".",
        color="black",
    )


def draw_dof_three_view(
    top_xy: Axes,
    front_xz: Axes,
    right_yz: Axes,
    dof: DoF,
    color: str = "gray",
):
    if dof.translation is not None:
        for ax, i, j in ((top_xy, 0, 1), (front_xz, 0, 2), (right_yz, 1, 2)):
            if abs(dof.translation[i]) < 1e-9 and abs(dof.translation[j]) < 1e-9:
                ax.plot([0], [0], marker=".", color=color, markersize=DOF_MARKERSIZE)
            else:
                ax.arrow(
                    0.0,
                    0.0,
                    DOF_TRANSLATION_LENGTH * dof.translation[i],
                    DOF_TRANSLATION_LENGTH * dof.translation[j],
                    color=color,
                    head_width=0.05,
                    length_includes_head=True,
                )
    if dof.rotation is not None:
        for ax, i, j in ((top_xy, 0, 1), (front_xz, 0, 2), (right_yz, 1, 2)):
            ax.plot(
                [dof.rotation.point[i], dof.rotation.point[i] + dof.rotation.direction[i]],
                [dof.rotation.point[j], dof.rotation.point[j] + dof.rotation.direction[j]],
                linewidth=DOF_LINEWIDTH,
                markersize=DOF_MARKERSIZE,
                marker="+",
                color=color,
            )
            if not (
                abs(dof.rotation.direction[i]) < 1e-9 and abs(dof.rotation.direction[j]) < 1e-9
            ):
                ax.axline(
                    (dof.rotation.point[i], dof.rotation.point[j]),
                    (
                        dof.rotation.point[i] + dof.rotation.direction[i],
                        dof.rotation.point[j] + dof.rotation.direction[j],
                    ),
                    linestyle="--",
                    linewidth=DOF_EXTENDED_LINEWIDTH,
                    color=color,
                )


def draw_dof_3d(
    ax: Axes3D,
    dof: DoF,
    color: str = "gray",
):
    if dof.translation is not None:
        ax.quiver(
            0.0,
            0.0,
            0.0,
            DOF_TRANSLATION_LENGTH * dof.translation[0],
            DOF_TRANSLATION_LENGTH * dof.translation[1],
            DOF_TRANSLATION_LENGTH * dof.translation[2],
            color=color,
        )
    if dof.rotation is not None:
        ax.plot(
            [dof.rotation.point[0], dof.rotation.point[0] + dof.rotation.direction[0]],
            [dof.rotation.point[1], dof.rotation.point[1] + dof.rotation.direction[1]],
            [dof.rotation.point[2], dof.rotation.point[2] + dof.rotation.direction[2]],
            linewidth=DOF_LINEWIDTH,
            markersize=DOF_MARKERSIZE,
            marker="+",
            color=color,
        )
