import numpy as np
from matplotlib import pyplot as plt

from tensegrity_force_balance import (
    calc_dofs,
    draw_constraint_3d,
    draw_constraint_three_view,
    draw_dof_3d,
    draw_dof_three_view,
    shortest_dist_between_lines,
    Constraint,
    get_rotation_linear_operator,
    constraints_allow_dof,
    DoF,
    Rotation,
)

# Three skew constraints example from Blanding Figure 6.4.11
constraints = [
    Constraint((1, 0, 0), (0, 1, 0)),
    Constraint((0, 1, 0), (0, 0, 1)),
    Constraint((0, 0, 1), (1, 0, 0)),
]
# The "screw" motion rotating about (1, 1, 1) and translating in the (-1, -1, -1)
# direction should be an allowed degree of freedom
A = get_rotation_linear_operator(constraints)
print(A @ np.array([1, 1, 1, 1, 1, 1]))
assert np.all(A @ np.array([1, 1, 1, 1, 1, 1]) == np.zeros(3))
assert constraints_allow_dof(constraints, DoF((-1, -1, -1), Rotation((0, 0, 0), (1, 1, 1))))

# constraints = [
#     Constraint((1, 1, 1), (1, 0, 0)),
#     Constraint((1, 1, 1), (0, 1, 0)),
#     Constraint((1, 1, 1), (0, 0, 1)),
# ]
# connection_points = [
#     (1.0, 1.0, 1.0),
#     (1.0, 1.0, 1.0),
#     (1.0, 1.0, 1.0),
# ]
# directions = [
#     (1.0, 0.0, 0.0),
#     (0.0, 1.0, 0.0),
#     (0.0, 0.0, 1.0),
# ]

# 0 T, 2 R example
# connection_points = [
#     (1.0, 0.0, -1.0),
#     (1.0, 0.0, 1.0),
#     (0.0, 0.0, 1.0),
#     (0.0, -1.0, 0.0),
# ]
# directions = [
#     (-1.0, 0.0, 0.0),
#     (-1.0, 0.0, 0.0),
#     (0.0, 0.0, -1.0),
#     (0.0, 1.0, 0.0)
# ]

# 1 T, 3 R example
# constraints = [
#     Constraint((1, 0, 0), (-1, 0, 0)),
#     Constraint((0, -1, 0), (0, 1, 0)),
# ]

dofs = calc_dofs(constraints)
print(dofs)

for i, dof in enumerate(dofs):
    if dof.rotation is None:
        continue
    for j, cst in enumerate(constraints):
        x = shortest_dist_between_lines(cst, dof.rotation)
        print(f"Shortest distance between rotation axis {i} and constraint line {j} = {x}")
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(15, 15))
axes[0, 1].remove()
axes[0, 1] = fig.add_subplot(2, 2, 2, projection="3d")
top_xy, ortho = axes[0]
front_xz, right_yz = axes[1]
ortho.set_proj_type("ortho")
for cst in constraints:
    draw_constraint_three_view(top_xy, front_xz, right_yz, cst)
    draw_constraint_3d(ortho, cst)
for i, dof in enumerate(dofs):
    color = f"C{i}"
    draw_dof_three_view(
        top_xy,
        front_xz,
        right_yz,
        dof,
        color=color,
    )
    draw_dof_3d(ortho, dof, color=color)
top_xy.set_title("Top")
top_xy.set_xlabel("$x$")
top_xy.set_ylabel("$y$")
front_xz.set_title("Front")
front_xz.set_xlabel("$x$")
front_xz.set_ylabel("$z$")
right_yz.set_title("Right")
right_yz.set_xlabel("$y$")
right_yz.set_ylabel("$z$")
ortho.set_xlabel("$x$")
ortho.set_ylabel("$y$")
ortho.set_zlabel("$z$")

for ax in (top_xy, front_xz, right_yz):
    ax.set_aspect("equal")
ortho.set_aspect("equal")

fig.tight_layout()
fig.subplots_adjust(right=0.94)
plt.show()
