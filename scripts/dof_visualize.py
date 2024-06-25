from tensegrity_force_balance import (
    calc_rotation_center,
    calc_dofs,
    _draw_vector_three_view,
    shortest_dist_between_lines,
)
import numpy as np
from matplotlib import pyplot as plt

connection_points = [
    (1.0, 1.0, 1.0),
    (1.0, 1.0, 1.0),
    (1.0, 1.0, 1.0),
]
directions = [
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
]

calc_rotation_center(connection_points, directions, np.array([1.0, 0.0, 0.0]))
calc_rotation_center(connection_points, directions, np.array([0.0, 1.0, 0.0]))
calc_rotation_center(connection_points, directions, np.array([0.0, 0.0, 1.0]))

calc_rotation_center(connection_points, directions, np.array([1.0, 1.0, 1.0]))

dofs = calc_dofs(connection_points, directions)
for i, dof in enumerate(dofs):
    if dof.rotation is None:
        continue
    for j, (c, d) in enumerate(zip(connection_points, directions)):
        x = shortest_dist_between_lines(c, d, dof.rotation.center, dof.rotation.axis)
        print(f"Shortest distance between rotation axis {i} and constraint line {j} = {x}")
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(15, 15))
top_xy, leg = axes[0]
front_xz, right_yz = axes[1]
for c, d in zip(connection_points, directions):
    _draw_vector_three_view(
        top_xy,
        front_xz,
        right_yz,
        c,
        d,
        color="black",
    )
for i, dof in enumerate(dofs):
    color = f"C{i}"
    if dof.rotation is not None:
        _draw_vector_three_view(
            top_xy,
            front_xz,
            right_yz,
            dof.rotation.center,
            dof.rotation.axis,
            color=color,
        )
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
plt.show()
