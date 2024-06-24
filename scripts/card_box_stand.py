from matplotlib import pyplot as plt
from tensegrity_force_balance import calc_wire_tensions_two_body, draw_three_view

# tensions = calc_wire_tensions_two_body(
#     wire_connection_points=[
#         (0.0, -0.040, 0.147),
#         (0.0, -0.100, 0.300),
#         (0.100, 0.100, 0.300),
#         (0.100, 0.100, 0.300),
#         (-0.100, 0.100, 0.300),
#         (-0.100, 0.100, 0.300),
#     ],
#     wire_directions=[
#         (0.0, 1.0, 1.0),
#         (0.0, -np.sin(np.deg2rad(70)), -np.cos(np.deg2rad(70))),
#         (1.0, -1.0, -3.0),
#         (-1.0, 1.0, -3.0),
#         (-1.0, -1.0, -3.0),
#         (1.0, 1.0, -3.0),
#     ],
#     platform_com=(0.0, 0.0, 0.350),
#     platform_weight=(0.0, 0.0, -10.0),
#     relative_stiffness=[2.0, 1.0, 1.0, 1.0, 1.0, 1.0],
# )


# wire_connection_points = [
#     (0.0, -0.1 * 2**0.5, 0.300),
#     (0.0, -0.1 * 2**0.5, 0.300),
#     (0.100, 0.100, 0.300),
#     (0.100, 0.100, 0.300),
#     (-0.100, 0.100, 0.300),
#     (-0.100, 0.100, 0.300),
# ]
# wire_directions = [
#     (0.0, -(2**0.5), -3.0),
#     (0.0, 2**0.5, -3.0),
#     (1.0, -1.0, -3.0),
#     (-1.0, 1.0, -3.0),
#     (-1.0, -1.0, -3.0),
#     (1.0, 1.0, -3.0),
# ]
# platform_com = (0.0, 0.0, 0.350)
# platform_weight = (0.0, 0.0, 10.0)

wire_connection_points = [
    # (0.0, -0.040, 0.147),
    (0.0, 0.0, 0.0),
    (0.0, -0.100, 0.300),
    (0.100, 0.100, 0.300),
    (0.100, 0.100, 0.300),
    (-0.100, 0.100, 0.300),
    (-0.100, 0.100, 0.300),
]
wire_directions = [
    # (0.0, 1.0, 1.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
    (1.0, -1.0, -3.0),
    (-1.0, 1.0, -3.0),
    (-1.0, -1.0, -3.0),
    (1.0, 1.0, -3.0),
]
platform_com = (0.0, 0.0, 0.350)
platform_weight = (0.0, 0.0, -10.0)

try:
    tensions = calc_wire_tensions_two_body(
        wire_connection_points,
        wire_directions,
        platform_com,
        platform_weight,
        relative_stiffness=[3.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        set_tensions=[(1, 1.0)],
    )

    print("\n".join(f"wire {i}: {t:.1f} N" for i, t in enumerate(tensions)))

    draw_three_view(
        wire_connection_points,
        wire_directions,
        platform_com,
        platform_weight,
        tensions,
        meter_per_newton=0.1,
    )
except ValueError as err:
    print(err)

    draw_three_view(
        wire_connection_points,
        wire_directions,
        platform_com,
        platform_weight,
        tensions=6 * [1.0],
        meter_per_newton=0.1,
    )

plt.show()
