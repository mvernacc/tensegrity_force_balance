"""Unit tests for calc_wire_tensions_two_piece."""

import numpy as np
from pytest import approx

from tensegrity_force_balance import calc_wire_tensions_two_body


def four_wire_geometry() -> dict:
    """Get the geometry arguments for a simple structure with four parallel wires.

    One central wire pulls up on the platform and three outer wires pull down.
    The three outer wires are attached at the points of an equilateral triangle
    which is centered on the central wire.

    An example structure with this arrangement is shown in assets/four_wire_tensegrity_structure.webp
    from https://www.thingiverse.com/thing:4182634 by jlcarter11.

    Note that this structure is under-constrained. The platform is free for small rotations
    about the vertical axis and small translations in the two horizontal axes.
    """
    r = 0.05  # [m] Distance from the axis of the center wire to the axis of each other wire.
    z_offset = 0.05  # [m]
    return dict(
        wire_connection_points=[
            (0.0, 0.0, 0.0),
            (r, 0.0, z_offset),
            (-np.sin(np.deg2rad(30)) * r, np.cos(np.deg2rad(30)) * r, z_offset),
            (-np.sin(np.deg2rad(30)) * r, -np.cos(np.deg2rad(30)) * r, z_offset),
        ],
        wire_directions=[
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
            (0.0, 0.0, -1.0),
            (0.0, 0.0, -1.0),
        ],
        platform_com=(0.0, 0.0, z_offset),
    )


def test_four_wire():
    """Check that the correct tensions are calculated for a simple structure with four parallel wires.

    An example structure with this arrangement is shown in assets/four_wire_tensegrity_structure.webp
    from https://www.thingiverse.com/thing:4182634 by jlcarter11.
    """

    weight = 9.81 * 0.02  # [N]

    tensions = calc_wire_tensions_two_body(
        **four_wire_geometry(),
        platform_weight=(0.0, 0.0, -weight),
    )

    assert tensions[0] == approx(weight, abs=1e-4)
    assert tensions[1] == approx(0.0, abs=1e-4)
    assert tensions[2] == approx(0.0, abs=1e-4)
    assert tensions[3] == approx(0.0, abs=1e-4)


def test_four_wire_set_tension():
    """Check that the correct tensions are calculated for a simple structure with four parallel wires.
    In this test, the tension in one of the three outside wires is set.
    """
    weight = 9.81 * 0.02  # [N]
    pre_tension = 1.0  # [N]

    tensions = calc_wire_tensions_two_body(
        **four_wire_geometry(),
        platform_weight=(0.0, 0.0, -weight),
        set_tensions=[(1, pre_tension)],
    )

    assert tensions[0] == approx(weight + 3 * pre_tension, abs=1e-4)
    assert tensions[1] == approx(pre_tension, abs=1e-4)
    assert tensions[2] == approx(pre_tension, abs=1e-4)
    assert tensions[3] == approx(pre_tension, abs=1e-4)


def test_fail():
    assert False
