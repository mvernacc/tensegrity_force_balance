"""Unit tests for degree of freedom calculations."""

from pytest import approx
import numpy as np
from tensegrity_force_balance import (
    get_translation_linear_operator,
    get_rotation_linear_operator,
    calc_dofs,
    shortest_dist_between_lines,
)


class TestGetTranslationLinearOperator:
    def test_translation_one_constraint(self):
        linop = get_translation_linear_operator([(1.0, 0.0, 0.0)])
        assert linop @ np.array([1, 0, 0]) == approx(1.0)
        assert linop @ np.array([0, 1, 0]) == approx(0.0)
        assert linop @ np.array([0, 0, 1]) == approx(0.0)


class TestGetRotationLinearOperator:
    def test_three_constraints_thru_origin(self):
        linop = get_rotation_linear_operator(
            connection_points=[
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ],
            directions=[
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ],
        )

        # No length changes for rotation about the x axis through the origin.
        r = [1, 0, 0]
        p = [0, 0, 0]
        assert linop @ np.concatenate((r, np.cross(r, p))) == approx(np.zeros(3))
        # No length changes for rotation about the y axis through the origin.
        r = [0, 1, 0]
        p = [0, 0, 0]
        assert linop @ np.concatenate((r, np.cross(r, p))) == approx(np.zeros(3))
        # No length change for rotation about the z axis through the origin.
        r = [0, 0, 1]
        p = [0, 0, 0]
        assert linop @ np.concatenate((r, np.cross(r, p))) == approx(np.zeros(3))

        # This rotation should change the length of only the z-aligned constraint
        # because it passes through the contact point of the x-aligned constraint
        # and is parallel to the y-aligned constraint.
        r = [0, 1, 0]
        p = [1, 0, 0]
        dl = linop @ np.concatenate((r, np.cross(r, p)))
        assert dl[0] == approx(0)
        assert dl[1] == approx(0)
        assert dl[2] > 1e-3


class TestShortestDistanceBetweenLines:
    def test_parallel(self):
        p1 = np.array((0, 0, 0))
        d1 = np.array((0, 1, 0))

        p2 = np.array((1, 0, 0))
        d2 = np.array((0, 1, 0))

        assert shortest_dist_between_lines(p1, d1, p2, d2) == approx(1.0)

    def test_skew(self):
        p1 = np.array((1, 1, -1))
        d1 = np.array((-1, 0, 1))

        p2 = np.array((-1, -1, -1))
        d2 = np.array((1, 0, 1))

        # Closest approach should be 2 apart, between points (0, 1, 0) and (0, -1, 0)
        assert shortest_dist_between_lines(p1, d1, p2, d2) == approx(2.0)

    def test_intersecting(self):
        p1 = np.array((1, 0, 0))
        d1 = np.array((-1, 1, 0))

        p2 = np.array((-1, 0, 0))
        d2 = np.array((1, 1, 0))

        # The lines intersect at (0, 1, 0)
        assert shortest_dist_between_lines(p1, d1, p2, d2) == approx(0.0)


class TestCalcDofs:
    def test_three_constraints_thru_origin(self):
        dofs = calc_dofs(
            connection_points=[
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ],
            directions=[
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ],
        )
        print(dofs)
        assert len(dofs) == 3
        for dof in dofs:
            assert dof.translation is None
            assert dof.rotation is not None
            assert dof.rotation.center == approx(np.zeros(3))

    def test_three_constraints_thru_111(self):
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
        dofs = calc_dofs(connection_points, directions)
        print(dofs)
        assert len(dofs) == 3
        for dof in dofs:
            assert dof.translation is None
            assert dof.rotation is not None
            # Axes of rotation will intersect or be parallel to all constraints.
            for c, d in zip(connection_points, directions):
                assert (
                    shortest_dist_between_lines(c, d, dof.rotation.center, dof.rotation.axis) < 1e-9
                )

    # TODO test with translation dofs.
