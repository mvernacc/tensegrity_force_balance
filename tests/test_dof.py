"""Unit tests for degree of freedom calculations."""

from pytest import approx
import numpy as np
from tensegrity_force_balance import (
    get_translation_linear_operator,
    get_rotation_linear_operator,
    calc_dofs,
    shortest_dist_between_lines,
    Constraint,
    Line3,
)


class TestGetTranslationLinearOperator:
    def test_translation_one_constraint(self):
        linop = get_translation_linear_operator([Constraint((0, 0, 0), (1, 0, 0))])
        assert linop @ np.array([1, 0, 0]) == approx(1.0)
        assert linop @ np.array([0, 1, 0]) == approx(0.0)
        assert linop @ np.array([0, 0, 1]) == approx(0.0)


class TestGetRotationLinearOperator:
    def test_three_constraints_thru_origin(self):
        linop = get_rotation_linear_operator(
            [
                Constraint((1, 0, 0), (1, 0, 0)),
                Constraint((0, 1, 0), (0, 1, 0)),
                Constraint((0, 0, 1), (0, 0, 1)),
            ]
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
        a = Line3((0, 0, 0), (0, 1, 0))
        b = Line3((1, 0, 0), (0, 1, 0))

        assert shortest_dist_between_lines(a, b) == approx(1.0)

    def test_skew(self):
        a = Line3((1, 1, -1), (-1, 0, 1))
        b = Line3((-1, -1, -1), (1, 0, 1))

        # Closest approach should be 2 apart, between points (0, 1, 0) and (0, -1, 0)
        assert shortest_dist_between_lines(a, b) == approx(2.0)

    def test_intersecting(self):
        a = Line3((1, 0, 0), (-1, 1, 0))
        b = Line3((-1, 0, 0), (1, 1, 0))

        # The lines intersect at (0, 1, 0)
        assert shortest_dist_between_lines(a, b) == approx(0.0)


class TestCalcDofs:
    def test_three_constraints_thru_origin(self):
        dofs = calc_dofs(
            [
                Constraint((1, 0, 0), (1, 0, 0)),
                Constraint((0, 1, 0), (0, 1, 0)),
                Constraint((0, 0, 1), (0, 0, 1)),
            ]
        )
        print(dofs)
        assert len(dofs) == 3
        for dof in dofs:
            assert dof.translation is None
            assert dof.rotation is not None
            assert dof.rotation.point == approx(np.zeros(3))

    def test_three_constraints_thru_111(self):
        constraints = [
            Constraint((1, 1, 1), (1, 0, 0)),
            Constraint((1, 1, 1), (0, 1, 0)),
            Constraint((1, 1, 1), (0, 0, 1)),
        ]
        dofs = calc_dofs(constraints)
        print(dofs)
        assert len(dofs) == 3
        for dof in dofs:
            assert dof.translation is None
            assert dof.rotation is not None
            # Axes of rotation will intersect or be parallel to all constraints.
            for cst in constraints:
                assert shortest_dist_between_lines(cst, dof.rotation) < 1e-9

    # TODO test with translation dofs.
    # TODO test on all examples from fig 2-21 of
    # https://ocw.mit.edu/courses/2-76-multi-scale-system-design-fall-2004/3aa5862a1724b75c3e4aa7a6fee6c511_reading_l3.pdf
