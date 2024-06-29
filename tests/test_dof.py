"""Unit tests for degree of freedom calculations."""

from pytest import approx
import pytest
import numpy as np
from numpy.typing import NDArray
from tensegrity_force_balance import (
    get_translation_linear_operator,
    get_rotation_linear_operator,
    calc_dofs,
    shortest_dist_between_lines,
    closest_points_on_lines,
    use_common_point_for_intersecting_lines,
    Constraint,
    Line3,
    DoF,
    Rotation,
    basis_contains_vector,
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


class TestClosestPointsOnLine:
    def test_parallel(self):
        a = Line3((0, 0, 0), (0, 1, 0))
        b = Line3((1, 0, 0), (0, 1, 0))

        closest_a, closest_b = closest_points_on_lines(a, b)
        assert np.linalg.norm(closest_a - closest_b) == approx(1.0)

    def test_skew(self):
        a = Line3((1, 1, -1), (-1, 0, 1))
        b = Line3((-1, -1, -1), (1, 0, 1))

        # Closest approach should be 2 apart, between points (0, 1, 0) and (0, -1, 0)
        closest_a, closest_b = closest_points_on_lines(a, b)
        assert closest_a == approx(np.array([0, 1, 0]))
        assert closest_b == approx(np.array([0, -1, 0]))

    def test_intersecting(self):
        a = Line3((1, 0, 0), (-1, 1, 0))
        b = Line3((-1, 0, 0), (1, 1, 0))

        # The lines intersect at (0, 1, 0)
        closest_a, closest_b = closest_points_on_lines(a, b)
        assert closest_a == approx(np.array([0, 1, 0]))
        assert closest_b == approx(np.array([0, 1, 0]))


class TestUseCommonPointForIntersectingLines:
    def test_triangle(self):
        # These three lines for a triangle in the x,y plane.
        # They intersect at:
        #  a, b: (0, 0, 0)
        #  a, c: (1, 0, 0)
        #  b, c: (0, 1, 0)
        a = Line3((0, 0, 0), (1, 0, 0))
        b = Line3((0, 1, 0), (0, 1, 0))
        c = Line3((2, -1, 0), (-1, 1, 0))

        use_common_point_for_intersecting_lines([a, b, c])

        # Each line's point should be set to one of it's
        # two intersection points, but it does not matter which one.
        assert a.point == approx(np.zeros(3)) or a.point == approx(np.array([1, 0, 0]))
        assert b.point == approx(np.zeros(3)) or b.point == approx(np.array([0, 1, 0]))
        assert c.point == approx(np.array([1, 0, 0])) or c.point == approx(np.array([0, 1, 0]))

    def test_three_thru_origin(self):
        # These three lines all intersect at the origin.
        a = Line3((2, 0, 0), (1, 0, 0))
        b = Line3((0, 2, 0), (0, 1, 0))
        c = Line3((0, 0, 2), (0, 0, 1))

        use_common_point_for_intersecting_lines([a, b, c])

        assert a.point == approx(np.zeros(3))
        assert b.point == approx(np.zeros(3))
        assert c.point == approx(np.zeros(3))


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

    @pytest.mark.parametrize(
        ["offset"],
        [
            (np.array([0.0, 0.0, 0.0]),),
            (np.array([0.1, 0.2, 0.3]),),
        ],
    )
    @pytest.mark.parametrize(
        ["constraints", "correct_dofs"],
        [
            # 3 R, 3 T
            (
                [],
                [
                    DoF((1, 0, 0), None),
                    DoF((0, 1, 0), None),
                    DoF((0, 0, 1), None),
                    DoF(None, Rotation((0, 0, 0), (1, 0, 0))),
                    DoF(None, Rotation((0, 0, 0), (0, 1, 0))),
                    DoF(None, Rotation((0, 0, 0), (0, 0, 1))),
                ],
            ),
            # 3 R, 2 T
            (
                [
                    Constraint((1, 0, 0), (-1, 0, 0)),
                ],
                [
                    DoF((0, 1, 0), None),
                    DoF((0, 0, 1), None),
                    DoF(None, Rotation((0, 0, 0), (1, 0, 0))),
                    DoF(None, Rotation((0, 0, 0), (0, 1, 0))),
                    DoF(None, Rotation((0, 0, 0), (0, 0, 1))),
                ],
            ),
            # 3 R, 1 T
            (
                [
                    Constraint((1, 0, 0), (-1, 0, 0)),
                    Constraint((0, -1, 0), (0, 1, 0)),
                ],
                [
                    DoF((0, 0, 1), None),
                    DoF(None, Rotation((0, 0, 0), (1, 0, 0))),
                    DoF(None, Rotation((0, 0, 0), (0, 1, 0))),
                    DoF(None, Rotation((0, 0, 0), (0, 0, 1))),
                ],
            ),
            # 3 R, 0 T
            (
                [
                    Constraint((1, 0, 0), (-1, 0, 0)),
                    Constraint((0, -1, 0), (0, 1, 0)),
                    Constraint((0, 0, 1), (0, 0, -1)),
                ],
                [
                    DoF(None, Rotation((0, 0, 0), (1, 0, 0))),
                    DoF(None, Rotation((0, 0, 0), (0, 1, 0))),
                    DoF(None, Rotation((0, 0, 0), (0, 0, 1))),
                ],
            ),
            # TODO add cases for 2, 1, and 0 R
        ],
    )
    def test_hale_2_21(
        self, constraints: list[Constraint], correct_dofs: list[DoF], offset: NDArray
    ):
        """Test on the example cases shown in Figure 2-21 of Hale [1].

        References:
            [1] L. C. (Layton C. Hale, "Principles and techniques for designing precision machines,"
                 Thesis, Massachusetts Institute of Technology, 1999. Accessed: Jun. 28, 2022.
                 [Online]. Available: https://dspace.mit.edu/handle/1721.1/9414
        """
        for cst in constraints:
            cst.point += offset

        for constraint in constraints:
            print(str(constraint))

        dofs = calc_dofs(constraints)

        assert len(dofs) == len(correct_dofs)
        for dof in dofs:
            assert dof.translation is None or dof.rotation is None
        translations = [dof.translation for dof in dofs if dof.translation is not None]
        rotations = [dof.rotation for dof in dofs if dof.rotation is not None]
        correct_translations = [
            dof.translation for dof in correct_dofs if dof.translation is not None
        ]
        correct_rotations = [dof.rotation for dof in correct_dofs if dof.rotation is not None]
        assert len(translations) == len(correct_translations)
        assert len(rotations) == len(correct_rotations)

        # "The axes of a body's rotational degrees of freedom will each intersect
        # all constraints applied to the body"
        # -- Hale, Section 2.6, Statement 4.
        for rotation in rotations:
            for constraint in constraints:
                assert shortest_dist_between_lines(constraint, rotation) < 1e-9
        # For good measure, check that we go this right when writing down the correct DoFs.
        for rotation in correct_rotations:
            for constraint in constraints:
                assert shortest_dist_between_lines(constraint, rotation) < 1e-9

        # There should be exactly one translation equal to each correct translation.
        for correct_translation in correct_translations:
            equal_count = 0
            for translation in translations:
                if translation == approx(correct_translation):
                    equal_count += 1
            assert equal_count == 1

        # All the rotations should be through the offset point.
        for rotation in rotations:
            if len(constraints) == 0:
                assert rotation.point == approx(np.zeros(3))
            else:
                # TODO this fails for the "3 R, 2 T" case with an offset.
                assert rotation.point == approx(offset)

        # The basis of rotation directions should contain each correct rotation direction.
        rotation_direction_basis = [rotation.direction for rotation in rotations]
        for correct_rotation in correct_rotations:
            assert basis_contains_vector(rotation_direction_basis, correct_rotation.direction)
