import numpy as np
import unittest
import esr
import matplotlib.pyplot as plt

__author__ = 'Michal Kononenko'


class TestBlochSystem(unittest.TestCase):
    def setUp(self):
        self.field = esr.MagneticField()
        self.field.field_functions = [
            lambda t: 0, lambda t: 0, lambda t: 20 + np.cos(t)
        ]

        self.gyromagnetic_ratio = 20e7
        self.time_list = np.linspace(0, 1e-8, 10000)
        self.initial_state = np.array([1e-3, 0, 0])

        self.system = esr.BlochSystem()
        self.system.magnetic_field = self.field
        self.system.gyromagnetic_ratio = self.gyromagnetic_ratio
        self.system.time_list = self.time_list
        self.system.initial_state = self.initial_state
        self.system.t1 = 5e-9
        self.system.t2 = 5e-9

    def test_is_solvable(self):
        self.assertTrue(self.system.is_solvable)

    def test_solution(self):
        solution = self.system.solve()
        self.assertIsInstance(solution, np.ndarray)
        plt.plot(solution.transpose())
        plt.show()


class TestEnvironmentMagneticField(unittest.TestCase):
    def setUp(self):
        self.perturbation_functions = [
            lambda t: t**2, lambda t: 2*t, lambda t: np.sin(t)
        ]
        self.time = np.pi

    def test_constructor_no_arg(self):
        expected_result = [
            np.cos(self.time), np.sin(self.time), 1
        ]
        field = esr.MagneticField()
        self.assertAlmostEqual(expected_result, field(self.time), 6)

    def test_constructor_with_arg(self):
        expected_result = [f(self.time) for f in self.perturbation_functions]
        field = esr.MagneticField(self.perturbation_functions)
        self.assertAlmostEqual(expected_result, field(self.time), 6)

    def test_repr(self):
        field = esr.MagneticField(self.perturbation_functions)
        self.assertEqual(
            field.__repr__(),
            '%s(%s)' % (field.__class__.__name__, field.field_functions)
        )

