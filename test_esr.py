import mock
import numpy as np
import unittest
import esr

__author__ = 'Michal Kononenko'


class TestMagneticField(unittest.TestCase):

    def setUp(self):
        self.field_functions = [
            lambda t: np.sin(10*t), lambda t: np.cos(10*t), lambda t: 20
        ]
        self.field = esr.MagneticField(self.field_functions)

    def test_constructor_with_custom_functions(self):
        field = esr.MagneticField(self.field_functions)
        self.assertEqual(field.field_functions, self.field_functions)

    def test_call(self):
        time = 1
        expected_output = np.array([f(time) for f in self.field_functions])
        np.testing.assert_array_equal(expected_output, self.field(time))

    def test_repr(self):
        expected_repr = '%s(%s)' % (self.field.__class__.__name__, self.field_functions)
        self.assertEqual(expected_repr, self.field.__repr__())


class TestBlochSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.gyromagnetic_ratio = 20e7
        cls.time_list = np.linspace(0, 1e-7, 1000)

        cls.field = esr.MagneticField()
        cls.bloch = esr.BlochSystem(magnetic_field=cls.field,
                                    gyromagnetic_ratio=cls.gyromagnetic_ratio,
                                    time_list=cls.time_list)


class TestIsSolvable(TestBlochSystem):

    def setUp(self):
        self.bloch.gyromagnetic_ratio = self.gyromagnetic_ratio
        self.bloch.time_list = self.time_list
        self.bloch.magnetic_field = self.field

        self.assertTrue(self.bloch.time_list is not None)
        self.assertTrue(self.bloch.gyromagnetic_ratio is not None)
        self.assertTrue(self.bloch.magnetic_field is not None)

    def test_is_solvable_true(self):

        self.assertTrue(self.bloch.is_solvable)

    def test_is_solvable_false(self):
        self.bloch.magnetic_field = None
        self.assertFalse(self.bloch.is_solvable)


class TestGetBlochMatrix(TestBlochSystem):

    def setUp(self):
        self.time_as_list = list(self.time_list)
        self.assertTrue(self.bloch.is_solvable)

    def test_call_with_unsolvable_system(self):
        self.bloch.magnetic_field = None

        self.assertFalse(self.bloch.is_solvable)

        with self.assertRaises(esr.UnableToSolveSystemError):
            self.bloch.get_bloch_matrix(self.time_list)

    def test_is_time_list(self):
        self.assertIsNotNone(
            self.bloch.get_bloch_matrix(self.time_as_list))

    def tearDown(self):
        self.bloch.magnetic_field = self.field
        self.bloch.gyromagnetic_ratio = self.gyromagnetic_ratio
        self.bloch.time_list = self.time_list


class TestSolveBlochSystem(TestBlochSystem):

    def setUp(self):
        self.bloch.initial_state = np.array([1e-3, 0, 0])
        self.assertTrue(self.bloch.is_solvable)

    def test_unable_to_solve(self):
        self.bloch.magnetic_field = None
        self.assertFalse(self.bloch.is_solvable)

        with self.assertRaises(esr.UnableToSolveSystemError):
            self.bloch.solve()

    def test_solve(self):
        solution = self.bloch.solve()

        target_shape = (len(self.bloch.initial_state), len(self.time_list))

        self.assertEqual(target_shape, solution.shape)

    def tearDown(self):
        self.bloch.magnetic_field = self.field
        self.bloch.gyromagnetic_ratio = self.gyromagnetic_ratio
        self.bloch.time_list = self.time_list

        self.assertTrue(self.bloch.is_solvable)


class TestGetLarmorFrequency(TestBlochSystem):

    def setUp(self):
        self.time = 1

    def test_get_larmor_freq(self):
        expected_freq = self.gyromagnetic_ratio * self.field(self.time)[2] / \
                        (2 * np.pi)
        self.assertEqual(
            expected_freq, self.bloch.get_larmor_frequency(self.time))


class TestSignalAnalyzer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mock_bloch_system = mock.MagicMock()
        cls.mock_bloch_system.is_solvable = False

    def test_constructor_bad_bloch(self):
        with self.assertRaises(esr.BadBlochSystemError):
            esr.SignalAnalyzer(self.mock_bloch_system)

    def test_constructor_good_bloch(self):
        self.mock_bloch_system.is_solvable = True
        sig = esr.SignalAnalyzer(self.mock_bloch_system)
        self.assertEqual(sig.bloch_system, self.mock_bloch_system)


class TestOscillatingMagneticField(unittest.TestCase):

    def setUp(self):
        self.time_list = np.linspace(0, 1e-7, 1000)

        self.frequency = 20e7
        self.x_axis_pulse_amplitude = 1
        self.z_axis_field_strength = 1
        self.pulse_start_time = 0
        self.pulse_end_time = 1e-8

        self.magnetic_field = esr.OscillatingMagneticField(
            self.frequency, self.x_axis_pulse_amplitude,
            self.z_axis_field_strength,
            self.pulse_start_time, self.pulse_end_time
        )

    def test_field_functions(self):
        field_functions = self.magnetic_field(self.time_list)
        self.assertEqual(field_functions.shape, (3, 1000))
