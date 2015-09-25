"""
Contains unit tests for :mod:`field_programs`
"""
import numpy as np
import numpy.testing as np_test
import unittest
import field_programs as fp

__author__ = 'Michal Kononenko'
__library_under_test__ = 'field_programs'


class TestProgram(unittest.TestCase):
    """
    Contains set-up and teardown logic common to all unit tests for
    :class:`field_programs.Program`
    """

    @classmethod
    def setUpClass(cls):
        cls.initial_state = np.array([1, 1, 1])
        cls.program = fp.Program(initial_state=cls.initial_state)


class TestProgramConstructor(unittest.TestCase):
    """
    Contains unit tests for :meth:`field_programs.Program.__init__`
    """

    def test_constructor_default_args(self):
        program = fp.Program()
        self.assertFalse(program.children)

        expected_initial_state = np.array([0, 0, 0])
        np_test.assert_array_equal(expected_initial_state,
                                   program.initial_state)

    def test_constructor_non_default_args(self):
        child_program = fp.Program()
        initial_state = np.array([1, 1, 1])

        parent_program = fp.Program(children=(child_program,),
                                    initial_state=initial_state)

        self.assertEqual(child_program, parent_program.children[0])
        np_test.assert_array_equal(initial_state, parent_program.initial_state)


class TestProgramEndState(TestProgram):

    def test_end_state(self):
        np_test.assert_array_equal(
            self.program.end_state, self.program.initial_state)


class TestProgramAdd(TestProgram):

    def setUp(self):
        self.program_to_add = fp.Program()
        self.program_to_add.initial_state = np.array([0, 0, 0])

        self.bad_program = 'not an instance of program'
        self.assertFalse(isinstance(self.bad_program, fp.Program))

    def test_add_bad_program(self):
        with self.assertRaises(fp.UnableToAddError):
            self.program + self.bad_program

    def test_add_program(self):
        new_program = self.program + self.program_to_add

        expected_children = (self.program, self.program_to_add)

        np_test.assert_array_equal(
            new_program.initial_state, self.program.initial_state)
        self.assertEqual(self.program.end_time, self.program_to_add.start_time)

        self.assertEqual(expected_children, new_program.children)

    def tearDown(self):
        self.program = fp.Program(initial_state=self.initial_state)


class TestProgramRepr(TestProgram):

    def setUp(self):
        self.expected_repr = '%s(children=%s, initial_state=%s)' % (
            self.program.__class__.__name__, self.program.children,
            self.program.initial_state
        )

    def test_repr(self):
        self.assertEqual(self.expected_repr, self.program.__repr__())


class TestProgramStartTime(TestProgram):

    def setUp(self):
        self.start_time = 0
        self.program.time_list = np.linspace(self.start_time, 1e-7, 1000)
        self.time_to_set = 1

    def test_getter(self):
        self.assertEqual(self.start_time, self.program.start_time)

    def test_setter(self):
        self.program.start_time = self.time_to_set
        self.assertEqual(self.time_to_set, self.program.time_list[0])

