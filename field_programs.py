"""
Contains utilities for creating magnetic fields in a more programmatic
fashion by combining sets of magnetic field regimes. This module
allows combination of magnetic fields into large programs, propagated in
time via analytic or numeric methods.

This module is tested in :mod:`test_field_programs`

**Example Usage**

Say that an ESR pulse consists of applying a sinusoidal pulse for 200 ms
at an amplitude of 0.1, followed by a waiting period of 300 ms where only
a static field is applied, and then a 400 ms pulse would be applied at an
amplitude of 0.1. In order to solve this problem, three programs will need
to be created, possibly as follows

.. code-block:: python

    from esr.field_programs import PulseProgram, WaitProgram

    pi_over_2_pulse = PulseProgram(amplitude=0.1, frequency=1e9,
        start_time=0, end_time=200e-3)
    wait_period = WaitProgram(start_time=200e-3, end_time=500e-3)
    pi_pulse = PulseProgram(amplitude=0.1, frequency=1e9,
        start_time=500e-3, end_time=900e-3)

    program = pi_over_2_pulse + wait_period + pi_pulse

The ``+`` operator is overloaded such that each program is concatenated into a
single pulse. The variable ``program`` will store this internally as a linked
list of programs, with each ``class:Program`` containing a reference to the

It is the responsibility of :meth:`Program.__add__` to ensure that the end
state of one program is mapped to the initial state of another,

"""
import numpy as np
__author__ = 'Michal Kononenko'


class InitializationError(TypeError):
    """
    Error that is thrown if a :class:`Program` is being constructed with
    :any:`previous_program` or :any:`next_program` not being objects of type
    :class:`Program`
    """
    pass


class UnableToAddError(TypeError):
    """
    Error that is thrown if an object of type :class:`Program` is being added
    to another object that is not a :class:`Program`
    """
    pass


class Program(object):
    """
    Base class for all magnetic fields and solutions for the ESR system as
    a function of time.

    :var children: Objects of type Program that form the ESR system to be
        solved
    :var initial_state: The initial state of the system, expressed as a row
        vector
    :var time_list: A list of floating point numbers representing the times
        for which the ESR system will be solved while in this program.
    """

    def __init__(self, next_program=None, previous_program=None,
                 initial_state=np.array([0, 0, 0]),
                 time_list=np.array([0.0, 0.0])):
        """
        Instantiates the variables described above
        """
        if not isinstance(next_program, Program) and next_program is not None:
            raise InitializationError(
                'The program %s is not a valid next for this program',
                next_program)
        else:
            self.next_program = next_program

        if not isinstance(previous_program, Program) \
                and previous_program is not None:
            raise InitializationError(
                'The program %s is not a valid previous program for this'
                'program', previous_program
            )
        else:
            self.previous_program = previous_program

        self._initial_state = initial_state
        self.time_list = time_list

    @property
    def initial_state(self):
        if self.previous_program is not None:
            return self.previous_program.end_state
        else:
            return self._initial_state

    @initial_state.setter
    def initial_state(self, initial_state):
        self.previous_program = None
        self._initial_state = initial_state

    @property
    def end_state(self):
        """
        After the propagator is complete with solving the system, return the
        state at the end of the program. This property is used when combining
        programs to pipe the end state of one program to the initial state of
        another.

        In the generic program, since no changes are performed, the returned
        state is the initial state of this program

        :return: The final state of the solved program. In this case, this is
            the initial state.
        """
        return self.initial_state

    def __add__(self, other):
        if not isinstance(other, Program):
            raise UnableToAddError('Cannot add Program %s to object %s',
                                   self, other)
        self.next_program = other
        other.previous_program = self

        return self

    def __repr__(self):
        return '%s(previous_program=%s, next_program=%s initial_state=%s)' % (
            self.__class__.__name__, self.previous_program,
            self.next_program, self.initial_state
        )

    @property
    def start_time(self):
        return self.time_list[0]

    @start_time.setter
    def start_time(self, new_start_time):
        self.time_list = np.linspace(new_start_time, self.end_time,
                                     self.number_of_points)

    @property
    def end_time(self):
        return self.time_list[len(self.time_list) - 1]

    @end_time.setter
    def end_time(self, new_end_time):
        self.time_list = np.linspace(self.start_time, new_end_time,
                                     self.number_of_points)

    @property
    def number_of_points(self):
        return len(self.time_list)

    @number_of_points.setter
    def number_of_points(self, new_number_of_points):
        self.time_list = np.linspace(self.start_time, self.end_time,
                                     new_number_of_points)

    @property
    def magnetic_field(self):
        return np.zeros(len(self.time_list), len(self.initial_state))