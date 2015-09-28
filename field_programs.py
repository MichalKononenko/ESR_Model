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
list of programs, with each class:`Program` containing a reference to the

It is the responsibility of :meth:`Program.__add__` to ensure that the end
state of one program is mapped to the initial state of another,

"""
import numpy as np
__author__ = 'Michal Kononenko'


class InitializationError(TypeError):
    """
    Error that is thrown if a :class:`Program` is being constructed with
    :any:`Program.previous_program` or :any:`Program.next_program` not being
    objects of type :class:`Program`
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

    :var next_program: An object of type :class:`Program` that will serve as
        the next program in the series of ESR pulses to be applied. The end
        state of this program will be piped in as an initial state of the next
        program. If this program does not have a :any:`next_program`, the ESR
        system will be considered solved by a solver.
    :var previous_program: An object of type :class:`Program`
    :var initial_state: The initial state of the system, expressed as a row
        vector
    :var time_list: A list of floating point numbers representing the times
        for which the ESR system will be solved while in this program.

    :raises: :class:`InitializationError` if the previous_program or
        :any:`Program.next_program` to be added are not of type
        :class:`Program`
    """

    def __init__(self, next_program=None, previous_program=None,
                 initial_state=np.array([0, 0, 0]),
                 time_list=np.array([0.0, 0.0])):
        """
        Instantiates the variables described above
        """
        if not isinstance(next_program, Program) \
                and next_program is not None:
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
        """
        Returns a 3-vector representing the initial state :math:`M_0` of the
        program. If this program has a :any:`Program.previous_program`, this
        method returns the end state of the previous program instead.

        :return: A 3*1 numpy array representing the x, y, and z components
            of the initial state in the lab frame
        :rtype array-like
        """
        if self.previous_program is not None:
            return self.previous_program.end_state
        else:
            return self._initial_state

    @initial_state.setter
    def initial_state(self, initial_state):
        """
        Sets the initial state. If an initial state is assigned to a Program,
        it is assumed that this program has no previous program, and this
        program's previous program will be assigned ``None``. Assigning an
        initial state to a :class:`Program` "detaches" the :class:`Program`
        from an ESR system

        :param array-like initial_state: a 1*3 numpy array representing the
            state to which the initial state is being assigned
        """
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
        """
        Return the start time of this program, given as the first entry
        in :attribute:`Program.time_list`
        """
        return self.time_list[0]

    @start_time.setter
    def start_time(self, new_start_time):
        """
        Sets :attribute:`Program.start_time` by overwriting
        :attribute:`Program.time_list` by creating a new array using
        :function:`numpy.linspace` with the first point being the supplied
        ``new_start_time``

        :param float new_start_time: The new start time of this program
        """
        self.time_list = np.linspace(new_start_time, self.end_time,
                                     self.number_of_points)

    @property
    def end_time(self):
        """
        Returns the last entry in the time list

        :return:
        """
        return self.time_list[len(self.time_list) - 1]

    @end_time.setter
    def end_time(self, new_end_time):
        """
        Sets :attribute:`Program.end_time` by overwriting
        :attribute:`Program.time_list` with a new numpy array with the last
        point being ``new_end_time``

        :param float new_end_time: The new required end time
        """
        self.time_list = np.linspace(self.start_time, new_end_time,
                                     self.number_of_points)

    @property
    def number_of_points(self):
        """
        Returns the length of the time list
        :return:
        """
        return len(self.time_list)

    @number_of_points.setter
    def number_of_points(self, new_number_of_points):
        """
        :param int new_number_of_points: The new number of points required for
         the program
        """
        self.time_list = np.linspace(self.start_time, self.end_time,
                                     new_number_of_points)

    @property
    def magnetic_field(self):
        """ Returns the magnetic field used in this system. As this is a
        generic :class:`Program`, this method returns a matrix of zeros
        with each row being a component of the field and each column being the
        value of the field at a given point in time
        :return:
        """
        return np.zeros(len(self.time_list), len(self.initial_state))


class PulseProgram(Program):
    """
    Applies a sinusoidal pulse
    """
    def __init__(self, pulse_amplitude, pulse_frequency, pulse_length,
                 z_field_strength, number_of_points=1000, phase_angle=0,
                 offset=0):

        super(self.__class__, self).__init__()

        self.pulse_amplitude = pulse_amplitude
        self.pulse_frequency = pulse_frequency

        self.number_of_points = number_of_points

        self.pulse_length = pulse_length

        self.end_time = self.start_time + self.pulse_length

        self.phase_angle = phase_angle
        self.offset = offset

        self.z_field_strength = z_field_strength

    def magnetic_field_x_component(self, time):
        return self.pulse_amplitude * np.cos(
            2 * np.pi * self.pulse_frequency * time + self.phase_angle
        ) + self.offset

    def magnetic_field_y_component(self, time):
        return np.zeros(time.shape)

    def magnetic_field_z_component(self, time):
        return self.z_field_strength * np.ones(time.shape)

    @property
    def magnetic_field(self):
        return np.array(
            [self.magnetic_field_x_component(self.time_list),
             self.magnetic_field_y_component(self.time_list),
             self.magnetic_field_z_component(self.time_list)]
        )
