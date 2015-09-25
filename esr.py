r""" Contains the model for an ESR spin system obeying the Bloch equations
    as well as utilities for analyzing the resulting signal.
"""
import numpy as np
from scipy.signal.waveforms import square as square_wave

__author__ = 'Michal Kononenko'


class UnableToSolveSystemError(Exception):
    """
    Exception class thrown when :class:`BlochSystem` is unable to solve
    the Bloch Equations for the given set of parameters
    """
    pass


class BadBlochSystemError(Exception):
    """
    Thrown when SignalAnalyzer is created with an unsolvable Bloch System
    """
    pass


class MagneticField(object):
    r"""
    Models the magnetization vector :math:`\mathbf{B} = (B_x, B_y, B_z)`
    perturbing the system for which a solution is to be found

    :arg perturbation_functions: A list of three functions of x, y, and z,
        that give the value of the magnetic field in each spatial dimension.
        Each function must take in a floating-point time value as an argument
        and return the strength of the magnetic field in Tesla. Defaults to
        :math:`\mathbf{B} = ( \cos(t), \sin(t), 1)`
    """
    def __init__(self, field_functions=None):
        """
        Instantiates the variables listed above
        """
        if field_functions is None:
            self.field_functions = [
                lambda t: np.cos(t), lambda t: np.sin(t), lambda t: 1
            ]
        else:
            self.field_functions = field_functions

    def __call__(self, time):
        """ Evaluates the magnetic field at a given time t

        :param float time: The time value for which the field is to be
        evaluated
        :return: A list of the magnetic field components
        """
        return np.array([f(time) for f in self.field_functions])

    def __setitem__(self, key, value):
        self.field_functions[key] = value

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__,
                           self.field_functions)


class BlochSystem(object):
    """
    Contains all constants and problem parameters for the Bloch equations
    """
    def __init__(self, magnetic_field=None,
                 time_list=None,
                 initial_state=np.array([0, 0, 0]),
                 gyromagnetic_ratio=None,
                 t1=np.inf, t2=np.inf,
                 equilibrium_field=np.array([0, 0, 0])
                 ):
        self.magnetic_field = magnetic_field
        self.time_list = time_list
        self.initial_state = initial_state
        self.gyromagnetic_ratio = gyromagnetic_ratio
        self.t1 = t1
        self.t2 = t2
        self.equilibrium_field = equilibrium_field

    def __repr__(self):
        return '%s(magnetic_field=%s)' % \
               (self.__class__.__name__, self.magnetic_field)

    @property
    def is_solvable(self):
        """
        Checks that a time list, gyromagnetic ratio, and a magnetic field
        are provided

        :return: boolean stating whether system is solvable or not
        """
        conditions = [
            self.time_list is not None,
            self.gyromagnetic_ratio is not None,
            self.magnetic_field is not None
        ]
        return all(conditions)

    def get_bloch_matrix(self, time):
        r"""
        Returns a 3 * 3 matrix corresponding to the factor by which the
        magentization vector is multiplied. This matrix corresponds to

        .. math::
            \left( \begin{array}{c c c}
                -\frac{1}{T_2} & \gamma B_z & -\gamma B_y \\
                - \gamma B_z & - \frac{1}{T_2} & \gamma B_x \\
                \gamma B_y & - \gamma B_x & -\frac{1}{T_1}
            \end{array} \right)

        :param array-like time: A list of times for which the matrix needs to
            be calculated
        :return: The matrix described above

        """
        if not self.is_solvable:
            raise UnableToSolveSystemError('Unable to solve system')

        if not isinstance(time, np.ndarray):
            time = np.array(time)

        t1 = self.t1*np.ones(time.shape)
        t2 = self.t2*np.ones(time.shape)

        gamma = self.gyromagnetic_ratio
        [bx, by, bz] = self.magnetic_field(time)

        return np.array(
            [
                [-1/t2, gamma * bz, -gamma*by],
                [-gamma*bz, -1/t2, gamma*bx],
                [gamma*by, -gamma*bx, -1/t1]
            ]
        )

    def solve(self):
        """
        Solves the Bloch equation system by propagating the solution via RK4

        :return: A numpy array with each row representing the respective
            x, y, and z components, and each column mapping to a point
            in the supplied time list
        """
        if not self.is_solvable:
            raise UnableToSolveSystemError('Cannot solve system')

        solution = np.zeros([len(self.initial_state), len(self.time_list)])

        solution[:, 0] = self.initial_state

        for index in range(0, len(self.time_list) - 1):
            solution[:, index + 1] = self._propagate_runge_kutta(
                solution, self.time_list, index)

        return solution

    def _propagate_runge_kutta(self, solution, time_list, index):
        """ Propagates the solution forward by one time step

        :param array-like solution: A partial solution to the problem that
            needs to be propagated forward in time, with each row corresponding
            to a component of the magnetization vector, and each column
            representing a point in time
        :param array-like time_list: The time list of the solution
        :param int index: The index of the column for which the solution needs
            to be propagated
        :return: The solution to the problem with the entry at :var:`index`
            filled in with the solution
        """
        if index >= len(time_list):
            raise UnableToSolveSystemError('Index exceeds time list')

        time = time_list[index]
        delta_t = self.time_list[index + 1] - self.time_list[index]

        mag_vector = solution[:, index]

        k1 = self._calculate_derivative(time, mag_vector)
        k2 = self._calculate_derivative(
            time + delta_t/2, mag_vector + delta_t/2 * k1)
        k3 = self._calculate_derivative(
            time + delta_t/2, mag_vector + delta_t/2 * k2
        )
        k4 = self._calculate_derivative(
            time + delta_t, mag_vector + delta_t * k3
        )
        return mag_vector + (delta_t / 6) * (k1 + 2*k2 + 2*k3 + k4)

    def _calculate_derivative(self, time, mag_vector):
        """
        Since the RK4 integral solves a problem of the form
        .. math::
            \frac{\partial y}{\partial t} = f(t, y)

        This method is syntactic sugar for implementing the RK4 algorithm, and
        calculates the derivative of the magnetization vector for a given time
        :math:`t` and a given magnetization vector :math:`y`

        :param float time: The time value for which the derivative is being
            calculated
        :param array-like mag_vector: The magnetization vector for which the
            derivative is to be calculated
        :return array-like: A vector of the same shape as the input
            magnetization vector for which the derivative is to be calculated
        """
        bloch_matrix = self.get_bloch_matrix(time)
        return np.dot(mag_vector, bloch_matrix) + \
            self._static_field_component

    @property
    def _static_field_component(self):
        r""" Returns the component of the Bloch equation system not multiplied
        by the magnetization vector. This component is

        .. math::
            \frac{1}{T_1} \left(\begin{array}{c}
                B_{x0} \\ B_{y0} \\ B_{z0}
            \end{array} \right)

        :return: The vector described above
        """
        return self.equilibrium_field / self.t1

    def get_larmor_frequency(self, time):
        r""" Returns the Larmor frequency as a function of time. The Larmor
        frequency is calculated using

        .. math::
            f_L(t) = \frac{\gamma B_z(t)}{2 \pi}

        :param float time: The time for which the
        :return array-like: The Larmor frequency as a function of time for the
            system
        """
        bz = self.magnetic_field(time)[2]
        return self.gyromagnetic_ratio * bz / (2 * np.pi)

    def get_steady_state_field(self):
        """
        Returns a set of three functions defining the steady state field in the
        x, y, and z directions respectively, with the frequency of the applied
        RF pulses matching the Larmor frequency of the system

        :return:
        """
        def omega_l(time):
            """ Shorthand for Larmor frequency
            :param array-like time: The times for which the Larmor frequency
             needs to be calculated
            :return: An array of the same size as :var:`time` with a given
            Larmor frequency
            """
            return self.get_larmor_frequency(time)
        t1 = self.t1
        t2 = self.t2

        mx, my, mz = self._static_field_component * self.t1

        def factor_x(time):
            """ Function defining the steady-state solution for the Bloch
            System in the x component of the lab frame
            :param array-like time: The time for which this factor is to be
                calculated
            :return: An array giving the required x component of the magnetic
                field
            """
            return (omega_l(time) * t2**2) / (1 + omega_l(time)**2 * t1 ** t2)

        def factor_y(time):
            """ Function defining steady-state solution for :math:`B_y(t)`
            :param array-like time: The required time
            :return: An array giving :math:`B_y`
            """
            return (omega_l(time) * t2) / (1 + omega_l(time)**2 * t1 * t2)

        def factor_z(time):
            """ Function defining the steady-state solution for :math:`B_z(t)`
            :param array-like time: The required time
            :return: An array giving :math:`B_z`
            """
            return (1 + t2 ** 2)/(1 + omega_l(time**2*t1*t2))

        return [
            lambda time: mx * factor_x(time) *
            np.cos(2 * np.pi * omega_l(time) * time),
            lambda time: my * factor_y(time) *
            np.sin(2 * np.pi * omega_l(time) * time),
            lambda time: mz * factor_z(time)
        ]


class SignalAnalyzer(object):
    """ Contains methods to analyze the return signal given a field and a
    Bloch system
    """
    def __init__(self, bloch_system):
        if not bloch_system.is_solvable:
            raise BadBlochSystemError('Cannot solve Bloch System')
        self.bloch_system = bloch_system

    @property
    def time_list(self):
        """ Returns the time list to be analyzed by this Analyzer
        :return array-like: The times for which this system was solved
        """
        return self.bloch_system.time_list

    @time_list.setter
    def time_list(self, time_list):
        """
        Allows setting of the time list from the Analyzer instead of the Bloch
        system
        :param array-like time_list: The new time list for which the system
            is to be solved
        :return:
        """
        self.bloch_system.time_list = time_list

    @property
    def solution(self):
        """ Returns the solution of the system defined by this analyzer's
        Bloch System in the time domain
        :return array-like: an array describing the solution of the system
        """
        return self.bloch_system.solve()

    @property
    def spectrum(self):
        """ Returns the solution of the system defined by this analyzer's
        Bloch System in the frequency domain
        :return array-like: an array describing the solution of the system
        """
        return np.fft.fft(self.solution)


class OscillatingMagneticField(MagneticField):

    def __init__(self, frequency, amplitude, base_field_strength,
                 start_time=None, end_time=None):

        self._field_functions = [
            lambda t: np.multiply((amplitude * np.cos(2 * np.pi * frequency * t)),
                             np.multiply(start_time <= t, t <= end_time)),
            lambda t: np.zeros(t.shape),
            lambda t: base_field_strength * np.ones(t.shape)
        ]

        super(self.__class__, self).__init__(
            field_functions=self._field_functions)


class PulsedMagneticField(MagneticField):
    def __init__(self, pulse_amplitude=0,
                 pulse_length=1,
                 off_pulse_length=1,
                 base_field_strength=0,
                 frequency=0):
        period = pulse_length + off_pulse_length
        duty_cycle = pulse_length / period

        def _square_component(time_list):
            return 0.5 * pulse_amplitude * square_wave(np.pi/period * time_list, duty=duty_cycle)

        def _sine_component(time_list):
            return np.cos(2 * np.pi * frequency * time_list)

        field_functions = [
            lambda t: np.multiply(_square_component(t), _sine_component(t)),
            lambda t: np.zeros(t.shape),
            lambda t: base_field_strength * np.ones(t.shape)
        ]
        super(self.__class__, self).__init__(field_functions)
