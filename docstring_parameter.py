"""
Contains a method to allow insertion of parameters into docstrings
"""

__author__ = 'Michal Kononenko'


def docstring_parameter(*args, **kwargs):
    """
    :param list-like args: A list of strings to be added to the docstring
    :param dict kwargs: A dictionary representing the strings to be added to
        the object's docstring
    :return:
    """
    def decorated_object(object_to_decorate):

        object_to_decorate.__doc__ = \
            object_to_decorate.__doc__.format(*args, **kwargs)

        return object_to_decorate
    return decorated_object



