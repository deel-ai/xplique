
"""
Collections of exceptions used in the library.
"""


class InvalidModelException(Exception):
    """Exception raised when the model is not supported by the library."""


def no_gradients_available():
    """Exception raised when no gradients are available for the specified model."""
    raise InvalidModelException("No gradients are available for the specified "
                          "model. Make sure it is a Keras or Tensorflow model.")


class InvalidOperatorException(Exception):
    """Exception raised when the operator is not supported by the library."""


def raise_invalid_operator():
    """Exception raised when the operator is not supported by the library."""
    raise InvalidOperatorException("The operator must be a function 'g(f,x,y) -> R' that take the"
                                   "model (f), the inputs (x), the label (y) and return a scalar.")
