import numpy as np
from typing import Callable
from typing import List

# A Function takes in an ndarray as an argument and produces an ndarray
Array_Function = Callable[[np.ndarray], np.ndarray]

# A Chain is a list of functions
Chain = List[Array_Function]

def deriv(func: Callable[[np.ndarray], np.ndarray], input_: np.ndarray, delta: float = 0.001) -> np.ndarray:
    # Evaluates the derivative of a function "func" at every element in the "input_" array.
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)

def multiple_inputs_add(x: np.ndarray, y: np.ndarray, sigma: Array_Function): # -> float:
    # Function with multiple inputs and addition, forward pass.
    assert x.shape == y.shape
    a = x + y
    return sigma(a)

def multiple_inputs_add_backward(x: np.ndarray, y: np.ndarray, sigma: Array_Function): # -> float:
    # Computes the derivative of this simple function with respect to both inputs

    # Compute "forward pass"
    a = x + y

    # Compute derivatives
    dsda = deriv(sigma, a)

    dadx, dady = 1, 1

    return dsda * dadx, dsda * dady

def matmul_forward(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    # Computs the forward pass of a matrix mutiplication
    assert X.shape[1] == W.shape[0], '''For matrix mutiplicatoin, the number of columns in
    the first array should match the number of rows in the second; instead the number of
    columns in the first array is {0} and the number of rows in the second array is {1}
    '''.format(X.shape[1], W.shape[0])

    # Matrix multiplication
    N = np.dot(X, W)

    return N

def matmul_backward_first(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    # Computes the backward pass of a matrix multiplication with respect to the first argument

    # Backward pass
    dNdX = np.transpose(W, (1, 0))

    return dNdX
