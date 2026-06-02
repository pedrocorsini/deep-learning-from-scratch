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

def chain_length_2(chain: Chain, a: np.ndarray) -> np.ndarray:
    # Evaluates two functions in a row, in a "Chain"
    assert len(chain) == 2, "Length input 'chain' should be 2"
    f1 = chain[0]
    f2 = chain[1]
    return f2(f1(a))

def sigmoid(x: np.ndarray) -> np.ndarray:
    # Apply the sigmoid function to each element in the input ndarray
    return 1/(1+np.exp(-x))

def chain_deriv_2(chain: Chain, input_range: np.ndarray) -> np.ndarray:
    # Uses the chain rule to compute the derivative of two nested functions: 
    # (f2(f1(x)))' = f2'(f1(x))*f1'(x)

    assert len(chain) == 2, "This function requires 'Chain' objects of length 2"

    assert input_range.ndim == 1, "Function requires a 1 dimensional ndarray as input_range"

    f1 = chain[0]
    f2 = chain[1]

    # df1/dx
    f1_of_x = f1(input_range)

    # df1/du
    df1dx = deriv(f1, input_range)

    # df2/du(f1(x))
    df2du = deriv(f2, f1(input_range))

    # Multiplying these quantities together at each point
    return df1dx * df2du

# PLOT_RANGE = np.arange(-3, 3, 0.01)

# chain_1 = [np.square, sigmoid]
# chain_2 = [sigmoid, np.square]

# plot_chain(chain_1, PLOT_RANGE)
# plot_chain_deriv(chain_1, PLOT_RANGE)
# plot_chain(chain_2, PLOT_RANGE)
# plot_chain_deriv(chain_2, PLOT_RANGE)

def chain_deriv_3(chain: Chain,
                  input_range: np.ndarray) -> np.ndarray:
    '''
    Uses the chain rule to compute the derivative of three nested functions:
    (f3(f2(f1)))' = f3'(f2(f1(x))) * f2'(f1(x)) * f1'(x)
    '''
    assert len(chain) == 3, \
    "This function requires 'Chain' objects to have length 3"
    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]
    # f1(x)
    f1_of_x = f1(input_range)
    # f2(f1(x))
    f2_of_x = f2(f1_of_x)
    # df3du
    df3du = deriv(f3, f2_of_x)
    # df2du
    df2du = deriv(f2, f1_of_x)
    # df1dx
    df1dx = deriv(f1, input_range)
    # Multiplying these quantities together at each point
    return df1dx * df2du * df3du