import numpy as np
from typing import Callable

def deriv(func: Callable[[np.ndarray], np.ndarray], input_: np.ndarray, delta: float = 0.001) -> np.ndarray:
    # Evaluates the derivative of a function "func" at every element in the "input_" array.
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)

f = lambda x: x**2

x = np.array([1.0, 2.0, 3.0])
print(deriv(f, x))
