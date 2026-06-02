import numpy as np
from typing import Callable
from typing import List

# A Function takes in an ndarray as an argument and produces an ndarray
Array_Function = Callable[[np.ndarray], np.ndarray]

# A Chain is a list of functions
Chain = List[Array_Function]

def deriv(func: Callable[[np.ndarray], np.ndarray], 
          input_: np.ndarray, 
          delta: float = 0.001) -> np.ndarray:
    # Evaluates the derivative of a function "func" at every element in the "input_" array.
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)

def sigmoid(x: np.ndarray) -> np.ndarray:
    # Apply the sigmoid function to each element in the input ndarray
    return 1/(1+np.exp(-x))

def matrix_function_forward_sum(X: np.ndarray, 
                                W: np.ndarray, 
                                sigma: Array_Function) -> float:
    '''
    Computing the result of the forward pass of this function with
    input ndarrays X and W and function sigma.
    '''
    assert X.shape[1] == W.shape[0]

    # matrix multiplication
    N = np.dot(X, W)

    # feeding the output of the matrix multiplication through sigma
    S = sigma(N)

    # sum all the elements
    L = np.sum(S)
    
    return L

def matrix_function_backward_sum_1(X: np.ndarray,
                                   W: np.ndarray,
                                   sigma: Array_Function) -> np.ndarray:
    '''
    Compute derivative of matrix function with a sum with respect to the
    first matrix input.
    '''
    assert X.shape[1] == W.shape[0]
    # matrix multiplication
    N = np.dot(X, W)
    # feeding the output of the matrix multiplication through sigma
    S = sigma(N)
    # sum all the elements
    L = np.sum(S)

    # note: I'll refer to the derivatives by their quantities here,
    # unlike the math, where we referred to their function names
    # dLdS - just 1s
    dLdS = np.ones_like(S)
    # dSdN
    dSdN = deriv(sigma, N)
    # dLdN
    dLdN = dLdS * dSdN
    # dNdX
    dNdX = np.transpose(W, (1, 0))
    # dLdX
    dLdX = np.dot(dSdN, dNdX)
    return dLdX

np.random.seed(190204)
X = np.random.randn(3, 3)
W = np.random.randn(3, 2)
print("X:")
print(X)
print("L:")
print(round(matrix_function_forward_sum(X, W, sigmoid), 4))
print()
print("dLdX:")
print(matrix_function_backward_sum_1(X, W , sigmoid))

X1 = X.copy()
X1[0, 0] += 0.001
print(round(
        (matrix_function_forward_sum(X1, W, sigmoid) - \
         matrix_function_forward_sum(X, W, sigmoid)) / 0.001, 4))