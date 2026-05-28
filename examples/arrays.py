import numpy as np

print('Python list operations:')
a = [1,2,3]
b = [4,5,6]
print('a+b:', a+b)
try:
    print(a*b)
except TypeError:
    print('a*b has no meaning for Python lists')
print()
print('Numpy array operations:')
a = np.array([1,2,3])
b = np.array([4,5,6])
print('a+b:', a+b)
print('a*b:', a*b)

print()
a = np.array([[1,2],[3,4]])
print('a:')
print(a)
print('a.sum(axis=0):', a.sum(axis=0))
print('a.sum(axis=1:)', a.sum(axis=1))
print()

a = np.array([[1,2,3],
              [4,5,6]])
b = np.array([10,20,30])
print('a+b:\n', a+b)

def square(x: np.ndarray) -> np.ndarray:
    # Square each element in the input ndarray
    return np.power(x, 2)
    
def leaky_relu(x: np.ndarray) -> np.ndarray:
    # Apply 'Leaky ReLU' function to each element in ndarray
    return np.maximum(0.2*x, x)

print()
print(square(10))
print(leaky_relu(10))
