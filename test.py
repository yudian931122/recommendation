import numpy as np

a = np.matrix([[1, 2, 3], [2, 3, 4], [1, 3, 5]])

norm_a = np.linalg.norm(a, axis=1).reshape(3, -1)

print(norm_a)

print(norm_a @ norm_a.T)

print((a @ a.T) / (norm_a @ norm_a.T))
