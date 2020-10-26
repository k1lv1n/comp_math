year_popul = [[1910, 92228496],
              [1920, 106021537],
              [1930, 123202624],
              [1940, 132164569],
              [1950, 151325798],
              [1960, 179323175],
              [1970, 203211926],
              [1980, 226545805],
              [1990, 248709873],
              [2000, 281421906]]
year = 1950
import numpy as np

A = np.ndarray(shape=(len(year_popul), len(year_popul)))
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        A[i, j] = (year_popul[j][0] - year) ** i
b = (0, 1, 0, 0, 0, 0, 0, 0, 0, 0)
x = np.linalg.solve(A, b)
result = 0
for i in range(len(x)):
    result += year_popul[i][1] * x[i]
print(result)
