import csv
import math
import matplotlib.pyplot as plt
from matrix import Matrix


def find_pivot(U, i):
    pivot = abs(U.data[i][i])
    pivot_index = i
    for j in range(i+1, U.N):
        if abs(U.data[j][i] > pivot):
            pivot = abs(U.data[j][i])
            pivot_index = j
    return pivot, pivot_index


def swap_elements(matrix, i, j, pivot_index):
    tmp = matrix.data[i][j]
    matrix.data[i][j] = matrix.data[pivot_index][j]
    matrix.data[pivot_index][j] = tmp


def pivoting(L, U, P, i):
    pivot, pivot_index = find_pivot(U, i)
    if pivot_index == i:
        return
    for j in range(U.N):
        if j < i:
            swap_elements(L, i, j, pivot_index)
        else:
            swap_elements(U, i, j, pivot_index)
        swap_elements(P, i, j, pivot_index)


def create_LUP_matrices(matrix):
    L = Matrix(matrix.N)
    U = matrix.get_copy()
    P = Matrix(matrix.N)  # permutation matrix
    for i in range(matrix.N-1):
        if U.data[i][i] < 10**-3:
            pivoting(L, U, P, i)
        for j in range(i+1, matrix.N):
            L.data[j][i] = U.data[j][i]/U.data[i][i]
            for k in range(i, matrix.N):
                U.data[j][k] = U.data[j][k] - L.data[j][i]*U.data[i][k]
    return L, U, P


def norm(vec):
    N = len(vec)
    n = 0  # norm
    for i in range(N):
        n += vec[i]**2
    return math.sqrt(n)


def residual(matrix, vec_x, vec_b):
    N = len(vec_x)
    product = matrix*vec_x
    res = [(product[i] - vec_b[i]) for i in range(N)]
    return res


def LU_decomposition(A, b):  # with pivoting
    x = [0 for _ in range(A.N)]
    y = [0 for _ in range(A.N)]
    L, U, P = create_LUP_matrices(A)
    b = P*b
    for i in range(A.N):
        series_sum = 0
        for j in range(i):
            series_sum += L.data[i][j]*y[j]
        y[i] = (b[i] - series_sum)
    for i in range(A.N-1, -1, -1):
        series_sum = 0
        for j in range(i+1, A.N):
            series_sum += U.data[i][j] * x[j]
        x[i] = (y[i] - series_sum)/U.data[i][i]
    res = residual(A, x, b)
    return x, y, norm(res)


def make_intervals(beg, end, n):
    vec = []  # vector of indexes of intervals
    x = beg
    dx = (end-beg)/(n-1)
    for i in range(n):
        m = min(round(x), end-1)  # avoiding oor exception
        vec.append(m)
        x += dx
    return vec


def read_input(path, delimiter):
    with open(path, 'r') as f:
        vec_x, vec_y = [], []
        cr = csv.reader(f, delimiter=delimiter)
        for x, y in cr:
            try:
                vec_x.append(float(x))
                vec_y.append(float(y))
            except ValueError as error:  # labels etc
                continue
    return vec_x, vec_y


def phi(vec_x, x, i, n):
    product = 1.0
    for j in range(n):
        if j == i:
            continue  # avoid diving by zero
        product *= (x-vec_x[j])/(vec_x[i]-vec_x[j])
    return product


def Lagrange_interpolation(vec_x, vec_y, x, n):
    series_sum = 0
    for i in range(n):
        series_sum += vec_y[i]*phi(vec_x, x, i, n)
    return series_sum


def Lagrange_part(name, path, delimiter, intervals_num):
    data_x, data_y = read_input(path, delimiter)
    intervals = make_intervals(0, len(data_x), intervals_num)
    vec_x = [data_x[i] for i in intervals]
    vec_y = [data_y[i] for i in intervals]
    interpol = []
    for i, x in enumerate(data_x):
        if intervals[-1] < i:
            break
        interpol.append(Lagrange_interpolation(vec_x, vec_y, x, len(vec_x)))
    plt.plot(data_x, data_y)
    plt.plot(vec_x, vec_y, 'o')
    plt.plot(data_x[:intervals[-1] + 1], interpol)
    tmp_str = 'Lagrange polynomial, ' + name + ', N = ' + str(intervals_num)
    plt.title(tmp_str)
    plt.show()


def main():
    Lagrange_part('Mount Everest', 'data/MountEverest.csv', ',', 15)


main()
