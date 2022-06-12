import math
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


def LU_decomposition(A, b):
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


def Lagrange_interpolation(n):
    pass


def main():
    a1 = 3
    a2 = a3 = -1
    f = 4
    N = 965
    A = Matrix(N, a1, a2, a3)
    b = [math.sin(i * (f + 1)) for i in range(N)]
    x, y, norm_res = LU_decomposition(A, b)
    print(norm_res)
    pass


main()
