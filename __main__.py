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


def pivoting(L, U, P, i):
    pivot, pivot_index = find_pivot(U, i)
    if pivot_index == i:
        return  # no need for interchanging rows
    for j in range(i, U.N):
        U.data[i][j], U.data[pivot_index][j] = U.data[pivot_index][j], U.data[i][j]
    for j in range(i):
        L.data[i][j], L.data[pivot_index][j] = L.data[pivot_index][j], L.data[i][j]
    for j in range(U.N):
        P.data[i][j], P.data[pivot_index][j] = P.data[pivot_index][j], P.data[i][j]


def create_LUP_matrices(matrix):
    L = Matrix(matrix.N)
    U = matrix.get_copy()
    P = Matrix(matrix.N)  # permutation matrix
    for i in range(matrix.N-1):
        pivoting(L, U, P, i)
        for j in range(i+1, matrix.N):
            L.data[j][i] = U.data[j][i]/U.data[i][i]
            for k in range(i, matrix.N):
                U.data[j][k] -= L.data[j][i]*U.data[i][k]
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


def Lagrange_method(name, path, delimiter, intervals_num):
    data_x, data_y = read_input(path, delimiter)
    intervals = make_intervals(0, len(data_x), intervals_num)
    vec_x = [data_x[i] for i in intervals]
    vec_y = [data_y[i] for i in intervals]
    interpol = []
    for i, x in enumerate(data_x):
        if intervals[-1] < i:
            break
        interpol.append(Lagrange_interpolation(vec_x, vec_y, x,
                                               len(vec_x)))
    plt.plot(data_x, data_y)
    plt.plot(data_x[:intervals[-1] + 1], interpol)
    plt.plot(vec_x, vec_y, 'o')
    tmp_str = 'Lagrange interpolation, ' + name + ', N = ' + \
              str(intervals_num)
    plt.title(tmp_str)
    filename = name.replace(" ", "_")
    plt.savefig(filename + '_Lagrange_N_'+str(intervals_num)+'.png')
    plt.show()


def spline_interpolation(vec_x, vec_y, n):
    intervals_number = len(vec_x) - 1
    size = intervals_number * 4  # due to 4 factors (a,b,c,d)
    A = Matrix(size, 0)
    b = [0 for i in range(size)]
    h = (vec_x[-1]-vec_x[0])/intervals_number  # Assuming that intervals are equal

    for i in range(0, intervals_number):
        A.data[2*i][4*i] = 1
        A.data[2*i + 1][4*i], A.data[2*i + 1][4*i + 1] = 1, h
        A.data[2*i + 1][4*i + 2], A.data[2*i + 1][4*i + 3] = h**2, h**3
        if i < intervals_number-1:
            A.data[i * 2 + intervals_number*2][4 * i + 1] = 1
            A.data[i * 2 + intervals_number*2][4 * i + 2] = 2 * h
            A.data[i * 2 + intervals_number*2][4 * i + 3] = 3 * (h ** 2)
            A.data[i * 2 + intervals_number*2][4 * i + 5] = -1
            A.data[i * 2 + intervals_number*2 + 1][4 * i + 2] = 2
            A.data[i * 2 + intervals_number*2 + 1][4 * i + 3] = 6 * h
            A.data[i * 2 + intervals_number*2 + 1][4 * i + 6] = -2
        b[2 * i], b[2 * i + 1] = vec_y[i], vec_y[i + 1]
    A.data[-1][-2], A.data[-1][-1] = 2, 6 * h
    A.data[-2][2] = 2

    lux, _, _ = LU_decomposition(A, b)  # LU decomposition with pivoting

    results_x, results_y = [], []
    for i in range(0, intervals_number):
        dx = make_intervals(vec_x[0] + i * h, vec_x[0] + (i + 1) * h, n//size)
        a, b, c, d = lux[4*i], lux[4*i+1], lux[4*i+2], lux[4*i+3]
        beg = (vec_x[0]+i*h)
        dy = [(d * (x - beg) ** 3 + c * (x - beg) ** 2 + b * (x - beg) + a) for x in dx]
        results_x += dx
        results_y += dy

    return results_x, results_y


def splines_method(name, path, delimiter, intervals_num):
    data_x, data_y = read_input(path, delimiter)
    intervals = make_intervals(0, len(data_x), intervals_num)
    vec_x = [data_x[i] for i in intervals]
    vec_y = [data_y[i] for i in intervals]
    result_x, result_y = spline_interpolation(vec_x, vec_y, intervals_num*100)
    plt.plot(data_x, data_y)
    plt.plot(result_x, result_y)
    plt.plot(vec_x, vec_y, 'o')
    tmp_str = 'Spline interpolation, ' + name + ', N = ' + str(intervals_num)
    plt.title(tmp_str)
    filename = name.replace(" ", "_")
    plt.savefig(filename + '_splines_N_' + str(intervals_num) + '.png')
    plt.show()


def main():
    dataset = [['Challenger\'s Deep', 'data/GlebiaChallengera.csv', ','],
               ['Grand Canyon', 'data/WielkiKanionKolorado.csv', ','],
               ['Mount Everest', 'data/MountEverest.csv', ','],
               ['Tczew and Starogard', 'data/tczew_starogard.txt', ' ']]

    for data in dataset:
        for i in range(10, 16, 5):  # only 10 and 15
            Lagrange_method(*data, i)
            splines_method(*data, i)
        splines_method(*data, 50)


main()

