class Matrix:
    def __init__(self, N, a1=1, a2=0, a3=0):
        # identity matrix by default, a2 and a3 are used for band matrices
        self.N = N
        self.data = [[0 for _ in range(N)] for _ in range(N)]
        for i in range(N):
            self.data[i][i] = a1
            if i + 1 < N:
                self.data[i][i + 1] = a2
                self.data[i + 1][i] = a2
            if i + 2 < N:
                self.data[i][i + 2] = a3
                self.data[i + 2][i] = a3

    def __str__(self):
        # graphic representation of a matrix, working well only with relatively small matrices
        string = ""
        for row in self.data:
            for value in row:
                string += str(round(value, 2))
                string += "\t\t"
            string += '\n'
        return string

    def __add__(self, other):
        for i in range(self.N):
            for j in range(self.N):
                self.data[i][j] += other.data[i][j]

    def __sub__(self, other):
        for i in range(self.N):
            for j in range(self.N):
                self.data[i][j] -= other.data[i][j]

    def __mul__(self, vector):  # only matrix times vector
        vtr = [0 for _ in range(self.N)]  # vector to return
        for i in range(self.N):
            for j in range(self.N):
                vtr[i] += self.data[i][j] * vector[j]
        return vtr

    def get_copy(self):  # returns deepcopy of a matrix
        copy = Matrix(self.N)
        for i in range(self.N):
            for j in range(self.N):
                copy.data[i][j] = self.data[i][j]
        return copy
