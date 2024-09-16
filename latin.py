import numpy as np
import matplotlib.pyplot as plt
import math
import sympy as sp
from IPython.display import display, Math, Latex, HTML
from matching_problem import *

class LatinRectangle():
    def __init__(self, L, N=None):
        if isinstance(L, list) and len(L) > 0 and \
           all([(isinstance(x, list) and len(x) == len(L[0])) for x in L]):
            P = list(range(len(L)))
            Q = list(range(len(L[0])))
            Lt = {(i,j):L[i][j] for i in range(len(L)) for j in range(len(L[0]))}
        elif isinstance(L, np.ndarray) and L.ndim == 2:
            P = list(range(L.shape[0]))
            Q = list(range(L.shape[1]))
            Lt = {(i,j):L[i,j] for i in range(len(L)) for j in range(len(L[0]))}
        elif isinstance(L, dict):
            PQ = list(L.keys())
            if not(all([isinstance(x, tuple) and len(x) == 2 for x in PQ])):
                raise ValueError('Keys must be tuples of length 2')
            P = list(set([x[0] for x in PQ]))
            Q = list(set([x[1] for x in PQ]))
            PQ1 = sorted([(i,j) for i in P for j in Q])
            if sorted(PQ) != PQ1:
                raise ValueError('Keys must be all possible combinations of P and Q')
            Lt = L
        else:
            raise ValueError('Input must be a list of lists or a dictionary')
        N0 = sorted(list(set(Lt.values())))
        if N is not None:
            if isinstance(N, int) and N > 0:
                N = list(range(N))
            if not all([n0 in N for n0 in N0]):
                raise ValueError('N must contain all the values in L')
            N0 = sorted(N)
        self.P = P
        self.Q = Q
        self.N = N0
        self.p = len(self.P)
        self.q = len(self.Q)
        self.n = len(self.N)
        self.num_rows = self.p
        self.num_cols = self.q
        self.num_values = self.n
        self.is_square = (self.p == self.n and self.q == self.n)
        self.L = Lt
        self.rows = [[Lt[(i,j)] for j in Q] for i in P]
        self.cols = [[Lt[(i,j)] for i in P] for j in Q]
        self.row_dict = {i : [Lt[(i,j)] for j in Q] for i in P}
        self.col_dict = {j : [Lt[(i,j)] for i in P] for j in Q}
        self.is_reduced = (self.rows[0] == list(range(self.n)) and self.cols[0] == list(range(self.n)))
        for i in range(self.p):
            for j1 in range(self.q):
                for j0 in range(j1):
                    if self.rows[i][j0] == self.rows[i][j1]:
                        raise ValueError(f'Repeated value at ({i},{j0}) and ({i},{j1})')
        for j in range(self.q):
            for i1 in range(self.p):
                for i0 in range(i1):
                    if self.cols[j][i0] == self.cols[j][i1]:
                        raise ValueError(f'Repeated value at ({i0},{j}) and ({i1},{j})')
        self.multiplicity = {k : 0 for k in self.N}
        for i in range(self.p):
            for j in range(self.q):
                self.multiplicity[self.rows[i][j]] += 1
        self.excess = {}
        self.deficient_values = []
        self.critical_values = []
        self.is_fully_extendable = True
        for k in self.N:
            self.excess[k] = self.multiplicity[k] - self.p - self.q + self.n
            if self.excess[k] < 0:
                self.deficient_values.append(k)
                self.is_fully_extendable = False
            elif self.excess[k] == 0:
                self.critical_values.append(k)
        if self.p < self.n:
            A = self.Q
            B = self.N
            E = [(q,k) for q in self.Q for k in self.N if k not in self.col_dict[q]]
            self.row_extension_problem = MatchingProblem(A, B, E)
        else:
            self.row_extension_problem = None
        if self.q < self.n:
            A = self.P
            B = self.N
            E = [(p,k) for p in self.P for k in self.N if k not in self.row_dict[p]]
            self.col_extension_problem = MatchingProblem(A, B, E)
        else:
            self.col_extension_problem = None

    def show_matrix(self):
        display(sp.Matrix(self.rows))

    def excess_report(self):
        mk = [self.multiplicity[k] for k in self.N]
        ek = [self.excess[k] for k in self.N]
        eks = []
        for e in ek:
            if e > 0:
                eks.append('<td>' + str(e) + '</td>')
            elif e == 0:
                eks.append('<td style="color:orange">' + str(e) + '</td>')
            else:
                eks.append('<td style="color:red">' + str(e) + '</td>')
        h = "<style>td {border: 1px solid black;}</style>\n<table>\n"
        h += '<tr><td>k</td>' + \
             ''.join([f'<td>{k}</td>' for k in self.N]) + "</tr>\n"
        h += '<tr><td>m(k)</td>' + \
             ''.join([f'<td>{x}</td>' for x in mk]) + "</tr>\n"
        h += '<tr><td>e(k)</td>' + ''.join(eks) + "</tr>\n"
        h += "</table>"
        display(HTML(h))

    @staticmethod
    def modular_square(n, u = 1):
        if not isinstance(n, int) or n <= 0:
            raise ValueError('n must be a positive integer')
        if not (isinstance(u, int) and math.gcd(u,n) == 1):
            raise ValueError('u must be an integer coprime to n')
        L = [[(i + u*j) % n for j in range(n)] for i in range(n)]
        return LatinRectangle(L)
    
    @staticmethod
    def check_orthogonal(L0, L1):
        if not isinstance(L0, LatinRectangle) or not isinstance(L1, LatinRectangle) or \
           not L0.is_square or not L1.is_square:
            raise ValueError('Both inputs must be Latin squares')
        if L0.P != L1.P or L0.Q != L1.Q or L0.N != L1.N:
            raise ValueError('Both Latin rectangles must have the same P, Q, and N')
        X = sorted([(L0.rows[i][j],L1.rows[i][j]) for i in L0.P for j in L0.Q])
        Y = sorted([(n0,n1) for n0 in L0.N for n1 in L1.N])
        return X == Y
