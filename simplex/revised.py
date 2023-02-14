"""
This module implements the revised simplex algorithm as presented on Wikipedia:

    https://en.wikipedia.org/wiki/Revised_simplex_method

It proceeds the same way as the standard one but moves the computation from the
pivot to the select function. In practice, the algorithm is numerically more
stable and the tableau stays sparse.

However, the current implementation uses library functionality to inverse the
basis matrix. In practice, some LU decomposition based algorithm should be used
that incrementally updates the inverse.

Note: the way global variables are used here is really ugly and should be
improved.
"""

import numpy as np
import numpy.linalg as npl

# pylint: disable=invalid-name,global-statement

np.set_printoptions(precision=2, suppress=True)

E = np.finfo(float).eps
A = None
N = None
B = None
b = None
c = None
x = None
s = None


def select():
    """
    Select pivot indices using Bland's rule.
    """
    global x, s
    x = npl.solve(A[:, B], b)  # x_B = B^-1 * b
    y = npl.solve(np.transpose(A[:, B]), c[B])  # y_B = B^T^-1 * c_B
    s = c[N] - np.transpose(A[:, N]).dot(y)  # s_N = c_N - N^T * y
    print("  x_B", x)
    print("  y_B", y)
    print("  s_N", s)
    if (s >= -E).all():
        return None
    # select entering variable q
    q = min(i for i, s_i in enumerate(s) if s_i < -E)
    d = npl.solve(A[:, B], A[:, [N[q]]].reshape(-1))  # d = B^-1 * N_q
    print("  d_B", d)
    if (d <= E).all():
        raise RuntimeError("problem is unbounded")
    # select leaving variable p
    xd = ((i, x_i / d_i) for i, (x_i, d_i) in enumerate(zip(x, d)) if d_i > E)
    p, _ = min(xd, key=lambda xd_i: (xd_i[1], B[xd_i[0]]))
    return N[q], B[p]


def pivot(q, p):
    """
    Pivot columns.
    """
    N.remove(q)
    B.remove(p)
    N.append(p)
    B.append(q)


def initialize():
    """
    Bring a linear program in slack form into canonical form.
    """
    global A, c
    p = np.argmin(b)

    # check if the basic solution is already feasible
    if b[p] >= -E:
        return

    # construct artificial problem
    a = A.shape[1]
    N.append(a)
    col = [[-1.0]] * len(B)
    A = np.append(A, col, axis=1)
    c_orig, c = c, np.zeros(len(N) + len(B))
    c[-1] = 1
    pivot(a, B[p])

    # solve artificial problem
    solve()

    if a in B:
        p = B.index(a)
        if x[p] > E:
            raise RuntimeError("problem is infeasible")
        q = min(i for i, s_i in enumerate(s) if s_i > E)
        pivot(N[q], B[p])

    # remove artificial variable and restore objective
    N.remove(a)
    A = np.delete(A, a, 1)
    c = c_orig


def solve():
    """
    Solve a linear program in canonical form.
    """
    i = 0
    while True:
        i += 1
        print(f"iteration {i}:")
        ret = select()
        print()
        if ret is None:
            return
        pivot(*ret)


def simplex(A_in, N_in, B_in, b_in, c_in):
    """
    Solve a linear program in slack form.
    """
    global A, N, B, b, c
    A, N, B, b, c = A_in[:], N_in[:], B_in[:], b_in[:], c_in[:]
    initialize()
    solve()
    return [x[B.index(i)] if i in B else 0 for i in range(len(N))]
