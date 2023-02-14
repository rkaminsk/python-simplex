"""
This module implements the revised simplex algorithm as presented on Wikipedia:

    https://en.wikipedia.org/wiki/Revised_simplex_method

It proceeds the same way as the standard one but moves the computation from the
pivot function to the variable selection. In practice, the algorithm is
numerically more stable and the tableau stays sparse.

However, the current implementation uses library functionality to inverse the
basis matrix. In practice, some LU decomposition based algorithm should be used
that incrementally updates the inverse.

Note: the way global variables are used here is really ugly and should be
improved.
"""

import numpy as np
import numpy.linalg as npl

E = np.finfo(float).eps


def pivot(N, B, q, p):
    """
    Pivot columns.
    """
    N.remove(q)
    B.remove(p)
    N.append(p)
    B.append(q)


def initialize(A, N, B, b, c):
    """
    Bring a linear program in slack form into canonical form.
    """
    p = np.argmin(b)

    # check if the basic solution is already feasible
    if b[p] >= -E:
        return

    # construct artificial problem
    a = A.shape[1]
    N.append(a)
    col = [[-1.0]] * len(B)
    A = np.append(A, col, axis=1)
    c = np.zeros(len(N) + len(B))
    c[-1] = 1
    pivot(N, B, a, B[p])

    # solve artificial problem
    x, z = solve(A, N, B, b, c)

    if a in B:
        p = B.index(a)
        if x[p] > E:
            raise RuntimeError("problem is infeasible")
        q = min(i for i, s_i in enumerate(z) if s_i > E)
        pivot(N, B, N[q], B[p])

    # remove artificial variable
    N.remove(a)


def solve(A, N, B, b, c):
    """
    Solve linear program in canonical form.

    The basis must be invertible and `B^-1 * b >= 0`.
    """
    it = 0
    while True:
        it += 1
        print(f"iteration {it}:")

        # compute intermediate values
        x = npl.solve(A[:, B], b)  # x_B = B^-1 * b
        y = npl.solve(np.transpose(A[:, B]), c[B])  # y_B = B^T^-1 * c_B
        z = c[N] - np.transpose(A[:, N]).dot(y)  # s_N = c_N - N^T * y_B
        print("  x = ", x)
        print("  l = ", y)
        print("  s = ", z)
        if (z >= -E).all():
            print()
            return x, z

        # select entering variable q
        q = min(i for i, s_i in enumerate(z) if s_i < -E)
        d = npl.solve(A[:, B], A[:, [N[q]]].reshape(-1))  # d = B^-1 * N_q
        print("  d = ", d)
        if (d <= E).all():
            raise RuntimeError("problem is unbounded")
        print()

        # select leaving variable p
        xd = ((i, x_i / d_i) for i, (x_i, d_i) in enumerate(zip(x, d)) if d_i > E)
        p, _ = min(xd, key=lambda xd_i: (xd_i[1], B[xd_i[0]]))

        pivot(N, B, N[q], B[p])


def simplex(A, N, B, b, c):
    """
    Solve linear program in slack form.

    The basis must be invertible.
    """
    N, B = N[:], B[:]

    with np.printoptions(precision=2, suppress=True):
        initialize(A, N, B, b, c)
        x, _ = solve(A, N, B, b, c)

    return [x[B.index(i)] if i in B else 0 for i in range(len(N))]
