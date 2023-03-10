"""
The main simplex algorithm.
"""

from fractions import Fraction
from typing import Callable, Tuple, cast

from .program import (
    Index,
    IndexSet,
    Matrix,
    VariableMap,
    Vector,
    matrix,
    slack_form_to_str,
    vector,
)


def pivot(
    N: IndexSet,
    B: IndexSet,
    A: Matrix,
    b: Vector,
    c: Vector,
    v: Fraction,
    l: Index,
    e: Index,
) -> Tuple[IndexSet, IndexSet, Matrix, Vector, Vector, Fraction]:
    """
    Pivot around the given leaving and entering variables.
    """
    An = matrix()
    bn = vector()
    cn = vector()

    # take care of row e
    bn[e] = b[l] / A[l][e]
    for j in N:
        if j == e:
            continue
        An[e][j] = A[l][j] / A[l][e]
    An[e][l] = 1 / A[l][e]

    # take care of the remaining rows
    for i in B:
        if i == l:
            continue
        bn[i] = b[i] - A[i][e] * bn[e]
        for j in N:
            if j == e:
                continue
            An[i][j] = A[i][j] - A[i][e] * An[e][j]
        An[i][l] = -A[i][e] * An[e][l]

    # compute objective function
    vn = v + c[e] * bn[e]
    for j in N:
        if j == e:
            continue
        cn[j] = c[j] - c[e] * An[e][j]
    cn[l] = -c[e] * An[e][l]

    # swap x_e and x_l
    Nn = N[:]
    Nn[N.index(e)] = l
    Bn = B[:]
    Bn[B.index(l)] = e

    return Nn, Bn, An, bn, cn, vn


def solve(
    M: VariableMap,
    N: IndexSet,
    B: IndexSet,
    A: Matrix,
    b: Vector,
    c: Vector,
    v: Fraction,
) -> Tuple[IndexSet, IndexSet, Matrix, Vector, Vector, Fraction]:
    """
    Solve a linear program in canonical form.

    This function selects an entering variable x_e among the non-basic variables
    that appears with a positive coefficient in the objective function. Since
    non-basic variables are currently pinned to value 0, pivoting it will
    assign a new value greater or equal to zero (zero if the pivot is
    degenerative).

    We then try to find a leaving variable x_l among the basic variables whose
    value can be lowered to zero such that the value of x_e increases. We
    increase x_e as much as we can ensuring that all basic variables are
    greater or equal to zero and the selected leaving variable is zero.

    We can then flip the roles of leaving and entering variables.

    This function just selects two suitable variables and then calls the pivot
    function to take care of rewriting the tableau. The actual assignemnt to
    variables is implicitely encoded in the tableau.
    """
    while True:
        # pylint: disable=undefined-loop-variable
        # select an entering variable
        for e in sorted(N):
            if c[e] > 0:
                break
        else:
            break

        # select a leaving variable
        l = None
        mm = None
        for i in B:
            if A[i][e] > 0:
                m = b[i] / A[i][e]
                if l is None or m < mm or (m == mm and i < l):
                    l = i
                    mm = m

        # do the pivot if the problem is bounded
        if l is None:
            raise RuntimeError("problem is unbounded")
        print(f"after pivoting around {M[l]} and {M[e]}")
        N, B, A, b, c, v = pivot(N, B, A, b, c, v, l, e)
        print(slack_form_to_str(M, N, B, A, b, c, v))
    return N, B, A, b, c, v


def initialize(
    M: VariableMap,
    N: IndexSet,
    B: IndexSet,
    A: Matrix,
    b: Vector,
    c: Vector,
    v: Fraction,
):
    """
    Bring a linear program in slack form into canonical form.
    """
    l = min(b, key=cast(Callable[[int], Fraction], b.get))

    # check if the basic solution is already feasible
    if b[l] >= 0:
        return N, B, A, b, c, v

    # construct artificial problem
    N = N + [0]
    for i in B:
        A[i][0] = Fraction(-1)
    c_orig, v_orig, c = c, v, vector([(0, Fraction(-1))])
    N, B, A, b, c, v = pivot(N, B, A, b, c, v, l, 0)

    print("artificial problem:")
    print(slack_form_to_str(M, N, B, A, b, c, v))

    # solve artificial problem
    N, B, A, b, c, v = solve(M, N, B, A, b, c, v)

    if b[0] > 0:
        raise RuntimeError("problem is infeasible")

    if 0 in B:
        # move variable x_0 out of the basis by an arbitrary degenerate pivot
        for e in N:
            if A[0][e] == 0:
                continue
            N, B, A, b, c, v = pivot(N, B, A, b, c, v, 0, e)
            break

    # remove the artificial variable
    N.remove(0)
    for i in A:
        A[i].pop(0, None)

    # restore the original objective function
    c, v = vector(), v_orig
    for i, val in c_orig.items():
        if i in N:
            c[i] += val
        elif i in B:
            v += c_orig[i] * b[i]
            for j in A[i]:
                c[j] -= c_orig[i] * A[i][j]

    print("initialized problem:")
    print(slack_form_to_str(M, N, B, A, b, c, v))
    return N, B, A, b, c, v


def simplex(
    M: VariableMap,
    N: IndexSet,
    B: IndexSet,
    A: Matrix,
    b: Vector,
    c: Vector,
    v: Fraction,
) -> Tuple[Vector, Fraction]:
    """
    Solve a linear program in slack form.
    """
    # initialize problem
    N, B, A, b, c, v = initialize(M, N, B, A, b, c, v)

    # solve problem
    N, B, A, b, c, v = solve(M, N, B, A, b, c, v)

    # return solution
    return b.copy(), v
