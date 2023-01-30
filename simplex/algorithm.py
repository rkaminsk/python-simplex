from typing import Tuple
from fractions import Fraction
from collections import defaultdict

from .program import vector, matrix, slack_form_to_str, Index, IndexSet, Vector, Matrix, VariableMap


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
    M: VariableMap, N: IndexSet, B: IndexSet, A: Matrix, b: Vector, c: Vector, v: Fraction
) -> Tuple[IndexSet, IndexSet, Matrix, Vector, Vector, Fraction]:
    while True:
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
        print(f"after pivoting around {l} and {e}")
        N, B, A, b, c, v = pivot(N, B, A, b, c, v, l, e)
        print(slack_form_to_str(M, N, B, A, b, c, v))
    return N, B, A, b, c, v


def initialize(M: VariableMap, N: IndexSet, B: IndexSet, A: Matrix, b: Vector, c: Vector, v: Fraction):
    l = min(b, key=b.get)

    print("initial problem:")
    print(slack_form_to_str(M, N, B, A, b, c, v))

    # check if the basic solution is already feasible
    if b[l] >= 0:
        return N, B, A, b, c, v

    # construct artificial problem
    N.append(0)
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
    M: VariableMap, N: IndexSet, B: IndexSet, A: Matrix, b: Vector, c: Vector, v: Fraction
) -> Tuple[Vector, Fraction]:
    # initialize problem
    N, B, A, b, c, v = initialize(M, N, B, A, b, c, v)

    # solve problem
    N, B, A, b, c, v = solve(M, N, B, A, b, c, v)

    # return solution
    return b.copy(), v

'''
def ex1():
    """
    An example with an initial basic feasible solution.
    """
    N = [1, 2, 3]
    B = [4, 5, 6]
    A = matrix(
        [
            (4, vector([(1, Fraction(1)), (2, Fraction(1)), (3, Fraction(3))])),
            (5, vector([(1, Fraction(2)), (2, Fraction(2)), (3, Fraction(5))])),
            (6, vector([(1, Fraction(4)), (2, Fraction(1)), (3, Fraction(2))])),
        ]
    )
    b = vector([(4, Fraction(30)), (5, Fraction(24)), (6, Fraction(36))])
    c = vector([(1, Fraction(3)), (2, Fraction(1)), (3, Fraction(2))])
    return N, B, A, b, c


def ex2():
    """
    An example without an initial basic feasible solution.
    """
    N = [1, 2]
    B = [3, 4]
    A = matrix(
        [
            (3, vector([(1, Fraction(2)), (2, Fraction(-1))])),
            (4, vector([(1, Fraction(1)), (2, Fraction(-5))])),
        ]
    )
    b = vector([(3, Fraction(2)), (4, Fraction(-4))])
    c = vector([(1, Fraction(2)), (2, Fraction(-1))])
    return N, B, A, b, c


def main():
    print("**** Exmaple 1 ****\n")
    N, B, A, b, c = ex1()
    x, z = simplex(N, B, A, b, c)
    print(solution_to_str(N, x, z))

    print("\n**** Exmaple 2 ****\n")
    N, B, A, b, c = ex2()
    x, z = simplex(N, B, A, b, c)
    print(solution_to_str(N, x, z))


if __name__ == "__main__":
    main()
'''
