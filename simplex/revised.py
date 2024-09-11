"""
The revised solve algorithm.

Provides a primal and a dual version as well as a two-phase variant to solve
problem that have no intial feasible solutions.
"""
from fractions import Fraction
from enum import Enum

import pymatrix


class Result(Enum):
    """
    Capture possible result states of the solve algorigthm.
    """

    BOUNDED = 1
    UNBOUNDED = 2
    INFEASIBLE = 3


def mat(x):
    """
    Construct a matrix given as list of rows.
    """
    return pymatrix.Matrix.from_list([[Fraction(v) for v in r] for r in x])


def vec(x):
    """
    Construct a vector given as list.
    """
    return mat([x]).trans()


def extract(idx, m):
    """
    Extract the given columns from the matrix.
    """
    res = []
    for i in idx:
        res.append(list(m.col(i - 1)))
    return mat(res).trans()


def solve_primal(m_a, i_n, i_b, v_x, v_z):
    """
    Simplex implementation asssuming that the input is primal feasible.
    """
    assert all(x >= 0 for x in v_x.col(0))

    i_n = i_n.copy()
    i_b = i_b.copy()
    v_x = v_x.copy()
    v_z = v_z.copy()
    res = Result.BOUNDED

    while True:
        j, z = min(enumerate(v_z.col(0)), key=lambda x: x[1])
        if z >= 0:
            break
        m_b = extract(i_b, m_a)
        m_n = extract(i_n, m_a)
        d_x = (m_b.inv() * m_n).colvec(j)
        if all(a <= 0 for a in d_x.col(0)):
            res = Result.UNBOUNDED
            break
        iab = enumerate(zip(d_x.col(0), v_x.col(0)))
        t, i = min((b / a, i) for i, (a, b) in iab if a > 0)
        d_z = -(m_b.inv() * m_n).trans().colvec(i)
        s = v_z[j][0] / d_z[j][0]
        v_x -= t * d_x
        v_x[i][0] = t
        v_z -= s * d_z
        v_z[j][0] = s
        i_b[i], i_n[j] = i_n[j], i_b[i]

    return res, i_n, i_b, v_x, v_z


def solve_dual(m_a, i_n, i_b, v_x, v_z):
    """
    Simplex implementation asssuming that the input is dual feasible.
    """
    assert all(z >= 0 for z in v_z.col(0))

    i_n = i_n.copy()
    i_b = i_b.copy()
    v_x = v_x.copy()
    v_z = v_z.copy()
    res = Result.BOUNDED

    while True:
        i, x = min(enumerate(v_x.col(0)), key=lambda x: x[1])
        if x >= 0:
            break
        m_b = extract(i_b, m_a)
        m_n = extract(i_n, m_a)
        d_z = -(m_b.inv() * m_n).trans().colvec(i)
        if all(a <= 0 for a in d_z.col(0)):
            res = Result.UNBOUNDED
            break
        iab = enumerate(zip(d_z.col(0), v_z.col(0)))
        s, j = min((b / a, j) for j, (a, b) in iab if a > 0)
        d_x = (m_b.inv() * m_n).colvec(j)
        t = v_x[i][0] / d_x[i][0]
        v_x -= t * d_x
        v_x[i][0] = t
        v_z -= s * d_z
        v_z[j][0] = s
        i_b[i], i_n[j] = i_n[j], i_b[i]

    return res, i_n, i_b, v_x, v_z


def solve(m_a, i_n, i_b, v_x, v_z):
    """
    A two-phase solve implementation.
    """
    primal_feasible = all(x >= 0 for x in v_x.col(0))
    dual_feasible = all(z >= 0 for z in v_z.col(0))
    if primal_feasible and dual_feasible:
        return Result.BOUNDED, i_n.copy(), i_b.copy(), v_x.copy(), v_z.copy()
    if primal_feasible:
        return solve_primal(m_a, i_n, i_b, v_x, v_z)

    s_z = v_z if dual_feasible else vec([1 for _ in v_z.col(0)])
    res, d_n, d_b, d_x, d_z = solve_dual(m_a, i_n, i_b, v_x, s_z)
    # if the dual solution is unbounded, d_x does not capture a primal solution
    if res == Result.UNBOUNDED:
        res = Result.INFEASIBLE
    if res == Result.INFEASIBLE or dual_feasible:
        return res, d_n, d_b, d_x, d_z

    # adjust objective coefficients for solving the second phase
    def val(t):
        return -v_z[i_n.index(t)][0] if t in i_n else 0

    m_n = extract(d_n, m_a)
    m_b = extract(d_b, m_a)
    c_n = [val(t) for t in d_n]
    c_b = [val(t) for t in d_b]
    r_z = (m_b.inv() * m_n).trans() * vec(c_b) - vec(c_n)
    return solve_primal(m_a, d_n, d_b, d_x, r_z)


def print_solution(i_n, i_b, v_x, v_z, res, s_n, s_b, s_x, s_z):
    """
    Print the given solution to the solve problem.
    """

    def ps(v, i_n, v_z, i_b, v_x):
        res = 0
        for i, t in enumerate(i_n):
            if t in i_b:
                j = i_b.index(t)
                print(f"{v}_{t} = {v_x[j][0]}")
                res -= v_z[i][0] * v_x[j][0]
            else:
                print(f"{v}_{t} = 0")
        return res

    print("**************************")
    if res == Result.INFEASIBLE:
        print("infeasible")
    else:
        rp = ps("x", i_n, v_z, s_b, s_x)
        print(f"lower: {rp}")
        if res == Result.BOUNDED:
            rd = -ps("z", i_b, v_x, s_n, s_z)
            print(f"upper: {rd}")
            assert rp == rd
    print("**************************")


def main():
    """
    Compute some examples.
    """
    # fmt: off

    # a bounded example
    i_n = [1, 2]
    i_b = [3, 4, 5]
    m_a = mat([
        [-1,  1, 1, 0, 0],
        [-2,  1, 0, 1, 0],
        [ 0, -1, 0, 0, 1]])
    v_x = vec([-1, -3, -5])
    v_z = vec([4, 3])

    res, s_n, s_b, s_x, s_z = solve(m_a, i_n, i_b, v_x, v_z)
    print_solution(i_n, i_b, v_x, v_z, res, s_n, s_b, s_x, s_z)

    # an unbounded example
    i_b = [3, 4, 5]
    i_n = [1, 2]
    m_a = mat([
        [-2,  3, 1, 0, 0],
        [ 0,  4, 0, 1, 0],
        [ 0, -1, 0, 0, 1]])
    v_x = vec([5, 7, 0])
    v_z = vec([-1, 1])

    print_solution(i_n, i_b, v_x, v_z, *solve(m_a, i_n, i_b, v_x, v_z))

    # a bounded two-phase example
    i_b = [4, 5]
    i_n = [1, 2, 3]
    m_a = mat([
        [-1, -1, -1, 1, 0],
        [ 1, -1,  1, 0, 1]])
    v_x = vec([-2, 1])
    v_z = vec([-2, 6, 0])

    print_solution(i_n, i_b, v_x, v_z, *solve(m_a, i_n, i_b, v_x, v_z))

main()
