"""
Implementation of the simplex algorithm.
"""
import heapq

from typing import List, Tuple

Vector = List[float]
Matrix = List[Vector]
Assignment = List[Tuple[int, float]]

def dot(a: Vector, b: Vector) -> float:
    """
    Compute the dot product of `a` and `b`.
    """
    return sum(x * y for x, y in zip(a, b))


def column_vector(A: Matrix, j: int) -> Vector:
    """
    Extract column `j` from matrix `A`.
    """
    return [row[j] for row in A]


def transpose(A: Matrix) -> Matrix:
    """
    Transpose the given matrix.
    """
    return [column_vector(A, j) for j in range(len(A[0]))]


def is_pivot_col(col: Vector) -> bool:
    """
    Check if the given column can be used for pivoting.
    """
    return (len([c for c in col if c == 0]) == len(col) - 1) and sum(col) == 1


def variable_value_for_pivot_column(tab: Matrix, col: Vector) -> float:
    pivot_row = [i for (i, x) in enumerate(col) if x == 1][0]
    return tab[pivot_row][-1]


def init_tableau(c: Vector, A: Matrix, b: Vector):
    """
    Initialize the given tableau.

    The tableau is composed as follows:

        (A|b)
        (c|0)

    Matrix A of dimension (n+m,m) holds the coefficients of the linear
    constraints, vector b of size m the guards of the constraints, and vector c
    of size n+m the coefficients of the cost function. The tableau is assumed
    to already contain slack variables in the last m columns and the costs to
    be zero for the slack variables.
    """
    tab = [row[:] + [x] for row, x in zip(A, b)]
    tab.append(c[:] + [0])
    return tab


def primal_solution(tab: Matrix) -> List[Tuple[int, float]]:
    """
    Extract variable assignment from tableau.
    """
    # the pivot columns denote which variables are used
    columns = transpose(tab)
    indices = [j for j, col in enumerate(columns[:-1]) if is_pivot_col(col)]
    return [
        (colIndex, variable_value_for_pivot_column(tab, columns[colIndex]))
        for colIndex in indices
    ]


def objective_value(tab: Matrix) -> float:
    """
    Extract objective value from tableau.
    """
    return -(tab[-1][-1])


def can_improve(tab: Matrix) -> bool:
    """
    Check if the objective value can still be improved.
    """
    lastRow = tab[-1]
    return any(x > 0 for x in lastRow[:-1])


def more_than_one_min(L: List[Tuple[int, float]]) -> bool:
    """
    Check if the given vector contains two times the same smallest element.
    """
    if len(L) <= 1:
        return False

    x, y = heapq.nsmallest(2, L, key=lambda x: x[1])
    return x == y


def find_pivot_index(tab: Matrix) -> Tuple[int, int]:
    """
    Find a suitiable point for pivoting.
    """
    # pick minimum positive index of the last row
    column_choices = [(i, x) for (i, x) in enumerate(tab[-1][:-1]) if x > 0]
    col = min(column_choices, key=lambda a: a[1])[0]

    # check if unbounded
    if all(row[col] <= 0 for row in tab):
        raise Exception("Linear program is unbounded.")

    # check for degeneracy: more than one minimizer of the quotient
    quotients = [
        (i, r[-1] / r[col]) for i, r in enumerate(tab[:-1]) if r[col] > 0
    ]

    # FIXME: this test can impossibly be correct
    # if the problem were really degenerate, we could also dynamically switch to Bland's rule for pivoting.
    if more_than_one_min(quotients):
        raise Exception("Linear program is degenerate.")

    # pick row index minimizing the quotient
    row = min(quotients, key=lambda x: x[1])[0]

    return row, col


def pivot_about(tab: Matrix, pivot: Tuple[int, int]) -> None:
    """
    Pivot the given tableau around the pivot point.
    """
    i, j = pivot

    pivot_denom = tab[i][j]
    tab[i] = [x / pivot_denom for x in tab[i]]

    for k in range(len(tab)):
        if k != i:
            pivot_row_multiple = [y * tab[k][j] for y in tab[i]]
            tab[k] = [x - y for x, y in zip(tab[k], pivot_row_multiple)]


def simplex(c: Vector, A: Matrix, b: Vector) -> Tuple[Matrix, Assignment, float]:
    """
    Solve the given standard-form linear program:

      max <c,x>
      s.t. Ax = b
           x >= 0

    providing the optimal solution x* and the value of the objective function
    """
    tab = init_tableau(c, A, b)
    print("Initial tableau:")
    for row in tab:
        print(row)
    print()

    while can_improve(tab):
        pivot = find_pivot_index(tab)
        print(f"Next pivot index is={pivot[0]},{pivot[1]}\n")
        pivot_about(tab, pivot)
        print("Tableau after pivot:")
        for row in tab:
            print(row)
        print()

    return tab, primal_solution(tab), objective_value(tab)


def main():
    """
    Run an example.
    """
    c = [300, 250, 450]
    A = [[15, 20, 25], [35, 60, 60], [20, 30, 25], [0, 250, 0]]
    b = [1200, 3000, 1500, 500]

    # add slack variables by hand
    A[0] += [1, 0, 0, 0]
    A[1] += [0, 1, 0, 0]
    A[2] += [0, 0, 1, 0]
    A[3] += [0, 0, 0, -1]
    c += [0, 0, 0, 0]

    t, s, v = simplex(c, A, b)
    print(s)
    print(v)

if __name__ == "__main__":
    main()
