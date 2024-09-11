"""
Main application to solve linear programs.
"""

import sys
from argparse import ArgumentParser, FileType
from fractions import Fraction

from . import revised
from .algorithm import simplex
from .parser import parse
from .program import (
    IndexSet,
    Matrix,
    Vector,
    program_to_str,
    slack_form,
    slack_form_to_str,
    solution_to_str,
)


def _revised_simplex(
    N: IndexSet, B: IndexSet, A: Matrix, b: Vector, c: Vector, v: Fraction
):
    mat = []
    for i in B:
        row = []
        for j in N:
            row.append(A[i][j])
        for ii in B:
            row.append(Fraction(1 if i == ii else 0))
        mat.append(row)

    a = revised.mat(mat)
    x = revised.vec([b[i] for i in B])
    z = revised.vec([-c[j] for j in N])

    nn = [j + 1 for j in range(len(N))]
    bb = [i + 1 + len(N) for i in range(len(B))]

    revised.print_solution(nn, bb, x, z, *revised.solve(a, nn, bb, x, z))


def main():
    """
    Read and solve linear pogram.
    """

    parser = ArgumentParser(prog="simplex", description="Solve a linear problem.")
    parser.add_argument(
        "file", type=FileType("r", encoding="utf-8"), nargs="?", default=sys.stdin
    )
    parser.add_argument(
        "-r", "--revised", action="store_true", help="Use revised simplex algorithm."
    )
    args = parser.parse_args()
    prg = args.file.read()

    lp = parse(prg)
    print("linear program:")
    print(program_to_str(*lp))
    print()

    sf = slack_form(*lp)
    print("initial problem:")
    print(slack_form_to_str(*sf))

    if args.revised:
        _revised_simplex(*sf[1:])
    else:
        sol = simplex(*sf)

        print("solution:")
        print(solution_to_str(*sf[:2], *sol))


main()
