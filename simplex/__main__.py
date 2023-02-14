"""
Main application to solve linear programs.
"""
import sys
from argparse import ArgumentParser, FileType
from fractions import Fraction
from typing import Tuple

import numpy as np

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
    vector,
)
from .revised import simplex as revised_simplex


def _revised_simplex(
    N: IndexSet,
    B: IndexSet,
    A: Matrix,
    b: Vector,
    c: Vector,
    v: Fraction,
) -> Tuple[Vector, Fraction]:
    """
    Bring into form usable by revised simplex algorithm and the solve.
    """
    # bring into form accepted by revised module.
    n = len(N)
    m = len(B)
    Nr = list(range(n))
    Br = list(range(n, n + m))
    Ar = np.zeros((m, m + n))
    br = np.zeros(m)
    cr = np.zeros(m + n)
    for i, ii in enumerate(B):
        for j, jj in enumerate(N):
            Ar[i][j] = float(A[ii][jj])
        Ar[i][n + i] = 1.0
        br[i] = b[ii]
    for j, jj in enumerate(N):
        cr[j] = -c[jj]

    # compute solution
    solr = revised_simplex(Ar, Nr, Br, br, cr)

    # adjust solution
    sol = vector()
    z = 0.0
    for j, jj in enumerate(N):
        sol[jj] = Fraction(solr[j]).limit_denominator(1000)
        z += c[jj] * sol[jj]

    return sol, Fraction(z + v).limit_denominator(1000)


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
        sol = _revised_simplex(*sf[1:])
    else:
        sol = simplex(*sf)

    print("solution:")
    print(solution_to_str(*sf[:2], *sol))


main()
