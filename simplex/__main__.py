"""
Main application to solve linear programs.
"""

import sys
from argparse import ArgumentParser, FileType

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


def main():
    """
    Read and solve linear pogram.
    """

    parser = ArgumentParser(prog="simplex", description="Solve a linear problem.")
    parser.add_argument(
        "file", type=FileType("r", encoding="utf-8"), nargs="?", default=sys.stdin
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

    sol = simplex(*sf)

    print("solution:")
    print(solution_to_str(*sf[:2], *sol))


main()
