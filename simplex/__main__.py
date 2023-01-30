"""
Main application to solve linear programs.
"""
import sys

from .program import program_to_str, slack_form, slack_form_to_str, solution_to_str
from .parser import parse
from .algorithm import simplex


def main():
    """
    Read and solve linear pogram.
    """
    if len(sys.argv) == 1:
        prg = sys.stdin.read()
    elif len(sys.argv) == 2:
        # pylint: disable=unspecified-encoding
        with open(sys.argv[1]) as hnd:
            prg = hnd.read()
    else:
        print("usage: simplex [file]")
        sys.exit(1)

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
