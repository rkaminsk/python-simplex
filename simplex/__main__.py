import sys

from .program import program_to_str, slack_form, slack_form_to_str, solution_to_str
from .parser import parse
from .algorithm import simplex


def main():
    lp = parse(sys.stdin.read())
    print("linear program:")
    print(program_to_str(*lp))
    print()

    print("slack form:")
    sf = slack_form(*lp)
    print(slack_form_to_str(*sf))

    sol = simplex(*sf)
    print("solution:")
    print(solution_to_str(*sf[:2], *sol))


main()
