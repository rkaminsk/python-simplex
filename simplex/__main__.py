from .program import program_to_str, slack_form, slack_form_to_str, solution_to_str
from .parser import parse
from .algorithm import simplex

lp = parse("(x + y+z)/7>=-5\nx-y=3\n#minimize 3+4+x")
print("linear program:")
print(program_to_str(*lp))
print()

print("slack form:")
sf = slack_form(*lp)
print(slack_form_to_str(*sf))

sol = simplex(*sf)
print(solution_to_str(*sf[:2], *sol))
