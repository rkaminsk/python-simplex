from typing import List, DefaultDict, Dict, Optional, Sequence, Tuple
from fractions import Fraction
from collections import defaultdict

Term = Tuple[Fraction, str]
Constraint = Tuple[List[Term], str, Fraction]
Objective = Tuple[Fraction, List[Term]]
Program = Tuple[List[Constraint], Objective]
RawTerm = List[Tuple[Fraction, Optional[str]]]
VariableMap = Dict[int, str]

Index = int
IndexSet = List[int]
Vector = DefaultDict[int, Fraction]
Matrix = DefaultDict[int, Vector]


def vector(elems: Sequence[Tuple[int, Fraction]] = ()) -> Vector:
    """
    Construct a sparse vector.
    """
    vec = defaultdict(lambda: Fraction(0))
    for idx, val in elems:
        vec[idx] = val
    return vec


def matrix(elems: Sequence[Tuple[int, Vector]] = ()) -> Matrix:
    """
    Construct a sparse matrix.
    """
    mat: Matrix = defaultdict(vector)
    for i, vec in elems:
        mat[i] = vec
    return mat


def slack_form_to_str(
    M: VariableMap, N: IndexSet, B: IndexSet, A: Matrix, b: Vector, c: Vector, v: Fraction
) -> str:
    """
    Convert the given linear program in slack form into a readable string.
    """
    ret = f"{M[-1]:<3} = {float(v):7.2f} + "
    ret += " + ".join(f"{float(c[j]):7.2f} {M[j]:<3}" for j in sorted(N))
    ret += "\n"
    for i in sorted(B):
        ret += f"{M[i]:<3} = ".rjust(6)
        ret += f"{float(b[i]):7.2f} - "
        ret += " - ".join(f"{float(A[i][j]):7.2f} {M[j]:<3}" for j in sorted(N))
        ret += "\n"

    return ret


def program_to_str(constraints: List[Constraint], objective: Objective) -> str:
    """
    Convert the given linear program into a readable string representation.
    """
    def simp(co, var):
        if co == 1:
            return var
        if co == -1:
            return f"-{var}"
        return f"{co} {var}"

    v, c = objective
    ret = "#maximize "
    ret += " + ".join([f"{v}"] + [simp(co, var) for co, var in c])
    ret += "\n"
    for i, (lhs, rel, b) in enumerate(constraints):
        if i > 0:
            ret += "\n"
        ret += " + ".join(simp(co, var) for co, var in lhs)
        ret += f" {rel} {b}"

    return ret


def slack_form(constraints: List[Constraint], objective: Objective):
    """
    Convert the given linear program into slack form.
    """
    M: Dict[str, int]
    M = {}
    N: IndexSet = []
    B: IndexSet = []
    A = matrix()
    b = vector()
    c = vector()
    v = objective[0]

    # extract variables
    def add_var(lhs):
        nonlocal M, N
        for co, var in lhs:
            if var not in M:
                M[var] = len(M) + 1
                N.append(M[var])

    for lhs, rel, rhs in constraints:
        add_var(lhs)
    add_var(objective[1])

    # create auxiliary variables
    s = 0
    def aux_var(fixed = None):
        nonlocal s
        while f'x_{s}' in M:
            s += 1
        if fixed is None:
            M[f'x_{s}'] = len(M) + 1
            B.append(M[f'x_{s}'])
        else:
            M[f'x_{s}'] = fixed

        return M[f'x_{s}']

    # add artificial variable
    aux_var(0)

    # build coefficient matrix
    def add_row(lhs, rhs, mul):
        i = aux_var()
        for co, var in lhs:
            j = M[var]
            A[i][j] = mul * co
        b[i] = mul * rhs

    for lhs, rel, rhs in constraints:
        if rel == "<=":
            add_row(lhs, rhs, 1)
        if rel == ">=":
            add_row(lhs, rhs, -1)
        if rel == "=":
            add_row(lhs, rhs, 1)
            add_row(lhs, rhs, -1)

    # build objective
    if 'z' not in M:
        M['z'] = -1
    else:
        aux_var(-1)
    for co, var in objective[1]:
        c[M[var]] = co

    return {v: k for k, v in M.items()}, N, B, A, b, c, v
