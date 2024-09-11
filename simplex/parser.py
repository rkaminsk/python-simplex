"""
A parser to read the string representation of a linear program.
"""
import re
from collections import defaultdict
from fractions import Fraction
from typing import DefaultDict, Generator, Iterator, List, Optional, Tuple, Union, cast

from .program import Constraint, Objective, Program, Term

RawTerm = List[Tuple[Fraction, Optional[str]]]


def _negate(term: RawTerm) -> RawTerm:
    """
    Negate the given term.
    """
    for i, (n, v) in enumerate(term):
        term[i] = (-n, v)
    return term


def _to_number(term: RawTerm) -> Fraction:
    """
    Convert the given term to a number or raise an error.
    """
    ret = Fraction(0)
    for n, v in term:
        if v is not None:
            raise RuntimeError("number expected")
        ret += n
    return ret


def _divide(lhs: RawTerm, rhs: RawTerm) -> RawTerm:
    """
    Devide the given terms lhs by the term rhs.

    The term rhs must evaluate to a number.
    """
    den = _to_number(rhs)
    for i, (n, v) in enumerate(lhs):
        lhs[i] = (n / den, v)
    return lhs


def _multiply(lhs: RawTerm, rhs: RawTerm) -> RawTerm:
    """
    Multiply the given terms.

    Raises an error if the term becomes nonlinear.
    """
    ret = []
    for n, v in lhs:
        for m, w in rhs:
            if w is None:
                ret.append((n * m, v))
            elif v is None:
                ret.append((n * m, w))
            else:
                raise RuntimeError("linear term expected")
    return ret


def _simplify(term: RawTerm, t: str) -> Union[Constraint, Objective]:
    """
    Simplify the given constraint or objective function combining
    coefficients.
    """
    d: DefaultDict[Optional[str], Fraction]
    d = defaultdict(Fraction)
    for n, v in term:
        d[v] += n
    term = [(n, v) for v, n in d.items() if v is not None]

    if t == "#minimize":
        t = "#maximize"
        d[None] *= -1
        term = _negate(term)

    if t == "#maximize":
        return (d[None], cast(List[Term], term))

    return (cast(List[Term], term), t, -d[None])


class Parser:
    """
    Simplistic parser for the grammar below.

    Program     ::= Constraints Objective SumTerm
    Constraints ::= Constraint | Constraint Constraints
    Constraint  ::= SumTerm Operator SumTerm
    SumTerm     ::= MulTerm SumCont
    SumCont     ::= '+' MulTerm SumCont | '-' MulTerm SumCont | epsilon
    MulTerm     ::= NegTerm MulCont
    MulCont     ::= '*  NegTerm MulCont | '/' NegTerm MulCont | epsilon
    NegTerm     ::= '-' NegTerm | Term
    Term        ::= Number | Variable | ( SumTerm )



    Note that only _constraints that can be transformed into linear terms (in a
    straightforward manner) are accepted.
    """

    _lexer: Iterator[Tuple[str, str]]
    _current: Tuple[str, str]

    def __init__(self, prg: str):
        """
        Create a parser for the given linear program.
        """
        self._lexer = self._lex(prg)
        self._current = next(self._lexer)

    @staticmethod
    def _lex(line: str) -> Generator[Tuple[str, str], None, None]:
        """
        The tokenizer for the parser.
        """
        n = 0
        regexp = re.compile(
            r"(?P<number>[0-9]+)|"
            r"(?P<identifier>[a-zA-Z][a-zA-Z_0-9]*)|"
            r"(?P<operator>[-+*/])|"
            r"(?P<relation>>=|<=|=)|"
            r"(?P<whitespace>[ \t\r]+)|"
            r"(?P<newline>\n)|"
            r"(?P<objective>#minimize|#maximize)|"
            r"(?P<parenthesis>[()])"
        )
        while n < len(line):
            match = regexp.match(line[n:])
            if match is None:
                raise RuntimeError("failed to match")
            n += match.span(0)[1]
            assert match.lastgroup is not None
            if match.lastgroup != "whitespace":
                yield match.lastgroup, match.group(0)
        yield "eof", "\0"

    def _consume(self) -> Tuple[str, str]:
        """
        Return the current token and advance to the next one.
        """
        ret = self._current
        self._current = next(self._lexer)
        return ret

    def _peek(self, c: str, val: Optional[str] = None) -> bool:
        """
        Check if the current token matches the given type and optionally its
        value.
        """
        return (val is None and self._current[0] == c) or self._current == (c, val)

    def _accept(self, c: str, val: Optional[str] = None) -> bool:
        """
        Check the current token and advance to the next in case of a match.
        """
        if self._peek(c, val):
            if self._current[0] != "eof":
                self._current = next(self._lexer)
            return True
        return False

    def _expect(self, c: str, val: Optional[str] = None) -> str:
        """
        Require that the current token matches and advance to the next in case.
        """
        ret = self._current[1]
        if not self._accept(c, val):
            if val is None:
                raise RuntimeError(f"Unexpected {self._current[1]}, expected {val}")
            raise RuntimeError(f"Unexpected {self._current[0]}, expected {c}")
        return ret

    def parse(self) -> Program:
        """
        Parse the linear program.
        """
        return self._program()

    def _program(self) -> Program:
        """
        See the matching production in the class docstring.
        """
        c = self._constraints()
        t = self._expect("objective")
        o = _simplify(self._sum_term(), t)
        while self._accept("newline"):
            pass
        self._expect("eof")
        return (c, cast(Objective, o))

    def _constraints(self) -> List[Constraint]:
        """
        See the matching production in the class docstring.
        """
        c = self._constraint()
        while self._accept("newline"):
            pass
        if self._peek("objective"):
            return [c]
        return [c] + self._constraints()

    def _constraint(self) -> Constraint:
        """
        See the matching production in the class docstring.
        """
        lhs = self._sum_term()
        rel = self._expect("relation")
        rhs = self._sum_term()
        return cast(Constraint, _simplify(lhs + _negate(rhs), rel))

    def _sum_term(self) -> RawTerm:
        """
        See the matching production in the class docstring.
        """
        return self._mul_term() + self._sum_cont()

    def _sum_cont(self) -> RawTerm:
        """
        See the matching production in the class docstring.
        """
        if self._accept("operator", "+"):
            return self._mul_term() + self._sum_cont()
        if self._accept("operator", "-"):
            return _negate(self._mul_term()) + self._sum_cont()
        return []

    def _mul_term(self) -> RawTerm:
        """
        See the matching production in the class docstring.
        """
        lhs = self._neg_term()
        return self._mul_cont(lhs)

    def _mul_cont(self, lhs) -> RawTerm:
        """
        See the matching production in the class docstring.
        """
        if self._accept("operator", "/"):
            return self._mul_cont(_divide(lhs, self._neg_term()))
        if self._accept("operator", "*"):
            return self._mul_cont(_multiply(lhs, self._neg_term()))
        return lhs

    def _neg_term(self) -> RawTerm:
        """
        See the matching production in the class docstring.
        """
        if self._accept("operator", "-"):
            return _negate(self._term())
        return self._term()

    def _term(self) -> RawTerm:
        """
        See the matching production in the class docstring.
        """
        if self._accept("parenthesis", "("):
            term = self._sum_term()
            self._expect("parenthesis", ")")
            return term
        kind, val = self._consume()
        if kind == "number":
            return [(Fraction(val), None)]
        if kind == "identifier":
            return [(Fraction(1), val)]
        raise RuntimeError("number or variable expected")


def parse(prg: str) -> Program:
    """
    Parse the given linear program.
    """
    return Parser(prg).parse()
