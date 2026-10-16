"""Safe arithmetic and equation solving for model-generated input.

`calculate` walks a Python AST and allows only numbers, + - * / ** % //, unary signs,
the constants pi and e, and a short list of math functions. It never calls eval/exec.
`solve_equation` uses SymPy behind a character whitelist that blocks quotes, brackets,
attribute access, dunder names and Python keywords.
"""
import ast
import concurrent.futures
import keyword
import math
import operator
import re

import sympy as sp
from sympy.parsing.sympy_parser import convert_xor, parse_expr, standard_transformations

MAX_LEN = 300
MAX_EXPONENT = 1000

_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "sqrt": math.sqrt, "ln": math.log, "log": math.log, "log10": math.log10, "exp": math.exp,
    "abs": abs, "round": round, "min": min, "max": max, "floor": math.floor, "ceil": math.ceil,
}
_CONSTS = {"pi": math.pi, "e": math.e}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError("Exponent too large.")
        return _BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS
            and not node.keywords and 1 <= len(node.args) <= 10):
        args = [_eval(a) for a in node.args]
        if node.func.id == "round" and len(args) == 2:
            args[1] = int(args[1])
        return float(_FUNCS[node.func.id](*args))
    if isinstance(node, ast.Tuple):
        raise ValueError("Write plain numbers without thousands separators, e.g. 2000000 not 20,00,000.")
    if isinstance(node, ast.Name):
        raise ValueError(f"Unknown name '{node.id}'. Only numbers, pi, e and {sorted(_FUNCS)} are allowed.")
    raise ValueError(f"Not allowed in an arithmetic expression: {type(node).__name__}.")


def calculate(expression: str) -> dict:
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("expression must be a non-empty string.")
    if len(expression) > MAX_LEN:
        raise ValueError(f"expression is longer than {MAX_LEN} characters.")
    normalized = expression.replace("^", "**").strip()
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Could not parse the expression: {exc.msg}.") from None
    try:
        value = _eval(tree)
    except ZeroDivisionError:
        raise ValueError("Division by zero.") from None
    except OverflowError:
        raise ValueError("The result is too large.") from None
    if not math.isfinite(value):
        raise ValueError("The result is not a finite number.")
    return {"inputs": {"expression": expression}, "formula": normalized,
            "result": {"value": value, "rounded": round(value, 6)}, "formatted": {}}


# ------------------------------------------------------------------ equations
_ALLOWED_CHARS = re.compile(r"^[0-9A-Za-z_+\-*/^().=,\s]+$")
_ATTRIBUTE = re.compile(r"\.\s*[A-Za-z_]")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_VARNAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,11}$")
_SYMPY_FUNCS = {"sqrt": sp.sqrt, "exp": sp.exp, "ln": sp.log, "log": sp.log, "pi": sp.pi, "E": sp.E}
_SAFE_GLOBALS = {"__builtins__": {}, "Integer": sp.Integer, "Float": sp.Float, "Rational": sp.Rational,
                 "Symbol": sp.Symbol, "factorial": sp.factorial, **_SYMPY_FUNCS}
_SOLVE_TIMEOUT_S = 8


def _parse(text: str, symbols: dict):
    return parse_expr(text, local_dict=dict(symbols), global_dict=dict(_SAFE_GLOBALS),
                      transformations=standard_transformations + (convert_xor,))


def solve_equation(equation: str, solve_for: str) -> dict:
    if not isinstance(solve_for, str) or not _VARNAME.match(solve_for) or keyword.iskeyword(solve_for):
        raise ValueError("solve_for must be a simple variable name such as x or r.")
    if not isinstance(equation, str) or not equation.strip() or len(equation) > MAX_LEN:
        raise ValueError(f"equation must be a non-empty string of at most {MAX_LEN} characters.")
    if not _ALLOWED_CHARS.match(equation) or "__" in equation or _ATTRIBUTE.search(equation):
        raise ValueError("Only numbers, variable names, + - * / ^ ( ) = , and sqrt/exp/ln/log/pi/E are allowed.")
    sides = equation.split("=")
    if len(sides) > 2:
        raise ValueError("Use at most one '=' sign.")
    names = set(_IDENT.findall(equation))
    for name in names:
        if keyword.iskeyword(name):
            raise ValueError(f"'{name}' is not allowed.")
        if name not in _SYMPY_FUNCS and not _VARNAME.match(name):
            raise ValueError(f"'{name}' is not a valid variable name.")
    symbols = {n: sp.Symbol(n) for n in names if n not in _SYMPY_FUNCS}
    if solve_for not in symbols:
        raise ValueError(f"The variable '{solve_for}' does not appear in the equation.")
    try:
        lhs = _parse(sides[0], symbols)
        rhs = _parse(sides[1], symbols) if len(sides) == 2 else sp.Integer(0)
    except Exception as exc:  # sympy raises many parser exception types
        raise ValueError(f"Could not parse the equation: {exc}") from None
    expr = sp.sympify(lhs - rhs)
    target = symbols[solve_for]

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        sols = pool.submit(sp.solve, expr, target).result(timeout=_SOLVE_TIMEOUT_S)
    except concurrent.futures.TimeoutError:
        raise ValueError("The equation is too complex to solve quickly; try a specific finance tool.") from None
    except (NotImplementedError, ValueError, TypeError):
        sols = []
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    numeric, symbolic = [], []
    for s in sols:
        if s.free_symbols:
            symbolic.append(str(s))
            continue
        v = complex(s.evalf())
        if abs(v.imag) < 1e-9:
            numeric.append(float(v.real))
    if not sols and not expr.free_symbols - {target}:
        for start in (0.1, 1.0, 10.0, -0.5):  # numeric fallback for transcendental equations
            try:
                v = float(sp.nsolve(expr, target, start))
                if all(abs(v - u) > 1e-9 for u in numeric):
                    numeric.append(v)
            except (ValueError, ZeroDivisionError, TypeError):
                continue
    if not numeric and not symbolic:
        raise ValueError("No real solution found.")
    result = {"solutions": sorted(numeric)}
    if symbolic:
        result["solutions_symbolic"] = symbolic
    return {"inputs": {"equation": equation, "solve_for": solve_for},
            "formula": f"{lhs} = {rhs}", "result": result, "formatted": {}}
