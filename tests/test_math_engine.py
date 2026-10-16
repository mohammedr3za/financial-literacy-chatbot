import pytest

from finbot.tools.math_engine import calculate, solve_equation


@pytest.mark.parametrize("expr,expected", [
    ("2*(3+4)", 14),
    ("sqrt(16) + 2**3", 12),
    ("100000*(1+0.08)**5", 146932.8077),
    ("245000*0.15*1.18", 43365),
    ("2^10", 1024),               # caret means power in finance notation
    ("ln(exp(2))", 2),
    ("log10(1000)", 3),
    ("max(3, 7, 5) - min(1, 2)", 6),
    ("10 % 3", 1),
    ("-5 + abs(-3)", -2),
])
def test_calculate_ok(expr, expected):
    assert calculate(expr)["result"]["value"] == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("bad", [
    "__import__('os').system('ls')",
    "().__class__.__bases__",
    "open('/etc/passwd')",
    "a.b",
    "[1]*10",
    "20,00,000 * 2",              # thousands separators are refused with a helpful error
    "10**10**10",                 # overflow
    "lambda: 1",
    "x + 1",                      # unknown names
    "1/0",
    "9" * 600,                    # too long
])
def test_calculate_rejects(bad):
    with pytest.raises(ValueError):
        calculate(bad)


def test_solve_quadratic():
    r = solve_equation("x**2 - 5*x + 6 = 0", "x")["result"]
    assert sorted(r["solutions"]) == pytest.approx([2, 3])


def test_solve_growth_rate_real_root_only():
    r = solve_equation("1000*(1+r)^5 = 2000", "r")["result"]
    assert r["solutions"] == pytest.approx([2 ** (1 / 5) - 1])


def test_solve_without_equals_means_equal_zero():
    assert solve_equation("3*y - 12", "y")["result"]["solutions"] == pytest.approx([4])


@pytest.mark.parametrize("bad", [
    "__import__('os')",
    "x.__class__ = 1",
    "x = 1; import os",
    "x = 'a'",
    "x[0] = 1",
    "x = 1 = 2",
])
def test_solve_rejects(bad):
    with pytest.raises(ValueError):
        solve_equation(bad, "x")


def test_solve_rejects_bad_variable_name():
    with pytest.raises(ValueError):
        solve_equation("x = 2", "__x")
