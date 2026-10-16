from finbot.followups import suggest


def trace(tools=(), **flags):
    return {"tools": [{"name": t} for t in tools], "flags": flags}


def test_statement_followups_include_health_and_investment():
    s = suggest(trace(["financial_ratios"], statement_data=True))
    assert "Is the company healthy? Give me the key ratios" in s and "Should I make this investment?" in s


def test_goal_and_emi_followups():
    assert any("step up" in x for x in suggest(trace(["goal_saving"])))
    assert any("9.5%" in x for x in suggest(trace(["loan_emi"])))


def test_at_most_three_and_never_empty():
    assert 1 <= len(suggest(trace())) <= 3
    assert len(suggest(trace(["loan_emi", "financial_ratios"], statement_data=True))) <= 3
