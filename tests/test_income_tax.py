"""SPEC 4.3: income_tax, checked against hand-worked cases from the verified FY 2025-26 / 2026-27 rules."""
import pytest

from finbot.tools.income_tax import income_tax


def total(regime, **kw):
    return income_tax(regime=regime, **kw)["result"]["by_regime"][regime]["total_tax"]


def test_new_regime_nil_up_to_12_75_lakh_salary():
    assert total("new", gross_salary=1_275_000) == 0            # taxable 12 lakh → slab tax 60,000 fully rebated


def test_new_regime_marginal_relief_just_above_12_lakh():
    r = income_tax(gross_salary=1_300_000, regime="new")["result"]["by_regime"]["new"]
    assert r["taxable_income"] == 1_225_000 and r["slab_tax"] == 63_750
    assert r["rebate"] == 38_750 and r["total_tax"] == 26_000      # tax capped at income above 12 lakh, + 4% cess


def test_testers_question_30_lakh_new_regime():
    r = income_tax(gross_salary=3_000_000, hra_received=600_000)["result"]
    new = r["by_regime"]["new"]
    assert new["taxable_income"] == 2_925_000 and new["slab_tax"] == 457_500 and new["total_tax"] == 475_800
    assert any("rent_paid" in m for m in r["missing_inputs"])      # old-regime HRA needs rent and basic


def test_old_regime_with_hra_metro_and_80c():
    r = income_tax(gross_salary=3_000_000, regime="both", hra_received=600_000, basic_da=1_200_000,
                   rent_paid=480_000, city="Mumbai", section_80c=150_000)["result"]
    old = r["by_regime"]["old"]
    assert old["total_deductions"] == 560_000                      # HRA 3,60,000 + std 50,000 + 80C 1,50,000
    assert old["taxable_income"] == 2_440_000 and old["total_tax"] == 566_280
    assert r["lower_tax_regime"] == "new" and r["saving_with_lower_regime"] == 90_480


def test_hra_metro_list_changes_in_fy_2026_27():
    kw = dict(gross_salary=2_400_000, regime="old", hra_received=600_000, basic_da=1_000_000, rent_paid=700_000, city="Bengaluru")
    old_2526 = income_tax(financial_year="2025-26", **kw)["result"]["by_regime"]["old"]
    old_2627 = income_tax(financial_year="2026-27", **kw)["result"]["by_regime"]["old"]
    assert old_2526["total_deductions"] == 400_000 + 50_000        # 40% of basic in FY 2025-26
    assert old_2627["total_deductions"] == 500_000 + 50_000        # 50% of basic from FY 2026-27


def test_surcharge_10_percent_above_50_lakh():
    r = income_tax(gross_salary=5_500_000, regime="new")["result"]["by_regime"]["new"]
    assert r["taxable_income"] == 5_425_000 and r["slab_tax"] == 1_207_500
    assert r["surcharge"] == 120_750 and r["total_tax"] == 1_381_380


def test_surcharge_marginal_relief_just_above_50_lakh():
    r = income_tax(gross_salary=5_085_000, regime="new")["result"]["by_regime"]["new"]   # taxable 50,10,000
    assert r["slab_tax"] == 1_083_000 and r["surcharge"] == 7_000 and r["total_tax"] == 1_133_600


def test_senior_citizen_old_regime():
    assert total("old", gross_salary=800_000, age=65) == 62_400     # taxable 7,50,000 → 10,000 + 20% × 2,50,000


def test_old_regime_rebate_up_to_5_lakh():
    assert total("old", gross_salary=550_000) == 0


def test_rejects_unknown_year_and_monthly_salary():
    with pytest.raises(ValueError):
        income_tax(gross_salary=3_000_000, financial_year="2024-25")
    with pytest.raises(ValueError):
        income_tax(gross_salary=50_000, hra_received=600_000)


def test_monthly_inputs_are_multiplied_by_the_tool():
    a = income_tax(gross_salary=3_000_000, hra_received_monthly=50_000, rent_paid_monthly=40_000, basic_da_monthly=100_000,
                   city="Delhi", regime="old")["result"]["by_regime"]["old"]
    b = income_tax(gross_salary=3_000_000, hra_received=600_000, rent_paid=480_000, basic_da=1_200_000, city="Delhi",
                   regime="old")["result"]["by_regime"]["old"]
    assert a == b


def test_tax_question_routes_to_income_tax_and_statement_tax_rate_does_not():
    from finbot.routing import detect
    q = "if my salary is 30 lakhs (= 3000000) per annum, and i have 50000 hra per month, how much should i pay in taxes"
    assert "income_tax" in detect(q, True, False)
    assert "income_tax" not in detect("Sales are $19,073 and the tax rate is 23 percent. What is net income?", True, True)


def test_registry_tool_runs_and_grounding_rejects_invented_age():
    import json
    from finbot.graph import ungrounded_values
    from finbot.tools.registry import TOOLS_BY_NAME
    out = json.loads(TOOLS_BY_NAME["income_tax"].invoke({"gross_salary": 3000000, "hra_received_monthly": 50000}))
    assert out["result"]["by_regime"]["new"]["total_tax"] == 475_800
    q = "salary 30 lakhs (= 3000000) per annum, 50000 hra per month, how much tax"
    assert ungrounded_values({"gross_salary": 3000000, "hra_received_monthly": 50000}, q, "income_tax") == []
    assert ungrounded_values({"gross_salary": 3000000, "age": 45}, q, "income_tax") == ["age=45"]
