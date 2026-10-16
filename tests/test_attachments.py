from finbot.attachments import table_text
from finbot.precheck import precheck


def test_csv_becomes_pipe_table_and_is_seen_as_statement():
    csv = b"Item,FY25,FY26\nCurrent assets,800,950\nCurrent liabilities,400,420\nInventory,300,310\nCash,120,140\n"
    text = table_text("bs.csv", csv)
    assert text.splitlines()[0] == "| Item | FY25 | FY26 |"
    assert "| Current assets | 800 | 950 |" in text
    assert precheck("Analyse this balance sheet\n" + text).flags["statement_data"]
