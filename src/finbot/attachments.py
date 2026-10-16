"""Turn an uploaded CSV/Excel statement into a plain table the model can read (SPEC 7.1)."""
import io

import pandas as pd

MAX_ROWS, MAX_COLS = 80, 12


def table_text(name: str, data: bytes) -> str:
    lower = name.lower()
    if lower.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(data))
    elif lower.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(data))
    elif lower.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")[:8000]
    else:
        raise ValueError("Upload a CSV, Excel or text file.")
    df = df.dropna(how="all").dropna(axis=1, how="all").iloc[:MAX_ROWS, :MAX_COLS].fillna("")
    cols = [str(c) if not str(c).startswith("Unnamed") else "" for c in df.columns]

    def cell(v):
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v).strip()

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)
