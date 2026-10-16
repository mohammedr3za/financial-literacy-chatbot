# FinBot — rules for every session

1. **Read `SPEC.md` and `PLAN.md` before doing any work.** `SPEC.md` is the source of truth.
2. **Work only on the current milestone,** which is the first one in `PLAN.md` with unchecked boxes. Do not start later milestones early.
3. **Tick a box only after its acceptance check has passed,** and record the evidence (command and result) in the PLAN evidence log.
4. **Change `SPEC.md` first.** Any change to scope, model, architecture, or tool contracts goes into `SPEC.md` with a dated changelog line before any code changes.
5. **Non-goals in SPEC section 11 are off-limits** unless the user changes the spec.
6. **Never commit** books, PDFs, `kb/raw/`, `kb/index/`, `.env`, secrets, passwords, or fine-tune data.
7. **Tools stay deterministic:** no `eval` or `exec`, and every number in an answer comes from a tool.
8. **Run tests with `uv run pytest`.** The tests must not need the model; use a fake chat model.
