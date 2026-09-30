# DEV - Decision Engine for Verdicts

A compact decision model: **typed questions in, typed answers with probabilities and confidence out.**
Instead of generating text, DEV reads a support ticket once and returns a department (choice),
an urgency (score) and an escalate flag (yes/no) in a single forward pass. Low-confidence
cases are routed to `review` instead of being auto-handled.

> Status: v0.1 - schema-driven multi-head model, training loop, confidence-gated API.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .
pytest -q                     # sanity tests (no downloads needed)
python -m dev.train --epochs 3   # downloads MiniLM once, trains on synthetic tickets
uvicorn api.main:app --reload
```
Then try it:
```bash
curl -X POST localhost:8000/decide -H "Content-Type: application/json" \
  -d '{"state": "I was charged twice this month, urgent!"}'
```

## How it works
`schema.yaml` defines the questions -> a shared MiniLM encoder reads the text -> one small head per
question -> logits become probabilities -> confidence decides `auto` vs `review`.

## Roadmap
- v0.2 temperature-scaling calibration + reliability plots
- v0.3 LLM fallback cascade + cost/latency benchmark
- v1.0 audit module (paraphrase + counterfactual tests), dashboard, model card

## Limitations
v0.1 trains on synthetic data, so its accuracy numbers are not meaningful yet. Replace
`data/samples/tickets.jsonl` with a real labeled dataset. Do not use for decisions about people.
