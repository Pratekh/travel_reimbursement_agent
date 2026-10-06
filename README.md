# Travel Reimbursement Approval Agent — Streamlit

This repository implements the Travel Reimbursement Approval Agent assignment as a
working Streamlit application with a LangGraph orchestration layer.

## Run locally

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### macOS / Linux

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app works without an OpenAI API key. LangGraph is used for the workflow and the
policy calculations are deterministic for reproducibility.

## Optional GenAI layer

Set:

```bash
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4o-mini
```

The LLM is used only to enrich the explanation. It does not control monetary calculations
or override the policy engine.

## Menu

- Dashboard
- Evaluate Claim
- Claims & Results
- Agent Trace
- Design Notes

## Supported input

1. Assignment's five sample claims
2. JSON claim / list of claims
3. CSV

CSV columns:

`claim_id, employee, trip_start, trip_end, submitted, category, description, amount, receipt_attached`

Optional columns:

`nights, days`

## Expected sample decisions

- CLM-001: APPROVE — $1,110.00
- CLM-002: REJECT — $0.00
- CLM-003: PARTIAL_APPROVE — $840.00 approved, $100.00 deducted
- CLM-004: MANUAL_REVIEW
- CLM-005: MANUAL_REVIEW

## Assignment alignment

The implementation covers:
- claim intake
- policy grounding
- multiple meaningful tools
- LangGraph agentic orchestration
- structured output
- manual review
- dashboard
- audit trace
- confidence
- sample claims
- design notes
